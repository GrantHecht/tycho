(explanation-vectorfunction)=
# The VectorFunction model

A **VectorFunction** is the central abstraction in Tycho. Every dynamics model,
constraint, and objective you hand to the optimizer is a VectorFunction, and
almost everything else in the library is built on top of them. This page
explains *what* a VectorFunction is, *why* it is designed the way it is, and
*how* the pieces fit together. It is conceptual rather than exhaustive — for the
full API surface see the {doc}`Python </reference/python/vector_functions>` and
{doc}`C++ </reference/cpp/vector_functions>` reference pages, and for a
step-by-step build see the {doc}`Tutorials </tutorials/index>`.

## What a VectorFunction is

Mathematically, a VectorFunction is a differentiable map from an input vector of
dimension `IR` (*input rows*) to an output vector of dimension `OR` (*output
rows*):

$$ f : \mathbb{R}^{\text{IR}} \rightarrow \mathbb{R}^{\text{OR}}. $$

What distinguishes it from an ordinary function is that it carries its own
**derivative machinery**. Given a point $x$ and (for second-order information) a
vector of adjoint variables $\lambda$, a VectorFunction can produce, on demand:

| Quantity | Symbol | Shape |
| --- | --- | --- |
| Value | $f(x)$ | `OR` |
| Jacobian | $J(x)$ | `OR × IR` |
| Adjoint gradient | $J(x)^\top \lambda$ | `IR` |
| Adjoint Hessian | $\sum_i \lambda_i \, H_i(x)$ | `IR × IR` |

The adjoint gradient and Hessian are exactly the quantities the optimizer needs
to assemble the Karush–Kuhn–Tucker (KKT) system: for a constraint the $\lambda_i$
are Lagrange multipliers; for an objective there is a single scaling factor.
Because each VectorFunction knows how to deliver all four quantities through one
fused call, the solver never has to reconstruct derivatives numerically or
re-walk the expression — the work is computed once, together, where the
intermediate values are already in registers.

The second defining property is that a VectorFunction is **symbolic at
construction time**. When you write `vf.sin(theta) * v` in Python, no arithmetic
happens. You are assembling a tree of C++ expression-template types that will be
evaluated later, at full native speed, inside the solver's hot loop.

::::{tab-set}
:::{tab-item} Python
```python
import tychopy as typy

vf = typy.vector_functions
args = vf.Arguments(3)
x, y, z = args.tolist()

f = vf.sin(x) * y + z**2   # builds an expression tree; nothing is evaluated yet

f.compute([1.0, 2.0, 3.0])   # sin(1)*2 + 9  ->  ~10.683
f.jacobian([1.0, 2.0, 3.0])  # 1 x 3 Jacobian
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;

auto args  = Arguments<3>();
auto x     = args.coeff<0>();
auto y     = args.coeff<1>();
auto z     = args.coeff<2>();

auto f = sin(x) * y + z * z;   // a concrete expression-template type

Eigen::Vector3d xv(1.0, 2.0, 3.0);
auto fx = f.compute(xv);       // evaluated at native speed
```
:::
::::

This split — symbolic assembly, then compiled evaluation — is the source of both
Tycho's performance and most of its design complexity. The rest of this page is
about the machinery that makes it work.

## Why CRTP

Tycho builds its function hierarchy with the **Curiously Recurring Template
Pattern (CRTP)**: each base class is parameterized by the concrete type that
derives from it. The user-facing base is

```cpp
template <class Derived, int IR, int OR,
          DenseDerivativeMode Jm = Analytic,
          DenseDerivativeMode Hm = Analytic>
struct VectorFunction;
```

and a concrete function (a `Norm`, a `Segment`, your own dynamics) inherits from
it as `struct MyFunc : VectorFunction<MyFunc, IR, OR> { ... }`.

CRTP exists here to solve one specific problem: **evaluation must be a function
template**. A VectorFunction has to be callable both with plain `double` and with
a SIMD batch type (`Eigen::Array<double, N, 1>`, discussed under
[Vectorization](#vectorization-superscalar-dispatch)), so its core method is

```cpp
template <class InType, class OutType>
void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx) const;
```

C++ does not allow virtual function templates. That rules out classical
inheritance for the hot path and leaves two realistic options: static
polymorphism (CRTP), or runtime type erasure. CRTP wins on the hot loop because:

- **Zero overhead.** Dispatch is `this->derived().compute_impl(...)`, resolved at
  compile time. There is no vtable indirection, and the compiler can inline the
  entire expression tree into a single tight kernel.
- **Global optimization.** A composite type such as
  `NestedFunction<Outer, Inner>` carries the *exact* types of its operands, so
  the optimizer sees through the whole composition and optimizes across operand
  boundaries.

CRTP is not free. Every distinct expression is a distinct C++ type, so deep
expressions produce deeply nested template types that are slow to compile, and
you cannot store two differently-shaped expressions in the same container. Tycho
accepts the compile-time cost on the hot path and recovers runtime flexibility
with a type-erasure layer ([GenericFunction](#type-erasure-genericfunction))
exactly where heterogeneity is needed.

:::{note}
CRTP is an *implementation choice*, not a mathematical necessity. C++23's
"deducing this" (P0847) could eventually replace the dispatch mechanism and
simplify the class declarations, but it would not change the expression-type
explosion, the type-erasure boundary, or the sizing parameters — those are the
genuinely hard parts of the design and are orthogonal to CRTP. The internal
developer notes discuss this trade-off in depth.
:::

### The layered hierarchy

A user type does not inherit from one class but from a *chain* of them, each
adding one concern. Reading code in `include/tycho/detail/vf/` is much easier
once you know the layers and what each contributes:

| Layer | Responsibility |
| --- | --- |
| `ComputableBase` | `compute()` dispatch and the solver-facing `constraints()` batch loop |
| `Computable` | scalar-output (`OR == 1`) specialization of the type aliases |
| `DenseFunctionBase` | the operator DSL: indexing, slicing, math methods, Jacobian products, KKT fill |
| `DenseFunction` | routing — for `OR == 1` it swaps in `DenseScalarFunctionBase`, which adds the objective/gradient interface the optimizer calls on scalar objectives |
| `DenseFirstDerivatives` / `DenseSecondDerivatives` | select the Jacobian and Hessian *strategy* (analytic, finite-difference, or Enzyme) by partial specialization |
| `VectorFunction` | the aggregate the user inherits from |

The separation matters for two reasons. First, the scalar-output specialization
is why a scalar objective gets `objective()`,
`objective_gradient()`, and `objective_gradient_hessian()` for free while a
vector constraint does not — the routing layer chooses the right base from `OR`.
Second, the derivative strategy is a *layer*, not a runtime branch: choosing
`FDiffFwd` instead of `Analytic` substitutes a different `DenseFirstDerivatives`
specialization at compile time, so a function never carries code for a mode it
does not use.

### The sizing parameters

`IR` and `OR` may be compile-time constants or `-1` for dynamic sizing. When the
dimensions are known at compile time, Eigen uses fixed-size matrices, which
enables stack allocation and better vectorization — prefer this when the
dimensions are inherent to the function (a 3-D cross product is always
`6 → 3`). Use `-1` when a function wraps arbitrary user input whose size is only
known at runtime; this is what the type-erased `GenericFunction` does.

## Primal evaluation: `compute_impl`

To author a VectorFunction you inherit from `VectorFunction<Derived, …>` and
supply, at minimum, a `compute_impl` that writes $f(x)$ into its output
argument. Everything else — indexing, the operator DSL, derivative dispatch — is
inherited.

::::{tab-set}
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
namespace tycho {

// f(x) = [x_0^2, x_1^2, ..., x_{n-1}^2]
template <int IR>
struct CwiseSquareExample : VectorFunction<CwiseSquareExample<IR>, IR, IR> {
    using Base = VectorFunction<CwiseSquareExample<IR>, IR, IR>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    static const bool is_vectorizable = true;   // safe: only Eigen array ops below

    CwiseSquareExample() {}
    CwiseSquareExample(int ir) { this->set_io_rows(ir, ir); }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x,
                             ConstVectorBaseRef<OutType> fx_) const {
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = x.array().square();
    }
};

} // namespace tycho
```
:::
:::{tab-item} Python
```python
import tychopy as typy
vf = typy.vector_functions

# In Python you rarely write a raw struct; you compose existing primitives.
# The equivalent of CwiseSquare is just an elementwise power:
args = vf.Arguments(4)
f = vf.stack([a**2 for a in args.tolist()])

f.compute([1.0, 2.0, 3.0, 4.0])   # [1, 4, 9, 16]
```
:::
::::

A few details earn their place in the contract:

- The output argument arrives `const` and is "un-`const`-ed" with
  `const_cast_derived()`. This is an Eigen idiom for writing through an
  expression reference; it does not copy.
- `compute_impl` is a *template* on the scalar type. Writing the body in terms of
  Eigen array operations (`.square()`, `.sin()`, `+`, `*`) means it works
  unchanged for `double` and for the SIMD batch type.
- `DENSE_FUNCTION_BASE_TYPES(Base)` pulls in the inherited type aliases the body
  relies on.

For C++ ODE definitions there is a more declarative path: a struct that returns a
VectorFunction expression from a static `Definition` method, wrapped by the
`BUILD_ODE_FROM_EXPRESSION` macro (defined in `<tycho/optimal_control.h>`). This
is the form used by the C++ examples.

### Input domains: skipping the zeros

A function that maps from $\mathbb{R}^{100}$ but reads only inputs $[3,8)$ has a
Jacobian with 94 zero columns. Tycho tracks, at the type level, which contiguous
blocks of the input each function actually touches — its **input domain**. During
Jacobian accumulation the system processes only the live columns. For composite
functions the domains of the operands are merged. You rarely interact with this
directly, but it is a large part of why sparse, high-dimensional collocation
problems stay fast: the cost of a Jacobian product scales with the inputs a
function *uses*, not with the size of the global decision vector.

## Derivative evaluation: modes and dispatch

The fourth and fifth template parameters, `Jm` and `Hm`, select *how* the
Jacobian and Hessian are obtained. They are independent — you can pair an
analytic Jacobian with a finite-difference Hessian — and each is chosen by
partial specialization of the derivative layers, so the cost of an unused mode is
never paid.

### Analytic (the default)

You provide hand-coded derivatives. Beyond `compute_impl`, you implement
`compute_jacobian_impl` (value + Jacobian together) and the fused
`compute_jacobian_adjointgradient_adjointhessian_impl` (value, Jacobian, adjoint
gradient, adjoint Hessian — all at once). This is the fastest mode and the one
the built-in function library uses, at the cost of deriving the math by hand.

The fused signature is what the optimizer actually calls. Computing all four
quantities in one pass lets intermediate values (already resident from the primal
evaluation) be reused for the derivatives, which is the dominant performance win
over computing them separately.

### Finite differences (`FDiffFwd`, `FDiffCentArray`)

`FDiffFwd` approximates $\partial f / \partial x_i \approx (f(x + h e_i) -
f(x))/h`; `FDiffCentArray` uses the higher-accuracy central stencil $(f(x + h
e_i) - f(x - h e_i))/(2h)$ at the cost of more evaluations. These modes need only
`compute_impl` and work for *any* function — including ones that call external
libraries whose derivatives you cannot obtain otherwise. They trade accuracy for
generality.

### Compiler automatic differentiation (`EnzymeAD`)

`EnzymeAD` differentiates the function at the LLVM IR level using the Enzyme
Clang plugin. No dual-number types are involved: Enzyme transforms the compiled
bitcode directly, so you write an ordinary `double`-arithmetic `compute_impl` and
receive machine-precision derivatives. The Jacobian uses forward-mode
(`__enzyme_fwddiff`, optionally propagating several tangents per call); the
Hessian uses forward-over-reverse. This mode is opt-in at configure time
(`ENABLE_ENZYME_AD=ON`) and is documented in detail in the build notes — it is
the right choice when a function's derivatives are painful to derive by hand but
finite-difference accuracy is not good enough.

:::{note}
The Python API does not expose the derivative-mode parameters. Every function you
build from the Python DSL is composed of analytic primitives, so its derivatives
are analytic and exact end to end. The mode parameters matter when authoring new
primitive function types in C++.
:::

To check a hand-written analytic derivative against finite differences, use
{py:func}`~tychopy.vector_functions.FDDerivChecker`, which sweeps a range of step
sizes and reports the Jacobian and Hessian error.

## Composition: expressions and operators

The reason you almost never write a raw struct in Python — and rarely in C++ — is
that VectorFunctions compose through an operator DSL. Every operator and free
function returns a *new* VectorFunction expression type; the algebra is closed.

::::{tab-set}
:::{tab-item} Python
```python
import tychopy as typy
vf = typy.vector_functions

args = vf.Arguments(6)
r = args.head(3)          # Segment: first 3 inputs
v = args.tail(3)          # Segment: last 3 inputs

# Arithmetic and elementwise math compose freely:
speed   = v.norm()                  # scalar:  ||v||
gravity = r.normalized_power3()     # vector:  r / ||r||^3
energy  = 0.5 * v.dot(v) - r.norm() # scalar

ode = vf.stack([v, gravity * (-1.0)])   # 6-output dynamics
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;

auto args = Arguments<6>();
auto r    = args.head<3>();
auto v    = args.tail<3>();

auto speed   = v.norm();
auto gravity = r.normalized_power3();
auto ode     = StackedOutputs{v, gravity * (-1.0)};
```
:::
::::

A handful of patterns underlie the whole DSL:

- **`Segment` / `Arguments`.** `Segment` extracts a contiguous slice of the
  input; `Arguments` is the identity special case that passes the whole input
  through. Both are linear and vectorizable — their Jacobians are selector
  matrices.
- **Scaling.** `f * 2.0`, `2.0 * f`, `f / 2.0`, and `-f` all produce a `Scaled`
  expression, and chained scalings are folded: `(f * 2.0) * 3.0` becomes a single
  `Scaled` with factor `6.0`, not a nested wrapper. Multiplying by a vector
  produces a per-row `RowScaled`.
- **Nesting (composition).** Calling one function on another —
  `outer(inner)` — produces a `NestedFunction` that applies the chain rule
  $J_f = J_{\text{outer}} \, J_{\text{inner}}$. When the inner function is a
  `Segment`, a partial specialization "absorbs" it so the outer function reads a
  sub-range of the input directly, with no intermediate buffer.
- **Stacking.** `vf.stack([f1, f2, f3])` concatenates outputs vertically; the
  Jacobian is the vertical concatenation of the operands' Jacobians. All operands
  must share the same input dimension.
- **Elementwise math and vector ops.** `sin`, `cos`, `sqrt`, `exp`, … each wrap a
  scalar function and chain the inner Jacobian through the known derivative of the
  operation. Norms, normalizations, dot, and cross products are first-class and
  analytically differentiated.

Because each of these returns a function with full derivative support, a deeply
composed expression is itself a VectorFunction indistinguishable, to the solver,
from a hand-written one. Vectorizability propagates too: a composite is
vectorizable exactly when all of its parts are.

### The built-in primitives at a glance

The DSL is backed by a sizeable library of pre-built primitive types under
`include/tycho/detail/vf/`, grouped by role. You compose these rather than
implement derivatives by hand:

| Category | Examples | Notes |
| --- | --- | --- |
| Structural | `Segment`, `Arguments`, `Constant`, `StackedOutputs` | selection, identity, constants, concatenation |
| Scaling | `Scaled`, `RowScaled`, `StaticScaled`, `IOScaled` | output and input/output scaling for non-dimensionalization |
| Composition | `NestedFunction` | the chain rule; absorbs an inner `Segment` |
| Elementwise math | `CwiseSin`, `CwiseCos`, `CwiseExp`, `CwiseSqrt`, … | scalar functions with diagonal Jacobians |
| Norms | `Norm`, `SquaredNorm`, `InverseNorm`, `NormPower` | scalar output; all vectorizable |
| Normalization | `Normalized`, `NormalizedPower` | `x / ‖x‖^p`, common in gravity models |
| Vector products | dot, cross, elementwise product, vector·scalar product | analytic, 3-D where applicable |
| Control flow | `Conditional`, `Comparative` | branch and min/max selection, type-erased |

Each carries the same derivative contract as a user-authored function, which is
what makes the whole algebra closed: any expression you can write is, to the
layers below it, just another VectorFunction.

## Type erasure: GenericFunction

CRTP gives every expression a distinct compile-time type. That is ideal for
speed but impossible for storage: a `Scaled<Segment<6,3,0>>` and a
`NestedFunction<Norm<3>, Segment<6,3,0>>` are unrelated types and cannot share a
container, a function signature, or a class member.

`GenericFunction<IR, OR>` solves this by wrapping *any* compatible expression in
a type-erased container that itself satisfies the VectorFunction interface. It is
the bridge between the compile-time world and the runtime world:

```cpp
GenericFunction<6, 3> f = some_complicated_expression;   // erased
GenericFunction<-1, -1> g = f;                            // deep copy, dynamic sizing
```

The crucial point about the cost is *where* the virtual dispatch lands. A
`GenericFunction` call is virtual at the **function** level, not the element
level — and *inside* that one virtual call the entire original CRTP expression
tree is still fully inlined. You pay a single indirect call per function
evaluation, not one per arithmetic operation.

This is exactly the boundary the solver sits behind. In Python, every expression
is erased to a `GenericFunction` the moment it is handed to the optimal-control
layer (`vf.stack`, `vf.sum`, and an ODE's registration all produce erased
functions); from there the solver calls through a uniform interface while each
call still runs the inlined kernel. The two-layer architecture — CRTP for speed,
type erasure for flexibility — is what lets the same VectorFunction be both a
zero-overhead kernel and a first-class runtime object.

## Vectorization: SuperScalar dispatch

During a solve, a single constraint or dynamics function is applied at *many*
points — once per collocation node. Rather than evaluate one point at a time,
Tycho can pack several independent inputs into a SIMD batch and evaluate them
together.

A **SuperScalar** is an Eigen fixed-size array used as the scalar type:

```cpp
template <class Scalar, int Sz>
using SuperScalarType = Eigen::Array<Scalar, Sz, 1>;
// DefaultSuperScalar is 8 doubles on AVX-512, 4 on AVX/AVX2, 2 on SSE2/NEON.
```

Packing four independent evaluations into one `Array<double, 4>` call turns four
scalar evaluations into a single SIMD evaluation:

```
scalar:       compute(x_1) ; compute(x_2) ; compute(x_3) ; compute(x_4)
superscalar:  compute( Array{x_1, x_2, x_3, x_4} )   // one SIMD call
```

Whether this happens is governed by the `is_vectorizable` flag and a small
amount of dispatch in the core base class:

- If `is_vectorizable = true`, the function's templated body is called directly
  with the batch type — Eigen's expression templates make `+`, `*`, `.sin()`,
  and friends work identically on a batch. This is the fast path.
- If `is_vectorizable = false` but the solver passes a batch anyway, the base
  class transparently unpacks the batch, calls `compute_impl` once per lane with
  plain `double`, and repacks the results. Correct, but it forfeits the SIMD
  speedup.

Set `is_vectorizable = true` when your body uses only Eigen array/matrix
operations, takes no branches on the scalar *value* (no `if (x[0] > 0)`), and
calls no `double`-only external code. A function that calls back into Python or a
scalar C library is, by definition, not vectorizable — and should leave the flag
at its default of `false`.

:::{note}
Vectorization is a batched-evaluation optimization, not a change in results. A
vectorizable function called with plain `double` still goes through the ordinary
scalar path and produces identical numbers; the batch path is taken only when the
solver decides batching is worthwhile and `enable_vectorization_` is set.
:::

## Where to go next

This page covered the model. To put it to work:

- **Reference.** The complete, curated API surface lives in the
  {doc}`Python </reference/python/vector_functions>` and
  {doc}`C++ </reference/cpp/vector_functions>` reference pages — every public
  class and free function, with signatures and notes.
- **Tutorials.** For a guided, end-to-end build of a real dynamics model and a
  small solve, start with the {doc}`Tutorials </tutorials/index>`.
- **How-to guides.** When you already know the concepts and just need the recipe
  for adding a new dynamics model, see the {doc}`How-to guides </how_to/index>`.

The two abstractions to keep in mind are the ones this page opened and closed
with: a VectorFunction is a *differentiable, symbolic* map that carries its own
derivatives, and it lives a double life — a zero-overhead compiled kernel via
CRTP, and a uniform runtime object via `GenericFunction`. Everything else is
detail in service of those two ideas.
