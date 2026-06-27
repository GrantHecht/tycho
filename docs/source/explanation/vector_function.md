(explanation-vectorfunction)=
# The VectorFunction model

A **VectorFunction** is the central abstraction in Tycho. Every dynamics model,
constraint, and objective you hand to the optimizer is a VectorFunction, and
nearly every other subsystem is built on them. This page explains what a
VectorFunction *is*, how you *build* one, and which operators you compose with.
The deeper "why it is designed this way" material — CRTP, type erasure, SIMD
dispatch — lives in [Under the hood](#under-the-hood) at the end; you can use
VectorFunctions productively without reading it.

For a guided, hands-on build see the
{doc}`first-VectorFunction tutorial </tutorials/basics/your_first_vectorfunction>`; for the
complete API surface see the
{doc}`Python reference </reference/python/vector_functions>` and
{doc}`C++ reference </reference/cpp/vector_functions>`.

## What a VectorFunction is

A VectorFunction is a map from an input vector of dimension `IR` (*input rows*)
to an output vector of dimension `OR` (*output rows*):

$$ f : \mathbb{R}^{\text{IR}} \rightarrow \mathbb{R}^{\text{OR}}. $$

Two properties make it more than an ordinary function.

**It carries its own derivatives.** Given a point $x$ — and, for second-order
information, a vector of adjoint variables $\lambda$ — a VectorFunction produces,
on demand:

| Quantity | Symbol | Shape |
| --- | --- | --- |
| Value | $f(x)$ | `OR` |
| Jacobian | $J(x)$ | `OR × IR` |
| Adjoint gradient | $J(x)^\top \lambda$ | `IR` |
| Adjoint Hessian | $\sum_i \lambda_i \, H_i(x)$ | `IR × IR` |

These are exactly the quantities the optimizer needs to assemble its
Karush–Kuhn–Tucker (KKT) system: for a constraint the $\lambda_i$ are Lagrange
multipliers; for an objective there is a single scaling factor. The derivatives
are **analytic and exact**, not finite differences, and they are delivered
through one fused call so the solver never reconstructs them numerically.

**It is symbolic.** When you write `vf.sin(theta) * v` in Python, no arithmetic
happens. You are assembling an *expression* — a recipe — that is compiled and
evaluated later, at full native speed, inside the solver's hot loop.

::::{tab-set}
:::{tab-item} Python
```python
from tychopy import vector_functions as vf

args = vf.Arguments(3)
x, y, z = args.tolist()

f = vf.sin(x) * y + z**2   # builds an expression; nothing is evaluated yet

f.compute([1.0, 2.0, 3.0])   # sin(1)*2 + 9  ->  ~10.683
f.jacobian([1.0, 2.0, 3.0])  # 1 x 3 Jacobian
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::vf;

auto args  = Arguments<3>();
auto x     = args.coeff<0>();
auto y     = args.coeff<1>();
auto z     = args.coeff<2>();

auto f = sin(x) * y + z * z;   // a concrete expression type

Eigen::Vector3d xv(1.0, 2.0, 3.0);
auto fx = f.compute(xv);       // evaluated at native speed
```
:::
::::

This split — symbolic assembly, then compiled evaluation — is the source of
Tycho's performance, and it is invisible from the API: you build expressions,
ask for values and derivatives, and the machinery handles the rest.

## Building by composition

You almost never write a VectorFunction from scratch. Instead you start from the
symbolic input and **compose** existing pieces with operators and helper
functions. Every operator returns a *new* VectorFunction, so the algebra is
closed: anything you build is itself a VectorFunction, indistinguishable to the
solver from a hand-written one, with full derivative support intact.

A typical build looks like this:

::::{tab-set}
:::{tab-item} Python
```python
from tychopy import vector_functions as vf

args = vf.Arguments(6)
r = args.head(3)          # first 3 inputs  (position)
v = args.tail(3)          # last 3 inputs   (velocity)

# Arithmetic and vector ops compose freely:
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
using namespace tycho::vf;

auto args = Arguments<6>();
auto r    = args.head<3>();
auto v    = args.tail<3>();

auto speed   = v.norm();
auto gravity = r.normalized_power<3>();   // r / ||r||^3
auto ode     = StackedOutputs{v, gravity * (-1.0)};
```
:::
::::

A handful of patterns underlie the whole DSL:

- **Selecting inputs.** `Arguments(n)` is the symbolic input — the identity map
  $\mathbb{R}^n \to \mathbb{R}^n$. Indexing or slicing it (`args[0]`,
  `args.head(3)`, `args.tail(3)`, `args.segment(i, k)`) extracts a scalar or a
  contiguous sub-vector — a `Segment`.
- **Arithmetic.** `+`, `-`, `*`, `/`, `**`, and unary `-` combine functions and
  scale them by constants. Scalings are folded automatically: `(f * 2.0) * 3.0`
  becomes a single scaling by `6.0`.
- **Composition (nesting).** Calling one function on another — `outer(inner)` —
  substitutes `inner`'s output as `outer`'s input, applying the chain rule
  $J_f = J_{\text{outer}} \, J_{\text{inner}}$ at the boundary. The inner output
  dimension must match the outer input dimension.
- **Stacking.** `vf.stack([f1, f2, f3])` concatenates outputs top to bottom into
  a longer vector; all pieces must share the same input dimension.
- **Math and vector operators.** Elementwise math (`sin`, `exp`, …), norms,
  normalizations, dot, and cross products are all first-class and analytically
  differentiated — see the catalog below.

Because each piece returns a function with full derivative support, a deeply
composed expression differentiates itself end to end with no extra work. This is
what keeps large dynamics models readable: intermediate quantities are named and
passed through layers, and Tycho keeps the derivatives consistent at each one.

## The operator catalog

The operators below are the building blocks you compose with. The one-line
descriptions give the *meaning*; for full signatures, overloads, and per-operand
dimension rules see the
{doc}`Python reference </reference/python/vector_functions>`. In the tables, $f$
and $g$ are VectorFunctions evaluated at the same input $x$; $r$, $a$, $b$, $c$
denote vector-valued operands.

### Elementwise math

Each takes a scalar VectorFunction (output dimension 1) and returns a new scalar
VectorFunction, chaining the inner Jacobian through the known derivative of the
operation.

| Python name | Computes |
| --- | --- |
| `sin` | $\sin f$ |
| `cos` | $\cos f$ |
| `tan` | $\tan f$ |
| `sinh` | $\sinh f$ |
| `cosh` | $\cosh f$ |
| `tanh` | $\tanh f$ |
| `arcsin` | $\arcsin f$ |
| `arccos` | $\arccos f$ |
| `arctan` | $\arctan f$ |
| `arcsinh` | $\operatorname{arcsinh} f$ |
| `arccosh` | $\operatorname{arccosh} f$ |
| `arctanh` | $\operatorname{arctanh} f$ |
| `arctan2` | Two-argument arctangent $\operatorname{atan2}(y, x)$, matching `numpy.arctan2` |
| `exp` | $e^{f}$ |
| `log` | Natural logarithm $\ln f$ |
| `sqrt` | $\sqrt{f}$ |
| `squared` | $f^2$ |
| `pow` | $f^{p}$ for a constant exponent $p$ |
| `abs` | $\lvert f\rvert$ |
| `sign` | $\operatorname{sign}(f)$ (locally constant; derivative is zero) |

### Vector operations

Operations over vector-valued functions: norms, normalizations, the
inner / cross / elementwise products, and matrix and quaternion products.
`normalize` is an alias of `normalized`.

| Python name | Computes |
| --- | --- |
| `norm` | $\lVert r\rVert$ |
| `squared_norm` | $\lVert r\rVert^2$ |
| `cubed_norm` | $\lVert r\rVert^3$ |
| `inverse_norm` | $1/\lVert r\rVert$ |
| `inverse_squared_norm` | $1/\lVert r\rVert^2$ |
| `inverse_cubed_norm` | $1/\lVert r\rVert^3$ |
| `inverse_four_norm` | $1/\lVert r\rVert^4$ |
| `normalize` | $r/\lVert r\rVert$ (alias of `normalized`) |
| `normalized` | $r/\lVert r\rVert$ |
| `normalized_power2` | $r/\lVert r\rVert^2$ |
| `normalized_power3` | $r/\lVert r\rVert^3$ (the two-body gravity term) |
| `normalized_power4` | $r/\lVert r\rVert^4$ |
| `normalized_power5` | $r/\lVert r\rVert^5$ |
| `dot` | Inner product $a \cdot b$ |
| `cross` | 3-vector cross product $a \times b$ |
| `doublecross` | Iterated cross product $(a \times b) \times c$ |
| `cwise_product` | Elementwise product $a \odot b$ |
| `cwise_quotient` | Elementwise quotient $a / b$ |
| `matmul` | Matrix product $M_1\, M_2$ (or $M\,v$) of matrix-valued functions |
| `quat_product` | Hamilton product $q_1 \otimes q_2$ of two scalar-last unit quaternions (4-vectors) |
| `quat_rotate` | Rotate a 3-vector by a scalar-last unit quaternion, $q \otimes (v,0) \otimes q^{-1}$ |

### Composition and reduction

Helpers that combine several functions into one, or reduce a function's outputs.

| Python name | Meaning |
| --- | --- |
| `stack` | Concatenate the outputs of several functions vertically into one longer vector |
| `stack_scalar` | `stack` restricted to scalar-output (dimension-1) functions |
| `sum` | Elementwise sum of several equal-output functions |
| `sum_scalar` | `sum` restricted to scalar-output functions |
| `sum_elems` | Sum several scalar segment functions into a single scalar (optionally weighted) |
| `ifelse` | Select between two branches by a predicate; differentiates the active branch (piecewise differentiable) |

(vf-authoring)=
## Authoring a custom primitive (C++)

If a function you need cannot be expressed by composing existing primitives, you
can author a new one in C++. You inherit from `VectorFunction<Derived, …>` and
supply, at minimum, a `compute_impl` that writes $f(x)$ into its output; the
operator DSL, indexing, and derivative dispatch are all inherited.

::::{tab-set}
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::vf;

// f(x) = [x_0^2, x_1^2, ..., x_{n-1}^2]
template <int IR>
struct CwiseSquareExample : VectorFunction<CwiseSquareExample<IR>, IR, IR> {
    using Base = VectorFunction<CwiseSquareExample<IR>, IR, IR>;
    VF_TYPE_ALIASES(Base);

    static constexpr bool is_vectorizable = true;   // safe: only Eigen array ops below

    CwiseSquareExample() {}
    CwiseSquareExample(int ir) { this->set_io_rows(ir, ir); }

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             const Eigen::MatrixBase<OutType> &fx_) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        fx = x.array().square().matrix();
    }
};
```
:::
:::{tab-item} Python
```python
from tychopy import vector_functions as vf

# In Python you rarely write a raw struct; you compose existing primitives.
# The equivalent of CwiseSquare is just an elementwise power:
args = vf.Arguments(4)
f = vf.stack([a**2 for a in args.tolist()])

f.compute([1.0, 2.0, 3.0, 4.0])   # [1, 4, 9, 16]
```
:::
::::

By default Tycho differentiates a custom `compute_impl` analytically only if you
also provide hand-coded derivative methods; otherwise you can select a
finite-difference or compiler-AD mode. The mechanics — the `compute_impl`
contract, the derivative modes, and how dispatch chooses among them — are
covered under [Under the hood](#under-the-hood). The Python DSL never exposes
these knobs: every function you build in Python is composed of analytic
primitives, so its derivatives are analytic and exact end to end.

:::{note}
To check a hand-written analytic derivative against finite differences, use
{py:func}`~tychopy.vector_functions.FDDerivChecker`, which sweeps a range of step
sizes and reports the Jacobian and Hessian error.
:::

## Under the hood

Everything above is enough to *use* VectorFunctions. This section explains *how*
they deliver zero-overhead, self-differentiating evaluation — useful when you
read the C++ in `include/tycho/detail/vf/`, author a new primitive, or debug a
performance issue. Skip it if you only compose existing pieces.

### CRTP: static polymorphism on the hot path

Tycho builds its function hierarchy with the **Curiously Recurring Template
Pattern (CRTP)**: each base class is parameterized by the concrete type that
derives from it. The user-facing base is

```cpp
template <class Derived, int IR, int OR,
          DenseDerivativeMode Jm = Analytic,
          DenseDerivativeMode Hm = Analytic>
struct VectorFunction;
```

and a concrete function inherits as `struct MyFunc : VectorFunction<MyFunc, IR,
OR> { ... }`. CRTP exists to solve one problem: **evaluation must be a function
template**, because a VectorFunction has to be callable both with plain `double`
and with a SIMD batch type (`Eigen::Array<double, N, 1>`, see
[Vectorization](#vectorization-superscalar-dispatch)). Its core method is

```cpp
template <class InType, class OutType>
void compute_impl(const Eigen::MatrixBase<InType> &x,
                  const Eigen::MatrixBase<OutType> &fx) const;
```

C++ does not allow virtual function templates, so classical inheritance is
unavailable on the hot path; CRTP is the static-polymorphism alternative. The
benefits: dispatch resolves at compile time (no vtable, full inlining of the
whole expression tree into one tight kernel), one templated body serves every
scalar type, and a composite type carries the *exact* types of its operands so
the compiler optimizes across operand boundaries. The costs: slow compiles (every
distinct expression is a distinct type), no shared container for differently
shaped expressions, and a deep inheritance chain. Tycho accepts these costs where
the hot path dominates and recovers runtime flexibility with type erasure
([GenericFunction](#type-erasure-genericfunction)) exactly where heterogeneity is
needed.

CRTP is an implementation choice, not a mathematical necessity. Virtual functions
over a fixed scalar set would bake the supported scalar types into the base
class; type erasure everywhere would forfeit cross-expression inlining; code
generation / JIT (as in JAX or CasADi) would add a codegen step and a runtime to
maintain. Tycho picks CRTP because solver iterations run millions of evaluations
where hot-path inlining dominates, and because compile-time types catch dimension
mismatches before the program runs.

### The layered hierarchy

A user type inherits from a *chain* of bases, each adding one concern:

| Layer | Responsibility |
| --- | --- |
| `ComputableBase` (+ `DomainHolder`) | `compute()` dispatch, the solver-facing `constraints()` batch loop, and input-domain tracking |
| `DenseFunctionBase` | the operator DSL (indexing, slicing, math methods, Jacobian products, KKT fill), plus the scalar-objective interface constrained to `OR == 1` |
| `DenseFirstDerivatives` → `DenseSecondDerivatives` | select the Jacobian and Hessian *strategy* (analytic, finite-difference, or Enzyme) by partial specialization |
| `VectorFunction` | the aggregate the user inherits from |

Two consequences are worth noting. The scalar-objective methods (`objective()`,
`objective_gradient()`, `objective_gradient_hessian()`) carry a C++20
`requires(OR == 1)` constraint, so they exist *only* on scalar functions — the
optimizer can call them on an objective while a vector constraint simply does not
have them. And the derivative strategy is a *layer*, not a runtime branch:
choosing `FDiffFwd` instead of `Analytic` substitutes a different
`DenseFirstDerivatives` specialization at compile time, so a function never
carries code for a mode it does not use.

### The sizing parameters

`IR` and `OR` may be compile-time constants or `-1` for dynamic sizing. When the
dimensions are known at compile time, Eigen uses fixed-size matrices, enabling
stack allocation and better vectorization — prefer this when the dimensions are
inherent to the function (a 3-D cross product is always `6 → 3`). Use `-1` when a
function wraps arbitrary user input whose size is only known at runtime; this is
what the type-erased `GenericFunction` does.

### The `compute_impl` contract

A few details of the primal-evaluation contract from the
[authoring section](#vf-authoring) above:

- The argument types are written as `const Eigen::MatrixBase<…>&` for clarity;
  the codebase abbreviates them with the aliases `CVecRef`/`VecRef` (and
  `CMatRef`/`MatRef` for matrices), which expand to exactly these types.
- The output arrives `const` and is "un-`const`-ed" with `const_cast_derived()`,
  an Eigen idiom for writing through an expression reference; it does not copy.
- `compute_impl` is a *template* on the scalar type. Writing the body in Eigen
  array operations (`.square()`, `.sin()`, `+`, `*`) makes it work unchanged for
  `double` and for the SIMD batch type.
- `VF_TYPE_ALIASES(Base)` pulls in the inherited type aliases (`Input`,
  `Output`, `Jacobian`, …) the body relies on.

For C++ ODE definitions there is a more declarative path: a struct returning a
VectorFunction expression from a static `Definition` method, wrapped by the
`BUILD_ODE_FROM_EXPRESSION` macro (in `<tycho/optimal_control.h>`).

### Input domains: skipping the zeros

A function that maps from $\mathbb{R}^{100}$ but reads only inputs $[3,8)$ has a
Jacobian with 94 zero columns. Tycho tracks, at the type level, which contiguous
blocks of the input each function actually touches — its **input domain** — and
processes only the live columns during Jacobian accumulation. For composites the
operand domains are merged. You rarely interact with this directly, but it is a
large part of why sparse, high-dimensional collocation problems stay fast: the
cost of a Jacobian product scales with the inputs a function *uses*, not with the
size of the global decision vector.

### Derivative modes and dispatch

The fourth and fifth template parameters, `Jm` and `Hm`, select *how* the
Jacobian and Hessian are obtained. They are independent — an analytic Jacobian
can pair with a finite-difference Hessian — and each is chosen by partial
specialization, so the cost of an unused mode is never paid.

- **Analytic (the default).** You implement `compute_jacobian_impl` (value +
  Jacobian) and the fused `compute_jacobian_adjointgradient_adjointhessian_impl`
  (value, Jacobian, adjoint gradient, adjoint Hessian, all at once). The fused
  signature is what the optimizer calls; computing everything in one pass lets
  intermediate values resident from the primal evaluation be reused. This is the
  fastest mode and the one the built-in library uses.
- **Finite differences (`FDiffFwd`, `FDiffCentArray`).** `FDiffFwd` uses the
  forward stencil $(f(x + h e_i) - f(x))/h$; `FDiffCentArray` uses the
  higher-accuracy central stencil $(f(x + h e_i) - f(x - h e_i))/(2h)$ at the
  cost of more evaluations. Both need only `compute_impl` and work for *any*
  function, including ones that call external libraries.
- **Compiler automatic differentiation (`EnzymeAD`).** Differentiates the
  function at the LLVM IR level via the Enzyme Clang plugin — no dual-number
  types; you write ordinary `double` arithmetic and receive machine-precision
  derivatives. The Jacobian uses forward mode, the Hessian forward-over-reverse.
  Opt-in at configure time (`ENABLE_ENZYME_AD=ON`); documented in the build
  notes.

### Type erasure: GenericFunction

CRTP gives every expression a distinct compile-time type — ideal for speed but
impossible for storage: a `Scaled<Segment<6,3,0>>` and a `NestedFunction<Norm<3>,
Segment<6,3,0>>` are unrelated types that cannot share a container, signature, or
class member.

`GenericFunction<IR, OR>` wraps *any* compatible expression in a type-erased
container that itself satisfies the VectorFunction interface:

```cpp
GenericFunction<6, 3> f = some_complicated_expression;   // erased
GenericFunction<-1, -1> g = f;                            // deep copy, dynamic sizing
```

The key point is *where* the virtual dispatch lands. A `GenericFunction` call is
virtual at the **function** level, not the element level — and *inside* that one
virtual call the entire original CRTP expression tree is still fully inlined. You
pay a single indirect call per function evaluation, not one per arithmetic
operation.

This is the boundary the solver sits behind. In Python, every expression is
erased to a `GenericFunction` the moment it is handed to the optimal-control
layer (`vf.stack`, `vf.sum`, and an ODE's registration all produce erased
functions). The two-layer architecture — CRTP for speed, type erasure for
flexibility — lets the same VectorFunction serve as both a zero-overhead compiled
kernel and a uniformly addressable runtime object.

### Vectorization: SuperScalar dispatch

During a solve, a single constraint or dynamics function is applied at *many*
points — once per collocation node. Rather than evaluate one point at a time,
Tycho can pack several independent inputs into a SIMD batch and evaluate them
together. A **SuperScalar** is an Eigen fixed-size array used as the scalar type:

```cpp
template <class Scalar, int Sz>
using SuperScalarType = Eigen::Array<Scalar, Sz, 1>;
// DefaultSuperScalar is 8 doubles on AVX-512, 4 on AVX/AVX2, 2 on SSE2/NEON.
```

Packing four evaluations into one `Array<double, 4>` call turns four scalar
evaluations into a single SIMD evaluation:

```
scalar:       compute(x_1) ; compute(x_2) ; compute(x_3) ; compute(x_4)
superscalar:  compute( Array{x_1, x_2, x_3, x_4} )   // one SIMD call
```

Whether this happens is governed by the `is_vectorizable` flag:

- If `is_vectorizable = true`, the templated body is called directly with the
  batch type — Eigen's expression templates make `+`, `*`, `.sin()`, and friends
  work identically on a batch. This is the fast path.
- If `is_vectorizable = false` but the solver passes a batch anyway, the base
  class transparently unpacks the batch, calls `compute_impl` once per lane with
  plain `double`, and repacks. Correct, but forfeits the SIMD speedup.

Set `is_vectorizable = true` when your body uses only Eigen array/matrix
operations, takes no branches on the scalar *value* (no `if (x[0] > 0)`), and
calls no `double`-only external code. A function that calls back into Python or a
scalar C library is, by definition, not vectorizable and should leave the flag at
its default of `false`. Vectorization is a batched-evaluation optimization, not a
change in results: a vectorizable function called with plain `double` still goes
through the scalar path and produces identical numbers.

### The built-in primitives at a glance

The DSL is backed by a library of pre-built primitive types under
`include/tycho/detail/vf/`, grouped by role. You compose these rather than
implement derivatives by hand:

| Category | Examples | Notes |
| --- | --- | --- |
| Structural | `Segment`, `Arguments`, `Constant`, `StackedOutputs` | selection, identity, constants, concatenation |
| Scaling | `Scaled`, `RowScaled`, `StaticScaled`, `IOScaled` | output and input/output scaling for non-dimensionalization |
| Composition | `NestedFunction` | the chain rule; absorbs an inner `Segment` |
| Elementwise math | `CwiseSin`, `CwiseCos`, `CwiseExp`, `CwiseSqrt`, … | scalar functions with diagonal Jacobians |
| Norms | `Norm`, `SquaredNorm`, `InverseNorm`, `NormPower` | scalar output; all vectorizable |
| Normalization | `Normalized`, `NormalizedPower` | $x/\lVert x\rVert^p$, common in gravity models |
| Vector products | `FunctionDotProduct`, `FunctionCrossProduct`, `CwiseFunctionProduct`, `VectorScalarFunctionProduct` | inner / cross / elementwise / vector·scalar |
| Control flow | `GenericConditional`, `GenericComparative` | branch and min/max selection, type-erased |

Each carries the same derivative contract as a user-authored function, which is
what makes the algebra closed: any expression you write is, to the layers below
it, just another VectorFunction.

## Where to go next

- **Tutorial.** For a guided build from a single symbolic input to complete
  dynamics, start with the
  {doc}`first-VectorFunction tutorial </tutorials/basics/your_first_vectorfunction>`.
- **Reference.** Every public class and free function, with full signatures, is
  in the {doc}`Python reference </reference/python/vector_functions>` and
  {doc}`C++ reference </reference/cpp/vector_functions>`.
- **How-to guides.** When you know the concepts and just need the recipe for
  adding a new dynamics model, see the {doc}`How-to guides </how_to/index>`.

The two organizing ideas: a VectorFunction is a *differentiable, symbolic* map
that carries its own derivatives, and it operates at two levels — a zero-overhead
compiled kernel via CRTP, and a uniform runtime object via `GenericFunction`.
Everything else is detail in service of those two ideas.
</content>
</invoke>
