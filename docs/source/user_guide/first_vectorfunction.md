(tutorial-first-vectorfunction)=
# Your first VectorFunction

A **VectorFunction** is the object used to describe *everything* passed to
Tycho — dynamics, constraints, and objectives are all VectorFunctions. This
tutorial teaches you to build them by composition, starting from a single
symbolic input and working up to a complete set of spacecraft dynamics. By the
end you will be able to read, write, and evaluate VectorFunctions of arbitrary
complexity, and you will have seen that **every VectorFunction differentiates
itself** — its Jacobian and Hessian are computed automatically.

This is a hands-on, learning-oriented walkthrough. For the *why* behind the
design — CRTP, expression templates, vectorization — read
{doc}`The VectorFunction model </user_guide/vectorfunctions>`. For the complete
catalog of every type and function, see the
{doc}`Python </reference/python/vector_functions>` and
{doc}`C++ </reference/cpp/vector_functions>` references.

Every `{doctest}` block below (the ones showing `>>>` prompts) is executed as
part of Tycho's test suite, so the results shown are real. The C++ equivalents
appear alongside in tabs; those are illustrative fragments that share context
across steps (each is compile-checked, but not run here). Each step builds on
the previous one; run them in order in a REPL or script to follow along.

:::{note}
`.compute(...)` returns a NumPy array. To keep results tidy in this tutorial we
display them with `.round(n).tolist()`, which rounds and converts to an ordinary
Python list. You do not need this in real code — operate on the array directly.
:::

## Setup

Import the module. The Python VectorFunction API is in
`tychopy.vector_functions`; we alias it to `vf`.

```{doctest}
>>> import numpy as np
>>> from tychopy import vector_functions as vf
```

## 1. Arguments: the symbolic input

Every VectorFunction is a map $f : \mathbb{R}^n \to \mathbb{R}^m$. You start by
declaring the input with {py:class}`~tychopy.vector_functions.Arguments`, which
represents the symbolic input vector. An `Arguments(n)` on its own is the
identity map $\mathbb{R}^n \to \mathbb{R}^n$.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> a = vf.Arguments(3)
>>> a.input_rows(), a.output_rows()
(3, 3)
>>> a.compute([5.0, 6.0, 7.0]).round(6).tolist()
[5.0, 6.0, 7.0]
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::vf;

auto a = Arguments<3>();              // identity map R^3 -> R^3
Eigen::Vector3d p(5.0, 6.0, 7.0);
auto fx = a.compute(p);              // -> [5, 6, 7]
```
:::
::::

Elements and sub-vectors are extracted from the input by **indexing** and
**slicing**. A single index gives a scalar function (one output); a slice or
`head`/`tail`/`segment` gives a sub-vector. `tolist()` unpacks the whole input
into individual scalar functions — a convenient way to name the input variables.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> x, y, z = a.tolist()             # three scalar functions
>>> x.compute([5.0, 6.0, 7.0]).round(6).tolist()
[5.0]
>>> a[0:2].compute([5.0, 6.0, 7.0]).round(6).tolist()
[5.0, 6.0]
>>> a.head(2).compute([5.0, 6.0, 7.0]).round(6).tolist()
[5.0, 6.0]
>>> a.tail(1).compute([5.0, 6.0, 7.0]).round(6).tolist()
[7.0]
>>> a.segment(1, 1).compute([5.0, 6.0, 7.0]).round(6).tolist()
[6.0]
```
:::
:::{tab-item} C++
```cpp
auto x  = a.coeff<0>();              // scalar element  R^3 -> R
auto y  = a.coeff<1>();
auto z  = a.coeff<2>();
auto h2 = a.head<2>();               // first two       R^3 -> R^2
auto t1 = a.tail<1>();               // last one
auto s1 = a.segment<1, 1>();         // size 1 at offset 1
```
:::
::::

`x`, `y`, and `z` are *functions* of the input, not numbers. Nothing is computed
until `.compute()` is called; the operators assemble a symbolic expression that
is evaluated later.

## 2. Scalar expressions

Scalar functions support the full set of arithmetic operators (`+`, `-`, `*`,
`/`, `**`, unary `-`). Combining them builds a new VectorFunction; still nothing
is evaluated until you call `.compute()`.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> f = vf.sin(x) * y + z**2
>>> f.compute([1.0, 2.0, 3.0]).round(6).tolist()
[10.682942]
```
:::
:::{tab-item} C++
```cpp
auto f = sin(x) * y + z * z;
auto fx = f.compute(p);              // -> [10.682942]
```
:::
::::

Because `f` carries its own derivatives, you can ask for the Jacobian at any
point. For a scalar function this is a $1 \times n$ row — the gradient
transposed. Here $\partial f/\partial x = \cos(x)\,y$, $\partial f/\partial y =
\sin(x)$, $\partial f/\partial z = 2z$, evaluated at $(1, 2, 3)$:

```{doctest}
>>> f.jacobian([1.0, 2.0, 3.0]).round(4).tolist()
[[1.0806, 0.8415, 6.0]]
```

## 3. Elementwise math

Tycho provides the usual elementwise math functions, each taking a scalar
VectorFunction and returning a new one:

`sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, the hyperbolic
variants, `exp`, `log`, `sqrt`, `squared`, `pow`, `abs`, and `sign`.

Because **the chain rule is applied automatically**, defining
$\rho = \sqrt{x^2 + y^2}$ and requesting its Jacobian at $(3, 4)$ gives
$\partial\rho/\partial x,\ \partial\rho/\partial y = x/\rho,\ y/\rho = 0.6,\ 0.8$
with no derivative code:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> rho = vf.sqrt(x**2 + y**2)
>>> rho.compute([3.0, 4.0, 0.0]).round(6).tolist()
[5.0]
>>> rho.jacobian([3.0, 4.0, 0.0]).round(4).tolist()
[[0.6, 0.8, 0.0]]
```
:::
:::{tab-item} C++
```cpp
auto rho = sqrt(x * x + y * y);
auto Jr  = rho.jacobian(Eigen::Vector3d(3.0, 4.0, 0.0));   // -> [0.6, 0.8, 0.0]
```
:::
::::

The whole catalog composes the same way. Stack several functions to evaluate
them together (`arctan2(y, x)` is the two-argument arctangent, matching
`numpy.arctan2`):

```{doctest}
>>> vf.stack([vf.exp(x), vf.log(y), vf.tanh(z), vf.abs(x), vf.sign(z),
...           vf.arctan2(y, x)]).compute([1.0, 2.0, 3.0]).round(6).tolist()
[2.718282, 0.693147, 0.995055, 1.0, 1.0, 1.107149]
```

The derivatives are the ones you would expect, including for the non-smooth
functions: `abs` differentiates to $\pm 1$ away from zero and `sign` is locally
constant, so its derivative is zero. At $x = 1$, $z = 3$:

```{doctest}
>>> vf.stack([vf.abs(x), vf.sign(z)]).jacobian([1.0, 2.0, 3.0]).round(4).tolist()
[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

## 4. Vectors: stacking and vector operations

So far our outputs have been scalars. To build a **vector-valued** function,
stack several functions together with
{py:func}`~tychopy.vector_functions.stack`. All the stacked pieces must share the
same input dimension; their outputs are concatenated top to bottom.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> s = vf.stack([x**2, y + z, x * y])
>>> s.compute([1.0, 2.0, 3.0]).round(6).tolist()
[1.0, 5.0, 2.0]
```
:::
:::{tab-item} C++
```cpp
auto s = StackedOutputs{x * x, y + z, x * y};   // R^3 -> R^3
```
:::
::::

Once you have a vector-valued function, the **vector operations** apply. These
are the workhorses of trajectory dynamics — norms, normalization, and the
inner/outer products. Build a 6-vector input, take the first three components as
a position `r` and the last three as a velocity `v`, and evaluate at
$r = (3, 0, 4)$, $v = (1, 0, 0)$:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> a6 = vf.Arguments(6)
>>> r = a6.head(3)
>>> v = a6.tail(3)
>>> pt = [3.0, 0.0, 4.0, 1.0, 0.0, 0.0]
>>> r.norm().compute(pt).round(6).tolist()             # ||r||
[5.0]
>>> r.squared_norm().compute(pt).round(6).tolist()     # ||r||^2
[25.0]
>>> r.dot(v).compute(pt).round(6).tolist()             # r . v
[3.0]
>>> r.cross(v).compute(pt).round(6).tolist()           # r x v
[0.0, 4.0, 0.0]
>>> r.normalized().compute(pt).round(6).tolist()       # r / ||r||
[0.6, 0.0, 0.8]
>>> r.normalized_power3().compute(pt).round(6).tolist()  # r / ||r||^3
[0.024, 0.0, 0.032]
```
:::
:::{tab-item} C++
```cpp
auto a6 = Arguments<6>();
auto r  = a6.head<3>();
auto v  = a6.tail<3>();
auto speed = r.norm();                    // ||r||
auto rxv   = r.cross(v);                  // r x v
auto rhat  = r.normalized();              // r / ||r||
auto grav  = r.normalized_power<3>();     // r / ||r||^3
```
:::
::::

`normalized_power3` — $r/\lVert r\rVert^3$ — appears frequently in gravity
models, as the two-body example below illustrates.

## 5. Composition: building bigger functions from smaller ones

Every operator and helper returns a VectorFunction, so expressions nest to any
depth: `rho` above already wrapped `sqrt` around `x**2 + y**2`. You can also
substitute one whole function into another by *calling* it — `outer(inner)`
evaluates `outer` at the output of `inner`, applying the chain rule at the
composition boundary. The inner function's output dimension must match the
outer's input dimension.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> inner = vf.stack([x + y, x * y])     # R^3 -> R^2  (uses x, y)
>>> u, w = vf.Arguments(2).tolist()
>>> outer = u + w                        # R^2 -> R^1  (sum of two inputs)
>>> composed = outer(inner)              # R^3 -> R^1: (x+y) + (x*y)
>>> composed.compute([2.0, 3.0, 0.0]).round(6).tolist()
[11.0]
```
:::
:::{tab-item} C++
```cpp
auto inner = StackedOutputs{x + y, x * y};   // R^3 -> R^2
auto o     = Arguments<2>();
auto outer = o.coeff<0>() + o.coeff<1>();    // R^2 -> R^1
auto composed = outer(inner);                // R^3 -> R^1
```
:::
::::

At $(2, 3)$ this is $(2 + 3) + (2 \cdot 3) = 11$. The derivative propagates
through the composition automatically — $\partial/\partial x = 1 + y = 4$ and
$\partial/\partial y = 1 + x = 3$ at $(2, 3)$:

```{doctest}
>>> composed.jacobian([2.0, 3.0, 0.0]).round(6).tolist()
[[4.0, 3.0, 0.0]]
```

Composition is what keeps large dynamics models readable: intermediate
quantities are named and passed through successive layers, and Tycho keeps the
derivatives consistent at each one.

## 6. Automatic derivatives

VectorFunctions exist to carry exact derivatives, which is what distinguishes
them from ordinary Python functions. Beyond `.jacobian()`, three more methods
provide everything an optimizer needs:

- `compute_jacobian(x)` returns the value **and** Jacobian together (cheaper than
  asking for each separately).
- `adjointgradient(x, lam)` returns $J(x)^\top \lambda$.
- `adjointhessian(x, lam)` returns the adjoint Hessian
  $\sum_i \lambda_i\, H_i(x)$.

For a scalar function with $\lambda = [1]$, the adjoint gradient is just the
gradient and the adjoint Hessian is the ordinary Hessian:

```{doctest}
>>> fx, jx = f.compute_jacobian([1.0, 2.0, 3.0])
>>> fx.round(6).tolist(), jx.round(4).tolist()
([10.682942], [[1.0806, 0.8415, 6.0]])
>>> f.adjointgradient([1.0, 2.0, 3.0], [1.0]).round(6).tolist()
[1.080605, 0.841471, 6.0]
>>> f.adjointhessian([1.0, 2.0, 3.0], [1.0]).round(4).tolist()
[[-1.6829, 0.5403, 0.0], [0.5403, 0.0, 0.0], [0.0, 0.0, 2.0]]
```

For a vector-valued function the multipliers $\lambda$ genuinely matter: the
adjoint gradient sums the rows of the Jacobian weighted by $\lambda$, and the
adjoint Hessian sums the per-output Hessians the same way. Take the 3-output
`s` from the stacking section with $\lambda = [1, 2, 0.5]$ — the `0.5` weight on
the $x\,y$ output produces the off-diagonal entries of the result:

```{doctest}
>>> s.adjointgradient([1.0, 2.0, 3.0], [1.0, 2.0, 0.5]).round(4).tolist()
[3.0, 2.5, 2.0]
>>> s.adjointhessian([1.0, 2.0, 3.0], [1.0, 2.0, 0.5]).round(4).tolist()
[[2.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

These derivatives are analytic and exact — not finite differences. If you ever
hand-write a custom derivative and want to check it, `FDDerivChecker` compares
the analytic Jacobian and Hessian against a finite-difference sweep:

```python
from tychopy.vector_functions import FDDerivChecker
FDDerivChecker(f, [1.0, 2.0, 3.0])   # prints an abs/rel error table
```

## 7. Worked example: brachistochrone dynamics

Now build something real. The brachistochrone problem has a state
$[x, y, v]$ and a control angle $\theta$. With gravity $g$, its dynamics are

$$
\dot{x} = v\sin\theta, \qquad
\dot{y} = -v\cos\theta, \qquad
\dot{v} = g\cos\theta.
$$

Lay the input out as $[x, y, v, t, \theta]$ (state, time, control) and assemble
the right-hand side as a single VectorFunction by composition — exactly the
operations from the previous sections:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> g = 9.81
>>> args = vf.Arguments(5)           # [x, y, v, t, theta]
>>> v_     = args[2]
>>> theta  = args[4]
>>> xdot = vf.sin(theta) * v_
>>> ydot = -1.0 * vf.cos(theta) * v_
>>> vdot = g * vf.cos(theta)
>>> ode = vf.stack([xdot, ydot, vdot])
>>> ode.compute([0.0, 10.0, 5.0, 0.3, 0.7]).round(6).tolist()
[3.221088, -3.824211, 7.503102]
```
:::
:::{tab-item} C++
```cpp
double g   = 9.81;
auto args  = Arguments<5>();          // [x, y, v, t, theta]
auto v_    = args.coeff<2>();
auto theta = args.coeff<4>();
auto ode   = StackedOutputs{sin(theta) * v_,
                            cos(theta) * v_ * (-1.0),
                            cos(theta) * g};
```
:::
::::

`ode` is a complete $\mathbb{R}^5 \to \mathbb{R}^3$ dynamics function, and its
$3 \times 5$ Jacobian is available immediately. Only the $v$ and $\theta$ columns
(indices 2 and 4) are non-zero, since the right-hand side depends on nothing
else:

```{doctest}
>>> ode.jacobian([0.0, 10.0, 5.0, 0.3, 0.7]).round(4).tolist()
[[0.0, 0.0, 0.6442, 0.0, 3.8242], [0.0, 0.0, -0.7648, 0.0, 3.2211], [0.0, 0.0, 0.0, 0.0, -6.3198]]
```

This is sufficient for the transcription layer to turn these dynamics into an
optimal-control problem; defining and solving a `Phase` is covered in a later
tutorial. The key point is that the dynamics *are* a VectorFunction, assembled
entirely from the building blocks above.

## 8. Worked example: two-body gravity

One more, slightly larger: the two-body problem in Cartesian coordinates. The
state is position and velocity stacked into a 6-vector, and the dynamics are
$\dot{r} = v$, $\dot{v} = -\mu\, r/\lVert r\rVert^3$. With `normalized_power3`
and `stack`, the whole right-hand side is two lines:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> mu = 1.0
>>> sc = vf.Arguments(6)             # [rx, ry, rz, vx, vy, vz]
>>> r6  = sc.head(3)
>>> v6  = sc.tail(3)
>>> rhs = vf.stack([v6, r6.normalized_power3() * (-mu)])
>>> # circular orbit: r = x_hat, v = y_hat  ->  rdot = v, vdot = -r
>>> # (the + 0.0 below clears display-only negative zeros)
>>> (rhs.compute([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) + 0.0).round(6).tolist()
[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
```
:::
:::{tab-item} C++
```cpp
double mu = 1.0;
auto sc  = Arguments<6>();            // [r (3), v (3)]
auto r6  = sc.head<3>();
auto v6  = sc.tail<3>();
auto rhs = StackedOutputs{v6, r6.normalized_power<3>() * (-mu)};
```
:::
::::

(The `+ 0.0` normalizes harmless negative zeros for display.) At the
unit circular state the result is exactly $\dot{r} = v = \hat{y}$ and
$\dot{v} = -\hat{x}$ — the centripetal acceleration of a circular orbit.

The Jacobian shows the structure of the dynamics. Its top-right $3\times3$ block
is the identity ($\partial\dot{r}/\partial v = I$) and its bottom-left block is
the gravity gradient
$\partial\dot{v}/\partial r = -\mu\bigl(I/\lVert r\rVert^3 - 3\,r r^\top/\lVert r\rVert^5\bigr)$ —
all produced by differentiating `normalized_power3`, with no hand-written
derivatives:

```{doctest}
>>> (rhs.jacobian([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) + 0.0).round(6).tolist()
[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]
```

## What you learned

- A VectorFunction is a **symbolic, differentiable map**; building one with
  operators and helpers constructs an expression that is evaluated later.
- `Arguments(n)` declares the input; indexing, slicing, and `tolist()` extract
  scalar and sub-vector functions from it.
- Arithmetic, elementwise math, vector operations (`norm`, `dot`, `cross`,
  `normalized`, `normalized_power3`), `stack`, and function composition all
  combine VectorFunctions into larger VectorFunctions.
- **Every VectorFunction differentiates itself** — `jacobian`,
  `compute_jacobian`, `adjointgradient`, and `adjointhessian` give exact
  derivatives with no extra work.
- Real dynamics (brachistochrone, two-body) are just VectorFunctions assembled
  from these same pieces.

## Next steps

- {doc}`The VectorFunction model </user_guide/vectorfunctions>` — the concepts
  and design behind what you just used.
- {doc}`Python reference </reference/python/vector_functions>` and
  {doc}`C++ reference </reference/cpp/vector_functions>` — the complete API.
- {doc}`User Guide </user_guide/index>` — task-oriented recipes, including
  authoring a custom dynamics model and using a VectorFunction in a `Phase`.
