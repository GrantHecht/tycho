(getting-started-first-program)=
# Your first Tycho program

You have a working `import tychopy` (if not, start at
{doc}`Installing Tycho <install>`). This page is the five-minute path from
there to a real program: you will build a set of **dynamics** as a
{py:class}`~tychopy.vector_functions.Arguments`-based VectorFunction, evaluate
it, and read off its Jacobian — which Tycho computes for you, with no
derivative code.

The whole program is below. Each `{doctest}` block runs as part of Tycho's
test suite, so the printed results are real. Paste the blocks into a Python
session (or a `.py` file) in your `tycho` environment and run them in order.

## The program

Import the VectorFunction API and NumPy:

```{doctest}
>>> import numpy as np
>>> from tychopy import vector_functions as vf
```

Build the two-body gravitational dynamics. The state is position and velocity
stacked into a 6-vector $[\,r\;(3),\ v\;(3)\,]$, and the equations of motion are
$\dot r = v$ and $\dot v = -\mu\, r / \lVert r\rVert^{3}$. With
`normalized_power3` ($r/\lVert r\rVert^{3}$) and
{py:func}`~tychopy.vector_functions.stack`, the entire right-hand side is two
lines:

```{doctest}
>>> mu = 1.0
>>> sc = vf.Arguments(6)              # symbolic input [rx, ry, rz, vx, vy, vz]
>>> r = sc.head(3)                    # position sub-vector
>>> v = sc.tail(3)                    # velocity sub-vector
>>> rhs = vf.stack([v, r.normalized_power3() * (-mu)])
```

`rhs` is a complete $\mathbb{R}^{6} \to \mathbb{R}^{6}$ dynamics function.
Evaluate it at the unit circular state $r = \hat{x}$, $v = \hat{y}$. The result
is $\dot r = v = \hat{y}$ and $\dot v = -\hat{x}$ — the centripetal
acceleration of a circular orbit:

```{doctest}
>>> x0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
>>> (rhs.compute(x0) + 0.0).round(6).tolist()
[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
```

(The `+ 0.0` only clears display-only negative zeros.)

Now the part that makes Tycho different: ask for the Jacobian. There is no
derivative code anywhere above — `rhs` differentiates itself. The top-right
$3\times3$ block is the identity ($\partial\dot r/\partial v = I$, the trivial
derivative of the velocity passthrough) and the bottom-left block is the gravity
gradient, produced automatically by differentiating `normalized_power3`:

```{doctest}
>>> (rhs.jacobian(x0) + 0.0).round(6).tolist()
[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]
```

That is a complete first program: a self-differentiating dynamics model in a
handful of lines. Every dynamics function, constraint, and objective you give
Tycho is built the same way.

## What just happened

- `vf.Arguments(6)` declared a symbolic 6-dimensional input; `.head(3)` and
  `.tail(3)` sliced out position and velocity as sub-functions.
- Operators and helpers (`normalized_power3`, `*`, `stack`) assembled a new
  VectorFunction — nothing was computed until `.compute()`.
- `.jacobian()` returned the exact analytic Jacobian. **Every VectorFunction
  carries its own derivatives**; the optimizer relies on this throughout.

## Next steps

- {doc}`Your first VectorFunction </user_guide/first_vectorfunction>`
  — the same building blocks in depth, working up from a single scalar to full
  dynamics, including `compute_jacobian`, `adjointgradient`, and
  `adjointhessian`.
- {doc}`The VectorFunction model </user_guide/vectorfunctions>` — the design
  behind what you just used.
- {doc}`User Guide </user_guide/index>` — task-oriented recipes, including using
  a VectorFunction inside a `Phase` and choosing an integrator.
