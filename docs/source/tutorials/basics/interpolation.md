(tutorial-interpolation)=
# Interpolation

Tycho turns tabulated data into a **VectorFunction** — the same composable object
used everywhere else in the library. An interpolation table is a map from one (or
more) inputs to one (or more) outputs that you can call numerically, compose into
dynamics, constraints, and objectives, and differentiate automatically. This
tutorial covers all of Tycho's interpolation methods: local piecewise tables
(`InterpTable1D`–`InterpTable4D`), global spectral tables (`ChebTable`), and
trajectory interpolation of a solved phase (`LGLInterpTable` / `InterpFunction`).

Every `{doctest}` block below (the ones with `>>>` prompts) is executed as part of
Tycho's test suite, so the results shown are real.

:::{note}
To keep results tidy we display arrays with `.round(n).tolist()`, which rounds and
converts to an ordinary Python list, or with boolean accuracy checks. You do not
need either in real code — operate on the returned NumPy array directly.
:::

## Setup

```{doctest}
>>> import numpy as np
>>> from tychopy import vector_functions as vf
```

## 1-D tables: `InterpTable1D`

`InterpTable1D` interpolates sampled data $v(t)$ on a 1-D grid. Construct it from a
strictly increasing array of sample points `ts` and matching values `vs`. The
`kind` argument selects `"cubic"` (the default, a C¹-continuous cubic Hermite
spline) or `"linear"` (C⁰ piecewise linear).

```{doctest}
>>> ts = np.linspace(0.0, 1.0, 11)
>>> table = vf.InterpTable1D(ts, np.sin(ts), kind="cubic")
>>> table.interp(0.5).round(6).tolist()
[0.479426]
```

Calling the table numerically with `__call__(t)` is equivalent to `interp(t)`:

```{doctest}
>>> np.asarray(table(0.5)).round(6).tolist()
[0.479426]
```

### Derivatives

A table reports its own derivatives with respect to `t`. `interp_deriv1` returns
`(value, dv/dt)` and `interp_deriv2` additionally returns `d²v/dt²` — useful when
you need slope or curvature of the tabulated quantity:

```{doctest}
>>> value, dvdt = table.interp_deriv1(0.5)
>>> bool(abs(float(dvdt[0]) - np.cos(0.5)) < 1e-3)
True
>>> value, dvdt, d2vdt2 = table.interp_deriv2(0.5)
>>> bool(abs(float(d2vdt2[0]) + np.sin(0.5)) < 1e-2)
True
```

### Multi-channel tables

Pass a 2-D `vs` (rows = samples, columns = channels) to interpolate several
quantities sharing one grid. The result of `interp` is then a vector with one
entry per channel:

```{doctest}
>>> vs = np.column_stack([np.sin(ts), np.cos(ts)])   # shape (11, 2)
>>> multi = vf.InterpTable1D(ts, vs)
>>> multi.interp(0.5).round(6).tolist()
[0.479426, 0.877583]
```

### Composing into a VectorFunction

The point of a table is to drop it into a larger expression. Call the table on a
VectorFunction argument (an `Element`, `Segment`, or `ScalarFunction`) and it
becomes a VectorFunction you can compose like any other. Here a two-input argument
vector feeds its first element into the table:

```{doctest}
>>> a = vf.Arguments(2)
>>> composed = table(a[0])
>>> composed.input_rows(), composed.output_rows()
(2, 1)
```

When the table has a single output channel you can also extract a standalone
`ScalarFunction` with `.sf()`; `.vf()` returns a `VectorFunction` for any number of
channels:

```{doctest}
>>> type(table.sf()).__name__
'ScalarFunction'
>>> type(multi.vf()).__name__
'VectorFunction'
```

The plot below shows both interpolants built from the same 8-point sample of a
wiggly function. The cubic spline tracks the underlying curve closely between nodes;
the piecewise linear interpolant hugs the straight chords between them.

```{eval-rst}
.. plot:: _plots/interp_cubic_vs_linear.py
```

:::{tip}
Outside the sample grid, `InterpTable1D` **extrapolates** along the boundary
polynomial. That is convenient near the edges but grows without bound far from the
data — if you need bounded behavior outside the domain, see Chebyshev tables below.
:::

## Multi-dimensional tables: `InterpTable2D` / `3D` / `4D`

For tabulated fields of two to four inputs, use `InterpTable2D`, `InterpTable3D`,
or `InterpTable4D`. A 2-D table interpolates $z(x, y)$ from per-axis grids `xs`,
`ys` and a value array `z`.

:::{important}
The value array is **row-major** with shape `(Ny, Nx)`: rows index `ys`, columns
index `xs`. This matches the default `np.meshgrid(xs, ys)` (Cartesian `"xy"`)
output, as used below.
:::

```{doctest}
>>> xs = np.linspace(0.0, 1.0, 11)
>>> ys = np.linspace(0.0, 2.0, 21)
>>> X, Y = np.meshgrid(xs, ys)            # shape (Ny, Nx) = (21, 11)
>>> z = np.sin(X) * np.cos(Y)
>>> field = vf.InterpTable2D(xs, ys, z)
>>> bool(abs(float(np.asarray(field(0.5, 1.0))) - np.sin(0.5) * np.cos(1.0)) < 1e-4)
True
```

The filled contour below evaluates the interpolated field $z(x,y)=\sin(x)\cos(y)$
on a dense grid, showing that the table reproduces the smooth spatial variation with
high fidelity.

```{eval-rst}
.. plot:: _plots/interp_field_2d.py
```

Like the 1-D table, a 2-D table composes into a VectorFunction when called on
arguments, e.g. `field(a[0], a[1])`. `InterpTable3D` and `InterpTable4D` extend the
same pattern to three and four input axes with value arrays shaped
`(Nz, Ny, Nx)` and `(Nw, Nz, Ny, Nx)` respectively.

## Chebyshev tables: `ChebTable` (global spectral)

`ChebTable` represents data as a single **global** polynomial sampled at Chebyshev
nodes. For *smooth* data this gives spectral (faster-than-any-power) accuracy with
no Runge oscillation, and the polynomial's derivatives are exact analytic
derivatives of the fit — not finite differences.

The Chebyshev nodes for a given order and domain are available directly:

```{doctest}
>>> nodes = vf.ChebTable.cheb_points(8, -1.0, 1.0)
>>> nodes.shape
(9,)
>>> [round(float(nodes[0]), 3), round(float(nodes[-1]), 3)]
[1.0, -1.0]
```

### Building from a Python callable

The most convenient constructors are the Python helpers `cheb_from_function`
(fixed order) and `cheb_adaptive` (refines the order until a relative coefficient
tolerance is met). Both sample your callable at the Chebyshev nodes for you:

```{doctest}
>>> ct = vf.cheb_from_function(np.sin, 0.0, np.pi, 16)
>>> bool(abs(float(ct.eval(np.pi / 2)[0]) - 1.0) < 1e-9)
True
>>> ct.order, ct.output_dim
(16, 1)
```

`cheb_adaptive` doubles the order until the trailing spectral coefficients fall
below `rtol` (default `1e-12`), so you do not have to guess the order. This is the
Runge-free payoff: on the classic Runge function $1/(1+25x^2)$ — where equally
spaced polynomial interpolation diverges at the edges — the adaptive Chebyshev fit
converges:

```{doctest}
>>> runge = lambda x: 1.0 / (1.0 + 25.0 * x * x)
>>> rt = vf.cheb_adaptive(runge, -1.0, 1.0)
>>> bool(abs(float(rt.eval(0.3)[0]) - runge(0.3)) < 1e-9)
True
```

The plot below makes this concrete: an order-18 equispaced polynomial fit to the
Runge function exhibits violent Runge oscillations near the edges, while a
`ChebTable` of the same order — using Chebyshev nodes instead — tracks the
function accurately everywhere.

```{eval-rst}
.. plot:: _plots/interp_runge_chebyshev.py
```

### Analytic derivatives

Like the 1-D table, `ChebTable` reports value plus derivatives — but here the
derivatives come from differentiating the spectral polynomial exactly:

```{doctest}
>>> value, deriv1 = ct.eval_deriv1(0.0)
>>> bool(abs(float(deriv1[0]) - np.cos(0.0)) < 1e-9)     # d/dt sin = cos
True
```

### Building from values directly

If you already have function values at the Chebyshev nodes (in descending node
order, matching `cheb_points`), build the table with `from_values`. The value array
is shaped `(output_dim, order + 1)`:

```{doctest}
>>> pts = vf.ChebTable.cheb_points(16, 0.0, np.pi)
>>> vals = np.sin(pts).reshape(1, -1)        # shape (1, 17)
>>> ct2 = vf.ChebTable.from_values(vals, 0.0, np.pi, 16)
>>> bool(abs(float(ct2.eval(np.pi / 2)[0]) - 1.0) < 1e-9)
True
```

For a smooth function the convergence gap between the two methods is dramatic: the
semilogy plot below compares the maximum interpolation error on $f(x) = e^{\sin(3x)}$
over $[0, 2]$ as the number of samples $N$ grows. The cubic spline error falls as a
power law (algebraic convergence); the Chebyshev error plunges to the floating-point
noise floor in just a handful of additional nodes (spectral convergence).

```{eval-rst}
.. plot:: _plots/interp_convergence.py
```

### Clamping and composition

Unlike `InterpTable1D`, a `ChebTable` **clamps** out-of-domain queries to the
nearest endpoint rather than extrapolating — querying past `ub` returns the value
at `ub`:

```{doctest}
>>> bool(float(rt.eval(2.0)[0]) == float(rt.eval(1.0)[0]))
True
```

The plot below makes the contrast visible: the `ChebTable` is evaluated on an
extended range $[-0.3,\,1.3]$ (built on $[0,1]$) and holds flat at the endpoint
values past each edge. A degree-11 global polynomial fit to the same samples (using
numpy) shows what unrestricted extrapolation looks like — it diverges rapidly away
from the training domain.

```{eval-rst}
.. plot:: _plots/interp_clamp_vs_extrapolate.py
```

A `ChebTable` composes into a VectorFunction the same way the other tables do —
call it on an argument, or use `.vf()`:

```{doctest}
>>> cf = ct.vf()
>>> cf.input_rows(), cf.output_rows()
(1, 1)
```

:::{note}
The `nthreads` argument on the constructors and `from_values` parallelizes the
**fit/construction** step only (default `1`; `0` selects an automatic count). It
has no effect on later evaluation, which is always serial per call.
:::

## Trajectory interpolation: `LGLInterpTable` / `InterpFunction`

The tables above interpolate data you tabulate yourself. To interpolate a **solved
trajectory** — a phase's state and control history as a continuous function of time
— use `LGLInterpTable` and `InterpFunction` from `tychopy.optimal_control`. These
build a piecewise polynomial on the Legendre–Gauss–Lobatto (LGL) collocation nodes
that the transcription already produced, so the interpolant is consistent with the
collocation scheme used to solve the problem.

A typical pattern, after solving a phase, looks like:

```python
from tychopy import optimal_control as oc

# After solving a phase, get its trajectory interpolation table directly.
# A trajectory is a list of packed node vectors ordered [x, t, u, p]
# (state, time, control, params) — the same layout phases use.
table = phase.return_traj_table()

# Evaluate the full packed node vector at an arbitrary time:
node = table(t)

# Wrap selected trajectory variables (here the state, indices 0:3) as a
# VectorFunction to reuse the interpolated trajectory inside another problem's
# dynamics or constraints:
pos_of_t = oc.InterpFunction(table, range(0, 3)).vf()
```

You can also build an `LGLInterpTable` directly from trajectory data (with or
without the ODE supplying node derivatives) — see the reference for the available
constructors.

Because constructing a full phase and solution is heavier than a tutorial doctest
should carry, this section is illustrative. For a complete, runnable end-to-end
example that solves a phase and reuses its trajectory through interpolation, see the
{doc}`Minimum time to climb example </tutorials/examples/min_time_to_climb>`, and
the {py:class}`~tychopy.optimal_control.LGLInterpTable` and
{py:class}`~tychopy.optimal_control.InterpFunction` reference entries.

## Choosing a method

| | Local piecewise (`InterpTable1D`–`4D`) | Global spectral (`ChebTable`) | Trajectory (`LGLInterpTable`) |
|---|---|---|---|
| **Locality** | Local — a sample affects only nearby region | Global — every node affects every point | Piecewise on collocation segments |
| **Best data** | General / noisy / non-smooth, any sampling | Smooth, analytic functions on a fixed domain | A solved phase's state/control history |
| **Accuracy on smooth data** | Algebraic (cubic ≈ O(h⁴)) | Spectral (super-algebraic), Runge-free | Order of the LGL scheme used |
| **Outside the domain** | Extrapolates (unbounded) | **Clamps** to the endpoint | Defined on the trajectory's time span |
| **Derivatives** | Analytic from the piecewise polynomial | Exact analytic from the spectral fit | Analytic from the LGL polynomial |
| **Sampling required** | Arbitrary grid you provide | Chebyshev nodes (helpers sample for you) | Collocation nodes from the solve |
| **Typical use** | Aero/gravity lookup tables, mission data | Smooth coefficient or basis functions | Re-using a solved trajectory as a VF |

As a rule of thumb: reach for `InterpTable*` for measured or non-smooth lookup data
and when you control the grid; reach for `ChebTable` when the underlying function is
smooth and you want spectral accuracy with exact derivatives and bounded
out-of-domain behavior; and reach for `LGLInterpTable` / `InterpFunction` to turn a
solved trajectory back into a composable VectorFunction.

## Where to go next

- {doc}`VectorFunction Python reference </reference/python/vector_functions>` —
  full API for {py:class}`~tychopy.vector_functions.InterpTable1D`,
  {py:class}`~tychopy.vector_functions.InterpTable2D`,
  {py:class}`~tychopy.vector_functions.ChebTable`, and the
  {py:func}`~tychopy.vector_functions.cheb_from_function` /
  {py:func}`~tychopy.vector_functions.cheb_adaptive` helpers.
- {doc}`Optimal control Python reference </reference/python/optimal_control>` —
  {py:class}`~tychopy.optimal_control.LGLInterpTable` and
  {py:class}`~tychopy.optimal_control.InterpFunction`.
- {doc}`Your first VectorFunction </tutorials/basics/your_first_vectorfunction>` —
  how composition and automatic differentiation work, which every table inherits.
