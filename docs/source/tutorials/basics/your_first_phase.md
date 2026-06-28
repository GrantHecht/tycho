(tutorial-first-phase)=
# Setting up a phase

A **Phase** is the object that turns dynamics, constraints, and an objective into
an optimal-control problem Tycho can solve. Where
{doc}`your first VectorFunction </tutorials/basics/your_first_vectorfunction>`
covered *defining* dynamics, this tutorial builds on that foundation: you will
wrap a VectorFunction in an ODE, build a `Phase`, pin its boundary conditions, add
an objective, and **solve it with PSIOPT** — Tycho's interior-point optimizer.

The worked problem is the **brachistochrone**: find the shape of the wire
down which a bead slides, under gravity, from one point to another in the least
time. The continuous answer is a cycloid; here we recover its travel time by
direct collocation. By the end you will have a converged trajectory and will read
the optimal travel time directly from the result.

This is a hands-on, learning-oriented walkthrough. For the *why* behind the
transcription — how a continuous problem becomes a finite-dimensional NLP — read
{doc}`Direct collocation in Tycho </explanation/direct_collocation>`. For the
complete catalog of every phase method, see the
{doc}`Python reference </reference/python/optimal_control>`.

Every `{doctest}` block below (the ones showing `>>>` prompts) is executed as part
of Tycho's test suite, so the results shown are real. The C++ equivalents appear
alongside in tabs; those use the builder API and are illustrative fragments that
share context across steps (faithful to
`examples/cpp_examples/builder/brachistochrone/main.cpp`, but not run here). Each
step builds on the previous one; run them in order in a REPL or script to
reproduce the example.

:::{note}
Several phase-building methods (`add_boundary_value`, `add_lu_var_bound`,
`add_delta_time_objective`, …) return an integer handle to the constraint or
objective they create. We assign these to `_` throughout to keep the transcript
clean; you do not need to in real code.
:::

## Setup

We need three modules: NumPy for the initial guess, `vector_functions` (`vf`) to
build the dynamics, and `optimal_control` (`oc`) for the ODE and phase machinery.

```{doctest}
>>> import numpy as np
>>> from tychopy import vector_functions as vf
>>> from tychopy import optimal_control as oc
```

## 1. Define the ODE

The brachistochrone has a three-component state $[x, y, v]$ — position and speed —
and a single control $\theta$, the wire angle. With gravity $g$ the dynamics are

$$
\dot{x} = v\sin\theta, \qquad
\dot{y} = -v\cos\theta, \qquad
\dot{v} = g\cos\theta.
$$

To use these dynamics in a phase we subclass {py:class}`~tychopy.optimal_control.ODEBase`.
The pattern is always the same: declare the variable layout with
{py:class}`~tychopy.optimal_control.ODEArguments` (here 3 state variables and 1
control), pull out the symbolic pieces, assemble the right-hand side as a single
VectorFunction with `vf.stack`, and hand it to `super().__init__` along with the
state and control counts.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> class Brachistochrone(oc.ODEBase):
...     def __init__(self, g):
...         args = oc.ODEArguments(3, 1)
...         x, y, v = args.x_vec().tolist()
...         theta = args.u_var(0)
...         xdot = vf.sin(theta) * v
...         ydot = -1.0 * vf.cos(theta) * v
...         vdot = g * vf.cos(theta)
...         ode = vf.stack([xdot, ydot, vdot])
...         super().__init__(ode, 3, 1)
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

double g = 9.81;
// ODEBuilder(n_state, n_control); the lambda builds the right-hand side.
auto ode = ODEBuilder(3, 1)
               .define([g](auto &args) {
                   auto v     = args.x_var(2);
                   auto theta = args.u_var(0);
                   return stack(sin(theta) * v,
                                cos(theta) * v * (-1.0),
                                g * cos(theta));
               })
               .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
               .build();
```
:::
::::

`ODEArguments(3, 1)` lays the input out as $[x, y, v, t, \theta]$ — the three
state variables, then time, then the one control. `args.x_vec()` is the state
sub-vector and `args.u_var(0)` is the first control; everything to the right of
`vf.` is exactly the VectorFunction algebra from the previous tutorial.

Instantiate it with $g = 9.81\ \mathrm{m/s^2}$:

```{doctest}
>>> g = 9.81
>>> ode = Brachistochrone(g)
```

## 2. An initial guess

Direct collocation is a local method: it needs a starting trajectory to refine.
The guess need not be accurate — a geometrically reasonable trajectory suffices. We supply one
state-and-time-and-control vector $[x, y, v, t, \theta]$ per sample, marching
straight from the start point $(0, 10)$ toward the end point $(10, 5)$ over a
unit time interval, with a constant guess angle of one radian.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> ts = np.linspace(0.0, 1.0, 100)
>>> Xs = [[10.0 * t, 10.0 - 5.0 * t, g * t * np.cos(1.0), t, 1.0] for t in ts]
>>> len(Xs), len(Xs[0])
(100, 5)
```
:::
:::{tab-item} C++
```cpp
// The guess is a std::vector of [x, y, v, t, theta] rows.
std::vector<Eigen::VectorXd> traj;
traj.reserve(100);
for (int i = 0; i < 100; ++i) {
    const double s = static_cast<double>(i) / 99;
    Eigen::VectorXd pt(5);
    pt[0] = 10.0 * s;
    pt[1] = 10.0 - 5.0 * s;
    pt[2] = g * s * std::cos(1.0);
    pt[3] = s;
    pt[4] = 1.0;
    traj.push_back(pt);
}
```
:::
::::

Each row is the full input vector the ODE expects: state, then time, then control.

## 3. Build the phase

{py:meth}`~tychopy.optimal_control.ODEBase.phase` transcribes the ODE onto a mesh.
The first argument is the collocation scheme (`"LGL3"` — third-order
Legendre–Gauss–Lobatto), the second is the initial guess, and the third is the
number of mesh segments.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> phase = ode.phase("LGL3", Xs, 32)
```
:::
:::{tab-item} C++
```cpp
auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
```
:::
::::

The phase now holds 32 segments of LGL3 collocation, seeded from our guess. It is
not yet a well-posed problem — nothing pins where the trajectory starts or ends,
and there is no objective.

## 4. Boundary values, bounds, and the objective

We pin the problem with three kinds of condition:

- **Front boundary** — the bead starts at $x = 0$, $y = 10$, $v = 0$ at time $0$.
  `range(0, 4)` selects the first four input slots $[x, y, v, t]$.
- **Back boundary** — it must arrive at $x = 10$, $y = 5$. Slots `[0, 1]` are
  $x$ and $y$; the arrival time is free (that is what we minimize).
- **Path bound** — keep the control angle $\theta$ (slot 4) in a sane range over
  the whole trajectory.

Finally, the objective: minimize elapsed time. `add_delta_time_objective(1.0)`
adds the phase's total time, scaled by `1.0`, to the cost.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> _ = phase.add_boundary_value("Front", range(0, 4), [0.0, 10.0, 0.0, 0.0])
>>> _ = phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
>>> _ = phase.add_boundary_value("Back", [0, 1], [10.0, 5.0])
>>> _ = phase.add_delta_time_objective(1.0)
```
:::
:::{tab-item} C++
```cpp
phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                         Eigen::Vector4d(0.0, 10.0, 0.0, 0.0));
phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);
phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"},
                         Eigen::Vector2d(10.0, 5.0));
phase.add_delta_time_objective(1.0);
```
:::
::::

The problem is now fully posed: a fixed start, a fixed end position, a bounded
control angle, and a minimum-arrival-time objective.

## 5. Solve

PSIOPT prints a detailed iteration log by default. For a clean run we turn its
console output down via the `optimizer` handle, then call `optimize()`. The call
returns a convergence flag; we keep it so we can confirm the solve succeeded.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> phase.optimizer.set_print_level(3)
>>> flag = phase.optimize()
>>> str(flag)
'ConvergenceFlags.CONVERGED'
```
:::
:::{tab-item} C++
```cpp
phase.optimizer().set_print_level(3);
auto flag = phase.optimize();   // or phase.solve_optimize() for a rough guess
```
:::
::::

`CONVERGED` means PSIOPT satisfied its KKT and feasibility tolerances — the
trajectory in the phase is now the optimized one.

## 6. Read the result

{py:meth}`~tychopy.optimal_control.PhaseInterface.return_traj` returns the converged
trajectory as a list of $[x, y, v, t, \theta]$ rows. The last row is the final
collocation point, so its time component (slot 3) is the optimal travel time:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> Traj = phase.return_traj()
>>> round(float(Traj[-1][3]), 4)
1.8013
```
:::
:::{tab-item} C++
```cpp
auto Traj = phase.return_traj();
double tf = Traj.back()[3];   // optimal travel time, ~1.8013 s
```
:::
::::

About **1.8013 seconds** — the known brachistochrone minimum for this geometry.
The boundary conditions we imposed are honored exactly: the bead arrives at
$(10, 5)$ as required.

```{doctest}
>>> [round(float(c), 4) for c in Traj[-1][0:2]]
[10.0, 5.0]
```

## 7. Visualize the trajectory

`return_traj()` returns the full state and control history at every collocation
node. The two most informative views are the spatial trajectory and the optimal
control schedule, both straightforward to produce with `matplotlib`. Stack the
trajectory into a NumPy array — each row is $[x, y, v, t, \theta]$ — and slice
out the columns you need.

```python
import matplotlib.pyplot as plt
import numpy as np

T = np.array(Traj)  # rows of [x, y, v, t, theta]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.plot(T[:, 0], T[:, 1], "-o", ms=3)
ax1.set(xlabel="x", ylabel="y", title="Brachistochrone path")

ax2.plot(T[:, 3], np.degrees(T[:, 4]), "-o", ms=3, color="C1")
ax2.set(xlabel="time [s]", ylabel=r"control $\theta$ [deg]", title="Optimal control")

fig.tight_layout()
plt.show()
```

```{eval-rst}
.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from tychopy import vector_functions as vf
   from tychopy import optimal_control as oc

   class Brachistochrone(oc.ODEBase):
       def __init__(self, g):
           args = oc.ODEArguments(3, 1)
           x, y, v = args.x_vec().tolist()
           theta = args.u_var(0)
           xdot = vf.sin(theta) * v
           ydot = -1.0 * vf.cos(theta) * v
           vdot = g * vf.cos(theta)
           ode = vf.stack([xdot, ydot, vdot])
           super().__init__(ode, 3, 1)

   g = 9.81
   ode = Brachistochrone(g)
   ts = np.linspace(0.0, 1.0, 100)
   Xs = [[10.0 * t, 10.0 - 5.0 * t, g * t * np.cos(1.0), t, 1.0] for t in ts]
   phase = ode.phase("LGL3", Xs, 32)
   phase.add_boundary_value("Front", range(0, 4), [0.0, 10.0, 0.0, 0.0])
   phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
   phase.add_boundary_value("Back", [0, 1], [10.0, 5.0])
   phase.add_delta_time_objective(1.0)
   phase.optimizer.set_print_level(3)
   phase.optimize()
   T = np.array(phase.return_traj())

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
   ax1.plot(T[:, 0], T[:, 1], "-o", ms=3)
   ax1.set(xlabel="x", ylabel="y", title="Brachistochrone path")
   ax1.grid(True, alpha=0.3)

   ax2.plot(T[:, 3], np.degrees(T[:, 4]), "-o", ms=3, color=_brand.STEEL)
   ax2.set(xlabel="time [s]", ylabel=r"control $\theta$ [deg]", title="Optimal control")
   ax2.grid(True, alpha=0.3)

   fig.tight_layout()
```

The path traces the characteristic cycloid — steep at the start to build speed,
flattening toward the target — while the control angle sweeps smoothly from near
vertical to past horizontal. That smooth, monotone control is the hallmark of a
cleanly converged collocation solution.

## What you learned

- A **Phase** turns an ODE plus constraints and an objective into a solvable
  optimal-control problem.
- You define dynamics by subclassing `oc.ODEBase`: declare the layout with
  `oc.ODEArguments`, build the right-hand side as a VectorFunction, and call
  `super().__init__(ode, n_state, n_control)`.
- `ode.phase(scheme, guess, segments)` transcribes the dynamics onto a
  collocation mesh seeded from an initial guess.
- `add_boundary_value`, `add_lu_var_bound`, and `add_delta_time_objective` pin
  the start and end, bound the control, and set the cost.
- `phase.optimize()` solves with PSIOPT and returns a convergence flag;
  `return_traj()` returns the optimized trajectory as a list of
  state-and-control rows.

## Next steps

- {doc}`Direct collocation in Tycho </explanation/direct_collocation>` — how the
  continuous problem becomes the NLP you just solved.
- {doc}`Python reference </reference/python/optimal_control>` — the complete phase
  and ODE API, including every constraint and objective type.
- {doc}`Your first VectorFunction </tutorials/basics/your_first_vectorfunction>` —
  the building blocks the dynamics above are made of.
