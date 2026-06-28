(tutorial-low-thrust-transfer)=
# Low-thrust orbit transfer

This tutorial puts together everything from the astrodynamics series and the
optimal-control series: you will build a low-thrust orbit-raising transfer using
the MEE dynamics model, set up a {py:class}`~tychopy.optimal_control.PhaseInterface`,
solve it with PSIOPT, and read the result.

It assumes you have worked through both
{doc}`Setting up a phase </tutorials/basics/your_first_phase>` (the
optimal-control tutorial) and
{doc}`Orbits and elements </tutorials/astrodynamics/orbits_and_elements>` (the
first astrodynamics tutorial). The phase and optimizer machinery is the same as
before; the only new ingredient is that the ODE comes from
{py:func}`~tychopy.astro.modified_dynamics` instead of being written by hand.

For the conceptual background on MEE and why it is the preferred element set for
low-thrust problems, read
{doc}`Astrodynamics in Tycho </explanation/astrodynamics>`.

Every `{doctest}` block below is executed as part of Tycho's test suite. The
C++ equivalents appear alongside in tabs; those are illustrative builder-API
fragments that assume the headers and `using namespace` lines shown in the
first tab (not run here). C++ does not subclass `ODEBase` — it assembles the
right-hand side as an expression and wraps it with the `ODE(...)` builder.

:::{note}
Return values of `phase.add_*` calls are integer constraint handles.  We assign
them to `_` throughout to keep the transcript clean.
:::

## Setup

```{doctest}
>>> import numpy as np
>>> from tychopy import astro
>>> from tychopy import vector_functions as vf
>>> from tychopy import optimal_control as oc
```

## The problem

We will raise a spacecraft from a circular orbit with semi-latus rectum
$p_0 = 1$ to a circular orbit of **double the size**, $p_f = 2$, using normalized
units ($\mu = 1$, distance unit = initial orbit radius, time unit chosen so that
the initial circular velocity is 1). The spacecraft has a maximum thrust
acceleration of $a_{\max} = 0.01$ (1% of the local gravitational acceleration),
applied in the RSW frame. Because the thrust is so small relative to gravity,
the spacecraft cannot transfer in a single arc — it must spiral outward over
many revolutions, which is the defining characteristic of low-thrust transfer.

In MEE the initial circular orbit is $[p, f, g, h, k, L] = [1, 0, 0,
0, 0, 0]$: the eccentricity components $f = g = 0$ mean circular, the
inclination components $h = k = 0$ mean equatorial, and the true longitude $L$
starts at zero. The target orbit has the same inclination and zero eccentricity
but twice the semi-latus rectum: $[p, f, g, h, k] = [2, 0, 0, 0, 0]$ with $L$
free (we do not care where on the final orbit the spacecraft arrives).

```{doctest}
>>> mu = 1.0
>>> p0, pf = 1.0, 2.0
>>> acc_max = 0.01
>>> mee0 = np.array([p0, 0.0, 0.0, 0.0, 0.0, 0.0])   # initial MEE state
```

## 1. Define the ODE

{py:func}`~tychopy.astro.modified_dynamics` returns a VectorFunction with input
layout $[\text{state}(6), \text{control}(3)]$ — nine inputs, six outputs. No
time slot is included because the MEE equations of motion are
time-autonomous. To wrap this function as an ODE for a
{py:class}`~tychopy.optimal_control.PhaseInterface`, which expects the full
$[\text{state}(6), t, \text{control}(3)]$ layout, we use
{py:class}`~tychopy.optimal_control.ODEArguments` to extract the state and
control sub-vectors, stack them into a nine-element input, and compose it with
the dynamics.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> class MEETransfer(oc.ODEBase):
...     def __init__(self, mu):
...         args = oc.ODEArguments(6, 3)
...         state   = args.x_vec()          # MEE state [p, f, g, h, k, L]
...         control = args.u_vec()          # RSW thrust [ur, ut, un]
...         ode_input = vf.stack([state, control])
...         ode_rhs = astro.modified_dynamics(mu)(ode_input)
...         super().__init__(ode_rhs, 6, 3)
```
:::
:::{tab-item} C++
```cpp
#include <tycho/tycho.h>
using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

// ODEArguments(6, 3, 0): full layout [p, f, g, h, k, L, t, ur, ut, un].
auto args  = ODEArguments(6, 3, 0);
auto state = args.head<6>();   // MEE state [p, f, g, h, k, L]
auto u     = args.tail<3>();   // RSW thrust [ur, ut, un]

// MEEDynamics is the C++ name for modified_dynamics (IR=9, OR=6).
auto mee_dyn = astro::MEEDynamics(mu);
auto ode_rhs = GenericFunction<-1, -1>(mee_dyn.eval(stack(state, u)));
auto ode     = ODE(ode_rhs, 6, 3)
                   .var_group("state", 0, 6)
                   .var_names({{"t", 6}})
                   .var_group("u", 7, 3);
```
:::
::::

`oc.ODEArguments(6, 3)` lays out the full ten-element ODE input
$[p, f, g, h, k, L, t, u_r, u_t, u_n]$.  `args.x_vec()` selects the six MEE
state components and `args.u_vec()` selects the three RSW control components.
Stacking them gives the nine-element vector that `modified_dynamics` expects.

The call `astro.modified_dynamics(mu)(ode_input)` composes the dynamics
VectorFunction with the nine-element selector — the resulting `ode_rhs` maps
the full ten-element ODE input to the six-element state derivative.

```{doctest}
>>> ode = MEETransfer(mu)
```

## 2. An initial guess

Direct collocation needs a starting trajectory. A constant along-track thrust
of magnitude $a_{\max}$ raises $p$ at a rate of roughly
$2 a_{\max} \sqrt{p/\mu}$ per time unit, which for $\mu = p = 1$ gives
$\dot{p} \approx 0.02$. The rate grows as the orbit expands, so doubling $p$
takes a few tens of time units spread over several revolutions.

We generate the initial guess by integrating the ODE forward with that
constant thrust. The `ode.integrator` call creates a numerical integrator; the
second and third arguments provide a VectorFunction that reads the MEE state
(indices 0–5 of the trajectory vector) and returns the constant control
$[0, a_{\max}, 0]$.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> u_prograde = vf.ConstantVector(6, np.array([0.0, acc_max, 0.0]))
>>> integ = ode.integrator(0.001, u_prograde, range(0, 6))
>>> integ.adaptive = True
```
:::
:::{tab-item} C++
```cpp
// Constant prograde thrust via the IntegratorBuilder constant-control overload.
// Adaptive step-size control is the integrator default (matching the Python
// integ.adaptive = True above); the builder exposes no separate toggle.
Eigen::Vector3d u_const(0.0, acc_max, 0.0);
auto integ = ode.integrator().step(0.001).control(u_const).build();
```
:::
::::

We integrate from the initial MEE state over eighteen time units to build a
multi-revolution spiral as the starting shape. The guess does not need to reach
the target exactly — it only has to be a reasonable spiral; the optimizer
stretches and refines it to hit $p_f$. Each row of the trajectory vector is
$[p, f, g, h, k, L, t, u_r, u_t, u_n]$.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> X0 = np.array([p0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, acc_max, 0.0])
>>> TrajIG = integ.integrate_dense(X0, 18.0, 200)
>>> len(TrajIG), len(TrajIG[0])
(200, 10)
```
:::
:::{tab-item} C++
```cpp
Eigen::VectorXd X0(10);
X0.setZero();
X0[0] = p0;        // initial semi-latus rectum
X0[8] = acc_max;   // ut
auto TrajIG = integ.integrate_dense(X0, 18.0, 200);   // 200 rows of 10 elements
```
:::
::::

## 3. Build the phase

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> phase = ode.phase("LGL3", TrajIG, 90)
```
:::
:::{tab-item} C++
```cpp
auto phase = ode.phase(TranscriptionModes::LGL3, TrajIG, 90);
```
:::
::::

90 LGL3 segments over a ~30-unit transfer — about 0.34 time units per segment
— gives enough mesh resolution to resolve the several-revolution spiral.

## 4. Boundary values, norm bound, and objective

We pin the front of the trajectory to the initial state at $t = 0$:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> _ = phase.add_boundary_value("Front", range(0, 7), list(mee0) + [0.0])
```
:::
:::{tab-item} C++
```cpp
Eigen::VectorXi front_vars(7);
front_vars << 0, 1, 2, 3, 4, 5, 6;
Eigen::VectorXd front_val(7);
front_val << p0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;   // [mee0, t = 0]
phase.add_boundary_value(PhaseRegionFlags::Front, front_vars, front_val);
```
:::
::::

For the back we fix the five shape elements $[p, f, g, h, k]$ of the target
circular equatorial orbit, leaving $L$ (the true longitude at arrival) free.
The optimizer is free to arrive anywhere on the target orbit.

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> _ = phase.add_boundary_value("Back", [0, 1, 2, 3, 4],
...                               [pf, 0.0, 0.0, 0.0, 0.0])
```
:::
:::{tab-item} C++
```cpp
Eigen::VectorXi back_vars(5);
back_vars << 0, 1, 2, 3, 4;
Eigen::VectorXd back_val(5);
back_val << pf, 0.0, 0.0, 0.0, 0.0;
phase.add_boundary_value(PhaseRegionFlags::Back, back_vars, back_val);
```
:::
::::

The control magnitude must not exceed $a_{\max}$.  Indices 7, 8, 9 of the
trajectory vector are the RSW controls $[u_r, u_t, u_n]$:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> _ = phase.add_lu_norm_bound("Path", [7, 8, 9], 0.0001, acc_max, 1.0)
```
:::
:::{tab-item} C++
```cpp
phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.0001, acc_max, 1.0);
```
:::
::::

Finally, minimize total transfer time:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> _ = phase.add_delta_time_objective(1.0)
```
:::
:::{tab-item} C++
```cpp
phase.add_delta_time_objective(1.0);
```
:::
::::

## 5. Solve

PSIOPT's console output goes directly to the terminal and does not appear here.
Print levels run from most verbose (`0`, the default — prints the full iteration
table) through progressively terser `1` and `2` to fully silent (`3`); pass
`set_print_level(3)` to keep the environment clean:

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
auto flag = phase.optimize();   // -> PSIOPT::ConvergenceFlags::CONVERGED
```
:::
::::

## 6. Read the result

The trajectory is returned as a list of rows, each containing
$[p, f, g, h, k, L, t, u_r, u_t, u_n]$.  Index 6 is time, so the optimal
transfer time is in the last row's time slot:

::::{tab-set}
:::{tab-item} Python
```{doctest}
>>> Traj = phase.return_traj()
>>> round(float(Traj[-1][6]), 2)
30.46
```
:::
:::{tab-item} C++
```cpp
auto Traj  = phase.return_traj();
double tf  = Traj.back()[6];   // optimal transfer time (index 6) -> ~30.46
```
:::
::::

About **30.5 time units** to double the orbit — roughly three full revolutions
of continuous thrust, compared with the fraction of a revolution a high-thrust
chemical burn would take. The target boundary conditions are met exactly:
$p = 2$ and $f = g = h = k = 0$ at arrival.

```{doctest}
>>> round(float(Traj[-1][0]), 4)
2.0
```

```{doctest}
>>> round(float(Traj[-1][1]), 4)
0.0
```

## 7. Visualize the trajectory

A picture makes the result concrete. `return_traj()` gives the full converged
trajectory, so a few `matplotlib` calls turn it into three views: the physical
spiral as the orbit is raised, the semi-latus rectum climbing to its target, and
the control magnitude over the transfer. Each row is
$[p, f, g, h, k, L, t, u_r, u_t, u_n]$; converting each MEE state back to
Cartesian with `astro.modified_to_cartesian` gives the spiral in the orbit plane.

```python
import matplotlib.pyplot as plt

T = np.array(Traj)                              # rows [p,f,g,h,k,L,t,ur,ut,un]
times  = T[:, 6]
p_vals = T[:, 0]
u_mag  = np.linalg.norm(T[:, 7:10], axis=1)
xy     = np.array([astro.modified_to_cartesian(row[:6], mu)[:2] for row in T])

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 3.6))

th = np.linspace(0, 2 * np.pi, 200)
ax0.plot(p0 * np.cos(th), p0 * np.sin(th), "k--", lw=0.8, label="initial orbit")
ax0.plot(pf * np.cos(th), pf * np.sin(th), color="C1", ls="--", lw=0.8, label="target orbit")
ax0.plot(xy[:, 0], xy[:, 1], color="C0", lw=1.5, label="transfer")
ax0.set(xlabel="x", ylabel="y", title="Transfer trajectory")
ax0.set_aspect("equal"); ax0.legend(fontsize=8)

ax1.plot(times, p_vals, color="C0")
ax1.axhline(pf, color="C1", ls="--", label=r"target $p$")
ax1.set(xlabel="time [TU]", ylabel=r"$p$", title="Semi-latus rectum"); ax1.legend(fontsize=8)

ax2.plot(times, u_mag, color="C0")
ax2.axhline(acc_max, color="C2", ls="--", label=r"$a_{\max}$")
ax2.set(xlabel="time [TU]", ylabel=r"$|u|$", title="Control magnitude")
ax2.set_ylim(0, acc_max * 1.3); ax2.legend(fontsize=8)

fig.tight_layout()
plt.show()
```

```{eval-rst}
.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from tychopy import astro
   from tychopy import vector_functions as vf
   from tychopy import optimal_control as oc

   mu = 1.0
   p0, pf = 1.0, 2.0
   acc_max = 0.01
   mee0 = np.array([p0, 0.0, 0.0, 0.0, 0.0, 0.0])

   class MEETransfer(oc.ODEBase):
       def __init__(self, mu):
           args = oc.ODEArguments(6, 3)
           super().__init__(astro.modified_dynamics(mu)(vf.stack([args.x_vec(), args.u_vec()])), 6, 3)

   ode = MEETransfer(mu)

   u_prograde = vf.ConstantVector(6, np.array([0.0, acc_max, 0.0]))
   integ = ode.integrator(0.001, u_prograde, range(0, 6))
   integ.adaptive = True
   X0 = np.array([p0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, acc_max, 0.0])
   TrajIG = integ.integrate_dense(X0, 18.0, 200)

   phase = ode.phase("LGL3", TrajIG, 90)
   phase.add_boundary_value("Front", range(0, 7), list(mee0) + [0.0])
   phase.add_boundary_value("Back", [0, 1, 2, 3, 4], [pf, 0.0, 0.0, 0.0, 0.0])
   phase.add_lu_norm_bound("Path", [7, 8, 9], 0.0001, acc_max, 1.0)
   phase.add_delta_time_objective(1.0)
   phase.optimizer.set_print_level(3)
   phase.optimize()
   T = np.array(phase.return_traj())

   times = T[:, 6]
   p_vals = T[:, 0]
   u_mag = np.linalg.norm(T[:, 7:10], axis=1)
   xy = np.array([astro.modified_to_cartesian(row[:6], mu)[:2] for row in T])

   fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 3.6))

   th = np.linspace(0, 2 * np.pi, 200)
   ax0.plot(p0 * np.cos(th), p0 * np.sin(th), color=_brand.FAINT, ls="--", lw=0.8, label="initial orbit")
   ax0.plot(pf * np.cos(th), pf * np.sin(th), color=_brand.STEEL, ls="--", lw=0.8, label="target orbit")
   ax0.plot(xy[:, 0], xy[:, 1], color=_brand.AMBER, lw=1.5, label="transfer")
   ax0.set(xlabel="x", ylabel="y", title="Transfer trajectory")
   ax0.set_aspect("equal")
   ax0.legend(fontsize=8)

   ax1.plot(times, p_vals, color=_brand.AMBER)
   ax1.axhline(pf, color=_brand.STEEL, ls="--", label="target p")
   ax1.set(xlabel="time [TU]", ylabel="p", title="Semi-latus rectum")
   ax1.legend(fontsize=8)

   ax2.plot(times, u_mag, color=_brand.AMBER)
   ax2.axhline(acc_max, color=_brand.STEEL_DARK, ls="--", label="a_max")
   ax2.set(xlabel="time [TU]", ylabel="|u|", title="Control magnitude")
   ax2.set_ylim(0, acc_max * 1.3)
   ax2.legend(fontsize=8)

   fig.tight_layout()
```

The optimal solution fires the engine at full throttle throughout the transfer.
This bang-bang character is the signature of a time-optimal low-thrust problem:
with only one binary decision (thrust or coast), the Pontryagin maximum
principle dictates that the maximum control magnitude is optimal at every point.

## What you learned

- {py:func}`~tychopy.astro.modified_dynamics` returns a VectorFunction for the
  MEE equations of motion.  To use it as a phase ODE, compose it with
  `oc.ODEArguments` to inject the time slot that the phase machinery requires.
- {py:class}`~tychopy.optimal_control.ODEBase` wraps any ODE right-hand side
  into an object that can spawn phases and integrators.
- `ode.integrator(dt, control_func, indices)` creates a numerical integrator
  whose control at each step is computed by `control_func` applied to the
  trajectory components at `indices`.
- `add_boundary_value` with a subset of variable indices lets you constrain
  only the orbital shape elements and leave the true longitude free at the
  target boundary.
- `add_lu_norm_bound("Path", [7, 8, 9], ...)` bounds the Euclidean norm of the
  RSW control vector along the entire trajectory.

## Next steps

- {doc}`Astrodynamics in Tycho </explanation/astrodynamics>` — the conceptual
  background on MEE, the RSW frame, and when each element set is preferable.
- {doc}`Python reference </reference/python/astrodynamics>` — the full listing
  of dynamics models and conversion VectorFunctions.
- {doc}`Python reference </reference/python/optimal_control>` — the complete
  phase and ODE API, including every constraint and objective type.
