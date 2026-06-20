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

Every `{doctest}` block below is executed as part of Tycho's test suite.

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
$p_0 = 1$ to a circular orbit with $p_f = 1.2$, using normalized units
($\mu = 1$, distance unit = initial orbit radius, time unit chosen so that the
initial circular velocity is 1). The spacecraft has a maximum thrust
acceleration of $a_{\max} = 0.01$ (1% of the local gravitational acceleration),
applied in the RSW frame.

In MEE the initial circular orbit is simply $[p, f, g, h, k, L] = [1, 0, 0,
0, 0, 0]$: the eccentricity components $f = g = 0$ mean circular, the
inclination components $h = k = 0$ mean equatorial, and the true longitude $L$
starts at zero. The target orbit has the same inclination and zero eccentricity
but a larger semi-latus rectum: $[p, f, g, h, k] = [1.2, 0, 0, 0, 0]$ with $L$
free (we do not care where on the final orbit the spacecraft arrives).

```{doctest}
>>> mu = 1.0
>>> p0, pf = 1.0, 1.2
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
$\dot{p} \approx 0.02$ and a rough transfer time of
$0.2 / 0.02 = 10$ normalized time units.

We generate the initial guess by integrating the ODE forward with that
constant thrust. The `ode.integrator` call creates a numerical integrator; the
second and third arguments provide a VectorFunction that reads the MEE state
(indices 0–5 of the trajectory vector) and returns the constant control
$[0, a_{\max}, 0]$.

```{doctest}
>>> u_prograde = vf.ConstantVector(6, np.array([0.0, acc_max, 0.0]))
>>> integ = ode.integrator(0.001, u_prograde, range(0, 6))
>>> integ.adaptive = True
```

We integrate from the initial MEE state over nine time units — slightly longer
than the rough estimate — to ensure the guess reaches (or passes) the target
orbit.  The trajectory vector each row contains is
$[p, f, g, h, k, L, t, u_r, u_t, u_n]$.

```{doctest}
>>> X0 = np.array([p0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, acc_max, 0.0])
>>> TrajIG = integ.integrate_dense(X0, 9.0, 200)
>>> len(TrajIG), len(TrajIG[0])
(200, 10)
```

## 3. Build the phase

```{doctest}
>>> phase = ode.phase("LGL3", TrajIG, 50)
```

50 LGL3 segments over a ~9-unit transfer — about 0.18 time units per segment
— gives enough mesh resolution for a clean solve.

## 4. Boundary values, norm bound, and objective

We pin the front of the trajectory to the initial state at $t = 0$:

```{doctest}
>>> _ = phase.add_boundary_value("Front", range(0, 7), list(mee0) + [0.0])
```

For the back we fix the five shape elements $[p, f, g, h, k]$ of the target
circular equatorial orbit, leaving $L$ (the true longitude at arrival) free.
The optimizer is free to arrive anywhere on the target orbit.

```{doctest}
>>> _ = phase.add_boundary_value("Back", [0, 1, 2, 3, 4],
...                               [pf, 0.0, 0.0, 0.0, 0.0])
```

The control magnitude must not exceed $a_{\max}$.  Indices 7, 8, 9 of the
trajectory vector are the RSW controls $[u_r, u_t, u_n]$:

```{doctest}
>>> _ = phase.add_lu_norm_bound("Path", [7, 8, 9], 0.0001, acc_max, 1.0)
```

Finally, minimize total transfer time:

```{doctest}
>>> _ = phase.add_delta_time_objective(1.0)
```

## 5. Solve

PSIOPT's console output goes directly to the terminal and does not appear here;
silence it with `set_print_level(0)` to keep the environment clean:

```{doctest}
>>> phase.optimizer.set_print_level(0)
>>> flag = phase.optimize()
>>> str(flag)
'ConvergenceFlags.CONVERGED'
```

## 6. Read the result

The trajectory is returned as a list of rows, each containing
$[p, f, g, h, k, L, t, u_r, u_t, u_n]$.  Index 6 is time, so the optimal
transfer time is in the last row's time slot:

```{doctest}
>>> Traj = phase.return_traj()
>>> round(float(Traj[-1][6]), 4)
9.9817
```

About **9.98 time units** to raise the orbit. The target boundary conditions
are met exactly: $p = 1.2$ and $f = g = h = k = 0$ at arrival.

```{doctest}
>>> round(float(Traj[-1][0]), 4)
1.2
```

```{doctest}
>>> round(float(Traj[-1][1]), 4)
0.0
```

## 7. Visualize the trajectory (illustrative)

The snippet below shows how to plot the semi-latus rectum and control magnitude
over the transfer.  It is not executed by the test suite because it opens a
display window.

```python
import matplotlib.pyplot as plt

T = np.array(Traj)          # rows of [p, f, g, h, k, L, t, ur, ut, un]
times   = T[:, 6]           # column 6: time
p_vals  = T[:, 0]           # column 0: semi-latus rectum
u_mag   = np.linalg.norm(T[:, 7:10], axis=1)  # |u|

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

ax1.plot(times, p_vals)
ax1.axhline(pf, color="C1", linestyle="--", label=r"target $p$")
ax1.set(xlabel="time [TU]", ylabel=r"$p$ [AU]", title="Semi-latus rectum")
ax1.legend()

ax2.plot(times, u_mag)
ax2.axhline(acc_max, color="C2", linestyle="--", label=r"$a_{\max}$")
ax2.set(xlabel="time [TU]", ylabel=r"$|u|$", title="Control magnitude")
ax2.legend()

fig.tight_layout()
plt.show()
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
