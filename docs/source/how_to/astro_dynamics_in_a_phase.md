# How to use astrodynamics dynamics in a phase

The dynamics models in `tychopy.astro` — `modified_dynamics`, `cartesian_dynamics`,
and `crtbp_dynamics` — each return a `VectorFunction` with input layout
`[state(6), a(3)]` (IR = 9, OR = 6).  The trailing three inputs are **not**
specifically a control.  They are the *non-two-body acceleration* acting on the
spacecraft — the total perturbing acceleration beyond central-body point-mass
gravity.  That acceleration may be a control you optimize (e.g. low-thrust), a
modeled environmental perturbation (J2, atmospheric drag, solar-radiation
pressure), or the sum of several such terms.  For `modified_dynamics` it is
expressed in the RSW (radial / along-track / cross-track) frame; for
`cartesian_dynamics` and `crtbp_dynamics` it is in the same frame as the state.

This layout does **not** include a time slot.  The phase machinery, however,
expects an ODE whose input layout is `[state(6), t, a(3)]` — the time occupies
the seventh slot.  Passing a dynamics model directly to a phase therefore
misaligns the time slot and produces wrong constraint structure.

The fix is to wrap the dynamics model in an `ODEBase` subclass that uses
`ODEArguments` to extract state and control from the full phase input, stack
them into the nine-element vector the dynamics model expects, and compose the
two.  This recipe shows the pattern for MEE dynamics and notes how to adapt it
for Cartesian and CR3BP dynamics.

This recipe assumes you can already build and solve a basic phase — see
{doc}`Setting up a phase </tutorials/basics/your_first_phase>`.  For the
conceptual background on the MEE dynamics model and the RSW control frame see
{doc}`Astrodynamics in Tycho </explanation/astrodynamics>`.

## 1. Define the ODE wrapper

```python
import numpy as np
from tychopy import astro
from tychopy import vector_functions as vf
from tychopy import optimal_control as oc

class MEEDynamicsODE(oc.ODEBase):
    def __init__(self, mu):
        # ODEArguments(6, 3) describes the full phase input:
        #   [p, f, g, h, k, L, t, ur, ut, un]  (10 elements)
        args = oc.ODEArguments(6, 3)
        state   = args.x_vec()    # selects [p, f, g, h, k, L]  (indices 0–5)
        control = args.u_vec()    # selects [ur, ut, un]          (indices 7–9)
        # Stack into a 9-element input and compose with the dynamics model.
        rhs = astro.modified_dynamics(mu)(vf.stack([state, control]))
        super().__init__(rhs, 6, 3)
```

`oc.ODEArguments(6, 3)` declares the ODE as having 6 state variables and 3
control variables.  `args.x_vec()` returns a VectorFunction selector for the
state, and `args.u_vec()` for the control.  `vf.stack([state, control])` forms
the 9-element input that `modified_dynamics` expects, skipping the time slot at
index 6.  The resulting `rhs` maps the full 10-element phase input to the
6-element MEE state derivative.

In this recipe the non-two-body acceleration is a pure thrust control, so it
comes straight from `args.u_vec()`.  The same three slots also accept a modeled
perturbation, or thrust plus a perturbation summed together — see
[Adding perturbations](#adding-perturbations) below.

```python
mu = 1.0    # normalized gravitational parameter
ode = MEEDynamicsODE(mu)
```

## 2. Build an initial guess

`ode.integrator` creates a numerical integrator driven by a constant or
trajectory-varying control.  Here a constant along-track thrust provides a
rough initial guess for an orbit-raising transfer:

```python
acc_max = 0.01                                          # max thrust acceleration
u_const = vf.ConstantVector(6, np.array([0.0, acc_max, 0.0]))
integ = ode.integrator(0.001, u_const, range(0, 6))
integ.adaptive = True

# Initial MEE state: circular equatorial orbit at p=1
X0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # MEE state
               0.0,                              # t=0
               0.0, acc_max, 0.0])               # control

TrajIG = integ.integrate_dense(X0, 9.0, 200)    # 200 points over 9 time units
```

Each row of `TrajIG` has the layout `[p, f, g, h, k, L, t, ur, ut, un]` —
the same 10-element layout the phase expects.

## 3. Build the phase and add constraints

```python
phase = ode.phase("LGL3", TrajIG, 50)   # 50 LGL3 segments

mee0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
pf   = 1.2   # target semi-latus rectum

# Pin the initial state and time at the front
_ = phase.add_boundary_value("Front", range(0, 7), list(mee0) + [0.0])

# Fix the target orbit shape at the back; leave true longitude L (index 5) free
_ = phase.add_boundary_value("Back", [0, 1, 2, 3, 4],
                              [pf, 0.0, 0.0, 0.0, 0.0])

# Bound the thrust magnitude along the whole trajectory
_ = phase.add_lu_norm_bound("Path", [7, 8, 9], 0.0001, acc_max, 1.0)

# Minimize total transfer time
_ = phase.add_delta_time_objective(1.0)
```

## 4. Solve and read the result

```python
phase.optimizer.set_print_level(0)
flag = phase.optimize()

Traj = phase.return_traj()
tf   = Traj[-1][6]   # time is at index 6 of each trajectory row
p_f  = Traj[-1][0]   # semi-latus rectum at arrival
```

## Adapting to other dynamics models

The same pattern works for `cartesian_dynamics` and `crtbp_dynamics` — both
have the same IR = 9, OR = 6 interface with input layout `[state(6), a(3)]`
and no time slot.  Substitute the appropriate factory function:

```python
class CartesianODE(oc.ODEBase):
    def __init__(self, mu):
        args = oc.ODEArguments(6, 3)
        rhs = astro.cartesian_dynamics(mu)(vf.stack([args.x_vec(), args.u_vec()]))
        super().__init__(rhs, 6, 3)

class CRTBPODE(oc.ODEBase):
    def __init__(self, mass_ratio):
        args = oc.ODEArguments(6, 3)
        # crtbp_dynamics takes the CR3BP mass ratio, not a gravitational parameter
        rhs = astro.crtbp_dynamics(mass_ratio)(vf.stack([args.x_vec(), args.u_vec()]))
        super().__init__(rhs, 6, 3)
```

For `crtbp_dynamics` the `mu` argument is the CR3BP **mass ratio**
$\mu = m_2 / (m_1 + m_2) \in (0, 1)$, not a gravitational parameter.

## Adding perturbations

The three acceleration slots are exactly where modeled perturbations belong.
Build the perturbing acceleration as a VectorFunction of the state and feed it
into the acceleration input — on its own, or summed with a thrust control.  For
example, to fly the orbit-raising transfer above under J2:

```python
class J2CartesianODE(oc.ODEBase):
    def __init__(self, mu, J2, Rb, north_pole):
        args = oc.ODEArguments(6, 3)
        state   = args.x_vec()
        control = args.u_vec()
        # j2_cartesian takes [rx, ry, rz, nx, ny, nz] (position + pole direction)
        # and returns the J2 acceleration in the same inertial frame as the state.
        # args.constant builds a constant whose input arity matches the phase input.
        north  = args.constant(np.array(north_pole))
        j2_acc = astro.j2_cartesian(mu, J2, Rb)(vf.stack([state.head(3), north]))
        # Total non-two-body acceleration = thrust control + J2 perturbation.
        total_accel = control + j2_acc
        rhs = astro.cartesian_dynamics(mu)(vf.stack([state, total_accel]))
        super().__init__(rhs, 6, 3)
```

Drop the `control` term for J2 with no thrust (a perturbed ballistic
propagation), or add further terms — drag, solar-radiation pressure — the same
way; they all sum into the single acceleration input.

A complete worked example is `examples/python_examples/SimpleLowThrust.py`,
which uses Cartesian dynamics to optimize a low-thrust orbit-raising transfer,
and {doc}`the low-thrust transfer tutorial </tutorials/astrodynamics/low_thrust_transfer>`,
which walks through the MEE composition end to end with a solved phase.

## See also

- {doc}`Python reference </reference/python/astrodynamics>` — `modified_dynamics`,
  `cartesian_dynamics`, `crtbp_dynamics`, `j2_cartesian`, and `non_ideal_solar_sail`.
- {doc}`Low-thrust orbit transfer tutorial </tutorials/astrodynamics/low_thrust_transfer>` —
  the complete MEE ODE wrapper pattern with a full solved phase.
- {doc}`Astrodynamics in Tycho </explanation/astrodynamics>` — conceptual
  background on all three dynamics models and the RSW control frame.
- {doc}`Direct collocation in Tycho </explanation/direct_collocation>` —
  how the phase transcription turns the ODE into defect constraints.
