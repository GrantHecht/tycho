# How to scale variables and constraints

PSIOPT is an interior-point optimizer, and interior-point methods are sensitive
to the *conditioning* of the problem. When a state is measured in meters
(~10⁷) sitting next to one measured in radians (~1) and a control near unity,
the KKT system is badly scaled: step lengths shrink, the line search stalls,
and convergence slows or fails outright. Scaling brings every variable and
every constraint residual to a comparable order of magnitude so the optimizer
sees a well-conditioned problem.

This recipe assumes you already have a phase set up — see
{doc}`Setting up a phase </user_guide/first_phase>`. For the
conceptual picture see
{doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>`.

The C++ tabs show the equivalent builder-API calls — illustrative fragments
that assume a `phase` (or `ocp`) already in scope, not standalone programs.

## Pick a scaling mode

The {py:class}`~tychopy.optimal_control.ScaleModes` enum has three values:

- `AUTO` — scales are computed automatically from the problem.
- `CUSTOM` — you supply the per-variable scales yourself.
- `NONE` — no scaling; unit scales everywhere.

There are two complementary mechanisms, and they work together: **units**
(per-variable scales you supply) and **auto-scaling** (the optimizer deriving
constraint/objective scales). The most robust recipe for a well-understood
physical problem is to supply units *and* turn on auto-scaling.

## Enable auto-scaling

Turn auto-scaling on per phase (and on the OCP for a multi-phase problem):

::::{tab-set}
:::{tab-item} Python
```python
phase.set_auto_scaling(True)
```
:::
:::{tab-item} C++
```cpp
using namespace tycho;
using namespace tycho::oc;

phase.set_auto_scaling(true);
```
:::
::::

With nothing else, the optimizer derives scales from the problem itself. The
boolean is also the read/write attribute `phase.auto_scaling`. For a
multi-phase problem, the same call on the
{py:class}`~tychopy.optimal_control.OptimalControlProblem` propagates to every
phase:

::::{tab-set}
:::{tab-item} Python
```python
ocp.set_auto_scaling(True)        # second arg apply_to_phases defaults True
```
:::
:::{tab-item} C++
```cpp
ocp.set_auto_scaling(true);       // second arg applytophases defaults true
```
:::
::::

## Supply per-variable units

When the natural magnitudes of your variables are known — characteristic
length, time, velocity, mass — supply them as *units* (the divisor that
non-dimensionalizes each variable). The cleanest way is by variable-group
name, which requires that your ODE registered named groups (`Vgroups`) at
construction:

::::{tab-set}
:::{tab-item} Python
```python
# Named units — order-independent, reads from the variable registry.
phase.set_units(h=Lstar, v=Vstar, t=Tstar)
```
:::
:::{tab-item} C++
```cpp
// Named units — order-independent, reads from the variable registry.
phase.set_units({{"h", Lstar}, {"v", Vstar}, {"t", Tstar}});
```
:::
::::

Names you did not mention default to a unit scale. You can also pass a full
units vector in packed `[x, t, u, p]` order:

::::{tab-set}
:::{tab-item} Python
```python
phase.set_units([xstar, tstar, ustar])
```
:::
:::{tab-item} C++
```cpp
// A full units vector in packed [x, t, u, p] order (Eigen::VectorXd).
phase.set_units(units_vec);
```
:::
::::

A convenience helper on the ODE builds that vector for you from named groups,
so you can construct it once and reuse it across phases:

::::{tab-set}
:::{tab-item} Python
```python
units = ode.make_units(v=Vstar, h=Lstar, r=Lstar, rad=Lstar)
aphase.set_units(units)
dphase.set_units(units)
```
:::
:::{tab-item} C++
```cpp
auto units = ode.make_units({{"v", Vstar}, {"h", Lstar}, {"r", Lstar}, {"rad", Lstar}});
aphase.set_units(units);
dphase.set_units(units);
```
:::
::::

Supplying units and enabling auto-scaling is the usual combination: units fix
the variable magnitudes from physics you know, and auto-scaling handles the
constraint and objective residual scales on top.

## Per-constraint scale control

Every constraint- and objective-adding method on a phase takes an optional
`auto_scale` argument (a {py:class}`~tychopy.optimal_control.ScaleModes` value
or its string name, defaulting to `"auto"`), plus a numeric `scale` factor on
the bound/value methods. This gives you per-constraint flexibility — the
`ScaleType` flexibility referenced in the API — without changing the phase's
global mode. Hand-scale a single constraint when its residual is naturally far
from order one:

::::{tab-set}
:::{tab-item} Python
```python
# Heating-rate path bound: scale so the right-hand side is order 1.
phase.add_upper_func_bound("Path", QFunc(), [0, 2, 6], Qlimit, 1.0 / Qlimit)

# Opt one constraint out of auto-scaling entirely:
phase.add_boundary_value("Back", "gamma", 0.0, auto_scale="none")
```
:::
:::{tab-item} C++
```cpp
// Heating-rate path bound: scale so the right-hand side is order 1.
Eigen::VectorXi q_vars(3);
q_vars << 0, 2, 6;
phase.add_upper_func_bound(PhaseRegionFlags::Path, QFunc(), q_vars,
                           Qlimit, 1.0 / Qlimit);

// Opt one constraint out of auto-scaling entirely:
phase.add_boundary_value(PhaseRegionFlags::Back, "gamma", 0.0, ScaleModes::NONE);
```
:::
::::

## When to use NONE

Use `ScaleModes.NONE` (or `set_auto_scaling(False)` with unit scales) only when
your problem is already non-dimensionalized to order one throughout — many of
the astrodynamics examples pre-scale by characteristic quantities, in which case
additional scaling adds nothing. If the optimizer exhibits short step lengths or
stalls in early iterations, enable scaling.

Two worked examples illustrate these settings:
`examples/python_examples/MultiPhaseCannon.py` combines `make_units` with
`set_auto_scaling`, and `examples/python_examples/Delta3Launch.py` shows both
the vector and named forms of `set_units`.

## See also

- {doc}`Python reference </reference/python/optimal_control>` — `ScaleModes`,
  `set_units`, `set_auto_scaling`, and the per-constraint `auto_scale`/`scale`
  arguments.
- {doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>` — how
  the transcribed NLP is assembled and why conditioning matters.
