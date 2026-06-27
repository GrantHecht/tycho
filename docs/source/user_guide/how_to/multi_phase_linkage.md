# How to link multiple phases

Many trajectories are naturally split into segments that differ in dynamics,
constraints, or control structure: an ascent followed by a descent, a coast
between two burns, a staged launch vehicle. You model each segment as its own
{py:class}`~tychopy.optimal_control.PhaseInterface` and then *link* them into a
single problem so the optimizer solves all segments simultaneously while
enforcing continuity across the boundaries.

This recipe assumes you can already build and solve a single phase — see
{doc}`Setting up a phase </user_guide/first_phase>`. For the
concepts behind transcription and linking see
{doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>`.

The C++ tabs show the equivalent builder-API calls — illustrative fragments
that assume an `ode` and `ocp` already in scope, not standalone programs.

## Build the phases

Build each phase exactly as you would a standalone problem: pick a
transcription, supply an initial guess, and attach the constraints and
objectives that belong to *that* segment. Boundary values that should hold at
the very start or very end of the whole trajectory go on the first and last
phases respectively; the interior boundaries are handled by the link
constraints below, so do **not** pin both sides of an internal seam.

::::{tab-set}
:::{tab-item} Python
```python
import tychopy.optimal_control as oc

aphase = ode.phase("LGL5", AscentIG, 128)
aphase.add_boundary_value("Front", ["h", "r", "t"], [h0, r0, 0])
aphase.add_boundary_value("Back", "gamma", 0.0)

dphase = ode.phase("LGL5", DescentIG, 128)
dphase.add_boundary_value("Back", "h", 0.0)
dphase.add_value_objective("Back", "r", -1.0)   # maximize final range
```
:::
:::{tab-item} C++
```cpp
using namespace tycho;
using namespace tycho::oc;

auto aphase = ode.phase(TranscriptionModes::LGL5, ascent_ig, 128);
Eigen::Vector3d front_vals(h0, r0, 0.0);
aphase.add_boundary_value(PhaseRegionFlags::Front, {"h", "r", "t"}, front_vals);
aphase.add_boundary_value(PhaseRegionFlags::Back, "gamma", 0.0);

auto dphase = ode.phase(TranscriptionModes::LGL5, descent_ig, 128);
dphase.add_boundary_value(PhaseRegionFlags::Back, "h", 0.0);
dphase.add_value_objective(PhaseRegionFlags::Back, "r", -1.0);   // maximize range
```
:::
::::

## Assemble the OptimalControlProblem

Create an empty {py:class}`~tychopy.optimal_control.OptimalControlProblem` and
register the phases. Add them one at a time with `add_phase`, or all at once
with `add_phases`:

::::{tab-set}
:::{tab-item} Python
```python
ocp = oc.OptimalControlProblem()
ocp.add_phase(aphase)
ocp.add_phase(dphase)
# or: ocp.add_phases([aphase, dphase])
```
:::
:::{tab-item} C++
```cpp
OptimalControlProblem ocp;
ocp.add_phase(aphase);
ocp.add_phase(dphase);
```
:::
::::

The registered phases are available afterward as `ocp.phases`. Phases can be
referred to in link calls by their integer index, by the phase object itself,
or by name.

## Link adjacent phases

The standard seam between two consecutive phases is a *forward link*:
`add_forward_link_equal_con` constrains the selected variables at the **back**
of the upstream phase to equal the same variables at the **front** of the
downstream phase. The `vars` argument indexes into the packed `[x, t, u, p]`
layout, so pass the indices you want to keep continuous across the boundary —
typically every state plus time:

::::{tab-set}
:::{tab-item} Python
```python
# States 0..4 and time continuous across the ascent/descent seam.
ocp.add_forward_link_equal_con(aphase, dphase, range(0, 5))
```
:::
:::{tab-item} C++
```cpp
// Same five vars by index, or by name: {"v", "gamma", "h", "r", "t"}.
Eigen::VectorXi link_vars(5);
link_vars << 0, 1, 2, 3, 4;
ocp.add_forward_link_equal_con(aphase, dphase, link_vars);
```
:::
::::

If a parameter must agree across phases (for example a shared design variable
like a radius), link it separately with `add_param_link_equal_con`, naming the
region it belongs to:

::::{tab-set}
:::{tab-item} Python
```python
ocp.add_param_link_equal_con(aphase, dphase, "ODEParams", "rad")
```
:::
:::{tab-item} C++
```cpp
// The builder API expresses a shared ODE parameter as a direct link between
// the ODEParams region of each phase (here raw ODE param index 0 = "rad").
Eigen::VectorXi pvar(1);
pvar << 0;
ocp.add_direct_link_equal_con(0, PhaseRegionFlags::ODEParams, pvar,
                              1, PhaseRegionFlags::ODEParams, pvar);
```
:::
::::

For non-adjacent or non-front/back couplings, `add_direct_link_equal_con`
links an explicit region/variable set of one phase to an explicit
region/variable set of another.

## Couple phases through the objective

Beyond equality seams, you can add an objective term that *depends* on
variables drawn from two phases at once with `add_link_objective`. It adds the
scalar value of a VectorFunction evaluated on
selected variables from a region of `phase0` and a region of `phase1` (and,
optionally, the shared link-parameter vector). Use it for costs that span a
seam — for example penalizing a mismatch or rewarding a combined quantity:

```python
# func reads v0 from the Back of phase0 and v1 from the Front of phase1.
ocp.add_link_objective(func, aphase, "Back", v0, dphase, "Front", v1)
```

The builder example does not use a link objective, so only the Python form is
shown here.

## Solve

Solve the whole linked problem with one call on the OCP — not on the
individual phases:

::::{tab-set}
:::{tab-item} Python
```python
ocp.optimize()              # or ocp.solve_optimize() for a rough guess
```
:::
:::{tab-item} C++
```cpp
ocp.optimize();             // or ocp.solve_optimize() for a rough guess
```
:::
::::

After it returns, pull each segment's trajectory from its own phase:

::::{tab-set}
:::{tab-item} Python
```python
Ascent = aphase.return_traj()
Descent = dphase.return_traj()
```
:::
:::{tab-item} C++
```cpp
auto Ascent  = aphase.return_traj();
auto Descent = dphase.return_traj();
```
:::
::::

A complete worked example is
`examples/python_examples/MultiPhaseCannon.py`, which links an ascent and a
descent phase, shares an ODE parameter across the seam, and maximizes range.

## See also

- {doc}`Python reference </reference/python/optimal_control>` — the full
  `OptimalControlProblem` API, including every link-constraint variant.
- {doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>` — how
  phases are transcribed and assembled into one NLP.
