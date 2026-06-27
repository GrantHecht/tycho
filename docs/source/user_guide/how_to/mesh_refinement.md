# How to refine the mesh adaptively

A fixed mesh chosen up front almost always under-resolves part of the
trajectory. Direct collocation approximates the dynamics with piecewise
polynomials of bounded order, so wherever the true solution bends sharply —
a control switch, an atmospheric pass, a thrust arc boundary — a coarse,
uniform mesh leaves a large discretization error that the optimizer may still
accept as feasible. Adaptive refinement addresses this: it estimates the
per-interval error, redistributes (and adds) mesh nodes where the error
concentrates, and re-solves until every interval is below a tolerance you set.

This recipe assumes you already have a working {py:class}`~tychopy.optimal_control.PhaseInterface`
with constraints and an objective attached. For the concepts behind
collocation error see {doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>`;
for end-to-end phase setup see {doc}`Setting up a phase </user_guide/first_phase>`.

The C++ tabs show the equivalent builder-API calls — illustrative fragments
that assume a `phase` is already in scope, not standalone programs.

## Enable adaptive refinement

Turn the loop on with `set_adaptive_mesh`, then set the tolerance the
aggregated error must reach:

::::{tab-set}
:::{tab-item} Python
```python
phase.set_adaptive_mesh(True)
phase.set_mesh_tol(1.0e-7)

# Keep the equality-constraint tolerance at least as tight as the mesh
# tolerance; otherwise the optimizer can satisfy its optimality conditions
# before the mesh-error criterion is met.
phase.optimizer.eq_con_tol = 1.0e-8
```
:::
:::{tab-item} C++
```cpp
using namespace tycho;
using namespace tycho::oc;

phase.set_adaptive_mesh(true);
phase.set_mesh_tol(1.0e-7);

// Keep the equality-constraint tolerance at least as tight as the mesh tolerance.
phase.optimizer().set_econ_tol(1.0e-8);
```
:::
::::

With adaptive mesh enabled, `phase.optimize()` (or `phase.solve_optimize()`
when your initial guess is poor) runs the refinement loop automatically: it
solves, estimates the error, refines the mesh, and repeats up to
`set_max_mesh_iters` times or until the tolerance is met. The boolean state is
also exposed as the read/write attribute `phase.adaptive_mesh`, and the
tolerance as `phase.mesh_tol`.

If you want to drive the loop by hand instead, call
`phase.refine_traj_auto()` to perform a single estimate-and-redistribute step
between your own `optimize` calls.

## Choose an error estimator and aggregation

Two knobs control *how* the error is measured and reduced to the scalar that
is compared against `mesh_tol`.

The **estimator** ({py:class}`~tychopy.optimal_control.MeshErrorEstimators`)
selects how per-interval error is computed:

::::{tab-set}
:::{tab-item} Python
```python
import tychopy.optimal_control as oc

# DEBOOR (default): de Boor collocation-residual estimate — cheap, local.
phase.set_mesh_error_estimator(oc.MeshErrorEstimators.DEBOOR)

# INTEGRATOR: re-integrate each interval with a reference RK stepper and
# compare — more expensive but more faithful, good for tight tolerances.
phase.set_mesh_error_estimator(oc.MeshErrorEstimators.INTEGRATOR)
```
:::
:::{tab-item} C++
```cpp
// DEBOOR (default): de Boor collocation-residual estimate — cheap, local.
phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);

// INTEGRATOR: re-integrate each interval with a reference RK stepper and
// compare — more expensive but more faithful, good for tight tolerances.
phase.set_mesh_error_estimator(MeshErrorEstimators::INTEGRATOR);
```
:::
::::

The **aggregation** ({py:class}`~tychopy.optimal_control.MeshErrorAggregation`)
reduces the per-interval errors to one number:

::::{tab-set}
:::{tab-item} Python
```python
# MAX (default), AVG, GEOMETRIC, or ENDTOEND.
phase.set_mesh_error_criteria(oc.MeshErrorAggregation.ENDTOEND)
```
:::
:::{tab-item} C++
```cpp
// MAX (default), AVG, GEOMETRIC, or ENDTOEND.
phase.set_mesh_error_criteria(MeshErrorAggregation::ENDTOEND);
```
:::
::::

`ENDTOEND` measures how far a re-integration of the converged solution drifts
from the collocated final state — use it when endpoint trajectory accuracy is
the primary concern. With `ENDTOEND` on a low-order
transcription you often need to raise the error safety factor:

::::{tab-set}
:::{tab-item} Python
```python
phase.set_mesh_err_factor(100.0)   # defaults to 10
```
:::
:::{tab-item} C++
```cpp
phase.set_mesh_err_factor(100.0);   // defaults to 10
```
:::
::::

Both setters also accept their string names (`"integrator"`, `"endtoend"`,
…), and the same values are exposed as the writable properties
`phase.mesh_error_estimator` and `phase.mesh_error_criteria`.

## Read the refinement history

After a solve, `phase.get_mesh_iters()` returns a list of
{py:class}`~tychopy.optimal_control.MeshIterateInfo` records — one per
refinement iteration. Each record carries the mesh node `times`, the
per-segment `error`, the `distribution` that drove the next redistribution,
its cumulative `distintegral`, plus the scalar summaries `max_error`,
`avg_error`, `numsegs`, and a `converged` flag:

```python
for i, mit in enumerate(phase.get_mesh_iters()):
    print(f"iter {i}: {mit.numsegs} segs, "
          f"max_error={mit.max_error:.2e}, converged={mit.converged}")
```

## Visualize the mesh error

The pure-Python helper `PhaseMeshErrorPlot` draws the whole history — the
per-segment error against the tolerance line, the normalized error
distribution, and its integral — one curve per iteration:

```python
from tychopy.optimal_control.mesh_error_plots import PhaseMeshErrorPlot

phase.optimize()
PhaseMeshErrorPlot(phase, show=True)
```

For a multi-phase problem, `OCPMeshErrorPlot(ocp)` from the same module calls
it for every phase.

A complete worked example is
`examples/python_examples/MeshRefinement/Reentry.py`, which uses the
`INTEGRATOR` estimator with `ENDTOEND` aggregation on a space-shuttle reentry.

## See also

- {doc}`Python reference </reference/python/optimal_control>` — the full
  `PhaseInterface` API, including every `set_mesh_*` setter.
- {doc}`Direct collocation in Tycho </user_guide/phases_and_collocation>` — why
  collocation error arises and what the estimators measure.
