(tutorials-examples)=
# Worked examples

Learning by example. This gallery collects complete, copy-paste-runnable
optimal-control programs drawn straight from Tycho's example suite — each page
is one self-contained problem you can run as-is and adapt to your own work.
Every Python program on these pages is executed when the documentation is
built, and the figure you see is the one it produced, so what you read is
exactly what runs.

Each page shows the whole program: imports, the ODE definition, the initial
guess, the phase (or multi-phase) setup, the solve, and a solution plot — plus
the matching C++ builder-API source.

## In this gallery

- **{doc}`Brachistochrone <brachistochrone>`** — single-phase basics, free
  final time, a control path bound.
- **{doc}`Bryson–Denham problem <bryson_denham>`** — a hard state path
  constraint together with an integral control-effort objective.
- **{doc}`Goddard rocket <goddard_rocket>`** — multi-phase linkage and a
  nonlinear equality *path constraint* defining a singular arc.
- **{doc}`Multi-phase cannonball <multi_phase_cannon>`** — ODE parameters,
  multi-phase parameter linking, and an event-detection initial guess.
- **{doc}`Simple low-thrust transfer <simple_low_thrust>`** — astrodynamics
  orbit-raising with a vector-norm control bound.
- **{doc}`Minimum time to climb <min_time_to_climb>`** — interpolated table
  data (`InterpTable1D`/`InterpTable2D`) with adaptive mesh refinement.

```{toctree}
:maxdepth: 1

brachistochrone
bryson_denham
goddard_rocket
multi_phase_cannon
simple_low_thrust
min_time_to_climb
```
