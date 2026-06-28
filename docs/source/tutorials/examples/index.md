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

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Brachistochrone
:link: brachistochrone
:link-type: doc
Single-phase basics — free final time and a control path bound.
:::

:::{grid-item-card} Bryson–Denham
:link: bryson_denham
:link-type: doc
A hard state path constraint with an integral control-effort objective.
:::

:::{grid-item-card} Goddard rocket
:link: goddard_rocket
:link-type: doc
Multi-phase linkage and a nonlinear path constraint defining a singular arc.
:::

:::{grid-item-card} Multi-phase cannonball
:link: multi_phase_cannon
:link-type: doc
ODE parameters, parameter linking across phases, and an event-detected guess.
:::

:::{grid-item-card} Simple low-thrust transfer
:link: simple_low_thrust
:link-type: doc
Astrodynamics orbit-raising with a vector-norm control bound.
:::

:::{grid-item-card} Minimum time to climb
:link: min_time_to_climb
:link-type: doc
Interpolated table data (`InterpTable1D`/`2D`) with adaptive mesh refinement.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

brachistochrone
bryson_denham
goddard_rocket
multi_phase_cannon
simple_low_thrust
min_time_to_climb
```
