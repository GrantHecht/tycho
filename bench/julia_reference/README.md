# Julia Reference Harness

Runs OrdinaryDiffEq.jl as numerical reference for Tycho's RK methods.

Uses **out-of-place ODE formulation with StaticArrays** — preferred over in-place
mutating `f!` for small fixed-size state vectors (6-dim here). See the
OrdinaryDiffEq.jl performance tips for rationale.

## Setup

    julia --project=. -e 'using Pkg; Pkg.instantiate()'

**Important:** Always use `Pkg.instantiate()`, NEVER `Pkg.update()`. Manifest.toml
pins package versions exactly; updating breaks reproducibility across machines
and invalidates regression-tied golden outputs.

## Usage

Two sibling runners are provided. They share the same ODE definitions, method
dispatch, and warm-up-then-measure pattern, but emit different binary formats.

### SP2 — final-state reference (`run_method.jl`)

    julia --project=. src/run_method.jl <method> <problem> <abstol> <reltol> <outfile>

Binary output format (**64 bytes** for n=6):

- int32    n       (state size)
- float64 × n      (final state)
- int32    nsteps  (accepted step count, from `sol.stats.naccept`)
- float64  walltime_sec

Reference binaries under `test/reference_outputs/<alg_lc>_<problem>.bin` are the
frozen SP2 golden outputs and **must not** be regenerated casually — downstream
regression tests are keyed off the exact byte contents.

### SP3 — step-stats reference (`run_method_stats.jl`)

    julia --project=. src/run_method_stats.jl <method> <problem> <abstol> <reltol> <outfile>

Extends the SP2 format with an explicit `nreject` field so Tycho's I/PI/PID
controllers can be validated against OrdinaryDiffEq.jl for both accepted and
rejected step counts.

Binary output format (**72 bytes** for n=6):

- int32    n         (state size)
- float64 × n        (final state)
- int32    naccept   (accepted step count, from `sol.stats.naccept`)
- int32    nreject   (rejected step count, from `sol.stats.nreject`)
- float64  walltime_sec

Reference binaries live at
`test/reference_outputs/<alg_lc>_<problem>_<tol_tag>.stats.bin`, with 48 files
covering 8 algorithms × 2 problems × 3 tolerance tiers. Tolerance tags encode
the absolute tolerance; relative tolerance is always `abs_tol / 10`:

| tag       | abstol | reltol |
| --------- | ------ | ------ |
| `tol1e6`  | 1e-6   | 1e-7   |
| `tol1e9`  | 1e-9   | 1e-10  |
| `tol1e12` | 1e-12  | 1e-13  |

### Method and problem selectors (both runners)

`method` ∈ {Tsit5, BS3, BS5, Vern7, Vern8, Vern9, DP5, DP8}
`problem` ∈ {two_body, crtbp}
