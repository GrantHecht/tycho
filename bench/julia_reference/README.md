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

    julia --project=. src/run_method.jl <method> <problem> <abstol> <reltol> <outfile>

`method` ∈ {Tsit5, BS3, BS5, Vern7, Vern8, Vern9, DP5, DP8}
`problem` ∈ {two_body, crtbp}

Binary output format:
- int32  n       (state size)
- float64 × n    (final state)
- int32  nsteps  (accepted step count, from `sol.stats.naccept`)
- float64 walltime_sec
