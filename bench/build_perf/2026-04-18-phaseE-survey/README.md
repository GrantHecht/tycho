# Phase E — Tests & benchmarks TU-split survey (documented non-goal)

The plan `kind-prancing-crown.md` Phase E catalogues why the TU-split pattern
from Phase A (statics) does **not** apply to the tests and benchmarks, so
future readers don't need to re-derive the finding. All numbers cite the
post-SP3 baseline in `bench/build_perf/2026-04-18-baseline/`.

## Scope

The candidates considered were the heavy non-example TUs from the baseline,
ranked by peak RSS:

| TU | wall (s) | peak RSS (GB) |
|---|---:|---:|
| `bench/cpp/solvers/bench_solvers.cpp` | 178.7 | 7.12 |
| `bench/cpp/optimal_control/bench_optimal_control.cpp` | 123.8 | 6.67 |
| `bench/cpp/integrators/bench_integrators.cpp` | 76.6 | 2.93 |
| `tests/cpp/integrators/regression/generate_golden.cpp` | 74.0 | 5.60 |
| `tests/cpp/integrators/regression/test_regression_ivp.cpp` | 73.2 | 5.58 |
| `tests/cpp/integrators/regression/test_regression_transcription.cpp` | similar | similar |
| `tests/cpp/integrators/test_method_*.cpp` | per-method, lighter | |
| `tests/cpp/integrators/test_julia_parity.cpp` | lighter | |

## Finding 1 — Bench TUs: RSS is solver-stack templates, not ODE codegen

`bench/cpp/solvers/bench_solvers.cpp` and
`bench/cpp/optimal_control/bench_optimal_control.cpp` do **not** instantiate
VectorFunction expression trees inline. Both delegate to
`make_brach_phase()` / `make_polar_lt_phase()` defined in the shared
`bench/cpp/optimal_control/bench_phases.h` header. The ODEs those helpers
construct (`BrachODE`, `PolarLTODE`) are declared in `bench_odes.h` and
already amortised across both benches.

What pushes these TUs to 7 GB is the full
`OptimalControlProblem` / `ODEPhase` / PSIOPT / LGL transcription template
tree that every `phase.*` method triggers. Applying the Phase A recipe —
moving the ODE behind a `GenericFunction<-1,-1>` factory — just moves code
that is *already* in a shared header, with zero reduction in what any of the
three benches' TUs have to instantiate.

Reducing these TUs' RSS would require extern-templating the solver stack
itself (`OptimalControlProblemBase::solve_optimize`,
`ODEPhase<GenericODE<…>>::transcribe_dynamics`, and friends). That is the
same failure mode as Phase C: any boundary shift on templates feeding
`tol=0` golden regressions has to be gated behind a golden-test tolerance
policy decision the team owns. **Out of scope for this plan.**

`bench/cpp/integrators/bench_integrators.cpp` (76.6 s / 2.93 GB) is already
the lightest heavy bench TU — its RSS is half of the optimal-control
benches because it sidesteps the PSIOPT stack. Splitting ODEs off would
move ~1 GB to a new TU and gain no measurable wall-clock at `-j2`.

**Action: no split.**

## Finding 2 — Regression tests: Kepler boundaries are FP-drift-risky

`tests/cpp/integrators/regression/generate_golden.cpp` and
`test_regression_ivp.cpp` (~5.6 GB each) contain Cases 1–3, 5–7 that
integrate `Integrator<Kepler>` / `Integrator<CR3BP_Substitute>` and compare
bytes against on-disk golden binaries with `tol = 0.0`
(`tests/cpp/integrators/regression/test_regression_ivp.cpp:24` and
`test_regression_transcription.cpp:26`).

Phase C already proved what happens when `Integrator<Kepler>` bodies cross
a TU boundary: codegen changes enough to shift FP rounding at the 1e-12
level, which a `tol=0` `EXPECT_NEAR` cannot absorb. Cases 02/06 failed under
the extern-template attempt and held green only once the instantiations
returned to the test TUs. Any ODE-factory split that puts Kepler/CR3BP
integrator bodies in a different TU from the calling regression code will
reproduce that failure mode.

**Action: do not split these TUs' Kepler-related bodies.**

A narrow safe carve-out would be feasible: Case 4 uses `BrachODE` (not
Kepler). Extracting the BrachODE fixture into
`tests/cpp/integrators/regression/regression_brach_factory.cpp` would shave
roughly 1 GB RSS from each of the two regression TUs without touching
Kepler codegen. Payoff is modest (the TUs are 5.6 GB today; the Case 4
fragment alone isn't the dominant contributor) and it adds file churn.
**Noted as an available future option, not executed in this plan.**

## Finding 3 — Method tests and julia-parity tests: already amortised

`tests/cpp/integrators/test_method_*.cpp` (per-algorithm test files
compiled into the `tycho_method_tests` executable) all include
`integrator_test_utils.h` and `regression/regression_utils.h`. Heavy ODE
instantiations (`SHO`, `Kepler`-based fixtures) are declared in those
shared headers, so each test file picks up the same instantiation once per
test binary, not once per test file. There's no duplication left to remove
via an ODE-factory split.

`tests/cpp/integrators/test_julia_parity.cpp` has 42 test cases that all
call a single `assert_parity_twobody()` helper. The inner template chain
compiles once per TU regardless of the number of cases.

**Action: no split needed; pattern is already applied implicitly via
shared headers.**

## Finding 4 — What would unblock a test/bench pass in the future

Every high-value candidate above is blocked by the same prerequisite that
blocks Phase C: the `tol=0` golden-regression compare in
`test_regression_{ivp,transcription}.cpp`. Any template-boundary refactor
that affects the Kepler/CR3BP integrator chain — extern templates in
libtycho, narrower solver-stack entry points, per-constraint sub-splits
inside benchmark TUs — will shift FP rounding enough to trip those
compares.

Before any of the above becomes tractable, the team needs to decide:

1. Relax the golden compares to a sensible tolerance
   (e.g. `1e-10 * max(|actual|, |golden|)`), OR
2. Adopt a policy where any codegen-touching refactor comes with a
   golden-regeneration commit and a documented diff.

Either choice is a test-policy decision beyond the scope of a build-perf
refactor, and should be made once rather than re-litigated per phase.

## Conclusion

Phase E commits no code. The findings are:

- **Benchmarks:** high RSS comes from the solver/phase/transcription
  template tree, not from anything Phase A/D could move. ODE-factory
  splits deliver zero measurable win.
- **Regression tests:** the Kepler/CR3BP halves of these TUs are
  FP-drift-risky at `tol=0`. Do not split them.
- **Method tests and julia-parity tests:** already amortised via shared
  headers.
- **Future unblocks:** wait for a `tol=0` policy decision on the golden
  regressions; Phase C's prerequisite is the same prerequisite here.

This artefact together with Phase D.0 (builder-example survey) closes the
build-performance refactor work planned in `kind-prancing-crown.md`.
