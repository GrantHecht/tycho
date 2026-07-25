# RESULTS: macOS verification of the Apple Accelerate interface overhaul (PR #88)

**Date:** 2026-07-25
**Machine:** Apple M1 Pro, 16 GB
**OS:** macOS 26.5 (build 25F84)
**Toolchain:** Homebrew clang 22.1.7 (`/opt/homebrew/opt/llvm/bin/clang++`), Xcode SDK MacOSX.sdk
**Python:** 3.13.12 (conda env `tycho`)
**Build:** `cmake --preset macos-llvm-release -DBUILD_CPP_TESTS=ON -DBUILD_CPP_EXAMPLES=ON`, `ninja -j4`
**HEAD verified:** `e3c221e` (+ this results commit)

This supersedes the original PR 7 handoff checks. The four inspection-only hunks (P1–P4)
grew into a six-invariant overhaul (I1–I6) with 20 regression tests and a committed
warning canary. See `docs/dev/plans/2026-07-24-accelerate-interface-overhaul-design.md`.

## Handoff check 2c — RESOLVED: PASS

The original blocking question was whether `SparseSolve` can clobber
`m_numericFactorization->status` on a successful solve, which would make the P2 fix
unsound.

**It cannot, by construction.** Every `SparseSolve` overload takes `Factored` **by value**
and passes `&Factored` — a pointer to its own stack copy — to `_SparseSolveOpaque`
(`SolveImplementationTyped.h:754-761`, shipped inline in the macOS SDK). The caller's status
field is unreachable from a solve. Confirmed empirically: status identical across a good
solve, `||Ax-b||_inf = 4.4e-16`.

Corollary: solve-time-specific failures are **not observable at all** through this API, so
the in-code comment on that call is accurate and the call itself can only ever rewrite
`Success` over `Success`.

Original checks 1, 2a and 2b are superseded by `tests/cpp/solvers/test_accelerate_interface.cpp`.

## Build — PASS

```
NINJA_EXIT_CODE=0        errors: 0
```

Built in two parts: an initial run reached `[139/174]` before its background task was
reaped (see "Process notes"), then an incremental resume completed `[35/35]`.

**Warning audit — PASS (the gate requirement):** zero warnings originate in
`accelerate_interface.h` or `accelerate_utils.h`.

```
grep -cE '(accelerate_interface|accelerate_utils)\.h:[0-9]+:[0-9]+: warning' <build log>   ->  0
```

Every build warning traces to three **pre-existing** sites, unrelated to this PR:

| Count | Site | Diagnostic |
|---|---|---|
| 66 | `include/tycho/detail/solvers/jet.h:227` | `-Wunused-result` (ignored `nodiscard`) |
| 13 | `include/tycho/detail/integrators/integrator.h:2306` | `-Wunused-result` |
| 13 | `include/tycho/detail/integrators/integrator.h:2337` | `-Wunused-result` |

Verified twice by independent measurement, and consistent with the A/B rebuild the I2 task
performed (which confirmed `-Wunused-result` was never in Eigen's suppressed set, so
balancing the pragma did not expose them). **Follow-up worth filing separately:** these are
ignored `nodiscard` results on `std::future`, which can silently swallow exceptions.

## C++ unit tests — RED (11 failures, triaged below)

```
99% tests passed, 11 tests failed out of 1668
CTEST_EXIT=8
```

**All 20 `AccelerateInterface` tests PASS.**

Note: four test executables in `build/` were stale from 2026-04-26 — three months old,
predating both this branch and the 13-commit merge from `main`. Any `ctest` result taken
before the full rebuild was meaningless; the run above is post-rebuild and therefore the
first meaningful full-suite result on macOS in months.

| # | Test | Reaches the Accelerate solver? |
|---|---|---|
| 1482 | `RegressionIVPTest.Case01_TwoBody_DOPRI54` | **No** |
| 1486 | `RegressionIVPTest.Case05_EventCrossing` | **No** |
| 1488 | `RegressionIVPTest.Case07_Backward` | **No** |
| 1489 | `RegressionTranscriptionTest.Case08_Jacobian` | **No** |
| 1490 | `RegressionTranscriptionTest.Case09_JacobianHessian` | **No** |
| 1491 | `RegressionTranscriptionTest.Case10_BatchJacobians` | **No** |
| 807 | `EventRefinementCoverageTest.ResetPerCall_SecondCallIndependent` | **No** |
| 29 | `cpp_example_optimal_docking_builder` | Possibly |
| 256 | `L1RestoStepApplication.ApplyMovesSlacksAndUpdatesPivots` | Possibly |
| 375 | `DivergencePersistence.MaratosCorpusConvergesAtDefaults` | Possibly |
| 1402 | `KeplerLCDKernelSS.UniformEllipticFourLanesHitsSimdPath` | Possibly |

**7 of 11 are provably not attributable to this PR.** `test_regression_ivp.cpp`,
`test_regression_transcription.cpp` and `test_event_refinement_coverage.cpp` contain **zero**
references to `PSIOPT` or `kkt_sol_` (grep-verified). They are integrator and event tests,
unreachable from the Accelerate *linear solver* interface, which is the only code this PR
modifies. Independently: the merged `main` commits never touched `accelerate_interface.h`
(0 commits for that path in `889faab~13..889faab`).

**The remaining 4 cannot be excluded by inspection.** The worst signature is
`DivergencePersistence`, which failed to converge — `iter_num_` 500 against an expected
`<= 60`, plus a flag mismatch (2 vs 0) at `test_divergence_persistence.cpp:234-237`.

Two facts bear on attribution, in opposite directions:

- **Against:** `test_divergence_persistence.cpp` was added by `889faab` (main's HEAD) and
  `test_l1_restoration.cpp` by `36fe4f5` — both from the globalization series merged into
  this branch. They are new and have **never executed against the Accelerate backend**: CI
  runs Linux/MKL, and the Accelerate path is CI-invisible, which is the premise of this PR.
- **For:** this PR's one plausible mechanism for perturbing PSIOPT convergence is the I3
  inertia reset. On success paths the changes are behaviorally inert — `doAnalysis()` →
  `doFactorization()` → `cacheInertia()` repopulates; `_solve_impl`'s guard only changes
  `info()` classification, which PSIOPT never reads for control flow; the alias-copy never
  triggers for any in-tree caller; `makeCompressed()` is a no-op on an already-compressed
  matrix; MT-METIS passes through unchanged on macOS 26.5. But on a **factorization
  failure**, the inertia-correction loop now reads zeroed inertia where it previously read
  stale values, which can change perturbation decisions.

**Resolution deferred by decision:** these are being investigated as part of the ongoing
globalization campaign rather than in this PR. Settling attribution requires building
`origin/main` (or reverting the two Accelerate headers) and re-running those 4 tests.

## C++ brachistochrone — PASS

```
Optimal Solution Found
Prim Obj : 1.800e+00        (expected ~1.8013)
exit 0
```

Positive evidence that PSIOPT driving the Accelerate backend converges correctly on the
happy path, which is the configuration all 32 examples and normal use exercise.

## Python examples — PASS (34/34)

```
Results: 34 passed, 0 failed, 0 skipped
EXAMPLES_EXIT=0
```

Run as `MPLBACKEND=Agg python scripts/run_examples.py` in the `tycho` conda env. Nothing
skipped, so every optional dependency (seaborn, spiceypy, basemap) was present.

**This is the strongest single piece of evidence bearing on the `ctest` triage above.** The
example suite is the project's stated integration/acceptance gate, every example drives
PSIOPT end-to-end, and on macOS PSIOPT's KKT solver *is* the Accelerate backend this PR
rewrites. If these changes had broken PSIOPT's normal operation, this suite would show it.
All 34 pass.

It does not fully exonerate the 4 PSIOPT-adjacent unit-test failures, which may probe
failure paths the examples never enter — and the I3 inertia reset only changes behavior on a
factorization failure. But it does rule out a broad regression in ordinary solver operation.

## Benchmarks — NOT RUN (no baseline)

```
ERROR: No results for HEAD (e3c221e2). Run: bench/bench_track.sh record
```

No baseline exists for this HEAD, so there is nothing to compare against. Recorded as
"not run" rather than as a pass or a failure. `bench/bench_track.sh record` would establish
one.

## Aliasing decision (I5) — measured, not assumed

The plan left the aliasing guard conditional on a probe with a 1e-10 threshold. Measured at
n=200 tridiagonal SPD, `nrhs` in {1,3}, 5 runs:

| Configuration | Aliased vs reference | Aliased residual |
|---|---|---|
| refinement **off** | `0.000e+00` (bit-identical) | `8.882e-16` |
| refinement **on**, pre-fix | `7.501e-01` | **`1.712e+00`** |
| refinement on, post-fix | `0.000e+00` | `2.220e-16` |

The initial probe covered only refinement-off and therefore certified aliasing as safe. With
refinement enabled, the refinement loop rebuilds `-b` from a buffer `SparseSolve` has already
overwritten — the same memory when aliased — producing **silently wrong answers reported as
`info() == Success`**. Reproduced at 3×3 with residual `2.2962962962962967`.

Resolution: copy `b` unconditionally when `b_ptr == x_ptr`, mirroring
`PardisoImpl::_solve_impl`. Deliberately *not* gated on the refinement flag, so correctness
does not depend on a runtime setting; and deliberately not an `eigen_assert`, which `NDEBUG`
would strip in the project's default Release build.

## Process notes

- A `dep/pocketfft` submodule arrived with `main` and blocks configure until initialized:
  `git submodule update --init dep/pocketfft`.
- Long builds must be launched from a persistent session. Two builds were killed mid-run
  when the subagent that started them ended its turn; the log signature is
  `ninja: build stopped: interrupted by user` with no exit code.
- `scripts/check_accelerate_warnings.sh` exits 0 (both checks PASS) on the final tree.
