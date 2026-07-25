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

## C++ unit tests — 11 failures, ALL exonerated by A/B; 9 fixed in-PR (2026-07-25)

```
Initial gate:            99% tests passed, 11 tests failed out of 1668
After commit b880f60:    99% tests passed,  2 tests failed out of 1668  (62 s wall)
```

**All 20 `AccelerateInterface` tests PASS.** The 9 calibration failures below were fixed
in this PR (`b880f60`) by making their FP bounds portable to arm64 — existing rel/close
matchers at justified bounds (≥8× headroom over observed drift, orders below any
behavioral-regression signal), plus one deterministic rewrite of EventRefinement's
satisfiable-on-arm64 "impossible tolerance" premise. Loosening cannot un-pass x86. The
2 remaining failures (`DivergencePersistence`, docking Form2) are the PSIOPT
rank-deficiency gap documented below, deferred to the globalization campaign.

Note: four test executables in `build/` were stale from 2026-04-26 — three months old,
predating both this branch and the 13-commit merge from `main`. Any `ctest` result taken
before the full rebuild was meaningless; the run above is post-rebuild and therefore the
first meaningful full-suite result on macOS in months.

### A/B method (no rebuild required)

The 6 `libpsiopt.a` TUs are the only objects in the build carrying `AccelerateImpl`
symbols (verified by `nm` across all archives), and they compile without the PCH. So the
decisive experiment cost seconds, not a 20–40 min rebuild: recompile those 6 TUs against
`origin/main`'s `accelerate_interface.h`/`accelerate_utils.h` via a shadow `-I` directory,
relink `tycho_tests` and the docking example against the swapped objects, and re-run every
failure. All 11 also re-ran deterministically beforehand (`ctest --rerun-failed`: same 11,
same values).

**Result: every failure reproduces bit-for-bit identically under `origin/main`'s
Accelerate headers** — down to the last printed digit (e.g. L1Resto residual
`6.763923865449443e-14`, DivergencePersistence 500/500 iters, Kepler lane delta
`1.624300693947589e-11`, docking Form2 flag 2 / Form1 PASS). **Zero of the 11 failures
are attributable to this PR.** They are macOS-first-run discoveries, invisible until now
because CI is Linux/MKL and no macOS build had run since 2026-04-26.

| # | Test | Verdict |
|---|---|---|
| 1482/1486/1488 | `RegressionIVPTest` Case01/05/07 | Golden-value drift: ulp-scale (≤2.6e-11 on values up to 7000) vs 1e-12 **absolute** tolerances; gtest itself flags the tolerance as below double spacing at that magnitude. No solver involvement. Recalibrate goldens/tolerances for arm64. |
| 1489/1490/1491 | `RegressionTranscriptionTest` Case08/09/10 | Same class (tolerances 1e-13/1e-14). |
| 1402 | `KeplerLCDKernelSS` | SIMD lane vs scalar at 1.29e-13 relative vs 1e-13 relative bound — NEON vs AVX ulp drift, no solver. |
| 807 | `EventRefinementCoverageTest.ResetPerCall` | Test premise ("impossibly tight tol must yield ≥1 nullopt") not portable: on this platform the refinement residual lands exactly within tol. No solver. |
| 256 | `L1RestoStepApplication` | 6.76e-14 vs 1e-14 bound — FP-noise scale through a KKT solve; backend/arch ulp drift. Fails identically pre/post PR. |
| 29 | `cpp_example_optimal_docking_builder` | Form2 flag 2 / Form1 PASS, identical pre/post PR. Backend-inherent convergence difference. |
| 375 | `DivergencePersistence.MaratosCorpus` | Identical pre/post PR. **Root cause fully diagnosed — see below.** |

### DivergencePersistence root cause: PSIOPT has no rank-deficiency correction

Instrumented the interface (debug prints in `doFactorization`/`doRefactorization`/
`_solve_impl`, same shadow-header relink) and probed the standalone 3×3 system:

1. At the start point (0,1), PSIOPT's least-squares multiplier initialization yields the
   multiplier that makes **∇²L exactly zero** (λ·∇²c = −∇²f: 4I − 2·2I). The assembled
   KKT matrix is exactly `[[0,0,0],[0,0,2],[0,2,0]]` (verified: refactor values print as
   `0 0 0 0 2 0`) with true inertia **(1 pos, 1 neg, 1 zero)** — exactly singular.
2. Accelerate's LDLT-TPP factors it, reports `status=OK`, inertia `p/n/z = 1/1/1` —
   **honest and correct** (a standalone probe of the well-conditioned variant returns
   (2,1,0) and exact solves, so the backend itself is sound).
3. `PSIOPT::factor_impl` (psiopt.cpp:1690) corrects inertia only when
   `neigs − m > 0`. Here `IncEigs = 1 − 1 = 0 ≤ 0` → **accepted without perturbation**;
   `zeigs > 0` only triggers the "Potential Rank Deficiency" *warning*. The singular
   solve returns a zero step (`b=(-1,0,0) → x=(0,0,0)`), the iterate and multipliers
   never move, the same singular matrix is refactored every iteration — 500 identical
   iterations to the max-iter flag.
4. On MKL the same test passes: consistent with Pardiso's documented automatic pivot
   perturbation for near-zero pivots, which prevents an exact `zeigs` from surfacing
   (it manifests as perturbed pivots / wrong inertia instead, which **does** enter the
   correction loop). The gap is latent on MKL, exposed by Accelerate's honest inertia.

**This is a PSIOPT globalization-campaign item, pre-existing on `main`:** the
inertia-correction loop needs a `neigs + peigs < n` (rank-deficient) branch that
regularizes — the standard IPOPT-style dual/primal regularization case — rather than
accepting the factorization. The docking Form2 failure is plausibly the same mechanism
and should be re-checked once that branch exists.

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
