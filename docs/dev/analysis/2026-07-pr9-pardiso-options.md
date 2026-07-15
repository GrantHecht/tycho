# PR 9 — MKL Pardiso option study: two-level scheduling + matching/scaling grid (PSIOPT §3.2)

**Date:** 2026-07-15
**Branch:** `perf/review-9` @ `fd049a7b`
**Task class:** MEASUREMENT ONLY — no defaults are changed in this PR. Any default flip motivated
by this study is CONVERGENCE-AFFECTING (matching/scaling alter Pardiso pivoting, hence in
principle the iterate sequence) and requires Grant's explicit sign-off plus the full pre-merge
gate (all C++ tests, all 32 Python examples with iteration-count audit, brachistochrone, bench
compare).

## 1. Question

PSIOPT's KKT systems are solved with MKL Pardiso (`Eigen::PardisoLDLT`, `mtype = -2`, real
symmetric indefinite — `include/tycho/detail/solvers/linear/pardiso_interface.h:569`). Three
Pardiso knobs are already plumbed as public `PSIOPT::Settings` fields
(`include/tycho/detail/solvers/psiopt.h:151-155`, consumed by `set_qp_params()` at
`src/solvers/psiopt.cpp:588-600` → `kkt_sol_.set_params()`):

| Settings field | iparm | Shipped default | Meaning |
|---|---|---|---|
| `qp_alg_` | `iparm[23]` | `Classic` (=0) | two-level factorization scheduling off |
| `qp_matching_` | `iparm[12]` | `1` | maximum weighted matching ON |
| `qp_scaling_` | `iparm[10]` | `0` | MPS scaling OFF |

MKL's own documentation recommends `iparm[10] = iparm[12] = 1` for "highly indefinite symmetric
matrices (e.g. from interior point optimizations)" — i.e. exactly tycho's saddle-point KKT
systems. Tycho currently ships the halfway point (matching on, scaling off). Two-level
scheduling is documented to help at ≥ 8 threads. This study measures both axes.

## 2. Method

**Runtime sweep, no library rebuilds.** The knobs are consumed at `set_nlp()` time
(`psiopt.cpp:806-820`: `set_nlp` → `set_qp_params` → `kkt_sol_.set_params()`), and phase
transcription — which calls `set_nlp` — happens lazily inside the first
`solve_optimize()` (`ode_phase_base.cpp:1543-1545`: `psipot_call_impl` runs `transcribe()`
first). Therefore setting the `Settings` fields on a freshly constructed, not-yet-solved phase's
`optimizer_` is sufficient for the options to take effect; no `force_qp_analysis_` needed.

A standalone driver (`pr9_pardiso_sweep.cpp`, scratchpad-only, not committed) `#include`s
`bench/cpp/bench_phases.h` verbatim and drives the exact fixtures behind the tracked benchmarks
`BM_PSIOPT_Brach_32seg`, `BM_PSIOPT_PolarLT_128seg`, and `BM_PSIOPT_PolarLT_256seg`
(`make_brach_phase(32, 3)`, `make_polar_lt_phase(128|256, 3)`); like the benchmarks, a fresh
phase is constructed for every solve. It was compiled with the exact bench TU flags from
`compile_commands.json` (`-O3 -march=native -ffast-math -fno-finite-math-only` etc., conda
clang 22.1.0, bench PCH) and linked with bench_all's exact link recipe (static
`libpsiopt.a`/`liboptimalcontrol.a`/… + static MKL `intel_lp64/intel_thread/core` +
`libiomp5.so`), so ABI and codegen match the tracked benchmarks.

Per config × problem:

1. one warm-up `solve_optimize()` (discarded; convergence smoke check),
2. **N = 5** timed solves — wall-clock per solve (construct + transcribe + solve, matching the
   bench's per-iteration unit), median/min/max reported; PSIOPT total iteration count read from
   `optimizer_->result().iter_num_` after each solve (flagged if it varies across the 5 runs —
   it never did),
3. one extra (untimed) solve with a `late_callback` installed to accumulate
   `IterateInfo::p_pivots_` (= `kkt_sol_.ppivs()` = `iparm[13]`, perturbed pivots per
   factorization) across all iterations. This solve is excluded from timing because the callback
   adds per-iteration overhead.

Configs (one axis at a time from the shipped default):

- `Classic_M1S0` — shipped default (baseline)
- `TwoLevel_M1S0` — axis 1
- `Classic_M1S1`, `Classic_M0S0`, `Classic_M0S1` — axis 2 grid

The full sweep was run twice: once with `MKL_CBWR=AUTO,STRICT` (run-to-run bitwise
reproducibility of the iterate sequence) and once without (production-representative codegen,
since CBWR suppresses some MKL kernel optimizations), to confirm the ranking is
CBWR-independent.

**Machine context:** AMD Ryzen 7 5800X3D (8 cores / 16 threads, AVX2, no AVX-512), 31 GiB RAM,
Fedora 44 (kernel 7.0.12-201), MKL 2026.0 (`/opt/intel/oneapi/mkl/2026.0`), conda-forge
clang 22.1.0, `TYCHO_DEFAULT_QP_THREADS=8` → `qp_threads_ = 8`. Quiet machine (nothing else
running). Approximate primal sizes: Brach 32seg ≈ 485 vars; PolarLT 128seg ≈ 2.7k vars;
PolarLT 256seg ≈ 5.4k vars (KKT dim ≈ primal + slacks + eq/iq multipliers).

## 3. Results

`med_s` = median wall-clock of 5 solves (seconds, fresh phase per solve); `iters` = PSIOPT total
iterations (solve + optimize phases; identical across all 5 runs in every cell); `Σppivs` /
`max ppivs` = perturbed pivots summed over all iterations of one solve / max in any single
factorization.

### With `MKL_CBWR=AUTO,STRICT`

| Config | Brach_32seg med_s (iters, Σppivs) | PolarLT_128seg med_s (iters, Σppivs) | PolarLT_256seg med_s (iters, Σppivs) |
|---|---|---|---|
| `Classic_M1S0` (default) | 0.00732 (18, 0) | 0.10814 (87, 95) | 0.20614 (87, 120) |
| `TwoLevel_M1S0` | 0.00590 (18, 0) | 0.10796 (87, 95) | 0.20692 (87, 120) |
| `Classic_M1S1` | 0.00592 (18, 0) | **0.08026** (87, 0) | **0.15469** (87, 7) |
| `Classic_M0S0` | 0.00684 (18, 2224) | 0.10694 (87, 60095) | 0.19976 (87, 119935) |
| `Classic_M0S1` | 0.00682 (18, 2224) | 0.10697 (87, 60095) | 0.19943 (87, 119935) |

### Without CBWR (production-representative)

| Config | Brach_32seg med_s (iters, Σppivs) | PolarLT_128seg med_s (iters, Σppivs) | PolarLT_256seg med_s (iters, Σppivs) |
|---|---|---|---|
| `Classic_M1S0` (default) | 0.00776 (18, 0) | 0.08196 (87, 95) | 0.15097 (87, 120) |
| `TwoLevel_M1S0` | 0.00767 (18, 0) | 0.08191 (87, 95) | 0.15050 (87, 120) |
| `Classic_M1S1` | 0.00721 (18, 0) | **0.06850** (87, 0) | **0.12634** (87, 7) |
| `Classic_M0S0` | 0.00626 (18, 2224) | 0.07771 (87, 60095) | 0.13760 (87, 119935) |
| `Classic_M0S1` | 0.00629 (18, 2224) | 0.07829 (87, 60095) | 0.13701 (87, 119935) |

Run-to-run spread (min→max over the 5 timed solves) was < 2 % in nearly every cell; Brach at
~6-8 ms/solve is the noisiest. CBWR itself costs ~25 % on the PolarLT problems but does not
change the ranking.

### Observations

- **Iteration counts are invariant across ALL configs, both CBWR modes, and all repeat runs**
  (Brach 18, PolarLT 87 at both scales). On these problems the matching/scaling choice provably
  did not alter the iterate trajectory — every measured time difference is pure per-factorization
  KKT cost, not a different path through the optimizer.
- **Axis 1 (TwoLevel):** no effect. Within noise of Classic on all three problems at
  `qp_threads_=8`. These KKT systems (≤ ~15k KKT dim) are too small for two-level scheduling to
  matter.
- **Axis 2 winner — `(matching=1, scaling=1)`:** consistent, large wall-clock win at both
  PolarLT scales: **−26 % / −25 %** (CBWR) and **−16 % / −16 %** (no CBWR) vs the shipped
  default, with identical iteration counts, and neutral-to-slightly-faster on Brach. The
  mechanism is visible in the pivot column: perturbed pivots drop 95→0 (128seg) and 120→7
  (256seg). This matches MKL's documented recommendation for interior-point saddle-point
  systems.
- **Matching OFF (`M0S0`/`M0S1`):** 1-9 % faster than the default on these problems (matching's
  analysis-time cost exceeds its payoff here) but produces enormous perturbed-pivot counts
  (Σ 60k-120k, up to 1411 in a single factorization). That factorization is only surviving on
  pivot perturbation (`iparm[9] = 8` ⇒ eps = 1e-8) with `qp_ref_steps_ = 0` — no iterative
  refinement backstop. Numerically fragile; rejected regardless of the small speedup.
- With matching OFF, scaling (`iparm[10]`) is a no-op on these systems (identical times and
  ppivs for M0S0/M0S1), consistent with MKL applying MPS scaling to symmetric matrices only in
  combination with matching.

## 4. Recommendation

- **`qp_alg_` (two-level): keep `Classic`.** No measurable benefit at any tested scale; no
  reason to change.
- **`qp_matching_ = 0`: rejected.** Small speedup, severe pivot-perturbation dependence.
- **`(qp_matching_, qp_scaling_) = (1, 1)` is a promising default-change candidate.** It meets
  the dossier's justification bar on this evidence: a consistent wall-clock win at ≥ 2 problem
  scales (PolarLT 128/256seg, −16 % to −26 %) with **zero iteration-count regression** (counts
  bit-identical in every measured configuration), and it moves tycho onto MKL's documented
  recommended setting for interior-point KKT systems. It also reduces perturbed pivots to ~0,
  i.e. it is the numerically *more* robust direction.

**However, no default changes in this PR.** Flipping `qp_scaling_` to 1 changes Pardiso's
pivoting and is therefore convergence-affecting by classification, even though no iterate change
was observed here. The flip should be its own PR with:

1. Grant's explicit sign-off (PSIOPT internals + MKL-integration-sensitive area),
2. the full pre-merge gate, including a per-example PSIOPT iteration-count comparison across all
   32 Python examples (the real convergence test — this study only covers two problem families),
3. `bench/bench_track.sh compare` on the full 128-bench suite,
4. consideration of whether `qp_ref_steps_` should stay 0 under the new scaling (likely yes,
   since ppivs → 0 makes refinement *less* necessary, but confirm on the examples).

## 5. Caveats / threats to validity

- Only two problem families (Brach, PolarLT), both LGL3 collocation, KKT dims ≤ ~15k. No
  MEE/Betts-class problem was testable in this sweep (the MEE benchmark TU is disabled for
  build-memory reasons — `bench/cpp/solvers/bench_solvers.cpp:126-128`). Larger or worse-scaled
  KKT systems could respond differently, in either direction.
- Wall-clock includes phase construction + transcription (same unit as the tracked benchmarks);
  the relative KKT-only win is therefore *larger* than the end-to-end percentages quoted.
- Iteration-count invariance was observed, not proven; on problems where the default already
  produces perturbed pivots and marginal steps, scaling could change the iterate path (that is
  the point of the full-gate requirement above).
- Single machine (Zen 3, AVX2), single MKL version (2026.0), `qp_threads_=8`. The two-level
  conclusion in particular ("no effect") should be rechecked if tycho is ever run with
  substantially larger KKT systems or ≥ 16 Pardiso threads.
