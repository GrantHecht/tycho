# E2 G0 — PSIOPT robustness corpus: 2026-07 defaults baseline

**Date:** 2026-07-16
**Branch:** `test/e2-g0-corpus`
**Data:** `tests/corpus/baselines/2026-07-defaults.jsonl` (`scripts/run_corpus.py --cbwr --repeat 2`,
17 problems × 2 repeats = 34 records)
**Task class:** MEASUREMENT ONLY. No PSIOPT/library code changes — the corpus and harness
(Tasks 1-4) only observe today's defaults. This document is the scorecard every subsequent
E2 PR (G1-G8) diffs against via `scripts/run_corpus.py --diff` to show whether a
globalization/robustness change actually helps.

## 1. Per-problem table (repeat 1)

`wall_s` is per-solve subprocess wall time (harness overhead — tychopy import + one PSIOPT
solve — not a meaningful perf number on its own). Determinism is judged over `--repeat 2`
of this baseline capture, cross-checked against the tier reports' own repeated manual runs.

| Tier | Problem | Status | Iterations | Objective | wall_s | Determinism |
| --- | --- | --- | ---: | ---: | ---: | --- |
| degenerate | `deg_dup_equality` | converged | 3 | 6.011497 | 1.03 | stable |
| degenerate | `deg_conflicting_equality` | failed | 500 | 24.045986 | 1.10 | stable |
| degenerate | `deg_zero_objective` | converged | 3 | 0.0 | 0.89 | stable |
| degenerate | `deg_redundant_defects` | converged | 3 | 6.011497 | 1.00 | stable |
| degenerate | `deg_near_infeasible` | failed | 500 | 0.484377 | 1.14 | stable |
| hard | `hard_vanderpol` | diverged | 1 | 0.0 | 0.86 | stable |
| hard | `hard_brach_coldstart` | converged | 24 | 1.801295 | 1.01 | stable |
| hard | `hard_brach_illscaled` | failed | 500 | 1.405983 | 1.16 | stable |
| hard | `hard_zermelo_wrongbasin` | diverged | 822 | -0.295412 | 1.62 | **order-sensitive** |
| hard | `hard_mountaincar_badguess` | failed | 1000 | -0.952637 | 1.03 | **order-sensitive** |
| hard | `hard_lowthrust_badguess` | diverged | 1 | 20.106193 | 1.19 | stable |
| hard | `hard_cartpole_tightbounds` | converged | 95 | 78.545622 | 1.05 | **order-sensitive (LSB)** |
| hard | `hard_hypersens_stiff` | acceptable | 103 | 969.808138 | 0.93 | stable |
| literature | `lit_wb2000` | failed | 500 | -0.968231 | 1.07 | stable |
| literature | `lit_maratos` | diverged | 2 | 1e16 | 0.98 | stable |
| literature | `lit_hs13` | acceptable | 77 | 0.985042 | 1.00 | stable |
| literature | `lit_powell_badscaled` | converged | 22 | 0.0 | 1.00 | stable |

### Determinism check on this baseline capture

Comparing repeat 1 vs repeat 2 record-by-record (see raw JSONL): 16 of 17 problems are
byte-identical on status, iterations, and objective. `hard_cartpole_tightbounds` shows
LSB-level objective float noise (`78.5456220302006` vs `78.54562203020075` — differs only
at the ~13th significant digit; iterations stay exactly 95 both times) — already documented
in the module's own docstring, attributed to `set_num_partitions(8, 8)` threaded evaluation
order and present in the unperturbed parent example too.

The corpus's other two known order-sensitive hard-tier problems, `hard_zermelo_wrongbasin`
and `hard_mountaincar_badguess`, happened to land byte-identical in this particular pair of
repeats — this is expected noise, not a claim that they became deterministic (Task 3's own
manual probing saw `hard_zermelo_wrongbasin` iteration counts range 795-831 and
`hard_mountaincar_badguess`'s *unperturbed parent* range 118-128, both persisting under
forced single-threading). **No other problem varied** across this baseline's two repeats —
if a future baseline capture shows visible variation in any problem outside this
three-problem list, that is a new finding to investigate, not expected noise.

## 2. Flag → status mapping

Discovered by introspecting `_tychopy.solvers.ConvergenceFlags` (`enum.IntEnum`, exactly 4
members) — see `tests/corpus/README.md` for the full discovery trail:

| `ConvergenceFlags` member | harness `status` |
| --- | --- |
| `CONVERGED` | `converged` |
| `ACCEPTABLE` | `acceptable` |
| `NOTCONVERGED` | `failed` |
| `DIVERGING` | `diverged` |
| *(unreachable)* | `error` (unrecognized flag name recorded in `notes`) |

Two more statuses are synthesized by the harness itself, never from a flag: `timeout`
(child killed after exceeding the module's `TIMEOUT`) and `error` (nonzero child exit,
uncaught exception, or malformed/missing result file).

## 3. Headline: how many problems strain today's defaults

Of the 17 corpus problems, **11 (65%) do not reach a clean `CONVERGED` on today's PSIOPT
defaults**:

- **5 failed** (`NOTCONVERGED`, hitting `max_iters`): `deg_conflicting_equality`,
  `deg_near_infeasible`, `hard_brach_illscaled`, `hard_mountaincar_badguess`, `lit_wb2000`.
- **4 diverged** (`DIVERGING`): `hard_vanderpol`, `hard_zermelo_wrongbasin`,
  `hard_lowthrust_badguess`, `lit_maratos`.
- **2 acceptable** (a compromised, not clean, solution): `hard_hypersens_stiff`,
  `lit_hs13`. `hard_hypersens_stiff` is a sharper case than the flag alone shows — the
  harness's `notes` field records `mesh_converged=False` and an objective (≈969.8) wildly
  off the true ≈1.669, a silent-wrong-answer failure the flag-based status does not
  surface on its own.
- **6 converged cleanly**: `deg_dup_equality`, `deg_zero_objective`,
  `deg_redundant_defects`, `hard_brach_coldstart`, `hard_cartpole_tightbounds`,
  `lit_powell_badscaled`. Of these, three (`deg_dup_equality`, `deg_zero_objective`,
  `deg_redundant_defects`) are degenerate-tier problems whose intended pathology did *not*
  manifest as non-convergence at all (see §4) — they are negative results about which
  degeneracies PSIOPT's defaults already tolerate, not evidence the corpus lacks bite.

This 11-of-17 count (or, counting only outright failure/divergence and setting aside the
two `acceptable` cases, 9-of-17 = 53%) is the E2 program's improvement target: every
subsequent G1-G8 PR is expected to move some of these rows from
failed/diverged/acceptable toward converged, evidenced by `scripts/run_corpus.py --diff`
against this baseline.

## 4. Findings from corpus construction

Notable discoveries made while building the corpus (Tasks 1-4), independent of the
baseline numbers above:

- **Pardiso tolerates exact rank-deficiency.** `deg_dup_equality` (an exact duplicate
  terminal boundary-equality block, 3 rows) and `deg_redundant_defects` (a dynamics-implied
  shadow-state redundancy replicated across all 65 interior LGL3 nodes) both `CONVERGED` in
  3 iterations, matching the well-posed baseline almost exactly. `deg_zero_objective` (no
  objective term at all — pure feasibility) also `CONVERGED` in 3 iterations. PSIOPT's KKT
  factorization evidently handles a rank-deficient-by-construction equality block, at
  either the boundary or replicated at scale through the interior, without perturbation or
  regularization trouble on this small linear system.
- **The acceptable band swallows small conflicts.** Both `deg_conflicting_equality` (Task
  2) and the original `stub_fails.py` (Task 1, since deleted) initially used a small
  conflicting-constraint gap (`1.0` vs `1.001`, a ~5e-4 residual) that PSIOPT reported as
  `ACCEPTABLE` rather than a genuine failure — the residual landed inside PSIOPT's default
  acceptable-equality-constraint tolerance. Both modules were widened to a `1.0` vs `2.0`
  gap, which reliably produces `NOTCONVERGED`. This is itself informative: PSIOPT's
  acceptable-tolerance band is wide enough to silently swallow small equality conflicts
  that a stricter contract might want flagged.
- **Time-pinning necessity in collocation phases.** The Task 1 stub double-integrator
  pattern never pinned the phase's node "t" (time) index at the `Back` region. Node time is
  itself a free decision variable in this collocation formulation — nothing ties it to
  `[0, 1]` just because the initial guess spans that range. Left unpinned, PSIOPT silently
  stretches the effective transfer duration to reduce the objective (observed: objective
  collapsing to ~6e-5 instead of the analytic minimum-energy optimum of 6.0, with node "t"
  values drifting to values like -14.6 or +31.4). Fixed across all five degenerate-tier
  modules by pinning `t(0)=0`/`t(1)=1` explicitly at both `Front` and `Back`; the two Task 1
  stubs shared the same gap but were deleted in Task 5 rather than patched.
- **WB2000 constant correction — a source-verification win.** The Task 4 brief's proposed
  constants for the Wächter-Biegler (2000) counterexample (`a=1, b=0.5`, start `(-2,3,1)`)
  do not actually satisfy the paper's own sufficient condition for interior-point failure
  (`m ≈ 0.571`, `a - m·b ≈ 0.714`, which is not `<= min(0,-0.5) = -0.5`). Corrected to
  `a=-1, b=1` — the exact numeric instance Benson, Shanno & Vanderbei ran through LOQO in
  their own reproduction of the example — which does satisfy the theorem with the brief's
  original start point unchanged. Caught only because the citation was verified against a
  fetched, directly-read secondary source rather than implemented from the brief's
  constants as given.
- **Maratos degenerate-start divergence.** The classic Maratos-effect example (Nocedal &
  Wright, Example 15.4) is textbook-framed as a *slow stall* near the solution under a
  merit-function line search. PSIOPT instead `DIVERGING`s outright in 2 iterations from a
  starting point already exactly on the constraint manifold — a short step at iteration 0
  followed by the equality residual blowing up to ~5e15 at iteration 1. A qualitatively
  different (and faster) failure mode than the textbook's, recorded as observed rather than
  forced to match the reference behavior.
- **VanDerPol toolchain divergence.** `hard_vanderpol` is the parent
  `examples/python_examples/VanDerPol.py` copied verbatim (modulo plotting) — no synthetic
  perturbation was needed because the example already diverges (`KKT = nan` at iteration 0)
  on the current clang22/MKL2026 fast-math toolchain. Tracked separately in project memory
  (`project_vanderpol_diverges`) as toolchain-dependent, not a property of the problem or of
  PSIOPT's algorithm in general, and dropped from the docs example gallery for this reason —
  but it is exactly the kind of live, in-tree, reproducible failure the corpus wants
  permanent regression coverage for regardless of root cause.
- **MountainCar parent nondeterminism.** `hard_mountaincar_badguess`'s perturbation is
  order-sensitive, but so is its *unperturbed parent*: repeated runs of the verbatim
  linear-interpolation-guess parent gave iteration counts of 118, 119, 121, 123, 124, 128
  across probing runs, persisting even with `phase.set_num_partitions(1)` and
  `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` forced in the environment — i.e. the
  nondeterminism is not solely a threading artifact of this problem's defaults. This
  generalizes the brief's SimpleLowThrust-specific noisy-parent caveat to two more parents
  (Zermelo and MountainCar) discovered empirically while building the hard tier — three of
  the eight hard-tier parents are order-sensitive on this toolchain.

## 5. Possible future hardening

- **Nonlinear-degenerate variants.** The three degenerate-tier "converges anyway" findings
  (§4, first bullet) suggest today's exact-duplicate-row degeneracies are too easy for
  PSIOPT's defaults on a small *linear* system. A sharper future perturbation — combining
  row duplication with poor scaling, or moving the degenerate construction onto a nonlinear
  ODE — might actually trip up the defaults where the current three do not, giving the
  degenerate tier more bite for G1-G8 comparisons.
- **A CPLP cycling problem, if the paper becomes accessible.** `lit_cycling` (the
  Chamberlain, Powell, Lemaréchal & Pedersen 1982 watchdog-technique paper's own cycling
  example) was skipped in Task 4: the paper is paywalled on SpringerLink with no accessible
  preprint, and no secondary source found reproduces its actual motivating example (as
  opposed to merely citing the paper's existence). If a future contributor obtains
  institutional access to Mathematical Programming Study 16 (1982), pp. 1-17, this is the
  one literature-tier problem class (merit-function cycling, as distinct from Maratos'
  single-step divergence and WB2000's jamming) not currently represented in the corpus.

## References

- `tests/corpus/README.md` — problem-module contract, harness CLI, JSONL schema.
- `tests/corpus/baselines/2026-07-defaults.jsonl` — raw data for §1.
- `.superpowers/sdd/g0-task-1-report.md` through `g0-task-4-report.md` — per-task
  construction reports (harness build, degenerate tier, hard tier, literature tier).
