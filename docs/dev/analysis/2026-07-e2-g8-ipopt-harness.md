# Shared-transcription Ipopt reference harness — what shipped and first head-to-head data

Date: 2026-07-25. Branch: `feat/g8-ipopt-harness` (off `287d71e9`).
Companion decision records: `2026-07-e2-g7-tr-decision.md` (trust-region cut;
composite-step successor gated on this harness's evidence),
`../notes/first-class-variable-bounds.md` (bounds deferral; decided by a future
lifted-bounds arm of this harness).

## 1. What shipped

Ipopt as an in-process alternative NLP solver on the IDENTICAL transcribed
problem PSIOPT receives — same variables, constraints, `IOScaled` scaling, and
start point — for head-to-head evaluation, promotable later to a user-facing
backend:

- **Backend dispatch seam**: `prob.nlp_solver ∈ {psiopt (default), ipopt}` on
  `OptimizationProblemBase`, routing all five solve modes through one
  `run_nlp_solver` helper. The psiopt path preserves the historic call/read
  order exactly (gate: default-path CBWR bit-identical, full corpus
  status/iteration-identical). The ipopt backend always runs a single NLP solve
  (the feasibility-then-optimize staging has no Ipopt analog). Adaptive-mesh
  refinement requires the built-in solver and is rejected at dispatch.
- **`Ipopt::TNLP` adapter** over the transcribed `NonLinearProgram`: one-time
  slot maps slice the fused upper-triangular KKT CSR into Ipopt's Jacobian
  triplets (constraint-row × primal-col; slack completion excluded) and
  lower-triangle Lagrangian-Hessian triplets; per-iteration evaluation is one
  fused eval plus a flat value copy, and the multi-partition threaded function
  evaluation is identical for both solvers (a fairness property: wall-time
  differences isolate solver algorithms, not evaluation paths). Multiplier
  convention is identity (pinned by an FD cross-check test and a cross-backend
  multiplier assertion). Bounds stay general inequalities on BOTH sides —
  Tycho has no variable-bound concept, so the NLP really is identical.
- **Build integration**: `ENABLE_IPOPT` CMake option (default OFF), pkg-config
  discovery, link-only (never built from source). Ipopt is EPL-2.0; nothing is
  redistributed. The Python surface is identical in both configs
  (`ipopt_available()` reports the build; dispatching without support raises a
  clear error), so the committed stubs are config-independent.
- **Termination mapping**: Ipopt runs stock defaults except tolerances/caps
  mapped from the PSIOPT settings (`tol ← kkt_tol`, `constr_viol_tol ←
  max(econ,icon)`, acceptable tier likewise, `max_iter ← max_iters`); user
  pass-through via `prob.ipopt_options` (type-directed routing through Ipopt's
  option registry). Status normalization to the corpus vocabulary; diagnostics
  in `prob.last_ipopt_result`.
- **Corpus harness backend seam**: the 17 problem modules split into
  `build()` + `SOLVE_MODE` (+ optional `NOTES`/`POST_SOLVE`) with a shared
  driver; `run_corpus.py --backend {psiopt,ipopt}` with a backend scorecard
  column. Problem construction verified byte-identical per module.

## 2. First head-to-head (corpus, 2026-07-25)

PSIOPT at **stock defaults** (none of the opt-in globalization stack enabled)
vs Ipopt 3.14.19 (conda-forge, **MUMPS** linear solver — the MKL-Pardiso Ipopt
build for linear-algebra parity is a planned campaign arm, not this run) at
stock options + matched tolerances. Same machine, same threaded evaluation.

| problem | PSIOPT (defaults) | Ipopt (stock, MUMPS) |
|---|---|---|
| deg_conflicting_equality | failed @500 | failed @5 |
| deg_dup_equality | converged @3 | converged @2 |
| deg_near_infeasible | failed @500 | failed @16 |
| deg_redundant_defects | converged @3 | failed (pre-iteration abort) |
| deg_zero_objective | converged @3 | converged @2 |
| hard_brach_coldstart | converged @24 | converged @43 |
| hard_brach_illscaled | failed @500 | converged @16 |
| hard_cartpole_tightbounds | converged @95 | failed @116 |
| hard_hypersens_stiff | acceptable @103 | converged @7 |
| hard_lowthrust_badguess | diverged @1 | failed @1 |
| hard_mountaincar_badguess | failed @1000 | converged @276 |
| hard_vanderpol | diverged @1 | failed (pre-iteration abort) |
| hard_zermelo_wrongbasin | diverged @907 | converged @28 |
| lit_hs13 | acceptable @77 | converged @46 |
| lit_maratos | converged @40 | converged @25 |
| lit_powell_badscaled | converged @22 | converged @11 |
| lit_wb2000 | failed @500 | failed @16 |

Solve-or-acceptable: **PSIOPT 9/17, Ipopt 10/17** — close, before any of the
opt-in stack is enabled. Honest observations:

- **Ipopt's wins overlap the shipped opt-in stack's known rescues**: wb2000 and
  near_infeasible (both failed here at defaults) are rescued by
  `inertia_mode = proximal_regularization` (converged @95 / acceptable @86 in
  that mode's own campaign); conflicting_equality's fast infeasibility
  certificate (@5 for Ipopt vs a 500-iteration burn at defaults) is what
  nested-ℓ1 restoration provides (@6–16). The full-factorial campaign with the
  stack enabled is the real comparison; this table is the defaults anchor.
- **Ipopt-only rescues to study**: brach_illscaled @16 (its native NLP scaling
  machinery is the prime suspect — PSIOPT has no counterpart), mountaincar
  @276, and zermelo_wrongbasin @28 — the latter converges to a KKT point in the
  wrong basin, a legitimate solve of a problem this program had classified as
  globalization-unfixable; worth a basin/objective comparison in the campaign.
- **PSIOPT-only wins**: redundant_defects (Ipopt/MUMPS aborts before iterating
  on the rank-deficient defect rows that the Pardiso-backed KKT path handles at
  @3) and cartpole_tightbounds.
- Iteration counts are not cross-solver comparable (different algorithms);
  status is the primary metric here. Wall-time comparison is deferred to the
  pardisomkl arm.

Raw scorecards: `g8-corpus-task4-psiopt.csv` / `g8-corpus-task4-ipopt.csv`
(session records; the table above is the complete status/iteration content).
The three ipopt-backend rows for NOTES-bearing modules re-run after a notes
formatting fix; status/iterations/objective unaffected.

## 3. Gates

- Default build 0-warn; ctest **1666/1666** (default) and **1642/1642**
  (ENABLE_IPOPT config incl. the adapter group: FD derivative cross-check,
  structural shape, cross-backend parity with multiplier assertions, Phase
  end-to-end, exception latching, option rejection). Full pytest 381 passed
  (both-config backend tests: 5+1 skip / 5+1 skip). Stubs regenerated.
- 34/34 Python examples; C++ brachistochrone "Optimal Solution Found" (final
  point (10, 5), objective ≈ 1.8013 s).
- Default-path CBWR: **bit-identical** across all 31 deterministic examples vs
  the pre-change baseline, at every task boundary and at final HEAD.
- Corpus psiopt backend: **status/iteration/objective-identical 17/17** vs the
  pre-change scorecard (captured on the pre-change install, same session).
- Bench: 128 lanes, one flag — `BM_Phase_Transcribe_64seg` +28.5% single-shot,
  **refuted** by 5-rep re-measurement (median 340 μs vs 319 μs baseline,
  +6.6%, cv 1%): under threshold, does not reproduce; residual consistent with
  code-layout displacement from the enlarged problem-base object (the branch
  touches no transcription code, and the backend is opt-in besides).
- ENABLE_IPOPT config validated end-to-end on this machine per the
  build-config rule: configure (pkg-config discovery from the conda env),
  full build, full ctest, Python solve, full-corpus backend run.
  `_tychopy` links `libipopt.so.3` (ldd-verified).

## 4. Known limitations / next steps

- The fair-comparison arm needs an Ipopt built with MKL Pardiso
  (`linear_solver=pardisomkl`) so both solvers share the sparse linear
  algebra; the conda MUMPS build is the adapter-validation vehicle only.
- The evaluation campaign proper: full-factorial PSIOPT configuration sweep ×
  corpus × examples, with this backend as the reference column; promotion
  criteria per the program design.
- Lifted-bounds arm (Ipopt-side `x_L/x_U` lifting) to price first-class
  variable bounds — see the bounds note.
- Secondary build dirs re-run the user-site install on relink (a
  `build-ipopt` build replaces the installed extension with the Ipopt-enabled
  one); harmless surface-wise but config-relevant for CBWR instruments —
  restore the intended config's install after secondary builds.
