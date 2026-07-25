# Proximal regularization inertia mode — corpus evidence and the elastic-relaxation decision

Date: 2026-07-24. Branch: `feat/e2-g6-implicit-tr-regularization` (HEAD `9dad7632`).
Corpus harness: `scripts/run_corpus.py`, 17 problems, `MKL_CBWR=AUTO,STRICT`, ×2
repeats per configuration.

## 1. What shipped

`inertia_mode = proximal_regularization` beside the (default, byte-identical)
`classic` ladder:

- **Always-on dual shift** −δ_c on the constraint-row diagonal (e/i pivot slots),
  δ_c = 1e-8·μ^0.25 — Ipopt's `jacobian_regularization_value/exponent` constants and
  its `perturb_always_cd` semantics (verified at coin-or/Ipopt 72a29c9). Suppressed
  while nested ℓ1 restoration is active (the elastic pivots own those slots and the
  condensed step-recovery algebra assumes exact pivot values).
- **Persistent decaying primal shift** ρ_k on the Hessian-block diagonal, floored at
  1e-10 (Cipolla–Gondzio floor), replacing the classic retry-zero attempt; decays by
  `decr_h` toward the floor while inertia stays correct, re-seeds from the decayed
  ρ_k + last-ladder-delta after a ladder episode. The persistence *dynamics* are
  Tycho-original (the convex-QP literature the mode draws on has no nonconvex inertia
  story; Ipopt has no always-on primal mode) — this corpus differential is their
  acceptance evidence.
- The escalation ladder, warm-start memory, and inertia predicate are unchanged; a
  singular base factorization is upgraded from warn-only to a ladder trigger on the
  mode branch only.
- Diagnostics: `result.last_prox_reg_primal` / `last_prox_reg_dual` (−1 sentinels;
  the shifts applied at the last factorized iteration otherwise).

Full mechanism documentation and citations:
`include/tycho/detail/solvers/globalization/inertia_regularization.h`.

## 2. Corpus A/B

All six configurations are status-stable 17/17 between their two repeats;
iteration counts are also repeat-identical everywhere except one wobble on the
documented jitter-prone problem (`hard_cartpole_tightbounds` under `merit-l1`:
diverged @160 vs @162). Single-run diffs below are therefore reliable, with
cartpole rows read loosely. The harness records status/iterations/wall-time;
per-solve factorization counts are not a harness field, so iteration deltas serve
as an (imperfect) proxy for the factorization-savings design intent.

### 2.1 Headline: defaults vs `proximal_regularization` alone

**Two genuine status rescues, zero status regressions** (the harness's own summary
says "3 improved" because it also ranks zermelo's diverged→failed relabel as an
improvement; both labels are non-solutions, so that row is a wash):

| problem | classic | prox |
|---|---|---|
| lit_wb2000 | failed @500 | **converged @95** |
| deg_near_infeasible | failed @500 | **acceptable @86** |
| deg_zero_objective | converged @3 | converged @2 |
| lit_maratos | converged @40 | converged @37 |
| lit_hs13 | acceptable @77 | acceptable @75 |
| hard_brach_coldstart | converged @24 | converged @44 |
| hard_cartpole_tightbounds | converged @95 | converged @105 |
| hard_hypersens_stiff | acceptable @103 | acceptable @104 |
| hard_zermelo_wrongbasin | diverged @907 | failed @1000 |

(Rows below the rule are the honest costs: brach_coldstart pays +20 iterations,
cartpole +10 — cartpole is the corpus's documented jitter-prone problem —
hypersens +1, and zermelo's genuine wrong-basin non-solve rides to the iteration
cap instead of the divergence abort; neither label is a solution, so it is a
wash.) Everything else is iteration-identical.

The two rescues are exactly the mode's design case: wb2000's near-rank-deficient
geometry and near_infeasible's degenerate constraint set were previously
factorization-quality failures; the persistent shifts turn both into solves.
Engagement is confirmed by the mode's own diagnostics on the wb2000 solve: at
convergence `last_prox_reg_primal ≈ 2.5e-4` (the persistent primal shift is live,
more than six orders above its 1e-10 floor — the implicit trust region did the
work) and
`last_prox_reg_dual = 1e-11` (= 1e-8·μ^0.25 at the converged μ ≈ 1e-12, the dual
shift decaying to negligible exactly as the μ-scaling intends).
The Maratos/hs13/zero-objective single-digit savings match the
implicit-trust-region prediction (no retry-zero factorizations while curvature is
temporarily bad).

### 2.2 Composed with the restoration stacks

`merit-l1` vs `merit-l1-prox` — 3 improved, 0 status-regressed, with costs:
near_infeasible failed@498 → **acceptable@224**; brach_coldstart 49 → 31;
zero_objective/redundant −1 each; BUT wb2000 converged 90 → 241 (+151 iterations,
still converged), cartpole diverged@160 → failed@498 (the jitter-prone problem),
zermelo diverged@900 → failed@998 (wash, as above).

`filter-mon-l1` vs `filter-mon-l1-prox` — 2 improved, **1 regressed**:
mountaincar_badguess acceptable@661 → **converged@969** (status win, iteration
cost); hypersens_stiff 152 → 120; brach_coldstart 36 → 26; conflicting_equality
fails-fast @16 → @6; wb2000 37 → 97 (+60, still converged); cartpole 95 → 105 and
the zermelo diverged→failed relabel (both as in the standalone comparison); and the
one real loss: **brach_illscaled acceptable@339 → failed@498**.

Two mechanism observations for reviewer attention (within-design behavior, possible
future tuning): (1) after a ladder episode the successful shift is remembered twice
— ρ_{k+1} carries its decayed value as the next base AND `Hpert0` warm-starts the
ladder from the same decayed delta, so the first ladder rung of a subsequent episode
lands at roughly twice the intended shift; a plausible mechanism for the composed
wb2000 iteration inflation. (2) The singular-base upgrade retries through the
primal ladder only (there is no escalating constraint-side remedy beyond the fixed
δ_c), so a persistently singular base spends 1 + `max_refac` factorizations per
iteration.

### 2.3 Parity verdict

Against the equivalent-or-better standard: **standalone, the mode meets it
outright** (zero status regressions, two major rescues, single-digit wins on the
literature problems, bounded iteration costs on two problems). **Composed with the
restoration stacks it is mixed** — real wins (near_infeasible, mountaincar,
hypersens) against real costs (wb2000 iteration inflation, the brach_illscaled loss
under the filter stack). As an opt-in mode with `classic` remaining the default,
this ships; the composed cells are recorded here so the evaluation campaign can
weigh them per-preset rather than folding the mode into the robust preset blindly.

## 3. Elastic/penalty-relaxation decision (required close-out)

The third restoration-trio member (LOQO/Knitro `bar_relaxcons` lineage —
formulation-level elastic relaxation) was cut from the nested-ℓ1 stage with the
commitment to reconsider it here, on this stage's degenerate-tier evidence. The
degenerate tier now reads:

| problem | best result | via |
|---|---|---|
| deg_dup_equality | converged @3–5 | every configuration (best: defaults/filter) |
| deg_redundant_defects | converged @3–6 | every configuration (best: filter; prox −1) |
| deg_zero_objective | converged @2 | prox configurations |
| deg_conflicting_equality (infeasible) | fails-fast @6–16 | ℓ1 certificate (+prox: @6–7) |
| deg_near_infeasible | acceptable @86 | prox alone |

**Decision: elastic is dropped from the program.** The tier is well-served by
regularization + ℓ1: the rank-deficient/duplicated/zero-curvature problems are
solved outright (the constraint-row regularization now does structurally what
elastic's slack pivots would have done), the genuinely infeasible problem gets its
fast infeasibility certificate from ℓ1 restoration (6–16 iterations), and the one
previously-open degenerate problem (near_infeasible) gets its best-ever result from
the regularization mode itself — there is no remaining gap that a formulation-level
relaxation would plausibly close. (A single configuration also covers the whole
tier by itself: `merit-l1-prox` solves or correctly dispositions all five.) The
earlier program note asking whether the dual shift recovers the elastic-pivot
speedup once observed on dup_equality (58 → 5 iterations in a pre-Pardiso-defaults
tree) is moot: that problem now converges @3 at classic defaults — the
factorization-defaults change absorbed it before this stage.
Reversal condition: a future workload class where
constraints must be *softly violated at the solution* (elastic's actual modeling
niche, distinct from restoration) would reopen this as a modeling feature, not a
robustness mechanism.

## 4. Gate results

- Build 0-warn; ctest **1661/1661** (11 new regularization tests incl. a
  rank-deficient-KKT convergence pin and an in-phase suppression composition test);
  full pytest incl. 17 new surface tests; stubs snapshot regenerated.
- 34/34 Python examples pass; C++ brachistochrone "Optimal Solution Found",
  1.801295 s.
- Default-path CBWR: bit-exact across all 34 examples against a
  same-toolchain rebuild of the pre-change tree (known-noisy trio excluded as
  always: MultiSpacecraftOptimization, ParallelParking, SimpleLowThrust).
- Bench: 128 lanes, two flags, both dispositioned. `BM_BumpAllocator_Resize`
  +12.1% — refuted by 5-rep re-measurement (133 ns median vs 129 baseline, the
  fourth single-shot flag/refutation of this lane in this series).
  `BM_GF_Clone_PolarLT` +28.6% (99→130 ns) — reproduces under 5 reps, but a
  same-machine-state rebuild of the pre-change tree also re-measured at 99 ns
  while every other micro-lane matches baseline within ±2.4%; the branch changes
  no clone/type-erasure/bench source (empty diff under those paths), so this is
  binary code-layout displacement of one alignment-sensitive 100 ns lane by the
  grown solver objects, not a runtime cost of the change (the lane executes no
  solver code, and the mode is opt-in besides).

Raw scorecards: `g6-score-*.jsonl` (session records; headline tables above are the
complete status/iteration deltas).
