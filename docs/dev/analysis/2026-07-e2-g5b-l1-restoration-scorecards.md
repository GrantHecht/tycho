# PSIOPT nested ℓ1 feasibility-restoration scorecards

**Date:** 2026-07-23
**Branch:** `feat/e2-g5b-nested-l1-restoration` (algorithmic HEAD `a1ddb02b`; two documentation commits follow)
**Data:** nine `scripts/run_corpus.py --cbwr --repeat 2` captures (17 problems × 2 repeats =
34 records each) — three engaged acceptance families (`merit`, `filter`+monitored governor,
`funnel`+monitored governor), each in the final `restoration_mode = l1_nested` state, plus the
proximal-restoration and no-restoration reference columns each is measured against — together
with two intermediate corpus snapshots taken between the two barrier-parameter defects this
document's own evidence campaign caught and fixed (the "both live" and "first fix only"
captures used for the before/after narratives in §4).
**Task class:** MEASUREMENT, plus the two barrier-parameter fixes this document's own campaign
discovered the need for and that ship as part of the record (`bfa869ca`, elastic
complementarity in the barrier oracle; `a1ddb02b`, the monotone in-phase barrier schedule), and
the earlier slack-completion seam fix (`71c17882`) a per-task gate test caught during
implementation. Everything here scores the nested ℓ1 restoration mode (`restoration_mode =
l1_nested`, default off, the second of the three-strategy restoration trio in `restoration.h`)
against the corpus, against the proximal mode-switch scored in
[`2026-07-e2-g5a-scorecards.md`](2026-07-e2-g5a-scorecards.md), and against the no-restoration
funnel/filter/merit references those documents established.

The nested ℓ1 restoration mode solves the ℓ1 elastic feasibility reformulation of the current
point as a **condensed, in-place** feasibility phase — reusing the outer barrier algorithm's own
KKT system rather than constructing and running a separate nested solver instance. Per
constraint row with residual `c` (equality rows `c = h_j(x)`; inequality rows `c = g_j(x) +
s_j`, the slack-completed residual), it introduces an elastic pair `(n, p) ≥ 0` with `c + n − p =
0` and minimizes `ρ·Σ(n + p) + (η(μ)/2)·‖D_R (x − x_R)‖²`. The elastic pair and its bound
multipliers are eliminated analytically per row (a positive pivot `n/z_n + p/z_p` the seam
negates into the KKT `(y,y)` diagonal, and a condensed row RHS), so no problem dimension
changes. Entry is refused at a near-feasible point (violation ≤ 0.1× `econ_tol_`) or once the
per-phase entry budget `max_feas_rest` is exhausted; a nested strategy additionally gets a
**soft feasibility pre-stage** (§6) before it commits to the full mode-switch. The formulation,
the closed-form elastic-slack initialization, the per-step condensation algebra, and the exit
machinery are transcribed from Ipopt at pinned commit
`72a29c9aab198afa0dbb940339022a22c415a4eb` (`src/Algorithm/IpRestoIpoptNLP.{hpp,cpp}`,
`IpRestoIterateInitializer.cpp`, `IpRestoMinC_1Nrm.cpp`, and `IpBacktrackingLineSearch.cpp` for
the soft pre-stage), with the disclosed deviations listed in §2.

## 1. Constants

All values are the shipped Ipopt option defaults at the pinned commit; the entry-guard and
stall-failure factors are shared with the proximal mode-switch (`proximal_restoration.h`) and
carried at the same literature default.

| Constant | Value | Role | Ipopt source (commit `72a29c9a`) |
| --- | --- | --- | --- |
| `kRestoPenaltyParameter` (ρ) | `1e3` | ℓ1 elastic penalty on `Σ(n+p)` | `resto_penalty_parameter`, `IpRestoIpoptNLP.cpp` |
| `kRestoProximityWeight` | `1.0` | factor in η(μ) = weight·√μ | `resto_proximity_weight`, `IpRestoIpoptNLP.cpp` (exponent 0.5 hardcoded) |
| `D_R` | `diag(1/max(1,|x_R_i|))` | per-coordinate proximity scaling | `IpRestoIpoptNLP.cpp` `InitializeStructures` |
| entry `resto_mu` | `max(μ_outer, ‖h‖∞, ‖g+s‖∞)` | elastic-slack init + phase start barrier | `IpRestoIterateInitializer.cpp` `SetInitialIterates` |
| `kBoundMultResetThreshold` | `1e3` | re-entry bound-multiplier reset threshold | `bound_mult_reset_threshold`, `IpRestoMinC_1Nrm.cpp` |
| `kNearFeasibleGuardFactor` | `0.1` | entry refused if violation ≤ factor·`econ_tol_` | proximal-switch shared constant |
| `kRestoFailureFeasibilityFactor` | `1e2` | stall classified infeasible above factor·`econ_tol_` | `resto_failure_feasibility_threshold`, shared |
| `kSoftRestoPdErrorReductionFactor` | `1 − 1e-4` | soft-step accept if trial PD-error ≤ factor·current | `soft_resto_pderror_reduction_factor`, `IpBacktrackingLineSearch.cpp` |
| `kMaxSoftRestoIters` | `10` | successive soft steps before escalating to the full switch | `max_soft_resto_iters`, `IpBacktrackingLineSearch.cpp` |

## 2. Disclosed deviations from a literal Ipopt restoration solve

Each is stated in the shipped header docstrings (`l1_restoration.h`,
`feasibility_switch_recovery.h`) with its consequence; they are the reasons iteration
trajectories differ from Ipopt's even where the per-step algebra is proven equivalent.

- **Condensed in-place, not a literal second solver.** Ipopt builds a separate five-block
  restoration NLP and runs a nested algorithm instance; this mode solves the same restoration
  problem with the same per-step algebra (established by block-elimination equivalence, verified
  numerically to 1e-13) inside the outer loop. There is no inner/outer iteration-count split —
  the phase counts iterations with a single in-mode counter, and iteration-level trajectories
  differ from Ipopt's.
- **η(μ) is recomputed live on every evaluation** (η = weight·√μ), matching Ipopt's `Eta(mu)`
  method — unlike the proximal mode-switch, which freezes its proximal coefficient ζ once at
  entry.
- **Constraint multipliers reset to zero on exit and entry, not recomputed by least squares.**
  Ipopt's re-entry path calls `least_square_mults` with the reset threshold at its shipped
  default 0, whose body sets the multipliers to zero at that default rather than computing the
  LSQ estimate. This mode transcribes the shipped-default behavior and does not implement the
  dormant LSQ branch (no knob exposes it).
- **Single-measure floors.** Ipopt carries separate scaled/unscaled tolerances; Tycho carries
  one `econ_tol_`, used by both the entry guard and the stall-failure classification.
- **No separate restoration iteration budget.** The outer iteration limit already bounds the
  in-place phase; the per-solve entry budget remains the shared `max_feas_rest`.
- **Soft pre-stage placement differs, with a disclosed consequence.** Ipopt tries soft
  restoration the moment its backtracking line search fails; this solver has a second-order
  correction / watchdog recovery ladder Ipopt lacks and tries the soft pre-stage only once that
  ladder is exhausted, so soft steps are attempted strictly later. The acceptance-strategy
  feasibility notification (which, for the filter/funnel/modern-merit strategies, augments the
  filter) stays at the full restoration entry and is never issued during the pre-stage — so a
  soft step is tested against the un-augmented optimality-phase acceptance state, one Ipopt
  (which augments at soft-stage start) could have rejected. The pre-stage cannot loop on this:
  the successive counter is cleared only by a genuine optimality acceptance, so at most
  `kMaxSoftRestoIters` such steps occur before escalation, which performs the augmentation.

## 3. Master table (final, repeat 1, status / iterations)

The `l1` columns report the shipped state at HEAD `a1ddb02b` — after both barrier-parameter
fixes (§4.2, §4.3). Each family is measured primarily against the **proximal** reference (the
mode this stage's trio succeeds), with the no-restoration reference shown for context. For the
`merit` family the fair no-restoration baseline is `merit` acceptance with restoration off
(`merit-alone`), not `defaults` (which is `classic_merit`); the filter and funnel families use
their own monitored-governor no-restoration captures, which reproduce
`2026-07-e2-g5a-scorecards.md` exactly. **Bold** marks a status change or a notable iteration
swing from the proximal column.

| Tier | Problem | Merit noR | Merit prox | Merit **l1** | Filter noR | Filter prox | Filter **l1** | Funnel noR | Funnel prox | Funnel **l1** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degen | `deg_dup_equality` | conv/56 | conv/58 | conv/**5** | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 |
| degen | `deg_conflicting_equality` | fail/500 | fail/498 | fail/**7** | fail/500 | fail/500 | fail/**16** | fail/500 | fail/500 | fail/500 |
| degen | `deg_zero_objective` | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 |
| degen | `deg_redundant_defects` | conv/56 | conv/58 | conv/**6** | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 | conv/3 |
| degen | `deg_near_infeasible` | fail/500 | fail/498 | fail/498 | fail/500 | fail/500 | fail/498 | fail/500 | fail/500 | fail/500 |
| hard | `hard_vanderpol` | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 |
| hard | `hard_brach_coldstart` | conv/35 | conv/48 | conv/49 | conv/36 | conv/36 | conv/36 | conv/22 | conv/22 | conv/22 |
| hard | `hard_brach_illscaled` | fail/500 | fail/498 | fail/498 | fail/500 | fail/499 | **acc**/**339** | fail/500 | fail/498 | fail/498 |
| hard | `hard_zermelo_wrongbasin` | div/973 | fail/1000 | **div**/**898** | div/845 | fail/1000 | **div**/**884** | div/667 | div/667 | div/667 |
| hard | `hard_mountaincar_badguess` | fail/1000 | fail/998 | fail/998 | fail/1000 | fail/999 | **acc**/**661** | fail/1000 | fail/1000 | fail/1000 |
| hard | `hard_lowthrust_badguess` | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 | div/1 |
| hard | `hard_cartpole_tightbounds` | fail/500 | fail/498 | **div**/**160** | conv/95 | conv/95 | conv/95 | conv/95 | conv/95 | conv/95 |
| hard | `hard_hypersens_stiff` | acc/103 | acc/220 | acc/**121** | acc/103 | acc/209 | acc/**152** | acc/103 | acc/103 | acc/103 |
| lit | `lit_wb2000` | conv/114 | conv/327 | conv/**124** | conv/43 | conv/46 | conv/37 | conv/33 | conv/34 | conv/32 |
| lit | `lit_maratos` | div/2 | div/2 | div/2 | div/2 | div/2 | div/2 | div/2 | div/2 | div/2 |
| lit | `lit_hs13` | acc/142 | acc/138 | acc/**127** | acc/76 | acc/76 | acc/76 | acc/76 | acc/76 | acc/76 |
| lit | `lit_powell_badscaled` | conv/21 | conv/24 | conv/**14** | conv/21 | conv/23 | conv/**14** | conv/12 | conv/12 | conv/12 |

`conv` = converged, `acc` = acceptable, `fail` = failed, `div` = diverged. Repeat stability: the
filter and funnel `l1` captures are repeat-1/repeat-2 identical on status, iteration count, and
objective; the merit `l1` capture is identical except for `hard_cartpole_tightbounds`, whose
thread-jitter is discussed in §5.3.

## 4. The three defects this stage's own evidence caught

The centerpiece of this stage is not the final table — it is the sequence of three distinct
defects, each visible only through a different verification instrument, that stood between the
first wired implementation and the numbers above. None of them was hypothetical; each froze or
broke a real corpus problem, and each is fixed in the shipped state.

### 4.1 The inequality slack-completion seam — caught by an end-to-end gate test

The condensed restoration seam runs inside `eval_nlp`, one step out of phase with the solver's
slack machinery. `eval_kkt_no` emits the raw constraint value `g(x)`; the slack-completed
residual `g(x) + s` is formed by a later `apply_reset_slacks` pass that runs *after* `eval_nlp`.
The first wired seam therefore condensed the **wrong residual** — it captured `c = g(x)` rather
than `c = g(x) + s` — and, worse, the condensed row RHS it wrote was then **clobbered** by that
same `apply_reset_slacks` pass, which zeroed any row whose value was negative. The inequality
Newton direction was destroyed and the exit measure mis-read the original-problem infeasibility.

This was invisible to the equality-only tests: when a problem has no inequalities the whole
slack-reset block is skipped, so the equality condensation was clean, and the seam-level sign
coverage had only ever exercised equality rows. It surfaced the moment a per-task end-to-end
gate test added an inequality: a 2-variable QP with one non-binding inequality, forced into the
nested phase, returned `NOTCONVERGED` where its equality-only twin converged. The trace showed
the phase stalling at θ = 98 = ‖g + s‖ at entry, with the inequality channel never moving. The
fix captures the slack-completed residual in the seam and suppresses `apply_reset_slacks` while
a nested phase is active (the seam has already produced the final condensed inequality RHS; the
elastic fraction-to-boundary caps keep the slacks strictly positive). Both edits are gated on
`restoration_ && is_active() && is_nested()`, leaving the default, equality-only, and proximal
paths byte-identical. Two through-the-assembled-KKT inequality-row sign tests were added to
close the coverage gap that let it slip. Commit `71c17882`.

### 4.2 The μ-collapse freeze — caught by the corpus evidence campaign

With the seam correct, the first full corpus scorecard still showed a systematic regression
under both the merit and filter families: `lit_wb2000` went from converged (without restoration)
to failed at the iteration cap, and `hard_zermelo_wrongbasin` from failed to diverged. A probe
reading the solver's restoration counters directly (session-local diagnostic trace, summarized in §4.2) pinned the
mechanism. During a nested phase the barrier-parameter oracle — the governor's `update_barrier`
and the monitored governor's KKT-error monitor — read complementarity from the **original**
slack/multiplier pairs only. When restoration is entered late in a solve, the original
complementarity is already at solve tolerance (~1e-12), so the oracle collapsed μ from the
restoration entry value (`resto_mu` ≈ 2.4) to the machine floor (1e-12) within about six
iterations, *before* the elastic slacks had shrunk. The condensed elastic pivot `(n² + p²)/μ`
then exploded to ~4e12, the condensed constraint rows decoupled, and the phase froze: subproblem
KKT ~1e-13, condensed residual stuck ~0.2, original infeasibility stuck ~2.0, no exit test able
to fire, burning to the iteration cap (entries = 2, in-mode iterations = 493). Ipopt is immune
because its restoration NLP has its own constraint qualification: complementarity is taken over
*all* restoration variables, including the elastic `(n, p, z_n, z_p)`.

The fix folds the elastic bound pairs into the complementarity aggregation that feeds the
barrier machinery whenever a nested phase is active: the elastic products are aggregated
separately (`NestedL1Restoration::nested_complementarity`) and combined with the original
aggregate as a union — union min is the min of the two mins, union max the max of the two maxes,
union average the count-weighted average. The original-pairs reduction is never re-run or
reordered, so off the nested path the aggregates are byte-identical and the CBWR invariant holds
by construction. This cleared the filter and funnel families outright — measured between the
"both defects live" and "first fix only" corpus snapshots:

| Problem (filter+monitored) | both live | after elastic-complementarity fix |
| --- | --- | --- |
| `lit_wb2000` | failed / 499 | **converged / 37** |
| `hard_mountaincar_badguess` | failed / 998 | **acceptable / 661** |
| `hard_brach_illscaled` | failed / 498 | **acceptable / 339** |

`hard_mountaincar_badguess` is the motivating stuck-in-mode problem the proximal scorecards
flagged as unrecovered and named this stage's target workload; the elastic-complementarity fix
is what moved it, under the filter+monitored family, from `failed` at the cap to `acceptable`.
Commit `bfa869ca`.

### 4.3 The free-oracle μ race — caught by a second probe, fixed with a gated schedule

The elastic-complementarity fix did **not** clear the merit family: `lit_wb2000` under merit
(classic adaptive governor) was still frozen at the cap. A second probe
(session-local diagnostic trace, summarized here) exposed a distinct mechanism. Under a *free*-mode oracle the
barrier machinery drives every complementarity product — including the elastic bound pairs —
toward whatever μ the oracle proposes, so any μ is self-consistent. Feeding the elastic
complementarity in cannot prevent the collapse, because the products simply chase μ down. The
oracle still races μ to the floor before the elastics shrink; the terminal state is the μ ≈ 0 ℓ1
optimality manifold reached in the **wrong basin** — a local ℓ1 minimizer with a positive
elastic `p`, the constraint multiplier `y` pinned exactly at the ℓ1 dual bound `ρ = 1e3`, the
pivot at ~4e12 and the rows decoupled — then frozen to the cap.

The fix transcribes Ipopt's default restoration `mu_strategy`, which is **monotone**: while a
nested phase is active the barrier parameter follows the safeguarded Fiacco–McCormick ladder
anchored at the entry `resto_mu`, never a free-mode oracle. It advances μ only when the
restoration barrier subproblem is sufficiently solved (`barrier_subproblem_error ≤
kBarrierTolFactor·μ`), and because the subproblem error reads the barrier-infeasibility measure
into which the previous fix already folds the elastic complementarity, the gate cannot fire
while the elastics are at restoration scale — μ stays anchored until they actually shrink.

The one deviation from a literal "force it regardless of governor" is evidence-driven and
disclosed. Forcing the monotone schedule on *every* governor fixed the merit family
(`lit_wb2000` → converged / 124) but **regressed the already-recovered filter+monitored
family**: `hard_mountaincar_badguess` fell from `acceptable` back to `NOTCONVERGED`, because
overlaying a second, differently-anchored monotone schedule on the monitored governor — which
already forces a monotone safeguarded decrease through its own free↔monotone monitor, the exact
reason that family did not suffer the race — let the restoration subproblem "converge" at high μ
and exit while still infeasible. The schedule is therefore gated by a new governor predicate,
`provides_restoration_barrier_safeguard()` (default `false`, so the free classic-adaptive
governor is routed to the monotone schedule; `MonitoredBarrierGovernor` overrides it to `true`
and keeps its own in-phase machinery byte-identical). This closes the free-oracle μ-race for the
family that suffers it while leaving the recovered monitored family untouched, and it makes the
free-mode oracles provably unreachable in-phase under a free governor. Commit `a1ddb02b`:

| Problem | after elastic-complementarity fix | after monotone schedule (shipped) |
| --- | --- | --- |
| `lit_wb2000` (merit) | failed / 499 | **converged / 124** |
| `lit_wb2000` (filter+monitored) | converged / 37 | converged / 37 (byte-identical) |

The merit `lit_wb2000` result (124 iterations) beats even the proximal-merit reference (327),
and the monitored path is byte-identical across the fix, confirming the gate leaves the
recovered family alone.

## 5. Findings

### 5.1 Where the nested mode wins: fast certificates, degenerate speedups, ill-scaled rescues

Measured against the proximal reference each family succeeds:

- **Fast infeasibility certificates.** On the genuinely-infeasible `deg_conflicting_equality`
  the elastic reformulation certifies infeasibility fast instead of grinding to the cap: 498 → 7
  iterations under merit, 500 → 16 under filter. The elastic penalty term drives the ℓ1
  infeasibility to its local minimum and the exit machinery recognizes the locally-infeasible
  point promptly.
- **Degenerate-tier speedups under merit.** `deg_dup_equality` (58 → 5) and
  `deg_redundant_defects` (58 → 6) collapse from dozens of iterations to a handful. The elastic
  pivots incidentally regularize the rank-deficient constraint rows these problems carry — a
  side effect worth flagging for the later stage that will reconsider whether an explicit
  elastic relaxation belongs in the regularization path, since it is direct evidence that the
  elastic machinery already helps the degenerate tier.
- **The motivating stuck-in-mode class, recovered under filter.** `hard_mountaincar_badguess`
  (failed / 999 → **acceptable / 661**) and `hard_brach_illscaled` (failed / 499 →
  **acceptable / 339**) are the problems the proximal scorecards left stuck in restoration mode
  to the budget. Under the filter+monitored family the nested mode moves both from `failed` at
  the cap to `acceptable` — the first genuine status improvement on `hard_mountaincar_badguess`
  in this series driven by a restoration strategy.
- **`lit_wb2000` throughout.** Converged in every engaged family, and cheaper than the proximal
  reference under merit (327 → 124) and filter (46 → 37).
- **Stiff and ill-conditioned literature problems.** `hard_hypersens_stiff` costs far fewer
  iterations than the proximal mode at the same `acceptable` status (merit 220 → 121, filter
  209 → 152), and `lit_hs13` (merit 138 → 127) and `lit_powell_badscaled` (merit 24 → 14,
  filter 23 → 14) all shorten.

### 5.2 Honest negative: `hard_zermelo_wrongbasin` is not recovered — and loses the proximal demotion

`hard_zermelo_wrongbasin` is the second problem this series has tracked as unrecovered, and the
nested mode does not clear it under any engaged family. Worse than a null result: the proximal
mode-switch had demoted it from `diverged` to a bounded `failed / 1000` (a severity improvement
the proximal scorecards counted as progress), and the nested mode **gives that demotion back** —
it returns to `diverged` under both merit (proximal `failed/1000` → `diverged/898`) and filter
(proximal `failed/1000` → `diverged/884`). The terminal state is the same wrong-basin signature
seen in the free-oracle race diagnosis: the constraint multiplier pinned at the ℓ1 dual bound,
the point a local ℓ1 minimizer with a positive elastic. This is a genuine basin-geometry
failure, not an under-budgeting one — the elastic feasibility subproblem has a local minimizer
that is infeasible for the original problem, and the restoration phase converges to it. It is
reported here as a clear regression against the proximal reference on this one problem, offsetting
the filter-family rescues above; whether the elastic relaxation (the third member of the
restoration trio) reaches this basin is an open question this stage does not answer.

### 5.3 `hard_cartpole_tightbounds`: thread-jitter noise, not a restoration effect

Under the merit family the final `l1` capture records `hard_cartpole_tightbounds` as
`diverged / 160`, versus `failed / 498` under the proximal reference — a status change flagged
in the master table. This is **not** attributable to restoration; it is the multithreaded
solver's known iteration-count nondeterminism on this problem. The evidence is in the repeat
diffs. Between the two final merit repeats, `hard_cartpole_tightbounds` lands `diverged / 160`
both times but with the objective jittering (90.0668 vs 90.0349) at identical iteration count;
and in the intermediate "both defects live" merit snapshot the same problem flipped
`diverged / 297` (repeat 1) versus `failed / 498` (repeat 2) — a diverged/failed boundary flip
between two runs of the same configuration. The problem sits on that boundary and tips across it
run to run regardless of the restoration mode. Under the filter and funnel families, where the
governor differs, `hard_cartpole_tightbounds` stays cleanly `converged / 95` in every capture.
No restoration attribution is claimed for this problem in either direction.

### 5.4 The funnel family is a clean null

The funnel+monitored family is unchanged across no-restoration, proximal, and nested columns on
every problem but two trivial iteration deltas (`lit_wb2000` 34 → 32, otherwise identical). The
funnel's own acceptance ladder rarely exhausts — the funnel test prevents most of the step
rejections that would trigger a restoration switch — so the nested mode almost never engages
under it, and where it does it engages harmlessly and hands the floor back. Nested restoration
under the funnel is neither a rescue nor a hazard on this corpus; it is inert.

## 6. The soft feasibility pre-stage

A nested strategy gets a soft pre-stage the proximal mode-switch does not. Before committing to
the full restoration switch at a ladder-exhausted step rejection, the outermost recovery link
returns a soft-step signal instead: the solver takes the full fraction-to-boundary step on the
current search direction and tests it under a primal–dual-error reduction rule (accept while the
trial primal–dual error is at most `1 − 1e-4` of the current). A successive-iteration counter
allows up to `kMaxSoftRestoIters` such steps in a row before escalating to the real
mode-switch; the counter resets whenever an ordinary optimality-phase step is accepted (the
pre-stage exiting because the outer acceptance test recovered on its own) or at any mode-switch
reset. The soft steps are ordinary optimality-phase steps — the acceptance strategy stays in the
optimality phase across the pre-stage, which is what lets the pre-stage exit be the outer loop's
own acceptance recovering, and guarantees the filter augmentation is issued exactly once per
episode and never during the pre-stage. Transcribed from Ipopt's soft restoration phase
(`IpBacktrackingLineSearch.cpp`, `TrySoftRestoStep` + the `in_soft_resto_phase_` continuation),
with the placement deviation and its consequence disclosed in §2.

Forced-entry unit scenarios on feasible problems confirm the pre-stage resolves via soft steps
without a full restoration entry, and a companion lifecycle test drives the counter to its cap
to confirm escalation runs the full phase to convergence. **How often the soft pre-stage engaged
across the corpus is not derivable from the scorecard data**: the corpus JSONL carries only
status, iteration count, objective, and wall time — it has no restoration-engagement field — so
this document deliberately makes no corpus-level claim about soft-pre-stage engagement frequency
rather than inventing one. The unit scenarios establish the mechanism works; the corpus captures
establish the end-to-end outcomes; the frequency between them is unmeasured here.

## 7. Verification

- **CBWR bit-identity at every gate, including the final.** Every per-task gate and the final
  gate reproduced status and iteration count byte-identically across two CBWR-pinned repeats on
  the 31 deterministic examples; the final gate log records `CBWR-FINAL2-EXACT`. The default and
  proximal paths are byte-identical to the pre-stage branch by construction — every nested-mode
  edit is gated on `restoration_ && is_active() && is_nested()` or the governor-capability
  predicate.
- **Proximal continuity is exact.** Re-running the proximal mode-switch (filter+monitored) on
  this branch reproduces the proximal reference from `2026-07-e2-g5a-scorecards.md` on **0 of 17**
  differing rows — the nested-mode wiring left the proximal path unchanged.
- **Full test suites.** `pytest` reports 358 passed, 4 skipped, and 168 subtests passed; the 34
  Python example scripts all pass; the C++ brachistochrone example converges to the optimal
  objective.
- **Benchmarks.** 128 lanes compared against the pre-restoration baseline. Three single-shot
  flags were all refuted by five-repetition re-measurement, landing back in the baseline's own
  historical range (543 vs 545 ns, 126 vs 129 ns, 12.8 vs 12.81 ns) — noise near this machine's
  measurement floor, not real regressions. The nested-restoration code path is dead on every
  benchmarked lane (all run with restoration off).

## 8. Theory posture

The nested ℓ1 restoration mode closes the same ladder-exhaustion gap the proximal mode-switch
closed, with a feasibility subproblem shaped closer to Ipopt's own: the ℓ1 elastic reformulation
with a live-μ proximity term and a monotone in-phase barrier schedule, rather than a single
frozen proximal center. On this 17-problem corpus that shape reaches problems the proximal mode
could not — `hard_mountaincar_badguess` and `hard_brach_illscaled` move from `failed` at the cap
to `acceptable` under the filter+monitored family — and it is cheaper on the problems both modes
solve. No convergence guarantee is claimed, and four qualifications should weigh on any decision
to enable it, in order of how much:

1. **It does not clear the wrong-basin class, and regresses `hard_zermelo_wrongbasin` relative
   to the proximal reference** (§5.2). The elastic feasibility subproblem has an infeasible
   local minimizer this mode converges to; that is a basin-geometry limit, and it costs the one
   severity demotion the proximal mode had earned on that problem. The third restoration
   strategy (the elastic relaxation) is the next candidate to test against this specific gap.
2. **This stage's own campaign shipped and then had to fix two distinct barrier-parameter
   defects** (§4.2, §4.3), each of which froze a real corpus problem at the iteration cap and
   each caught by a different instrument — a counter probe on `lit_wb2000`, then a second probe
   after the first fix proved insufficient for the free-oracle family. The final numbers are the
   first trustworthy scorecard for this mode; the intermediate snapshots are recorded as part of
   the evidentiary record, not as configurations anyone should run.
3. **The monotone-schedule gate is an evidence-driven deviation, not a transcription.** The task
   of forcing Ipopt's monotone restoration schedule regardless of governor regressed the
   already-recovered monitored family, so the schedule is gated to governors that lack their own
   safeguard (§4.3). This is the honest resolution of a real conflict between two requirements,
   and it means the "free oracle unreachable in-phase" guarantee holds unconditionally only
   under a free governor.
4. **No formal guarantee is claimed for the composition of nested restoration with any
   acceptance strategy or barrier governor.** The formulation and exit machinery are transcribed
   from Ipopt at the pinned commit with the disclosed single-measure and placement adaptations
   (§2), not re-derived and proven for Tycho's solver shape. A 17-problem corpus with the filter
   rescues above, one honest regression, and two defects caught and fixed mid-campaign is strong
   operational evidence the mode is safe to enable as an opt-in default-off setting; it is not
   proof of convergence on arbitrary problems.

## 9. Recommendation

On this 17-problem corpus (not a general claim about PSIOPT on arbitrary problems):

- **`restoration_mode = l1_nested` is worth enabling under the filter + monitored governor**,
  where it delivers the two target-workload rescues (`hard_mountaincar_badguess`,
  `hard_brach_illscaled` from `failed` to `acceptable`) and the fast infeasibility certificate
  on `deg_conflicting_equality`, with no status regression on that family except the
  `hard_zermelo_wrongbasin` demotion relative to the proximal mode.
- **Under the merit family** it converges `lit_wb2000` cheaply and speeds the degenerate and
  stiff tiers, but note the `hard_zermelo_wrongbasin` regression and the
  `hard_cartpole_tightbounds` thread-jitter (§5.3) — the latter not a restoration effect.
- **Under the funnel family it is inert** — a clean null (§5.4). Enable it there only for
  composability with the other families, not for any expected effect.
- **It is not a fix for the wrong-basin class.** `hard_zermelo_wrongbasin` is not recovered by
  any restoration strategy in the trio so far, and the nested mode gives back the proximal mode's
  one bounded-exit improvement on it. Do not enable this mode expecting to clear that problem.
- All configurations are strictly opt-in (`restoration_mode` defaults to `off`, reproducing
  today's behavior bit-identically, confirmed in §7); enabling the nested mode never changes
  `defaults` behavior.

All results are reproducible via:

```
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=merit restoration_mode=l1_nested
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter barrier_governor=monitored restoration_mode=l1_nested
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel barrier_governor=monitored restoration_mode=l1_nested
```

with the proximal and no-restoration reference columns produced by the same invocations at
`restoration_mode=proximal_switch` and `restoration_mode=off` respectively.

## References

- `tests/corpus/README.md` — problem-module contract, harness CLI, JSONL schema, subprocess
  isolation rationale, and the status severity ranking used in §3–§5.
- `docs/dev/analysis/2026-07-e2-g5a-scorecards.md` — the proximal feasibility mode-switch
  scorecards this document's proximal reference columns reproduce exactly (§7), the source of the
  stuck-in-mode target-workload framing, and the mode this stage's trio succeeds.
- `docs/dev/analysis/2026-07-e2-g0-baseline.md` — the 2026-07 defaults baseline and the
  order-sensitivity notes for `hard_zermelo_wrongbasin` and `hard_mountaincar_badguess` reused
  in §5.
- `include/tycho/detail/solvers/globalization/restoration.h` — the `RestorationStrategy`
  interface, the planned three-strategy trio, and the ownership rule.
- `include/tycho/detail/solvers/globalization/l1_restoration.h` — the nested ℓ1 mode's full
  formulation, the Ipopt citations (commit `72a29c9aab198afa0dbb940339022a22c415a4eb`:
  `IpRestoIpoptNLP.{hpp,cpp}`, `IpRestoIterateInitializer.cpp`, `IpRestoMinC_1Nrm.cpp`), the
  closed-form elastic-slack initialization, the per-step condensation algebra, and the disclosed
  deviations in §2.
- `include/tycho/detail/solvers/globalization/feasibility_switch_recovery.h` — the outermost
  recovery link, the soft feasibility pre-stage (§6), and its disclosed placement consequence
  (Ipopt `IpBacktrackingLineSearch.cpp`).
- `include/tycho/detail/solvers/globalization/barrier_governor.h` and
  `src/solvers/psiopt_globalization.cpp` — `update_barrier_monotone` and the
  `provides_restoration_barrier_safeguard()` gate (§4.3).
- `src/solvers/psiopt.cpp` — `augment_complementarity_nested` and the two gated call sites that
  fold the elastic complementarity into the barrier machinery (§4.2), and the nested-restoration
  evaluation and trial seams.
- Commit `71c17882` — the slack-completion seam fix (§4.1). Commit `bfa869ca` — the
  elastic-complementarity fix (§4.2). Commit `a1ddb02b` — the monotone in-phase barrier schedule
  and its governor gate (§4.3).
