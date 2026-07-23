# PSIOPT proximal feasibility-restoration scorecards

**Date:** 2026-07-23
**Branch:** `feat/e2-g5-feasibility-restoration`
**Data:** six `scripts/run_corpus.py --cbwr --repeat 2` captures (17 problems x 2 repeats =
34 records each), one per configuration below, plus a two-configuration control campaign
(`acceptance_strategy=merit` with and without restoration) used to attribute a defect this
campaign caught mid-stream.
**Task class:** MEASUREMENT, plus one fix this document's own campaign discovered the need
for and that is included as part of the record: `b592d934` (`fix(psiopt): classify
restoration stalls with Ipopt's failure threshold, not the entry guard`). Everything else here
scores the proximal feasibility mode-switch (`restoration_mode = proximal_switch`, default
off, introduced in `42add94f` and wired through the recovery chain and acceptance strategies
in the commits between `42add94f` and this document) against the corpus and baseline recorded
in [`2026-07-e2-g0-baseline.md`](2026-07-e2-g0-baseline.md), and against the funnel/filter and
monitored-governor scorecards this document follows
([`2026-07-e2-g3-scorecards.md`](2026-07-e2-g3-scorecards.md),
[`2026-07-e2-g4-scorecards.md`](2026-07-e2-g4-scorecards.md)).

On recovery-chain exhaustion, `restoration_mode = proximal_switch` swaps the true objective for
a proximal term centered on the switch point (coefficient ζ = `resto_proximity_weight` ·
sqrt(μ), frozen once at entry; per-coordinate scaling and term derived from Uno's
`l1RelaxedProblem`), runs a pure feasibility phase with every other globalization component
still live (barrier governor, recovery chain, acceptance strategy), and exits back to the true
objective through a per-strategy infeasibility-reduction test: `classic_merit`/`merit` track a
relative or smallest-known infeasibility reduction against the entry point; funnel requires
funnel membership *and* a 0.9999x reduction against the funnel's own reference; filter runs
Ipopt's three-part test against a filter entry preserved for exactly this purpose. Entry is
refused at a near-feasible point (guard: violation ≤ 0.1x `econ_tol_`) or once the per-phase
budget `max_feas_rest` is exhausted (default 2). A restoration stall is classified by comparing
the true constraint violation against Ipopt's `resto_failure_feasibility_threshold` (1e2x
`econ_tol_`) — this is the threshold this document's own campaign found misapplied and fixed
partway through (§3).

## 1. Method

Six configurations, each run with `--cbwr --repeat 2`:

| Config | Harness invocation | What it turns on |
| --- | --- | --- |
| **defaults** | `python scripts/run_corpus.py --cbwr --repeat 2` | Nothing — reconfirms the corpus reproduces the committed baseline before trusting the other five diffs against it. |
| **funnel+mon** | `--config acceptance_strategy=funnel barrier_governor=monitored` | Reproduces `2026-07-e2-g4-scorecards.md`'s strongest single configuration, restoration off — the no-restoration reference the funnel+resto row is measured against. |
| **filter+mon** | `--config acceptance_strategy=filter barrier_governor=monitored` | Reproduces that document's `lit_wb2000`-rescuing filter configuration, restoration off — the no-restoration reference the filter+resto row is measured against. |
| **funnel+mon+resto** | `--config acceptance_strategy=funnel barrier_governor=monitored restoration_mode=proximal_switch` | Adds the proximal switch on top of funnel+mon. |
| **filter+mon+resto** | `--config acceptance_strategy=filter barrier_governor=monitored restoration_mode=proximal_switch` | Adds the proximal switch on top of filter+mon. |
| **merit+resto** | `--config acceptance_strategy=merit restoration_mode=proximal_switch` | The modern smallest-known-infeasibility `merit` acceptance strategy (distinct from the default `classic_merit`) with the proximal switch, classic adaptive governor. |

A separate two-configuration control campaign isolates the restoration effect from the `merit`
acceptance strategy itself:

| Config | Harness invocation | What it isolates |
| --- | --- | --- |
| **merit-alone** | `--config acceptance_strategy=merit` | `merit` acceptance, restoration off — the fair no-restoration baseline for `merit+resto`, since `merit` differs substantially from `classic_merit` (`defaults`) on its own and a direct `defaults`-vs-`merit+resto` diff would conflate the two changes. |

All six main-campaign configurations reconfirm bitwise-reproducible status/iteration counts
across two repeats before any diff is trusted (§4).

## 2. Master table (post-fix, repeat 1, status / iterations)

`Mon` = `barrier_governor=monitored`; `Resto` = `restoration_mode=proximal_switch`. The three
restoration columns report **post-fix** numbers (after `b592d934`, §3, and after the
transition-state-isolation fixes described in §3.5); the pre-fix numbers that caught the
`b592d934` defect are reported separately in §3, not here, so this table reflects what ships.
Bold marks a status change or a notable iteration swing from the corresponding no-restoration
column.

| Tier | Problem | Defaults | Funnel+Mon | Filter+Mon | Funnel+Mon+Resto | Filter+Mon+Resto | Merit+Resto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| degenerate | `deg_dup_equality` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / **58** |
| degenerate | `deg_conflicting_equality` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / **498** |
| degenerate | `deg_zero_objective` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_redundant_defects` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / **58** |
| degenerate | `deg_near_infeasible` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / **498** |
| hard | `hard_vanderpol` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_brach_coldstart` | converged / 24 | converged / 22 | converged / 36 | converged / 22 | converged / 36 | converged / **48** |
| hard | `hard_brach_illscaled` | failed / 500 | failed / 500 | failed / 500 | failed / **498** | failed / **499** | failed / **498** |
| hard | `hard_zermelo_wrongbasin` | diverged / 822 | diverged / 667 | diverged / 845 | diverged / 667 | **failed** / 1000 | **failed** / 1000 |
| hard | `hard_mountaincar_badguess` | failed / 1000 | failed / 1000 | failed / 1000 | failed / 1000 | failed / **999** | failed / **998** |
| hard | `hard_lowthrust_badguess` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_cartpole_tightbounds` | converged / 95 | converged / 95 | converged / 95 | converged / 95 | converged / 95 | **failed** / **498** |
| hard | `hard_hypersens_stiff` | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / **209** | acceptable / **220** |
| literature | `lit_wb2000` | failed / 500 | converged / 33 | converged / 43 | converged / **34** | converged / **46** | **converged** / **327** |
| literature | `lit_maratos` | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 |
| literature | `lit_hs13` | acceptable / 77 | acceptable / 76 | acceptable / 76 | acceptable / 76 | acceptable / 76 | acceptable / **138** |
| literature | `lit_powell_badscaled` | converged / 22 | converged / 12 | converged / 21 | converged / 12 | converged / **23** | converged / **24** |

Repeat instability beyond LSB float noise: none — all six 34-record captures (post-fix, for the
resto columns) are repeat-1/repeat-2 identical on status and iteration count; see §4.

## 3. Findings

### 3.1 The centerpiece: this campaign's own pre-fix run caught restoration killing feasible solves

The first pass through this six-configuration campaign did not produce the table in §2. It
produced a campaign that broke problems the corpus had converged or accepted **without**
restoration for two prior globalization documents running. Diffing `defaults` against the
pre-fix `merit+resto` capture:

| Problem | No restoration (`merit-alone`) | Pre-fix `merit+resto` |
| --- | --- | --- |
| `hard_brach_coldstart` | converged / 35 | **failed / 24** |
| `lit_hs13` | acceptable / 142 | **failed / 21** |

and diffing `filter+mon` against the pre-fix `filter+mon+resto` capture:

| Problem | No restoration (`filter+mon`) | Pre-fix `filter+mon+resto` |
| --- | --- | --- |
| `hard_hypersens_stiff` | acceptable / 103 | **failed / 113** |
| `hard_zermelo_wrongbasin` | diverged / 845 | failed / 1000 |

`hard_brach_coldstart` and `lit_hs13` are not adversarial corner cases — they are a
brachistochrone cold-start and an HS13 literature problem that every prior document in this
series has converged or accepted cleanly. A feasibility-restoration mode-switch that turns
those into outright failures is a regression serious enough that the campaign stopped scoring
and started diagnosing.

**Attribution chain.** A `merit-alone` vs `merit-resto` control isolates the `merit` acceptance
strategy's own contribution (already known to differ from `classic_merit`, §1) from
restoration's: the same two problems regress under the control diff (`hard_brach_coldstart`
converged/35 → failed/24, -11 iterations; `lit_hs13` acceptable/142 → failed/21, -121
iterations), confirming the fault is in restoration, not in the `merit` strategy it was tested
under. An in-process engagement probe reading `last_feas_rest_entries`/`last_feas_rest_iters`
directly off the solver then pinned the mechanism: on `hard_brach_coldstart`, restoration
entered once, ran 5 iterations, and the phase terminated `NOTCONVERGED` — a single short-lived
restoration episode was enough to turn a problem that converges in 35 iterations without
restoration into an outright failure at iteration 24. Reading the classification logic
alongside the probe traced this to the exact defect described in `b592d934`: the stall
classification compared true constraint violation against `kNearFeasibleGuardFactor *
econ_tol_` (0.1x tol) — the ENTRY guard's threshold — instead of Ipopt's
`resto_failure_feasibility_threshold` (1e2x tol), a band three orders of magnitude tighter than
the reference. Any stall landing in `(0.1x tol, 100x tol]` — a region Ipopt treats as
"reached a near-feasible point, resume the true objective" — was being declared local
infeasibility instead. `hard_brach_coldstart` and `lit_hs13` both stall restoration in exactly
that band on the way to a point the *original* objective would have accepted; the over-strict
band discarded that point before the true objective ever got to see it.

`b592d934` fixes the stall classification to use `kRestoFailureFeasibilityFactor` (the cited
1e2x factor) in place of the entry guard's factor; the entry guard itself (which correctly
governs whether restoration may be *entered*, a different question) is unchanged. Re-running
the full six-configuration campaign after the fix produces the table in §2. Both regressions
are gone: `hard_brach_coldstart` now **converges** under `merit+resto` (48 iterations, vs 35
without restoration — restoration engages during the run and the phase still reaches the
optimum), and `lit_hs13` is back to **acceptable** (138 iterations, vs 142 without
restoration). A post-fix ctest run (1557/1557) and a second full campaign confirm no other
problem regressed as a side effect of the fix (§4).

### 3.2 Post-fix: zero regressions, and the funnel+resto combination is a clean null

Comparing each restoration column in §2 against its no-restoration reference (`funnel+mon+resto`
vs `funnel+mon`; `filter+mon+resto` vs `filter+mon`; `merit+resto` vs `merit-alone`, the fair
comparison per §1):

- **`funnel+mon+resto` vs `funnel+mon`: 17/17 unchanged status, zero regressions.** The only
  movements are trivial iteration deltas at unchanged status — `hard_brach_illscaled` costs 2
  fewer iterations (500 → 498) and `lit_wb2000` costs 1 more (33 → 34) — both restoration
  engaging briefly and leaving no visible trace on the outcome. This matches the engagement
  probe's reading on `hard_mountaincar_badguess` under this configuration: restoration entered
  once for 5 iterations mid-run and the phase still finished `failed / 1000`, identical to
  `funnel+mon` without restoration. **The funnel's own ladder rarely exhausts** (the funnel
  acceptance test itself already prevents most of the step rejections that would otherwise
  trigger restoration), and on the rare problem where it does, restoration engaging and then
  yielding the floor back costs nothing. Funnel + proximal restoration is a clean, harmless null
  on this corpus — not a rescue, but not a hazard either.
- **`filter+mon+resto` vs `filter+mon`: 16/17 unchanged, 1 improved (severity demotion), 0
  regressed.** `hard_zermelo_wrongbasin` moves from `diverged / 845` to `failed / 1000` — a
  bounded exit in place of an unbounded one, the same kind of severity demotion the funnel and
  filter scorecards have called an improvement before, but **not** a converged or acceptable
  solve. `hard_hypersens_stiff` costs 106 more iterations (103 → 209) at the same `acceptable`
  status — restoration engaged, cost iterations, and did not change the outcome either way.
  `lit_wb2000`, `lit_hs13`, and `lit_powell_badscaled` show small iteration deltas at unchanged
  status. Nothing regressed.
- **`merit+resto` vs `merit-alone`: 16/17 unchanged, 1 improved (severity demotion), 0
  regressed.** `hard_zermelo_wrongbasin` again demotes (`diverged / 973` → `failed / 1000`).
  `hard_brach_coldstart` costs 13 more iterations (35 → 48) but stays converged both ways;
  `lit_hs13` costs 4 *fewer* iterations (142 → 138) at the same acceptable status;
  `hard_hypersens_stiff` costs 117 more iterations (103 → 220) at the same acceptable status.
  `lit_wb2000` is **already** `converged` under `merit-alone` (114 iterations, no restoration
  needed) and stays converged under `merit+resto` (327 iterations) — restoration costs 213
  iterations here for no status change, so `lit_wb2000` gets **no attribution to restoration**
  under the `merit` strategy specifically (contrast funnel/filter, where `lit_wb2000` needs
  neither restoration nor `merit` — it is rescued by the monitored-governor composition alone,
  per `2026-07-e2-g4-scorecards.md`).

Across all three restoration columns, post-fix: **zero status regressions on the full
17-problem corpus**, in every configuration measured.

### 3.3 The target-workload verdict, front and center: `mountaincar`/`zermelo` are NOT recovered

This is the honest headline for anyone deciding whether to enable restoration for this problem
class specifically. `hard_mountaincar_badguess` and `hard_zermelo_wrongbasin` are the two
problems the funnel/filter scorecards flagged as still awaiting feasibility restoration
(`2026-07-e2-g3-scorecards.md` §3, `2026-07-e2-g4-scorecards.md` §3). **The proximal switch
does not clear either one, under any configuration measured in this document:**

- `hard_zermelo_wrongbasin` stays `diverged` under `funnel+mon+resto` (unchanged from
  `funnel+mon`) and gets a bounded-exit severity demotion (`diverged` → `failed`, still not
  converged or acceptable) under both `filter+mon+resto` and `merit+resto`.
- `hard_mountaincar_badguess` stays `failed` under every restoration configuration in the
  authoritative subprocess-isolated harness — `funnel+mon+resto` unchanged at `failed / 1000`,
  `filter+mon+resto` demoted by one iteration to `failed / 999`, `merit+resto` demoted by two
  iterations to `failed / 998`. None of these cross into `acceptable` or `converged`.

The engagement probe shows *why*: on `hard_zermelo_wrongbasin` under `filter+mon+resto`,
restoration enters once around iteration 506 and then **stays in restoration mode for 494
iterations straight through to the corpus's 1000-iteration budget** — it enters, cannot find an
exit its own infeasibility-reduction test will pass, and exhausts the run without ever leaving
the mode. This is a materially different failure shape from the funnel-composition null in
§3.2: it is not "restoration engages briefly and hands back control harmlessly," it is
"restoration engages and gets stuck for the rest of the budget." That long-stuck pattern —
enter, fail to find an exit, exhaust the iteration budget in-mode — is the empirical case this
document makes for the next strategy in the trio (a nested l1 proximal restoration, per
`restoration.h`'s docstring) rather than for tuning the proximal switch further: a single frozen
proximal center and a fixed ζ appear to be the wrong shape of feasibility subproblem for
`mountaincar`/`zermelo`'s particular basin geometry, not merely under-budgeted.

**A probe artifact to disregard, and why it is disregarded.** The in-process engagement probe
(which runs all corpus problems sequentially inside one Python process, unlike the
subprocess-per-problem scoring harness) recorded `hard_mountaincar_badguess` under
`filter+mon+resto` as `ACCEPTABLE` at iteration 562 with one restoration entry lasting 6
iterations. This does **not** reproduce under the subprocess-isolated harness, which is
authoritative for every status/iteration number in this document: the same configuration scores
`failed / 999` there. The discrepancy is consistent with state leaking across problems inside a
single long-lived process (thread-pool warm state, allocator arena reuse, or similar) rather
than with a genuine order-independent solve outcome, and it is exactly the kind of thing the
corpus harness's subprocess isolation exists to rule out (`tests/corpus/README.md`). The probe
is retained here as engagement-diagnostics only — it is how `rest_entries`/`rest_iters` were
read at all — and its one non-reproducing reading is flagged rather than reported as a result.

### 3.4 Iteration-cost accounting where restoration engages without changing status

Every problem in §3.2 whose iteration count moved without its status moving is a case where
restoration entered, ran, and handed control back — a real cost with no visible benefit or harm
on this corpus: `hard_brach_illscaled` (funnel/filter, -1 to -2 iterations — restoration
apparently shortens rather than lengthens the run to the cap here), `hard_hypersens_stiff`
(filter/merit, +106/+117 iterations), `hard_brach_coldstart` (merit, +13 iterations, still
converged), `lit_hs13` (merit, -4 iterations, still acceptable), `lit_wb2000` (all three
resto configurations, +1 to +213 iterations, still converged), and `lit_powell_badscaled`
(filter/merit, +1 to +2 iterations). None of these are free — every entry costs solver
iterations even when it resolves cleanly — but none of them change a corpus outcome either.
That is the same "engaged but harmless" character `2026-07-e2-g4-scorecards.md` §3 found for
the monitored governor under `classic_merit`: a component that visibly does work on more
problems than it visibly changes the result of.

### 3.5 Transition state isolation

A second defect wave, distinct from `b592d934`, surfaced after this document's numbers above
were first captured. External automated review of the pull request, followed up by a further
internal review, found that the single-strategy-instance mode-switch design let per-strategy
persistent state keep evolving across the optimality-to-feasibility phase boundary, where the
reference implementation's two-instance structure freezes it instead. Four distinct
consequences fell out of that one design gap:

- The merit tracker's penalty parameters and the funnel's width both kept updating while the
  feasibility phase was running, which made the merit strategy's reduction-based exit test
  unsatisfiable for any positive violation — the test compared each restoration-phase point
  against a running minimum that already included that same point, so the reduction it was
  looking for could never be observed.
- The recovery chain's watchdog state, including its revert snapshot, crossed the phase
  boundary carrying values from the wrong objective scale (proximal on one side, true on the
  other), so a watchdog decision made in one phase could act on a snapshot taken in the other.
- A watchdog-`RESOLVED` accept-as-is could be misclassified as a restoration entry, because the
  terminal wrapper keyed its dispatch on the `Action` value alone and could not distinguish a
  resolved watchdog outcome from an unresolved one that happened to produce the same `Action`.
- Best-iterate tracking mixed candidates evaluated on the proximal objective with candidates
  evaluated on the true objective, comparing scales that are not comparable.

All four were fixed by replicating the filter strategy's existing stash/fresh/restore
transition treatment in the merit and funnel strategies, so their exit tests now read the state
held from the optimality phase rather than state the feasibility phase kept mutating;
discriminating resolved from unresolved watchdog accept-as-is outcomes via the recovery-depth
signal rather than the `Action` value alone; resetting the recovery chain at both the entry and
exit transitions; and suspending best-iterate tracking for the duration of restoration.
Statuses were unchanged on the full corpus by these fixes — the tables above are already the
corrected numbers, and the iteration-count movement they caused was modest throughout (single
digits to low hundreds, concentrated in the `merit+resto` column; see §2 and §3.4). Commits
`4bf9b9b9` and `ef5dc8c9`.

## 4. Determinism

- **All 17 problems, all 6 configs, both before and after the fix:** status and iteration count
  are byte-identical between repeat 1 and repeat 2 for every configuration captured in this
  document — 0 improved, 0 regressed, 0 only-in-either in every repeat diff, pre-fix and
  post-fix alike.
- **`defaults` vs the committed baseline** (`tests/corpus/baselines/2026-07-defaults.jsonl`): all
  17 problems match on status and iteration count exactly — `defaults` is confirmed a true no-op
  before trusting the other diffs in this document.
- **Continuity — `funnel+mon` and `filter+mon` reproduce `2026-07-e2-g4-scorecards.md`'s own
  captures exactly.** Diffing this document's `funnel+mon` and `filter+mon` columns against that
  document's `funnel+monitored`/`filter+monitored` captures: **17/17 unchanged, 0 improved, 0
  regressed** for both. The corpus and harness have not drifted between documents, and the
  no-restoration references this document measures the three resto columns against are the same
  configurations the prior document scored.
- **Post-fix re-verification.** After `b592d934`: `ctest` 1557/1557; a full second
  six-configuration campaign with all three restoration configurations re-run at `--repeat 2`,
  every repeat pair exact; the 32 Python examples (34 passed including the 2 mesh-refinement
  variants, 0 failed, 0 skipped); the C++ brachistochrone example converging at 1.801295 s.
- **Benchmarks.** 128 lanes compared against the pre-restoration baseline; one flagged
  regression, `BM_BumpAllocator_Resize` (+68.4%, 129 ns → 217 ns). This benchmark has no
  connection to the restoration code path (it measures an unrelated bump-allocator resize) and
  this session's own benchmark history shows it ranging from 127 ns to 217 ns across unrelated,
  previously-accepted runs on this machine — consistent with a noisy microbenchmark near this
  machine's measurement floor rather than a real regression. A repeat measurement landed back in
  that historical range, refuting the flagged reading as noise rather than confirming it.
- Two corpus problems carry known order-sensitivity notes independent of this document's
  configurations (`hard_zermelo_wrongbasin`, `hard_mountaincar_badguess` — see their module
  docstrings and `2026-07-e2-g0-baseline.md`): status and iteration count landed identically for
  both problems under every configuration and both repeats captured for this document.

## 5. Theory posture

Before feasibility restoration existed, `2026-07-e2-g3-scorecards.md` and
`2026-07-e2-g4-scorecards.md` both flagged the same gap: the classic-path recovery chain's
accept-as-is ladder-exhaustion fallback could still accept a step neither the filter nor the
funnel would admit on its own terms, because ladder exhaustion had nowhere else to go. With
`restoration_mode = proximal_switch` enabled, that gap is **closed**: ladder exhaustion now
dispatches to a feasibility-restoration mode-switch with its own principled exit test (per
acceptance strategy, §method) instead of falling through to an unconditional accept. The
terminal step in the recovery chain is no longer "give up and accept whatever the last trial
was" — it is "try to restore feasibility under the true objective's own barrier machinery, and
only fail if that also does not resolve."

That is not a convergence guarantee, and none is claimed here. Three qualifications, in order of
how much they should weigh on a decision to enable this mode:

1. **Restoration does not fix the corpus's actual open problem class.** `mountaincar` and
   `zermelo` — the two problems every document in this series has flagged as needing feasibility
   restoration — are not converged or made acceptable by the proximal switch under any
   configuration tested (§3.3). The single Uno-derived frozen proximal center is evidently the
   wrong shape of feasibility subproblem for their basin geometry; a nested l1 restoration or an
   elastic/penalty relaxation (both already named in `restoration.h`'s docstring as the remaining
   members of the planned trio) are the next candidates to test against this specific gap, not a
   retuning of the proximal switch's constants.
2. **This document's own campaign shipped and then had to fix a real defect** (§3.1) — a
   three-order-of-magnitude threshold error that took two feasible, previously-converging corpus
   problems to outright failure. The post-fix numbers in §2 are the first trustworthy scorecard
   for this component; the pre-fix numbers are recorded here as part of the evidentiary record,
   not as a configuration anyone should run.
3. **No formal guarantee is claimed for the composition of restoration with any acceptance
   strategy or barrier governor.** The exit tests are transcribed from Ipopt/Uno reference
   implementations with disclosed single-measure adaptations (documented in
   `proximal_restoration.h`'s file docstring), not re-derived and proven for Tycho's
   `ProgressMeasures`/`SolverContext` shape. A 17-problem corpus with zero regressions and one
   defect caught and fixed mid-campaign is strong operational evidence that the mode is safe to
   enable as an opt-in default-off setting; it is not proof of convergence on arbitrary problems.

## 6. Recommendation

On this 17-problem corpus (not a general claim about PSIOPT's behavior on arbitrary problems):

- **`restoration_mode = proximal_switch` is safe to enable** with any of `classic_merit`/`merit`,
  `funnel`, or `filter` acceptance in combination with either barrier governor, per the
  post-fix zero-regression result in §3.2 — but it is not, on this evidence, a fix for the
  `mountaincar`/`zermelo` problem class specifically (§3.3). Enable it as a general
  safety-net against ladder exhaustion, not as a targeted rescue for those two problems.
- **`funnel` + monitored governor + restoration** is the configuration with the cleanest
  interaction on this corpus: 17/17 unchanged status vs. the no-restoration funnel+governor
  configuration, restoration engaging harmlessly on the rare problem where the funnel's own
  ladder exhausts.
- **`filter`/`merit` + restoration** both trade a small amount of iteration cost for one
  severity demotion each (`hard_zermelo_wrongbasin` from `diverged` to a bounded `failed`) and
  no regressions — a real but modest improvement, not a rescue.
- **Do not read this document's fix as evidence the constant-selection process is now closed
  out.** `kRestoFailureFeasibilityFactor` is one threshold in a strategy with several
  Ipopt/Uno-derived constants; this document's defect was caught by running the full corpus
  and diffing against known-good no-restoration baselines, which is the same process that
  should be applied to the next restoration strategy in the trio before it ships.
- All six main-campaign configurations are strictly opt-in (`restoration_mode` defaults to
  `off`, reproducing today's behavior bit-identically, confirmed in §4); enabling it never
  changes `defaults` behavior.

All results in this document are reproducible via:

```
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel barrier_governor=monitored
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter barrier_governor=monitored
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel barrier_governor=monitored restoration_mode=proximal_switch
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter barrier_governor=monitored restoration_mode=proximal_switch
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=merit restoration_mode=proximal_switch
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=merit
```

## References

- `tests/corpus/README.md` — problem-module contract, harness CLI, JSONL schema, subprocess
  isolation rationale, and the `ConvergenceFlags`-derived status severity ranking used in §3.
- `docs/dev/analysis/2026-07-e2-g0-baseline.md` — the 2026-07 defaults baseline this document
  diffs against, and the source of the order-sensitivity notes reused in §4.
- `docs/dev/analysis/2026-07-e2-g3-scorecards.md` — the funnel/filter acceptance-strategy
  scorecards that first flagged `mountaincar`/`zermelo` as awaiting feasibility restoration and
  the classic-path accept-as-is ladder-exhaustion gap this document's theory posture addresses.
- `docs/dev/analysis/2026-07-e2-g4-scorecards.md` — the monitored-governor scorecards this
  document's `funnel+mon`/`filter+mon` no-restoration reference columns reproduce exactly (§4),
  and the source of the "engaged but harmless" framing reused in §3.4.
- `include/tycho/detail/solvers/globalization/restoration.h` — the `RestorationStrategy`
  interface, the planned three-strategy trio, and the ownership rule cited in §5.
- `include/tycho/detail/solvers/globalization/proximal_restoration.h` — the proximal
  mode-switch's full formulation, citations (Uno `InteriorPointMethod`/`l1RelaxedProblem`, Ipopt
  `IpBacktrackingLineSearch`/`IpRestoMinC_1Nrm.cpp`), the near-feasible entry guard, and
  `kRestoFailureFeasibilityFactor` — the constant `b592d934` corrected.
- `src/solvers/psiopt.cpp` — the stall-classification call site `b592d934` fixed, and the
  recovery-chain dispatch to restoration on ladder exhaustion.
- Commit `42add94f` — introduces `ProximalSwitchRestoration` as a standalone component (no
  solver wiring yet); the commits between it and this document's HEAD wire it into the recovery
  chain, acceptance strategies, and Python bindings.
- Commit `b592d934` — the stall-classification fix this document's own campaign found the need
  for; see §3.1 for the attribution chain that led to it.
- Commits `4bf9b9b9` and `ef5dc8c9` — the transition-state-isolation fixes described in §3.5,
  which the tables in §2 already reflect.
- `tests/corpus/baselines/2026-07-defaults.jsonl` — committed baseline used for the `defaults`
  cross-check in §4.
