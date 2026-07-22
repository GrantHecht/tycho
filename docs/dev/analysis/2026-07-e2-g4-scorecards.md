# PSIOPT barrier-governor corpus scorecards

**Date:** 2026-07-20
**Branch:** `feat/e2-g4-barrier-governor`
**Data:** six `scripts/run_corpus.py --cbwr --repeat 2` captures (17 problems x 2 repeats =
34 records each), one per configuration below.
**Task class:** MEASUREMENT ONLY. No PSIOPT/library code changes in this document — it scores
the free<->monotone monitored barrier governor (`barrier_governor = monitored`,
`include/tycho/detail/solvers/globalization/monitored_governor.h`) against the corpus and
baseline recorded in [`2026-07-e2-g0-baseline.md`](2026-07-e2-g0-baseline.md), using the same
corpus, harness, and flag-to-status mapping described there and in `tests/corpus/README.md`.
This document follows directly from the funnel/filter acceptance scorecards
([`2026-07-e2-g3-scorecards.md`](2026-07-e2-g3-scorecards.md)), which established that the
reference-faithful filter strategy does not rescue the `lit_wb2000` counterexample and traced
that gap to a plausible dependency: the filter's global-convergence argument, in its source
(Wächter & Biegler; the Ipopt implementation this strategy transcribes), is stated for a filter
line search running under a *monotone* barrier-parameter update scheme, not the pre-existing
adaptive governor the filter strategy ran under in that document. `2026-07-e2-g4-forensics.md`
worked out the exact mechanism on `lit_wb2000` — a current-relative, objective-coupled H-type
acceptance test that a monotone barrier schedule was predicted to unblock — and set up the
experiment this document runs: does adding the monotone governor change the filter's outcome on
that problem, and what does it cost elsewhere on the corpus.

## 1. Method

Six configurations, each run with `--cbwr --repeat 2` (bitwise-reproducible MKL reductions, two
repeats to separate genuine behavior change from run-to-run float noise):

| Config | Harness invocation | What it turns on |
| --- | --- | --- |
| **defaults** | `python scripts/run_corpus.py --cbwr --repeat 2` | Nothing — `acceptance_strategy` at its default `classic_merit` value, `barrier_governor` at its default `classic_adaptive` value. Reconfirms the corpus reproduces the committed baseline before trusting the other five diffs against it. |
| **funnel+monitored** | `--config acceptance_strategy=funnel barrier_governor=monitored` | Funnel acceptance with the free<->monotone monitored governor in place of the classic adaptive one. |
| **filter+monitored** | `--config acceptance_strategy=filter barrier_governor=monitored` | Filter acceptance with the monitored governor — the configuration the forensics document's mechanism section predicted would change the filter's `lit_wb2000` outcome. |
| **funnel+never_monotone** | `--config acceptance_strategy=funnel never_monotone=1` | Funnel acceptance with the classic adaptive governor, `never_monotone` forcing the monitor off (an explicit escape hatch back to the funnel scorecard's own configuration) — reproduces `2026-07-e2-g3-scorecards.md`'s `funnel` column. |
| **filter+never_monotone** | `--config acceptance_strategy=filter never_monotone=1` | Same escape hatch under filter acceptance — reproduces `2026-07-e2-g3-scorecards.md`'s `filter` column. |
| **classic_merit+monitored** | `--config barrier_governor=monitored` | The monitored governor alone, with `acceptance_strategy` left at its default `classic_merit` — isolates whether the governor by itself (no acceptance-strategy change) moves anything on this corpus. |

`barrier_governor=monitored` and `never_monotone=1` are mutually exclusive in the sense that
`never_monotone` only has an effect when a strategy other than `classic_merit` composes with the
classic adaptive governor's monotone safeguard; both are opt-in and default off, so `defaults` is
expected to reproduce `tests/corpus/baselines/2026-07-defaults.jsonl` exactly.

## 2. Master table (repeat 1, status / iterations)

`Mon` = `barrier_governor=monitored`; `NM` = `never_monotone=1` (classic adaptive governor,
monitor forced off). Bold marks a status change or a notable iteration swing from `defaults`.

| Tier | Problem | Defaults | Funnel+Mon | Filter+Mon | Funnel+NM | Filter+NM | Classic+Mon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| degenerate | `deg_dup_equality` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_conflicting_equality` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| degenerate | `deg_zero_objective` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_redundant_defects` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_near_infeasible` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| hard | `hard_vanderpol` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_brach_coldstart` | converged / 24 | converged / **22** | converged / **36** | converged / **21** | converged / **27** | converged / 24 |
| hard | `hard_brach_illscaled` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| hard | `hard_zermelo_wrongbasin` | diverged / 822 | diverged / **667** | diverged / **845** | diverged / **667** | **failed** / 1000 | diverged / **819** |
| hard | `hard_mountaincar_badguess` | failed / 1000 | failed / 1000 | failed / 1000 | failed / 1000 | **acceptable** / 739 | failed / 1000 |
| hard | `hard_lowthrust_badguess` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_cartpole_tightbounds` | converged / 95 | converged / 95 | converged / 95 | converged / 95 | converged / 95 | converged / 95 |
| hard | `hard_hypersens_stiff` | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 |
| literature | `lit_wb2000` | failed / 500 | **converged / 33** | **converged / 43** | **converged / 29** | failed / 500 | failed / 500 |
| literature | `lit_maratos` | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 |
| literature | `lit_hs13` | acceptable / 77 | acceptable / **76** | acceptable / **76** | acceptable / 77 | acceptable / 77 | acceptable / **76** |
| literature | `lit_powell_badscaled` | converged / 22 | converged / **12** | converged / **21** | converged / **12** | converged / **21** | converged / 22 |

Repeat instability beyond LSB float noise: none observed for any config — all six 34-record
captures are repeat-1/repeat-2 identical on status and iteration count; see §4.

## 3. Findings

**The headline: the forensics document's predicted experiment ran, and the
governor-dependency hypothesis held.** `2026-07-e2-g4-forensics.md` traced the filter's
`lit_wb2000` failure to a current-relative, objective-coupled H-type acceptance test that walls
off the counterexample's escape route, and predicted that a monotone barrier safeguard would
change *where* the current iterate sits when that escape step is offered, possibly admitting it.
That prediction is confirmed: **filter+monitored converges `lit_wb2000` in 43 iterations**, a
status the filter has not reached under any other configuration measured in this program.

The clean part of this result is that it decomposes. Neither ingredient alone moves the problem:

| Configuration | `lit_wb2000` result |
| --- | --- |
| filter alone (filter+never_monotone) | failed / 500 |
| governor alone (classic_merit+monitored) | failed / 500 |
| filter + governor (filter+monitored) | **converged / 43** |

The filter's reference-faithful acceptance rule and the monitored governor's monotone safeguard
are each individually insufficient and jointly sufficient on this problem — exactly the
composition the forensics document's mechanism section predicted, not merely a status change
found by sweeping configurations.

**The honest trade: filter+monitored loses the adaptive filter's other robustness wins.** The
funnel/filter scorecards found two problems where the classic-adaptive-governed filter
(filter+never_monotone, this document's escape-hatch reproduction of that earlier `filter`
configuration) improved on `defaults`: `hard_mountaincar_badguess` moved to `acceptable / 739`,
and `hard_zermelo_wrongbasin` was demoted from an unbounded `diverged / 822` to a bounded
`failed / 1000`. Under filter+monitored, both of those wins are gone: `hard_mountaincar_badguess`
reverts to `failed / 1000`, the same status and iteration count as `defaults`, and
`hard_zermelo_wrongbasin` stays `diverged`, at `845` iterations (compare `822` under `defaults`
and `667` under funnel — the monitored governor's filter run diverges in more iterations than the
`defaults` run, not fewer). The monotone schedule the governor imposes changes *which* problems
the filter's acceptance rule helps; it is not a strict superset of the adaptive filter's
robustness. Neither filter configuration dominates the other on this corpus: filter+never_monotone
wins `mountaincar`/`zermelo`-severity but not `wb2000`; filter+monitored wins `wb2000` but not
`mountaincar`/`zermelo`. The `mountaincar`/`zermelo` problem class still awaits feasibility
restoration under either filter configuration.

filter+monitored's other movements: `hard_brach_coldstart` costs `24 -> 36` (`+12`), the largest
single-problem governor cost observed anywhere in this document; `lit_hs13` improves slightly
(`77 -> 76`); `lit_powell_badscaled` improves by the same margin the adaptive filter already
achieved (`22 -> 21`, matching filter+never_monotone exactly).

**funnel+monitored is nearly governor-insensitive and is currently the strongest single
configuration measured in this program.** Comparing funnel+monitored to funnel+never_monotone
(the funnel's own escape-hatch reproduction of `2026-07-e2-g3-scorecards.md`'s `funnel` column):
`hard_zermelo_wrongbasin`'s shortened divergence (`822 -> 667`) and `lit_powell_badscaled`'s
speedup (`22 -> 12`) are identical between the two governor settings; `hard_brach_coldstart`
(`21` vs `22`) and `lit_hs13` (`77` vs `76`) differ by exactly one iteration each — small
governor-induced perturbations that touch no status. `lit_wb2000` itself costs a little more
under the governor (`converged / 33` vs `converged / 29`) but stays comfortably converged. Set
against `defaults`, funnel+monitored converges seven problems (`deg_dup_equality`,
`deg_zero_objective`, `deg_redundant_defects`, `hard_brach_coldstart`, `hard_cartpole_tightbounds`,
`lit_wb2000`, `lit_powell_badscaled`), regresses nothing, and is the only configuration in this
document (or the prior funnel/filter document) that combines a `lit_wb2000` rescue with zero
losses anywhere on the corpus and a monotone-compatible barrier schedule.

**classic_merit+monitored is a status-level null, but the governor visibly engages —
this is "engaged but harmless," not "never engages."** All 17 problems land on the same
status as `defaults` under classic_merit+monitored. Two problems do show small iteration shifts
at the same status (`hard_zermelo_wrongbasin` `822 -> 819`, `lit_hs13` `77 -> 76`), and several
`failed`/`diverged` problems that hit their iteration cap land on markedly different final
objective values under the same cap (e.g. `hard_brach_illscaled` `1.40598` under `defaults` vs
`-157.351` under classic_merit+monitored; `lit_wb2000` `-0.968231` vs `-0.9804`) — evidence the
governor is perturbing the trajectory, not a true no-op. A direct probe confirms this
quantitatively: calling `opt.last_monotone_switches` / `opt.last_monotone_iters` after a
classic_merit+monitored solve (outside `--cbwr`, so the exact iteration counts differ slightly
from the table above; only the switch/mode-time reading matters here) shows the governor handing
off to monotone mode and staying there for most of the run — one free-to-monotone switch and 485
of 500 iterations spent in monotone mode on `lit_wb2000`; one switch and 326 of 832 iterations on
`hard_zermelo_wrongbasin`; one switch and 2 of 76 iterations on `lit_hs13`. The monitor engages
on the classic merit path on every problem checked; it simply never changes what the classic
merit acceptance test does with the steps it is offered, so the outcome is unchanged — zero cost
in status terms, zero benefit, but not because the governor is idle.

**Honest nulls: five problems are unmoved by every configuration in this document.**
`deg_conflicting_equality`, `deg_near_infeasible` (the degenerate tier's genuinely
conflicting/infeasible problems), `hard_brach_illscaled`, `hard_lowthrust_badguess`, and
`hard_vanderpol` hold their exact `defaults` status and iteration count across all five governor
and acceptance-strategy combinations tested here — as do `deg_dup_equality`, `deg_zero_objective`,
`deg_redundant_defects`, `hard_cartpole_tightbounds`, `hard_hypersens_stiff`, and `lit_maratos`.
`hard_brach_illscaled` is a useful illustration: it stays `failed / 500` under all six
configurations while its objective at the cap wanders across a wide range (`1.40598` down to
`-417.364` depending on configuration) — consistent with hitting the iteration cap rather than a
meaningful stopping point, not evidence of any configuration doing better or worse there. Neither
the acceptance strategy nor the barrier governor changes how a badly conditioned Jacobian is
handled or how a genuinely infeasible constraint set is detected; these problems await
feasibility restoration, already flagged as future work in `2026-07-e2-g3-scorecards.md`'s nulls
section.

## 4. Determinism

- **All 17 problems, all 6 configs:** status and iteration count are byte-identical between
  repeat 1 and repeat 2 for every configuration — 0 improved, 0 regressed, 0 only-in-either in
  every repeat diff.
- **`defaults` vs the committed baseline** (`tests/corpus/baselines/2026-07-defaults.jsonl`): all
  17 problems match on status and iteration count exactly (0 improved, 0 regressed, 0
  only-in-either) — `defaults` is confirmed a true no-op before trusting the five configuration
  diffs above.
- **`hard_cartpole_tightbounds`:** the same LSB-level objective float noise documented in the
  corpus baseline (`2026-07-e2-g0-baseline.md`) and reconfirmed in
  `2026-07-e2-g3-scorecards.md` persists here — e.g. `defaults` repeat 1
  `78.54562203020092` vs. repeat 2 `78.54562203020079`, and `classic_merit+monitored` repeat 1
  `78.54562203020157` vs. repeat 2 `78.54562203020188`, differing only at the ~13th significant
  digit in every case. Status (`converged`) and iteration count (`95`) are unaffected under any
  configuration.
- **Continuity check — `never_monotone` reproduces the pre-governor scorecards exactly.** Diffing
  `funnel+never_monotone` against `2026-07-e2-g3-scorecards.md`'s own recorded `funnel` capture,
  and `filter+never_monotone` against that document's `filter` capture, both come back **17/17
  unchanged, 0 improved, 0 regressed, 0 only-in-either** — every status and iteration count
  matches exactly. `never_monotone` is a faithful escape hatch: turning it on reproduces the
  monitor-free behavior byte-for-byte on status/iterations, not merely approximately.
- Two corpus problems carry known order-sensitivity notes independent of this document's
  configurations (`hard_zermelo_wrongbasin`, `hard_mountaincar_badguess` — see their module
  docstrings and `2026-07-e2-g0-baseline.md`): flag and iteration-cap behavior is stable across
  repeats even though objective/iteration-path details near the failure are not guaranteed
  byte-identical run to run in general. On the specific repeat pairs captured for this document,
  status and iteration count landed identically for both problems under every configuration.

## 5. Theory posture

The monitored governor changes the filter's `lit_wb2000` outcome in the direction the forensics
document's mechanism analysis predicted, and does so through the same monotone-Fiacco-McCormick
machinery Wächter & Biegler's and Ipopt's global-convergence argument assumes — the filter now
operates closer to its designed regime than it did under the classic adaptive governor. That is
not, on its own, a formal guarantee, and none is claimed here. Feasibility restoration is not yet
implemented, and the classic-path recovery chain's accept-as-is ladder-exhaustion fallback still
exists and can accept a step neither the filter nor the funnel would admit on their own terms
(documented in the strategies' own headers and in `2026-07-e2-g3-scorecards.md` §5). A single
corpus problem moving to `converged` under the predicted configuration is corpus evidence
consistent with the dependency the earlier document flagged; it is not proof that the filter
strategy carries a convergence guarantee under `barrier_governor=monitored`, and this document
draws no such conclusion.

## 6. Recommendation

On this 17-problem corpus (not a general claim about PSIOPT's behavior on arbitrary problems —
the corpus is deliberately small and adversarially selected, see `tests/corpus/README.md`):

- **`funnel` + `barrier_governor=monitored`** is the strongest single configuration measured in
  this program to date: seven problems converged (including `lit_wb2000`), zero regressions
  anywhere on the corpus, and a barrier schedule compatible with the filter/funnel literature's
  monotone assumption rather than the classic adaptive one. Reach for it by default among the
  configurations in this document.
- **`filter` + `barrier_governor=monitored`** is the configuration to reach for specifically to
  clear `lit_wb2000` under filter acceptance — it is the only configuration that does — but it
  gives up the adaptive filter's `hard_mountaincar_badguess` and `hard_zermelo_wrongbasin`
  robustness wins in exchange. Neither filter configuration dominates the other; picking between
  them requires knowing whether the counterexample class or the `mountaincar`/`zermelo` class
  matters more for a given use case.
- **`classic_merit` + `barrier_governor=monitored`** is safe to enable with `classic_merit`
  acceptance — confirmed engaged (nonzero switches, substantial monotone-mode dwell time) but a
  status-level no-op on this corpus — and is not a reason to prefer or avoid the monitored
  governor on its own.
- **`never_monotone=1`** is a faithful escape hatch back to the classic adaptive governor's
  behavior for either acceptance strategy, confirmed byte-for-byte identical to the pre-governor
  funnel/filter scorecards in §4.
- All six configurations are strictly opt-in (`acceptance_strategy` defaults to `classic_merit`,
  `barrier_governor` defaults to `classic_adaptive`); `defaults` behavior is unchanged, confirmed
  byte-for-byte against the committed baseline in §3 and §4.

All results in this document are reproducible via:

```
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel barrier_governor=monitored
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter barrier_governor=monitored
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel never_monotone=1
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter never_monotone=1
conda run -n tycho python scripts/run_corpus.py --cbwr --repeat 2 --config barrier_governor=monitored
```

## References

- `tests/corpus/README.md` — problem-module contract, harness CLI, JSONL schema, and the
  `ConvergenceFlags`-derived status severity ranking used in §3.
- `docs/dev/analysis/2026-07-e2-g0-baseline.md` — the 2026-07 defaults baseline this document
  diffs against, and the source of the order-sensitivity notes reused in §4.
- `docs/dev/analysis/2026-07-e2-g3-scorecards.md` — the funnel/filter acceptance-strategy
  scorecards this document follows; source of the `funnel`/`filter` captures reproduced exactly
  by `never_monotone` in §4, and of the dependency hypothesis this document's headline result
  confirms.
- `docs/dev/analysis/2026-07-e2-g4-forensics.md` — the per-iteration mechanism analysis on
  `lit_wb2000` (why the funnel escapes and the classically-governed filter jams) whose
  implications section set up the experiment run in this document.
- `include/tycho/detail/solvers/globalization/monitored_governor.h` — the monitored governor's
  formulation: the KKT-error monitor, the free<->monotone handoff and re-entry rules, the
  Fiacco-McCormick monotone update, and the `mu_event`/diagnostics contract cited in §3.
- `include/tycho/detail/solvers/globalization/funnel_acceptance.h` /
  `filter_acceptance.h` — the acceptance-strategy formulations composed with the governor in
  this document.
- `tests/corpus/baselines/2026-07-defaults.jsonl` — committed baseline used for the `defaults`
  cross-check in §4.
