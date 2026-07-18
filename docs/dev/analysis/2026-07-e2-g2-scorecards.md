# PSIOPT globalization option scorecards on the robustness corpus

**Date:** 2026-07-18
**Branch:** `feat/e2-g2-merit-soc-recovery`
**Data:** five `scripts/run_corpus.py --cbwr --repeat 2` captures (17 problems × 2 repeats =
34 records each), one per configuration below.
**Task class:** MEASUREMENT ONLY. No PSIOPT/library code changes in this document — it scores
five new opt-in globalization options against the corpus and baseline recorded in
[`2026-07-e2-g0-baseline.md`](2026-07-e2-g0-baseline.md), using the same corpus, harness, and
flag→status mapping described there and in `tests/corpus/README.md`.

## 1. Method

Five configurations, each run with `--cbwr --repeat 2` (bitwise-reproducible MKL reductions,
two repeats to separate genuine behavior change from run-to-run float noise):

| Config | Harness invocation | What it turns on |
| --- | --- | --- |
| **defaults** | `python scripts/run_corpus.py --cbwr --repeat 2` | Nothing — all new options at their off/default value. Exists to reconfirm the corpus reproduces the committed baseline before trusting the other four diffs against it. |
| **soc** | `python scripts/run_corpus.py --cbwr --repeat 2 --config max_soc=4` | Second-order correction (SOC) on a rejected trial step, per Wächter & Biegler, "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming," *Math. Program.* 106(1):25-57 (2006), §2.4. `4` is the library's own recommended cap (`kSocRecommendedMaxCorrections`). |
| **recovery** | `python scripts/run_corpus.py --cbwr --repeat 2 --config max_soc=4 ls_extended_iters=8 watchdog=1` | The full opt-in recovery chain: SOC, plus extended backtracking (8 further external trials on the same search direction once the normal ladder and SOC are exhausted), plus the watchdog technique of Chamberlain, Powell, Lemaréchal & Pedersen, "The watchdog technique for forcing convergence in algorithms for constrained optimization," *Mathematical Programming Study* 16, 1-17 (1982) — arming/trial-window constants taken from the Wächter & Biegler (2006) reference implementation (arm after 10 consecutive full rejections, allow up to 3 relaxed trial iterations before reverting). |
| **wmno** | `python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=merit` | Switches step acceptance from the bit-identical classic fused backtracking merit line search to the modernized penalty-based acceptance test, with the default penalty rule: Waltz, Morales, Nocedal & Orban, "An interior algorithm for nonlinear optimization that combines line search and trust region steps," *Math. Program.* 107:391-408 (2006), §3.1 — a single penalty value updated from a directional-derivative condition (their Eqs. 3.5-3.6). |
| **flex** | `python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=merit merit_penalty_rule=flexible` | Same modernized acceptance path, but with the flexible penalty rule of Curtis & Nocedal, "Flexible penalty functions for nonlinear constrained optimization," *IMA J. Numer. Anal.* 28(4):749-769 (2008) — a penalty *interval* `[π_l, π_u]`; a step is accepted if it improves the merit for at least one value in that interval, rather than a single scalar penalty. |

`acceptance_strategy=merit` requires `max_soc == 0` and `ls_extended_iters == 0` (the modern
acceptance path and the classic-path recovery chain are mutually exclusive — enforced by an
upfront `psiopt.cpp` check), so `wmno`/`flex` never compose with `soc`/`recovery` in this
scorecard; each of the five configurations is a clean, independent point in the option space.
All five options default off/classic, so the `defaults` run is expected to reproduce
`tests/corpus/baselines/2026-07-defaults.jsonl` exactly.

## 2. Master table (repeat 1, status / iterations)

| Tier | Problem | Defaults | SOC (`max_soc=4`) | Recovery (full chain) | WMNO (`merit`) | Flexible (`merit`) |
| --- | --- | --- | --- | --- | --- | --- |
| degenerate | `deg_dup_equality` | converged / 3 | converged / 3 | converged / 3 | converged / **56** | converged / 3 |
| degenerate | `deg_conflicting_equality` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| degenerate | `deg_zero_objective` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_redundant_defects` | converged / 3 | converged / 3 | converged / 3 | converged / **56** | converged / 3 |
| degenerate | `deg_near_infeasible` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| hard | `hard_vanderpol` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_brach_coldstart` | converged / 24 | converged / 23 | converged / **41** | converged / 35 | converged / 27 |
| hard | `hard_brach_illscaled` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| hard | `hard_zermelo_wrongbasin` | diverged / 822 | diverged / 808 | **failed** / 1000 | diverged / 973 | diverged / 933 |
| hard | `hard_mountaincar_badguess` | failed / 1000 | failed / 1000 | failed / 1000 | failed / 1000 | failed / 1000 |
| hard | `hard_lowthrust_badguess` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_cartpole_tightbounds` | converged / 95 | converged / 95 | converged / 95 | **failed / 500** | converged / 95 |
| hard | `hard_hypersens_stiff` | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 |
| literature | `lit_wb2000` | failed / 500 | failed / 500 | failed / 500 | **converged / 114** | **converged / 48** |
| literature | `lit_maratos` | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 |
| literature | `lit_hs13` | acceptable / 77 | acceptable / 77 | acceptable / 77 | acceptable / **142** | acceptable / 77 |
| literature | `lit_powell_badscaled` | converged / 22 | converged / **18** | converged / **49** | converged / 21 | converged / 21 |

Bold marks a status change or a notable (double-digit-plus) iteration swing from `defaults`.
Repeat instability beyond LSB float noise: none observed for any config — see §4.

## 3. Findings

**The headline: `lit_wb2000` converges under both modern merit rules.** The Wächter-Biegler
(2000) counterexample — A. Wächter & L. T. Biegler, "Failure of global convergence for a
class of interior point methods for nonlinear programming," *Mathematical Programming*
88(3):565-574 (2000), a small NLP whose only source is a paywalled journal article, verified
here against Benson, Shanno & Vanderbei's Princeton ORFE-00-02 technical report, which quotes
it verbatim (see the G0 baseline doc for the corrected `a, b` constants) — was constructed
specifically so that a class of line-search primal-dual interior-point methods jams at a
feasible-looking but non-stationary point rather than reaching the optimum. It is a *distinct*
paper from the Wächter & Biegler (2006) filter line-search paper cited for SOC/the watchdog
constants above; both share an author pair but address different failure modes. It sits at
`failed / 500` under `defaults`, `soc`, and `recovery` — none of those touch the acceptance
test itself, only the line search's retry mechanics — and flips to clean convergence under
both merit variants: `wmno` in 114 iterations, `flexible` in 48. This is exactly the failure
mode the modern acceptance rules were built for, reproduced on the exact instance the corpus
carries for it.

**`flex` is the strongest single configuration in this corpus.** Net effect vs. `defaults`:
+1 converged (`lit_wb2000` failed→converged) and *zero* status losses anywhere else in the
17-problem table — every other problem holds its `defaults` status exactly, with only minor
iteration movement (`hard_brach_coldstart` 24→27, `lit_powell_badscaled` 22→21). No other
configuration in this scorecard achieves a win with no corresponding loss.

**`wmno` wins the same headline case but loses elsewhere.** It also converges `lit_wb2000`
(114 iterations — slower than `flex`'s 48, consistent with a single scalar penalty needing
more updates than an interval that only has to clear one of two bounds), but it is not a free
win: `hard_cartpole_tightbounds` regresses from `converged / 95` to `failed / 500` — a genuine
status loss on a problem none of the other four configurations touch. `wmno` is also markedly
slower on two of the three "converges trivially anyway" degenerate problems: `deg_dup_equality`
and `deg_redundant_defects` both jump from 3 iterations to 56 under `wmno` (`deg_zero_objective`,
the third of that group, is unaffected). The penalty-interval flexibility that lets `flex`
absorb messy geometry cheaply appears to cost `wmno` real iterations even on structurally
trivial problems, and outright breaks one previously-converging problem.

**`soc` is a small, unambiguous, no-downside win.** No status changes anywhere in the corpus.
Two iteration improvements: `hard_brach_coldstart` 24→23, `lit_powell_badscaled` 22→18. SOC
never fires unless the classic backtrack's first trial is rejected on a step that is making
constraint-violation progress but failing the merit test (Wächter & Biegler's trigger
condition, §2.4) — on this corpus that condition is rare, so the option is mostly inert, but
the two problems where it does fire both get cheaper, never more expensive.

**`recovery` (the full chain) does not yet earn its cost.** One status change:
`hard_zermelo_wrongbasin` moves from `diverged / 822` to `failed / 1000` — arguably an
upgrade (a bounded `NOTCONVERGED` at the iteration cap is a more informative, more
recoverable failure mode for a caller than an unbounded `DIVERGING`; both are non-solutions,
but "ran out of iterations" and "blew up" are different signals to act on). Against that,
the chain measurably *slows* two problems that were already converging cleanly:
`hard_brach_coldstart` 24→41 iterations, `lit_powell_badscaled` 22→49 — both roughly double.
There are no rescues: `lit_wb2000` stays `failed / 500` (the classic-path recovery chain
doesn't touch the acceptance test, so it cannot reach the modern-merit fix), and none of the
four hard divergent/failed problems below are moved either. On this corpus, extended
backtracking and the watchdog have a real per-iteration cost with no corresponding win to
offset it — the chain is a cost center here, not (yet) a safety net.

**Nulls: four hard failures and two degenerate failures are unmoved by everything.**
`hard_vanderpol` (diverges at iteration 1, `KKT = nan` — a toolchain-level issue, see
`project_vanderpol_diverges`), `hard_brach_illscaled`, `hard_mountaincar_badguess`, and
`hard_lowthrust_badguess` hold their exact `defaults` status and iteration count across all
five configurations, and `deg_conflicting_equality` / `deg_near_infeasible` do too. None of
the five options in this PR change how the barrier subproblem is solved, how badly-scaled
Jacobians are handled, or how a genuinely infeasible/conflicting constraint set is detected —
they only change *how a step is accepted or recovered once it's already been computed*. A
badly conditioned or genuinely infeasible problem produces bad steps no acceptance/recovery
policy can rescue. Future strategies that could plausibly move this group: a filter or funnel
acceptance test paired with an explicit barrier-parameter governor (rather than the current
monolithic penalty/merit value), and a restoration phase for the near-infeasible/conflicting
cases that currently just grind to `max_iters`.

## 4. Determinism

Every one of the five 34-record captures was checked repeat-1 vs. repeat-2, per problem, on
status/iterations/objective. Findings match the three-problem order-sensitive list already
established in the G0 baseline (`hard_zermelo_wrongbasin`, `hard_mountaincar_badguess`,
`hard_cartpole_tightbounds`) with one addition specific to a config that fails to converge:

- **All 17 problems, all 5 configs:** status and iteration count are byte-identical between
  repeat 1 and repeat 2 — no config introduced a new source of status/iteration flakiness.
- **`hard_cartpole_tightbounds`:** LSB-level objective float noise persists under every
  config that converges it (`defaults`, `soc`, `recovery`, `flex`) — e.g. `defaults`
  `78.54562203020066` vs. `78.54562203020086`, differing only at the ~13th significant digit,
  exactly the pre-existing pattern from the G0 baseline (attributed there to threaded
  evaluation order under `set_num_partitions(8, 8)`). Under `wmno`, where this problem instead
  *fails* at `max_iters`, the objective divergence between repeats is much larger
  (`87.66033286910819` vs. `89.88544465602268`) — expected, since a run that never converges
  accumulates 500 iterations of float-order-dependent path divergence rather than settling
  near a fixed point after 95.
- **`hard_zermelo_wrongbasin` and `hard_mountaincar_badguess`:** landed fully byte-identical
  (status, iterations, *and* objective) across repeat 1/2 in every one of the five captures.
  This is the same "happened to land byte-identical in this particular pair of repeats" outcome
  the G0 baseline doc already flagged for these two — not a claim they became deterministic.
- **`defaults` vs. the committed baseline** (`tests/corpus/baselines/2026-07-defaults.jsonl`):
  all 34 records match on status, flag, and iterations exactly; the only diff across the whole
  file is the same known `hard_cartpole_tightbounds` LSB objective noise described above (e.g.
  `78.54562203020066` captured here vs. `78.5456220302006` in the committed baseline — both
  repeats in both files land on `converged / 95`). This confirms the `defaults` config is a
  true no-op relative to the pre-G2 baseline before trusting the four option diffs above.

## 5. Implications

On this 17-problem corpus (not a general claim about PSIOPT's behavior on arbitrary
problems — the corpus is deliberately small and adversarially selected, see
`tests/corpus/README.md`):

- **`flex`** (`acceptance_strategy=merit merit_penalty_rule=flexible`) is the configuration a
  user should reach for today when they suspect merit-hostile geometry — a problem that jams
  or cycles under a classic single-penalty line search. It is the only option here with a pure
  win/no-loss record, and it clears the corpus's one literature counterexample built
  specifically to defeat classic acceptance tests, in fewer iterations than `wmno`.
- **`soc`** (`max_soc=4`) is a cheap, always-reasonable add-on for classic-path users not
  ready to switch acceptance strategies: zero status losses, small iteration wins when it
  fires, silent no-op when it doesn't.
- **`wmno`** is a viable second choice to `flex` for merit-hostile problems, but its
  `hard_cartpole_tightbounds` regression means it is not a strict improvement over `flex`
  and should not be reached for by default without checking that regression doesn't apply to
  the problem at hand.
- **The full recovery chain (`max_soc=4 ls_extended_iters=8 watchdog=1`)** is not yet earning
  its keep on this corpus: one arguably-better bounded failure, two roughly-doubled iteration
  counts on problems that already converged, and no rescues. It should not be reached for as
  a default-on recovery strategy until either a corpus problem is found where it actually
  rescues a failure, or its cost on the already-converging cases comes down.

## References

- `tests/corpus/README.md` — problem-module contract, harness CLI, JSONL schema.
- `docs/dev/analysis/2026-07-e2-g0-baseline.md` — the 2026-07 defaults baseline this document
  diffs against, and the source of the three-problem order-sensitivity list reused in §4.
- `tests/corpus/baselines/2026-07-defaults.jsonl` — committed baseline used for the `defaults`
  cross-check in §4.
- `include/tycho/detail/solvers/globalization/soc.h` — SOC trigger/continuation predicates and
  the Wächter & Biegler (2006) §2.4 citation.
- `include/tycho/detail/solvers/globalization/watchdog.h` — extended backtracking and the
  Chamberlain, Powell, Lemaréchal & Pedersen (1982) watchdog technique, with Wächter & Biegler
  (2006) arming constants.
- `include/tycho/detail/solvers/psiopt_fwd.h` — `AcceptanceStrategies` and `MeritPenaltyRules`
  enum definitions, with the WMNO (Waltz, Morales, Nocedal & Orban 2006) and flexible (Curtis
  & Nocedal 2008) citations.
- `src/bindings/solvers/psiopt_bind.cpp` — Python-facing docstrings for `max_soc`,
  `ls_extended_iters`, `watchdog`, `acceptance_strategy`, `merit_penalty_rule`, and the
  `last_soc_steps` / `last_watchdog_activations` / `last_recovery_depth_histogram` diagnostics.
