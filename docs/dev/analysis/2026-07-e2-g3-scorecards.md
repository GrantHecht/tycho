# PSIOPT funnel/filter acceptance corpus scorecards

**Date:** 2026-07-19
**Branch:** `feat/e2-g3-funnel-filter`
**Data:** five `scripts/run_corpus.py --cbwr --repeat 2` captures (17 problems x 2 repeats =
34 records each), one per configuration below.
**Task class:** MEASUREMENT ONLY. No PSIOPT/library code changes in this document — it scores
two new opt-in acceptance strategies, and their composition with the watchdog, against the
corpus and baseline recorded in
[`2026-07-e2-g0-baseline.md`](2026-07-e2-g0-baseline.md), using the same corpus, harness, and
flag-to-status mapping described there and in `tests/corpus/README.md`. It follows directly
from the merit and recovery-chain scorecards
([`2026-07-e2-g2-scorecards.md`](2026-07-e2-g2-scorecards.md)), which established that a
modernized penalty-based acceptance test (rather than the classic fused backtracking merit
line search) is what rescues the `lit_wb2000` counterexample; this document scores the two
strategies purpose-built to replace a monolithic penalty/merit value with an explicit
progress-measures history — a set of dominating pairs (filter) or a single shrinking width
(funnel) — which the merit-series scorecards (2026-07-e2-g2-scorecards.md), whose nulls section flagged as the plausible next lever for the
hard/degenerate problems that no acceptance or recovery-chain change had moved.

## 1. Method

Five configurations, each run with `--cbwr --repeat 2` (bitwise-reproducible MKL reductions,
two repeats to separate genuine behavior change from run-to-run float noise):

| Config | Harness invocation | What it turns on |
| --- | --- | --- |
| **defaults** | `python scripts/run_corpus.py --cbwr --repeat 2` | Nothing — `acceptance_strategy` at its default `classic_merit` value. Exists to reconfirm the corpus reproduces the committed baseline before trusting the other four diffs against it. |
| **funnel** | `python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel` | Step acceptance switches to `FunnelAcceptance` — a single monotonically non-increasing scalar bound (the funnel width) on constraint violation, following Kiessling, Leyffer & Vanaret, "A Unified Funnel Restoration SQP Algorithm," arXiv:2409.09208, with constants taken from Vanaret's Uno solver's shipped option defaults. |
| **filter** | `python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter` | Step acceptance switches to `FilterAcceptance` — the classic Wächter & Biegler (θ, φ)-pair filter, per Wächter & Biegler, "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming," *Math. Program.* 106(1):25-57 (2006), with the practical bookkeeping (barrier-ceiling test, filter-reset heuristic, dominance comparison) transcribed rule-by-rule from the COIN-OR Ipopt reference implementation. |
| **funnel-wd** | `python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=funnel watchdog=1` | Funnel acceptance plus the watchdog technique (Chamberlain, Powell, Lemaréchal & Pedersen, 1982) layered on top, to check whether the two opt-in mechanisms interact on this corpus. |
| **filter-wd** | `python scripts/run_corpus.py --cbwr --repeat 2 --config acceptance_strategy=filter watchdog=1` | Filter acceptance plus the watchdog, same purpose as `funnel-wd`. |

Both `funnel` and `filter` are generic-path acceptance strategies in the same sense as `merit`
in the merit-series scorecards (2026-07-e2-g2-scorecards.md): the classic-path recovery links (`max_soc`, `ls_extended_iters`) require
`acceptance_strategy=classic_merit` and are rejected upfront by an `std::invalid_argument`
check in `psiopt.cpp` otherwise, so neither strategy composes with SOC or extended
backtracking in this scorecard. The watchdog is architecturally independent of that guard —
"[t]he watchdog alone is compatible with every strategy" (`src/solvers/psiopt.cpp`, the
strategy-combination guard comment) — which is what `funnel-wd`/`filter-wd` are built to
verify empirically. All five configurations default `acceptance_strategy` and `watchdog` off,
so the `defaults` run is expected to reproduce `tests/corpus/baselines/2026-07-defaults.jsonl`
exactly.

## 2. Master table (repeat 1, status / iterations)

| Tier | Problem | Defaults | Funnel | Filter | Funnel + watchdog | Filter + watchdog |
| --- | --- | --- | --- | --- | --- | --- |
| degenerate | `deg_dup_equality` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_conflicting_equality` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| degenerate | `deg_zero_objective` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_redundant_defects` | converged / 3 | converged / 3 | converged / 3 | converged / 3 | converged / 3 |
| degenerate | `deg_near_infeasible` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| hard | `hard_vanderpol` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_brach_coldstart` | converged / 24 | converged / **21** | converged / **27** | converged / 21 | converged / 27 |
| hard | `hard_brach_illscaled` | failed / 500 | failed / 500 | failed / 500 | failed / 500 | failed / 500 |
| hard | `hard_zermelo_wrongbasin` | diverged / 822 | diverged / **667** | **failed** / 1000 | diverged / 667 | failed / 1000 |
| hard | `hard_mountaincar_badguess` | failed / 1000 | failed / 1000 | **acceptable** / 739 | failed / 1000 | acceptable / 739 |
| hard | `hard_lowthrust_badguess` | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 | diverged / 1 |
| hard | `hard_cartpole_tightbounds` | converged / 95 | converged / 95 | converged / 95 | converged / 95 | converged / 95 |
| hard | `hard_hypersens_stiff` | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 | acceptable / 103 |
| literature | `lit_wb2000` | failed / 500 | **converged / 29** | **converged / 133** | converged / 29 | converged / 133 |
| literature | `lit_maratos` | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 | diverged / 2 |
| literature | `lit_hs13` | acceptable / 77 | acceptable / 77 | acceptable / 77 | acceptable / 77 | acceptable / 77 |
| literature | `lit_powell_badscaled` | converged / 22 | converged / **12** | converged / 21 | converged / 12 | converged / 21 |

Bold marks a status change or a notable iteration swing from `defaults`. Repeat instability
beyond LSB float noise: none observed for any config — see §4.

## 3. Findings

**`defaults` reconfirms the committed baseline exactly.** All 17 problems land on the same
status and iteration count as `tests/corpus/baselines/2026-07-defaults.jsonl`, with zero
improved/regressed/only-in-either entries in the corpus diff. The new acceptance-strategy
wiring — the `FunnelAcceptance`/`FilterAcceptance` construction branches added to
`psiopt.cpp`'s strategy dispatch, and the watchdog composition check — is inert when
`acceptance_strategy` stays at its default `classic_merit` value and `watchdog` stays off.

**Funnel: fastest rescue of the counterexample so far, plus three more iteration wins, zero
regressions.** `lit_wb2000` moves from `failed / 500` to `converged / 29`. Set against the
merit-based rescues in 2026-07-e2-g2-scorecards.md — `wmno` (`acceptance_strategy=merit`) took 114
iterations, `flex` (`acceptance_strategy=merit merit_penalty_rule=flexible`) took 48 — the
funnel's 29 is the cheapest fix for this instance across the whole series. Three more problems
get cheaper without changing status: `lit_powell_badscaled` 22 -> 12, `hard_brach_coldstart`
24 -> 21, and `hard_zermelo_wrongbasin`'s divergence shortens from 822 to 667 iterations
(still `diverged`, not a status change, but the same failure reached in materially fewer
steps). No problem in the 17-problem corpus regresses under `funnel`.

**Filter: two status upgrades no other configuration in this series has produced, plus a small
efficiency mix.** `lit_wb2000` also converges (`133` iterations — slower than the funnel's 29,
consistent with maintaining a growing set of dominating pairs costing more bookkeeping per
trial than updating one scalar width). `hard_mountaincar_badguess` moves from `failed / 1000`
to `acceptable / 739` — the first status movement on that problem by any configuration in
this program (it held its exact `defaults` status across every option in 2026-07-e2-g2-scorecards.md
too). `hard_zermelo_wrongbasin` moves from `diverged / 822` to `failed / 1000`: per the
`ConvergenceFlags` severity ordering documented at
`include/tycho/detail/solvers/psiopt_fwd.h` (`CONVERGED < ACCEPTABLE < NOTCONVERGED <
DIVERGING`, and reused by the harness's own status ranking in `tests/corpus/README.md`), a
`NOTCONVERGED` (`failed`, rank 2) outcome is strictly less severe than a `DIVERGING`
(`diverged`, rank 3) one — a bounded, iteration-capped non-solution instead of an unbounded
blow-up. `lit_powell_badscaled` improves slightly (22 -> 21); `hard_brach_coldstart` costs a
little more (24 -> 27). No problem regresses in status under `filter`; the one status-adjacent
movement (`hard_zermelo_wrongbasin`) is a severity improvement, not a loss.

**Watchdog composition is an honest null.** `funnel-wd` is identical to `funnel`, and
`filter-wd` is identical to `filter`, on all 17 problems — status, iteration count, and (per
the repeat-stability check in §4) even the LSB-level float pattern land the same way. The
corpus never puts either acceptance strategy through the run of consecutive full rejections
the watchdog arms on, so the composition is architecturally sound (confirmed by the earlier
error-check reading — the strategy-combination guard in `psiopt.cpp` explicitly allows
watchdog with any acceptance strategy) but has nothing to demonstrate on this corpus. This
mirrors the finding in 2026-07-e2-g2-scorecards.md that the classic-path recovery chain's watchdog leg rarely
fires here; a corpus problem engineered to stall a trial sequence for 10+ consecutive
rejections would be needed to observe the composition doing anything.

**Funnel vs. filter head-to-head matches the literature's efficiency-vs-robustness split.**
Kiessling, Leyffer & Vanaret's CUTEst study reports the funnel slightly ahead of the filter on
efficiency; this corpus agrees on the two problems both strategies move: `lit_wb2000` (29 vs.
133 iterations) and `lit_powell_badscaled` (12 vs. 21 iterations) both favor the funnel. The
filter is ahead on robustness: it is the only one of the two to move
`hard_mountaincar_badguess` to `acceptable`, and its `hard_zermelo_wrongbasin` demotion from
`diverged` to `failed` is a severity win the funnel does not produce (the funnel shortens that
divergence but does not change its status). Net status count on this corpus: funnel gets one
extra `converged`; filter gets one extra `converged` and one extra `acceptable`, at the cost of
being slower on the two problems both strategies solve.

**Honest nulls: five problems are unmoved by every configuration in this document, and in
several cases by every configuration in the whole two-PR series.** `deg_conflicting_equality`
and `deg_near_infeasible` (the degenerate tier's genuinely infeasible/conflicting problems),
`hard_brach_illscaled`, `hard_lowthrust_badguess`, `hard_vanderpol` (a known toolchain-level
divergence — `KKT = nan` at iteration 0 on this clang/MKL combination, see the
`project_vanderpol_diverges` note), and `lit_maratos` hold their exact `defaults` status and
iteration count across `funnel`, `filter`, and both watchdog compositions. Neither strategy
changes how the barrier subproblem is solved, how a badly conditioned Jacobian is handled, or
how a genuinely infeasible constraint set is detected — they only change how a step, once
computed, is judged acceptable. These problems await feasibility restoration and the barrier
governor work; a step-acceptance test alone, however it measures progress, cannot rescue a
problem whose steps are bad to begin with.

## 4. Determinism

Every one of the five 34-record captures was checked repeat-1 vs. repeat-2, per problem, on
status and iteration count:

- **All 17 problems, all 5 configs:** status and iteration count are byte-identical between
  repeat 1 and repeat 2 — no config introduces a new source of status/iteration flakiness.
- **`hard_cartpole_tightbounds`:** the same LSB-level objective float noise already documented
  in the corpus baseline (2026-07-e2-g0-baseline.md) and reconfirmed in 2026-07-e2-g2-scorecards.md persists under every configuration in
  this document — e.g. `defaults` `78.54562203020066` vs. `78.54562203020087`, differing only
  at the ~13th significant digit; `funnel`, `filter`, `funnel-wd`, and `filter-wd` all show the
  same pattern. Status (`converged`) and iteration count (`95`) are unaffected.
- **`defaults` vs. the committed baseline** (`tests/corpus/baselines/2026-07-defaults.jsonl`):
  all 17 problems match on status and iteration count exactly (0 improved, 0 regressed, 0
  only-in-either) — the `defaults` config is confirmed a true no-op relative to the
  pre-existing baseline before trusting the four strategy diffs above.
- Two corpus problems carry known order-sensitivity notes independent of this document's
  configurations (`hard_zermelo_wrongbasin`, `hard_mountaincar_badguess` — see their module
  docstrings and the corpus baseline doc (2026-07-e2-g0-baseline.md)): their flag and iteration-cap behavior is stable across
  repeats even though objective/iteration-path details near the failure are not guaranteed
  byte-identical run to run in general. On the specific repeat pairs captured for this
  document, status and iteration count landed identically for both problems under every
  configuration.

## 5. Theory posture

Neither strategy carries a convergence guarantee in this implementation, and none is claimed.
For the filter, the standard Wächter & Biegler global-convergence argument is stated for their
full algorithm, including a monotone barrier-parameter update scheme; this implementation
still drives the barrier parameter through the pre-existing monolithic governor, not a
provably-compatible one, so the filter's guarantee story is incomplete pending the barrier
governor work already flagged as future work in 2026-07-e2-g2-scorecards.md's nulls section. For the
funnel, the situation is more open: the trust-region funnel method has a published convergence
proof, but a *line-search* funnel's convergence proof is, per Kiessling, Leyffer & Vanaret's
own framing, an open question in the literature, and funnel acceptance embedded inside an
interior-point method (rather than the SQP setting the funnel literature analyzes) is
undocumented territory beyond that. The funnel earns its place in this codebase on corpus
evidence, not on a proof — the results in §3 are that evidence, not a substitute for one.

Both strategies also carry known, deliberate divergences from their reference
implementations, each disclosed with its consequence in the corresponding header's file-top
formulation comment:

- **Funnel** (`include/tycho/detail/solvers/globalization/funnel_acceptance.h`): Uno gates
  every accepted step — both switching/Armijo (F-type) and funnel (H-type) — on the funnel
  width; this implementation's shared switching skeleton consults the funnel only on the
  H-type path, so an F-type step is bounded by the base's θ_max ceiling but not by the funnel
  width itself. The strict funnel invariant (every accepted iterate inside the funnel) holds
  for H-type accepts only.
- **Filter** (`include/tycho/detail/solvers/globalization/filter_acceptance.h`): Ipopt drives
  its filter-reset heuristic once per solver iteration, from the *last* rejection's cause;
  this implementation's shared skeleton only exposes a per-trial H-type verdict, so the reset
  counter counts consecutive filter-caused rejections per trial rather than per iteration. The
  reset trigger is consequently easier to satisfy than Ipopt's, and can fire mid-line-search.

Readers who need the full derivation — every constant's source equation and option name, and
every divergence with its consequence spelled out — should read the headers directly rather
than rely on this summary.

## 6. Recommendation

On this 17-problem corpus (not a general claim about PSIOPT's behavior on arbitrary problems —
the corpus is deliberately small and adversarially selected, see `tests/corpus/README.md`):

- **`funnel`** (`acceptance_strategy=funnel`) is the configuration to reach for when
  efficiency matters most: it clears the corpus's literature counterexample in the fewest
  iterations of any configuration measured across this document and 2026-07-e2-g2-scorecards.md, wins on every
  other problem it touches, and regresses nothing.
- **`filter`** (`acceptance_strategy=filter`) is the configuration to reach for when
  robustness matters most: it is the only configuration in this program to move
  `hard_mountaincar_badguess` off `failed`, and it demotes `hard_zermelo_wrongbasin` from an
  unbounded divergence to a bounded failure — at some iteration cost on the problems both
  strategies solve.
- **The watchdog composes cleanly with both** but has not been observed to change any outcome
  on this corpus; enabling it alongside either strategy is architecturally sound and currently
  a no-op in practice here.
- Both strategies are strictly opt-in (`acceptance_strategy` defaults to `classic_merit`);
  `defaults` behavior is unchanged, confirmed byte-for-byte against the committed baseline in
  §3 and §4.

All results in this document are reproducible via:

```
conda run -n tycho python scripts/run_corpus.py --cbwr --config acceptance_strategy=funnel
conda run -n tycho python scripts/run_corpus.py --cbwr --config acceptance_strategy=filter
conda run -n tycho python scripts/run_corpus.py --cbwr --config acceptance_strategy=funnel watchdog=1
conda run -n tycho python scripts/run_corpus.py --cbwr --config acceptance_strategy=filter watchdog=1
```

## References

- `tests/corpus/README.md` — problem-module contract, harness CLI, JSONL schema, and the
  `ConvergenceFlags`-derived status severity ranking used in §3.
- `docs/dev/analysis/2026-07-e2-g0-baseline.md` — the 2026-07 defaults baseline this document
  diffs against, and the source of the order-sensitivity notes reused in §4.
- `docs/dev/analysis/2026-07-e2-g2-scorecards.md` — the merit and recovery-chain scorecards
  this document follows; source of the `wmno`/`flex` `lit_wb2000` iteration counts cited in
  §3.
- `tests/corpus/baselines/2026-07-defaults.jsonl` — committed baseline used for the `defaults`
  cross-check in §4.
- `include/tycho/detail/solvers/globalization/funnel_acceptance.h` — the funnel formulation,
  its Kiessling-Leyffer-Vanaret / Uno citations, and the disclosed divergences from those
  sources.
- `include/tycho/detail/solvers/globalization/filter_acceptance.h` — the filter formulation,
  its Wächter-Biegler / Ipopt citations, and the disclosed divergences from those sources.
- `include/tycho/detail/solvers/psiopt_fwd.h` — `AcceptanceStrategies` enum definition and the
  `ConvergenceFlags` severity-ordering operator cited in §3.
- `src/solvers/psiopt.cpp` — the strategy-combination guard establishing that the watchdog
  composes with every acceptance strategy while the classic-path recovery links do not.
