# Globalization evaluation campaign — sweep evidence, promotion recommendation, presets

Date: 2026-07-25. Branch: `feat/g8-campaign-sweep`. Artifacts of record, all
committed under `tests/corpus/campaign/`: the sweep aggregate
(`2026-07-globalization-sweep.{csv,json}`; **144 valid cells** — 48 of the 192
enumerated cells rejected by `Settings::validate()`, all 48 being exactly the
funnel/filter acceptance × classic-governor exclusion) and the four
example-arm captures (`2026-07-example-arm-{baseline,balanced,robust,proxonly}.csv`).
Driver and instrument: `scripts/run_campaign.py`,
`scripts/capture_example_iters.py`.

## 1. Corpus sweep (17 problems, ×2 CBWR repeats per cell)

Context columns: defaults-PSIOPT **9/17** solve-or-acceptable, Ipopt 3.14.19
stock + matched tolerances (both directions of the acceptable tier matched, so
the flag comparison is symmetric) at **10/17** (MUMPS) and **9/17** (MKL
Pardiso — linear-solver sensitivity is real: mountaincar flips
converged→failed between them; the two pre-iteration Ipopt aborts,
redundant_defects and vanderpol, are identical under both linear solvers and
therefore not linear-algebra artifacts).

**Headline: the best stack configuration reaches 12/17 at flag level — above
every Ipopt reference column on identical NLPs — with one caveat the flag
alone hides.** The winner (`62994231856d`) scores 8 converged + 4 acceptable;
one of those acceptables is `hard_brach_illscaled`, where the exit lands at
objective **−2.410** while Ipopt converges to **1.801** on the identical NLP.
The objective is a delta-time and the accepted point has NEGATIVE transfer
time — but a follow-up investigation resolved this as FORMULATION, not solver:
the ODE is time-reversal symmetric, time is pinned only at the front, nothing
bounds delta-time positive, and the NLP is unbounded below (pinning delta-time
at −1000 also solves cleanly, in the ill-scaled AND unit-scale variants). The
accepted point is feasible to well inside even the STRICT equality tolerance
(ECons 3.1e-8); only the KKT norm sat in the acceptable band. The acceptable
tier applied its tolerances exactly as written. Ipopt applied NO problem
scaling here (its scaled and unscaled residual columns are identical) — its
1.801 exit is basin selection on an unbounded problem, not a scaling
capability. This row is a legitimate exit on a pathological formulation; the
corpus module now warrants a docstring note about the unboundedness. Read strictly, the winner is
**11 solid + 1 flag-only**; the campaign machinery classifies on exit flag
alone and never checks objective agreement (recorded in §6). Under either
reading it beats or ties every Ipopt column.

The cap-8 shortlist is drawn from a **27-cell qualifying band** (within one of
best AND repeat-stable), tie-broken by iteration total on the commonly solved
set, and splits into exactly two families:

| family | axes | best cells |
|---|---|---|
| filter + monitored governor + ℓ1 restoration | ± prox inertia, ± recovery | **12/17** (`62994231856d`, classic inertia); prox variant `da972cd2de12` 11/17 with more full convergences (9+2) |
| classic merit + prox inertia | ± SOC, ± restoration | 11/17 (shortlisted variants incl. `0c5469fd8c67` prox+ℓ1 and `2825960df439` prox alone; ~20 stable 11/17 cells of this family exist in the band) |

The second family keeps the classic acceptance path entirely — its wins come
from the regularization mode plus restoration certificates alone. Notably,
every measured candidate's solved set is a **strict superset** of the defaults
column (gains deg_near_infeasible and lit_wb2000, loses nothing) — corpus
dominance is set-inclusion, not just a bigger count.

## 2. Example-suite arm (promotion criterion: ≤ +2% median iteration delta, no
example regressing to failure)

Measured natively: temporary local builds with each candidate's configuration
as the constructed `Settings{}` defaults (the only faithful way to measure "as
the shipped default"), against a same-session stock baseline
(`2026-07-example-arm-baseline.csv`). Failure checks cover all 34 examples;
percentage deltas are defined over the 23 with nonzero deterministic iteration
counts (34 − the 3 known-noisy − 8 that print no iteration lines). **Rule: an
example whose run fails is reported as a failure and excluded from the median**
(its partial iteration count would otherwise pollute the statistic — with the
failed runs excluded, the balanced candidate's median is **+8.16%**, not the
+0.00% the polluted set shows).

| cell (as defaults) | median Δ (fails excluded) | new failures | worst tail regressions |
|---|---|---|---|
| `0c5469fd8c67` prox + ℓ1 ("balanced", 11/17) | **+8.16%** | **MinimumTimeToClimb fails** | Dionysus +441%, Goddard +63%, Zermelo +53% |
| `62994231856d` filter + monitored + ℓ1 ("robust", 12/17) | +0.00% | BettsLowThrustCentralShooting fails | Dionysus +609%, MinTimeToClimb +395%, MultiPhaseCannon +340% (but OptimalDocking −41%, BrysonDenham −30%) |
| `2825960df439` prox alone (11/17) | +0.00% | none | **MinTimeToClimb +137%, OptimalDocking +99%**, OrbitContinuation +34%, BettsLowThrust +23% (BrysonDenham −27%, Delta3 −25%); and on the noise-excluded set, **MultiSpacecraftOptimization 1535 → 11969 (+680%)** |

The ℓ1 component caused the balanced candidate's outright failure (dropping it
recovers MinimumTimeToClimb at 154 iterations, ok); the prox component alone
still doubles two deterministic examples and blows up one noisy-set example by
7.8×. The noisy-trio exclusion exists for bit-exactness diffing, not for
absorbing magnitude changes of that size — hence the widened reversal
condition below.

**Both "new failures" were diagnosed to root cause and are evaluation-domain
exceptions, not solver regressions.** MinimumTimeToClimb under prox+ℓ1 dies on
`InterpTable2D: query x=1.80012 outside table x range [0, 1.8]` — the iterate
path grazes 0.007% past the aero-table edge and the table throws.
BettsLowThrustCentralShooting under the robust configuration dies inside the
shooting integrator (`ParallelDriver: step size underflowed ... non-finite
derivatives`). In both cases the mechanisms legitimately visit trial points
the legacy path never did, and the evaluation layer converts an out-of-domain
state into a hard exception that aborts the entire solve. Notably, the Ipopt
backend built alongside this campaign already handles this correctly for
Ipopt — evaluator exceptions are latched and reported as a failed evaluation,
which Ipopt treats as a rejected trial point — while the built-in solver has
no equivalent: a thrown evaluation exception bypasses the step-rejection
machinery entirely. Treating trial-evaluation exceptions as rejected steps
(bounded retries, matching the reference solver's semantics) is recorded as a
follow-up mechanism with direct evidence from both failures. The tail
magnitudes on the prox cells are additionally consistent with the shift
double-memory observation from the regularization mode's review (the first
ladder rung after an episode landing at roughly twice the intended shift) —
that tuning target now has measured example-suite costs attached.

## 3. Promotion recommendation: NO DEFAULT FLIP

Prox-only formally meets the written promotion criteria — set-inclusion corpus
dominance, median +0.00%, zero new failures. The recommendation is still
**no flip**, on the tails: the median criterion was written before this data
existed, and a change that doubles two production-class examples (+137%, +99%)
and multiplies a third by 7.8× is not a balanced default — it is a robustness
trade users should opt into knowingly. Every rescue the stack provides remains
one `apply_preset` call away. This is the options-menu outcome the program
design explicitly sanctions as success. (Promotion criterion (c) — bench
within noise — is deliberately unevaluated here: it is only measurable in a
default-flip PR, and none is being opened.)

Reversal condition: a future configuration meeting a tail-aware bar — no
example above +25%, judged over ALL 34 examples by magnitude (the known-noisy
trio included at magnitude level, exempt only from bit-exactness) — reopens
the flip with the same procedure: enumerated per-example deltas,
CA-classified PR, maintainer sign-off. Mechanism tuning that could plausibly
get there: the regularization mode's shift double-memory and singular-base
observations recorded in its review.

## 4. Preset nominations (for the presets/docs stage)

- `default` — tracks the constructed `Settings{}` defaults (stock today;
  automatically follows any future flip).
- `classic` — pinned historical configuration (identical to today's defaults;
  kept distinct so the name survives a future flip).
- `robust` — filter + monitored governor + ℓ1 nested restoration
  (`62994231856d`). Documented contract: maximum flag-level solve rate (12/17,
  of which one is the brach_illscaled flag-only acceptable), at a measured
  example-suite cost (tails above; median flat) including one central-shooting
  example failure — the troubleshooting guide carries the numbers.
- The balanced cell (`0c5469fd8c67`) is documented in the troubleshooting
  guide as the first thing to try on degenerate/infeasible formulations, NOT
  shipped as a preset name, since its example-arm failure disqualifies it from
  blanket recommendation.

## 5. Reference-solver checkpoint (composite-step activation reading)

Initially the sweep showed two problems only stock Ipopt solves; a follow-up
deep-dive dissolved both. `hard_zermelo_wrongbasin`: under a MATCHED single
`optimize()` call (exactly what the Ipopt backend always runs), PSIOPT with
merit acceptance + nested-ℓ1 restoration converges to Ipopt's point —
objective agreement 2.4e-15 relative (1.7009270229362865 @40 vs
1.7009270229362905 @28), repeat-stable, both solvers passing through their
restoration phases. The campaign's DIVERGING row traces to the module's
`solve_optimize` call shape, whose feasibility-only first stage stalls with
the globalization stack structurally unreachable (see §6a). Prior editions of
this note attributed `hard_brach_illscaled` to Ipopt's native NLP scaling;
that attribution was WRONG and is retracted — Ipopt applied no scaling there,
and the difference is basin selection on an unbounded formulation (§1).
**With both anomalies explained, no corpus problem remains that stock Ipopt
solves and the PSIOPT stack cannot; the composite-step checkpoint stays
closed, now without residue.** The restoration head-to-head is the strongest
correctness evidence the program has produced: two independent
Wächter–Biegler-lineage implementations agreeing to machine precision on a
wrong-basin recovery.

### 6a. New finding: the feasibility-only stage has no globalization

The zermelo investigation exposed that `solve()` (and the feasibility stage of
`solve_optimize`) forces the objective scale to zero, under which every trial
step is accepted — line-search backtracks, the recovery chain, and restoration
are all structurally unreachable (0 backtracks / 0 restoration entries / empty
filter over a 500-iteration stall in which the equality residual grew 1.9×).
The globalization program scoped the optimize path; the feasibility path
predates it and never gained any of it. Recorded as a refinement area with
mechanism and candidate fixes in the refinement note — arguably the largest
remaining robustness gap in the solver, and invisible to any optimize-path
comparison.

## 6. Campaign mechanics record

Sweep: 144 valid cells × 2 repeats in ~93 min wall (config-hash-resumable
store; the aggregate now prints an aggregated/invalid/incomplete summary line
so count errors surface at generation time). Shortlist per the
band/stability/intersection-tie-break rule (band 27, cap 8). Example arm via
three temporary default-flip builds (bindings target only), stock defaults
restored and `git diff --quiet`-verified after each; captures committed beside
the aggregate. Classification caveat: all corpus statuses are exit-flag-level;
no objective-agreement check is applied anywhere in the sweep machinery — the
brach_illscaled observation in §1 came from manual JSON inspection, and any
future campaign wanting objective-aware scoring needs a driver extension.
Cross-backend caveat: the corpus driver runs the psiopt backend through each
module's `SOLVE_MODE` but the ipopt backend always through a single solve, so
rows for non-`optimize` modules (zermelo, mountaincar, hypersens) compare
different call shapes — the zermelo deep-dive shows this can flip a status.
A matched-call option is recorded as a harness follow-up.
