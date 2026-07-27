# Post-fixes evidence refresh — sweep, matched-call column, co-leader example arms

Date: 2026-07-26. Solver state: main at the merge of the two globalization
robustness fixes (trial-evaluation exceptions as rejected rungs; feasibility
stage stall dispatch with the sustained-worsening detector) plus the corpus
harness's matched-call flag. This refresh re-measures the evaluation
campaign's full configuration matrix on the fixed solver and adds the
matched-call column the earlier campaign structurally could not see.
Committed artifacts: `tests/corpus/campaign/2026-07-globalization-sweep-postfixes.{csv,json}`
and the three `2026-07-postfixes-arm-*.csv` example-arm captures. Baseline
comparisons are against `2026-07-globalization-sweep.csv` and
`2026-07-example-arm-baseline.csv` from the original campaign.

## Sweep vs the pre-fixes campaign

144 valid cells, ×2 CBWR repeats, zero incomplete. Stability first: 139 of
144 cells carry byte-identical per-problem status strings versus the
pre-fixes sweep, and repeat-stability ticked up (142 vs 141) — the fixes are
surgical, not a re-roll of the landscape.

The leaderboard changed at the top. The original winner holds, and two cells
rose by one problem each to join it, giving three co-leaders at 12/17
(converged + acceptable, 8 converged each):

- `62994231856d` — filter acceptance + monitored governor + nested-ℓ1
  restoration (the original campaign winner; unchanged at 12/17).
- `8417a47846c1` — classic-merit + monitored + proximal-regularization
  inertia + SOC(4) + recovery (extended backtrack + watchdog) + nested-ℓ1
  (was 11/17; +1 from the fixes).
- `8d8397c915b2` — classic-merit + monitored + proximal-regularization
  inertia + SOC(4) + proximal-switch restoration (was 11/17; +1).

One cell fell by one (`da972cd2de12`, filter + monitored + ℓ1 + prox
inertia, 11 → 10). The shortlist (within-1-of-best, repeat-stable, cap 8)
contains all three co-leaders plus the classic+prox family variants.

## Matched-call column (the new harness flag)

The eight shortlisted cells were re-run under `--call-shape optimize`
(single `optimize()` per problem, the ipopt backend's shape), ×2 repeats,
all repeat-stable. For the shortlist the column is nearly shape-invariant:
the only mover is `hard_mountaincar_badguess` (drops its acceptable under
the matched shape for the three co-leaders; gains it for one prox cell).

The column's value shows outside the shortlist: the merit-acceptance +
nested-ℓ1 cell (`acceptance=merit`, classic governor, ℓ1) scores 7+2 under
its module shapes with zermelo diverging, but 8+2 under the matched shape
with **zermelo converging** — reproducing, through the shipped harness, the
investigation result that this family solves zermelo's wrong-basin guess in
a single optimize call (@40 iterations to objective 1.7009270229362865, the
Ipopt-agreement reference). The zermelo class is a call-shape problem, not
an acceptance-mechanism problem, and the harness can now measure that
directly.

## Co-leader example arms (temporary default-flip captures)

Each co-leader was applied as the Settings defaults, the extension rebuilt,
and the 34-example suite captured (MKL_CBWR pinned), then reverted. The
three machine-unstable examples documented by the CBWR gates
(MultiSpacecraftOptimization, SimpleLowThrust, ParallelParking) are excluded
from ratio statistics.

All three arms: median iteration ratio at parity with the stock baseline,
33/34 examples passing, and the same single failure —
BettsLowThrustCentralShooting, the known committed-point integrator failure
inside the feasibility stage (out of scope for the shipped fixes and
recorded as the stage's next frontier). The differences are in the tails:

| Cell | Aggregate iterations vs baseline | Worst tails |
| --- | --- | --- |
| filter+monitored+ℓ1 | +31% | DionysusLowThrust +609%, MinimumTimeToClimb +395%, MultiPhaseCannon +387% |
| classic+mon+prox+SOC4+recovery+ℓ1 | +42% | OptimalDocking +100%, Zermelo +64%, BettsLowThrust +56% |
| classic+mon+prox+SOC4+proximal | +27% | MinimumTimeToClimb +219%, OptimalDocking +100%, MultiPhaseCannon +67% |

## Consequences for the presets stage

- The no-default-flip ruling stands unchanged: every candidate carries
  example-arm tails far past the reversal condition, at median parity.
- The robust preset is now a three-way decision rather than a coronation:
  the filter family has the worst single-example blowups but the original
  campaign pedigree; the classic+prox+SOC+proximal cell has the lowest
  aggregate cost and no ℓ1 machinery; the recovery-equipped ℓ1 cell has the
  flattest worst-case at the highest aggregate cost. Maintainer adjudication
  happens in the presets PR with these tables.
- The matched-call evidence argues the presets documentation should teach
  the call-shape lever explicitly for wrong-basin/feasibility-hostile
  problems, independent of preset choice.
