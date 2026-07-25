# Globalization refinement areas from the evaluation campaign

Date: 2026-07-25. Source evidence: `docs/dev/analysis/2026-07-e2-g8-campaign.md`
and the committed artifacts under `tests/corpus/campaign/`. Status: recorded
work items — none scheduled until the campaign evidence PR and the presets/docs
stage land.

## 1. Trial-evaluation exceptions abort the solve instead of rejecting the step

**Evidence:** both example-arm "new failures" are evaluation-domain exceptions,
not solver regressions: `InterpTable2D: query x=1.80012 outside table x range
[0, 1.8]` (MinimumTimeToClimb under prox+ℓ1 — the iterate grazes 0.007% past
the aero-table edge) and `ParallelDriver: step size underflowed ... non-finite
derivatives` (Betts central shooting under the filter/monitored/ℓ1
configuration). The globalization mechanisms legitimately visit trial points
the classic path never did; the evaluation layer converts an out-of-domain
state into a process-killing exception that bypasses the step-rejection
machinery entirely.

**Reference behavior:** Ipopt treats a failed evaluation as a rejected trial
point and backtracks — and Tycho's own Ipopt adapter already implements
exactly that (evaluator exceptions latch and return false). The built-in
solver has no equivalent.

**Refinement:** catch evaluation exceptions at trial-point evaluation inside
the acceptance backtrack, treat as a rejected trial (bounded retries), latch
the message into diagnostics; a committed-point evaluation failure remains
fatal. PSIOPT step-path change: own CA-gated PR, CBWR bit-identity on the
default path required (the classic path should be unaffected unless an
exception occurs, which today is fatal anyway — byte-identity expected by
construction).

## 2. Regularization-mode shift double-memory (post-episode over-shift)

**Evidence:** the review of the regularization mode flagged that after a ladder
episode the successful shift is remembered twice — the persistent shift carries
its decayed value as the next base AND the ladder warm-start seeds from the
same delta — so the first rung of a subsequent episode lands at roughly twice
the intended shift. The campaign attached measured costs consistent with that
mechanism: with the mode as default, MinimumTimeToClimb +137%, OptimalDocking
+99%, MultiSpacecraftOptimization 1535 → 11969 iterations (+680%), while the
median stays +0.00%.

**Refinement:** a scoped A/B experiment — seed the ladder warm-start OR the
persistent shift from the episode outcome, not both — measured on the three
tail examples AND the corpus rescues (wb2000, near_infeasible must not
regress). If the tails collapse with rescues intact, this is the most
plausible path to a future default flip under the recorded tail-aware
reversal condition.

## 3. Acceptable-tier semantics on structurally meaningless points

**Evidence:** the campaign winner's `hard_brach_illscaled` exit is an
ACCEPTABLE at objective −2.410 — a negative transfer time — where Ipopt
converges to 1.801 on the identical NLP.

**Open question (investigate before changing code):** does that point satisfy
the constraints (in scaled and original units)? If yes, the formulation
genuinely admits it (nothing forbids negative delta-time in the corpus
problem) and this is a problem-classification note, not a solver deficiency;
if no, the acceptable tier's loosened tolerances signed off on an infeasible
point and need an original-units feasibility guard.

## Deep-dive addendum: the two Ipopt-only successes

Across all 144 swept configurations, exactly two corpus problems are solved by
stock Ipopt and by no PSIOPT configuration at flag level (treating the
negative-time acceptable as a non-solve): `hard_zermelo_wrongbasin` (Ipopt
converges @28 where every restoration-bearing PSIOPT configuration rides its
iteration cap) and `hard_brach_illscaled` (native NLP scaling). Both implicate
mechanisms this program implemented from the same literature Ipopt implements
— the zermelo case specifically pits Ipopt's restoration against the nested
ℓ1 restoration on the same start point, making it the sharpest available
probe for implementation-vs-intention deviations. Findings from that probe
belong in this note when they land.
