///////////////////////////////////////////////////////////////////////////////
// Unit tests for FunnelAcceptance — the scalar-funnel H-type acceptance
// strategy layered on the Wächter–Biegler switching skeleton
// (SwitchingAcceptance).
//
// FunnelAcceptance is concrete: it is exercised through the base's public
// template method is_iterate_acceptable(), observing the public accessor
// funnel_width() and the accept/reject verdict. To drive the funnel's own
// (H-type) rules deterministically, every "H-type" call below passes
// predicted_reduction.objective = 0: the base's switching condition requires a
// strict descent direction (m_f > 0), so with m_f = 0 the switching test never
// fires and the trial is always routed to the H-type delegate — regardless of
// θ_min. F-type is driven separately with a descent-satisfying predicted
// reduction (see FunnelFTypeAcceptLeavesWidthUnchanged).
//
// Every scenario's arithmetic is computed BY HAND (shown in the comments) from
// the rules documented in funnel_acceptance.h:
//   init:  τ = max(τ̄, κ̄·θ₀),        τ̄ = 1.0 (kFunnelInitialUpperBound),
//                                      κ̄ = 1.5 (kFunnelInfeasibilityFactor)
//   H-type verdict:  θ_trial ≤ τ  AND  θ_trial ≤ β·τ,  β = 0.9999 (kFunnelBeta)
//   update (accepted H-type, strategy 1):
//     θ_trial ≤ θ_current:  τ⁺ = max(β·τ, κ·θ_current + (1−κ)·θ_trial)
//     else:                 τ⁺ = β·τ,             κ = 0.5 (kFunnelKappa)
//
// UNITY RULE: anonymous namespace does not protect names against the unity
// build — every helper here is prefixed Funnel* to stay globally unique across
// tests/cpp/ (grep-confirmed no other "Funnel" symbol exists).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/funnel_acceptance.h"

#include "tycho/detail/solvers/globalization/progress_measures.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace {

using tycho::solvers::FunnelAcceptance;
using tycho::solvers::kFunnelBeta;
using tycho::solvers::kFunnelInitialUpperBound;
using tycho::solvers::kFunnelKappa;
using tycho::solvers::ProgressMeasures;

// File-unique helper (see UNITY RULE above).
ProgressMeasures FunnelMakePm(double infeasibility, double objective, double auxiliary) {
    ProgressMeasures pm;
    pm.infeasibility = infeasibility;
    pm.objective = objective;
    pm.auxiliary = auxiliary;
    return pm;
}

// Priming: run the FIRST is_iterate_acceptable() so the width is derived from
// θ₀ = current.infeasibility, WITHOUT the priming call mutating the width. The
// trial sits OUTSIDE the funnel (θ_trial > τ) so the H-type verdict rejects and
// register_accepted_h_type() never runs — the width is left exactly at its
// initialized value. Returns nothing; the caller asserts funnel_width().
//   pred.objective = 0 ⇒ H-type; trial_outside must be > τ but ≤ θ_max.
void FunnelPrimeRejecting(FunnelAcceptance &a, double theta0, double trial_outside) {
    const bool ok = a.is_iterate_acceptable(FunnelMakePm(theta0, 0.0, 0.0),
                                            FunnelMakePm(trial_outside, 0.0, 0.0),
                                            FunnelMakePm(0.0, 0.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok); // trial outside the funnel ⇒ rejected, width untouched
}

// Convenience: an H-type call (m_f = 0) with the given current/trial θ.
bool FunnelHType(FunnelAcceptance &a, double theta_current, double theta_trial) {
    return a.is_iterate_acceptable(FunnelMakePm(theta_current, 0.0, 0.0),
                                   FunnelMakePm(theta_trial, 0.0, 0.0), FunnelMakePm(0.0, 0.0, 0.0),
                                   1.0, 1.0);
}

// ===========================================================================
// Width initialization from θ₀:  τ = max(τ̄, κ̄·θ₀).
// ===========================================================================

// Multiplier branch: θ₀ = 4.0 ⇒ κ̄·θ₀ = 1.5·4 = 6.0 > τ̄ = 1.0 ⇒ τ = 6.0.
// Priming trial 10.0 > 6.0 ⇒ rejected ⇒ width stays 6.0.
TEST(FunnelAcceptance, InitWidthMultiplierBranch) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/4.0, /*trial_outside=*/10.0);
    EXPECT_DOUBLE_EQ(a.funnel_width(), 6.0);
}

// Floor branch: θ₀ = 0.2 ⇒ κ̄·θ₀ = 1.5·0.2 = 0.3 < τ̄ = 1.0 ⇒ τ = 1.0.
// Priming trial 5.0 > 1.0 ⇒ rejected ⇒ width stays 1.0.
TEST(FunnelAcceptance, InitWidthFloorBranch) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/0.2, /*trial_outside=*/5.0);
    EXPECT_DOUBLE_EQ(a.funnel_width(), kFunnelInitialUpperBound); // = 1.0
}

// Before the first call the width is the uninitialized sentinel (+∞).
TEST(FunnelAcceptance, WidthUninitializedBeforeFirstCall) {
    FunnelAcceptance a;
    EXPECT_TRUE(std::isinf(a.funnel_width()));
}

// ===========================================================================
// H-type verdict: within the funnel AND sufficient reduction.
// ===========================================================================

// θ₀ = 4.0 ⇒ τ = 6.0, β·τ = 0.9999·6 = 5.9994.
// Trial θ = 3.0: inside (3 ≤ 6) AND sufficient (3 ≤ 5.9994) ⇒ ACCEPT.
TEST(FunnelAcceptance, TrialInsideFunnelSufficientDecreaseAccepts) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    EXPECT_TRUE(FunnelHType(a, /*theta_current=*/4.0, /*theta_trial=*/3.0));
}

// Trial θ = 7.0 > τ = 6.0: outside the funnel ⇒ REJECT, and (proving the
// "only then" rule) the width is left unchanged at 6.0.
TEST(FunnelAcceptance, TrialOutsideFunnelRejectsAndLeavesWidth) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    EXPECT_FALSE(FunnelHType(a, /*theta_current=*/4.0, /*theta_trial=*/7.0));
    EXPECT_DOUBLE_EQ(a.funnel_width(), 6.0); // rejected H-type ⇒ no update
}

// Sufficient-decrease boundary. τ = 6.0, threshold β·τ = 5.9994.
//   • trial exactly at the threshold (5.9994): 5.9994 ≤ 6.0 AND 5.9994 ≤ 5.9994
//     ⇒ ACCEPT (the condition is "≤").
//   • trial just above (threshold·1.00001 ≈ 5.99946): still inside the funnel
//     (≤ 6.0) but fails sufficient reduction (> 5.9994) ⇒ REJECT.
// Fresh instances: the accepting case mutates the width.
TEST(FunnelAcceptance, SufficientDecreaseBoundaryPasses) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);         // τ = 6.0
    const double threshold = kFunnelBeta * 6.0; // = 5.9994
    EXPECT_TRUE(FunnelHType(a, /*theta_current=*/6.0, /*theta_trial=*/threshold));
}

TEST(FunnelAcceptance, SufficientDecreaseBoundaryFailsJustAbove) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);                    // τ = 6.0
    const double just_above = kFunnelBeta * 6.0 * 1.00001; // ≈ 5.99946, still ≤ 6.0
    ASSERT_LE(just_above, 6.0);                            // guard: still inside the funnel
    EXPECT_FALSE(FunnelHType(a, /*theta_current=*/6.0, /*theta_trial=*/just_above));
    EXPECT_DOUBLE_EQ(a.funnel_width(), 6.0); // rejected ⇒ width unchanged
}

// ===========================================================================
// Width update on an accepted H-type step (strategy 1) — exact values.
// ===========================================================================

// Floor branch dominates. τ = 6.0, β·τ = 5.9994.
// current θ = 4.0, trial θ = 2.0 (≤ current) ⇒
//   convex = κ·4 + (1−κ)·2 = 0.5·4 + 0.5·2 = 3.0
//   τ⁺ = max(5.9994, 3.0) = 5.9994.
TEST(FunnelAcceptance, WidthUpdateFloorDominates) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    ASSERT_TRUE(FunnelHType(a, /*theta_current=*/4.0, /*theta_trial=*/2.0));
    EXPECT_DOUBLE_EQ(a.funnel_width(), kFunnelBeta * 6.0); // 5.9994
}

// Convex-combination branch dominates. τ = 150.0, β·τ = 0.9999·150 = 149.985.
// current θ = 150.0, trial θ = 149.98 (≤ current, and ≤ β·τ ⇒ accepted) ⇒
//   convex = 0.5·150 + 0.5·149.98 = 0.5·299.98 = 149.99
//   τ⁺ = max(149.985, 149.99) = 149.99  (< 150 ⇒ still strictly decreasing).
TEST(FunnelAcceptance, WidthUpdateConvexCombinationDominates) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/100.0, /*trial_outside=*/200.0); // τ = 150.0
    ASSERT_TRUE(FunnelHType(a, /*theta_current=*/150.0, /*theta_trial=*/149.98));
    const double expected =
        std::max(kFunnelBeta * 150.0, kFunnelKappa * 150.0 + (1.0 - kFunnelKappa) * 149.98);
    EXPECT_DOUBLE_EQ(a.funnel_width(), expected); // 149.99
    EXPECT_LT(a.funnel_width(), 150.0);
}

// "else" branch (trial infeasibility ABOVE current). τ = 6.0, β·τ = 5.9994.
// current θ = 2.0, trial θ = 4.0 (> current, but ≤ β·τ ⇒ accepted) ⇒
//   τ⁺ = β·τ = 5.9994  (the convex combination is NOT taken).
TEST(FunnelAcceptance, WidthUpdateTrialAboveCurrentUsesFloor) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    ASSERT_TRUE(FunnelHType(a, /*theta_current=*/2.0, /*theta_trial=*/4.0));
    EXPECT_DOUBLE_EQ(a.funnel_width(), kFunnelBeta * 6.0); // 5.9994
}

// ===========================================================================
// F-type accept leaves the width unchanged (only accepted H-type updates it).
// ===========================================================================

// θ₀ = 1.0 ⇒ τ = max(1.0, 1.5·1.0) = 1.5; θ_min = 1e-4·max(1,1) = 1e-4.
// Prime rejecting (trial 5.0 > 1.5) ⇒ τ stays 1.5.
// F-type call: θ_current = 1e-5 ≤ θ_min, m_f = 0.01 > 0 ⇒ switching tested:
//   lhs = 1·(0.01/1)^2.3 = 0.01^2.3 ≈ 2.511886e-05
//   rhs = 1·(1e-5)^1.1        ≈ 3.162278e-06 ⇒ lhs > rhs ⇒ switching HOLDS.
// Armijo on φ = obj + aux: φ_trial = 9.9999 ≤ φ_current(10) − 1e-8·0.01
//   = 9.9999999999 ⇒ ACCEPT as F-type ⇒ register_accepted_h_type NOT called
//   ⇒ width stays 1.5.
TEST(FunnelAcceptance, FTypeAcceptLeavesWidthUnchanged) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/1.0, /*trial_outside=*/5.0); // τ = 1.5
    const bool ok =
        a.is_iterate_acceptable(FunnelMakePm(1.0e-5, 10.0, 0.0), FunnelMakePm(1.0e-6, 9.9999, 0.0),
                                FunnelMakePm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);                         // F-type accept
    EXPECT_DOUBLE_EQ(a.funnel_width(), 1.5); // untouched by the F-type step
}

// ===========================================================================
// reset() restores the uninitialized state; the next θ₀ re-derives the width.
// ===========================================================================

// First arm: θ₀ = 4.0 ⇒ τ = 6.0. reset() ⇒ τ = +∞ (sentinel). Second arm:
// θ₀ = 0.5 ⇒ κ̄·θ₀ = 0.75 < τ̄ = 1.0 ⇒ τ = 1.0.
TEST(FunnelAcceptance, ResetReDerivesWidthFromNewThetaZero) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);
    EXPECT_DOUBLE_EQ(a.funnel_width(), 6.0);

    a.reset();
    EXPECT_TRUE(std::isinf(a.funnel_width())); // uninitialized sentinel restored

    FunnelPrimeRejecting(a, 0.5, 5.0);
    EXPECT_DOUBLE_EQ(a.funnel_width(), 1.0); // re-derived from the new θ₀
}

// ===========================================================================
// Monotone non-increasing width across a synthetic accepted-H-type sequence.
// ===========================================================================

// τ₀ = 150.0. Each step: current θ = 10.0, trial θ = 5.0 (5 ≤ current, and
// 5 ≤ β·τ throughout since τ stays ≈ 150) ⇒ accepted; convex = 0.5·10+0.5·5 =
// 7.5 ≪ β·τ ⇒ the floor dominates ⇒ τ⁺ = β·τ = 0.9999·τ. Hence τ shrinks by a
// constant factor 0.9999 each step: strictly decreasing, always > 0.
TEST(FunnelAcceptance, WidthMonotoneNonIncreasingAcrossSequence) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/100.0, /*trial_outside=*/200.0); // τ = 150.0
    double prev = a.funnel_width();
    ASSERT_DOUBLE_EQ(prev, 150.0);

    for (int step = 0; step < 5; ++step) {
        ASSERT_TRUE(FunnelHType(a, /*theta_current=*/10.0, /*theta_trial=*/5.0));
        const double now = a.funnel_width();
        EXPECT_LT(now, prev);                      // strictly decreasing
        EXPECT_GT(now, 0.0);                       // never collapses to ≤ 0
        EXPECT_DOUBLE_EQ(now, kFunnelBeta * prev); // exact factor-β shrink
        prev = now;
    }
}

} // namespace
