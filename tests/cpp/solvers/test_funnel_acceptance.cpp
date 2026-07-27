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

#include "progress_measures_test_utils.h"

#include "tycho/detail/solvers/globalization/funnel_acceptance.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

using tycho::solvers::FunnelAcceptance;
using tycho::solvers::kFunnelBeta;
using tycho::solvers::kFunnelInitialUpperBound;
using tycho::solvers::kFunnelKappa;
using TychoTest::pm;

// Priming: run the FIRST is_iterate_acceptable() so the width is derived from
// θ₀ = current.infeasibility, WITHOUT the priming call mutating the width. The
// trial sits OUTSIDE the funnel (θ_trial > τ) so the base's MEMBERSHIP test
// (checked for every trial) rejects it and register_accepted_step() never runs
// — the width is left exactly at its initialized value. Returns nothing; the
// caller asserts funnel_width().
//   pred.objective = 0 ⇒ H-type were membership to pass; trial_outside must be
//   > τ but ≤ θ_max.
void FunnelPrimeRejecting(FunnelAcceptance &a, double theta0, double trial_outside) {
    const bool ok = a.is_iterate_acceptable(pm(theta0, 0.0, 0.0),
                                            pm(trial_outside, 0.0, 0.0),
                                            pm(0.0, 0.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok); // trial outside the funnel ⇒ rejected, width untouched
}

// Convenience: an H-type call (m_f = 0) with the given current/trial θ.
bool FunnelHType(FunnelAcceptance &a, double theta_current, double theta_trial) {
    return a.is_iterate_acceptable(pm(theta_current, 0.0, 0.0),
                                   pm(theta_trial, 0.0, 0.0), pm(0.0, 0.0, 0.0),
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

// Pinning test — the ONE residual re-widening edge (funnel_acceptance.h
// note (3)). At the strategy level the base gates EVERY accept on membership
// (θ_trial ≤ τ), so no strategy-accepted iterate can land outside the funnel and
// the width is unconditionally monotone. The residual edge is OUTSIDE the
// strategy: when the backtracking ladder exhausts, the solver's recovery
// fallback can accept a strategy-rejected trial (accept-as-is), leaving the
// CURRENT iterate outside the funnel (θ_current ≫ τ). A subsequent accepted
// H-type step then reads that oversized θ_current in the convex-combination
// update and can transiently RE-WIDEN the funnel. This test drives the hook
// directly with an out-of-funnel θ_current to hold that recovery-fallback edge
// fixed (the H-type verdict itself checks θ_trial, not θ_current, so the update
// still runs).
//
// τ = 1.5 (θ₀ = 1.0 ⇒ τ = max(1.0, 1.5·1.0) = 1.5; same init as
// FTypeAcceptLeavesWidthUnchanged above). β·τ = 0.9999·1.5 = 1.49985.
// Membership + H-type verdict read θ_trial only: θ_trial = 0.5 ≤ τ (1.5) AND
// θ_trial ≤ β·τ (1.49985) ⇒ ACCEPT, regardless of θ_current.
// Update (θ_trial ≤ θ_current ⇒ convex-combination branch):
//   convex = κ·θ_current + (1−κ)·θ_trial = 0.5·100 + 0.5·0.5 = 50 + 0.25 = 50.25
//   τ⁺ = max(β·τ, convex) = max(1.49985, 50.25) = 50.25
// 50.25 ≫ 1.5: the width RE-WIDENS by more than 33x on this single step.
//
// If a future change clamps the update (τ⁺ = min(τ, ...)) to restore
// monotonicity even across the recovery-fallback edge, this test must be updated
// deliberately, and the corpus scorecards analysis doc must be re-validated
// against the new behavior before the clamp ships.
TEST(FunnelAcceptance, WidthUpdateReWidensWhenCurrentIterateOutsideFunnel) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/1.0, /*trial_outside=*/5.0); // τ = 1.5
    ASSERT_TRUE(FunnelHType(a, /*theta_current=*/100.0, /*theta_trial=*/0.5));
    EXPECT_DOUBLE_EQ(a.funnel_width(), 50.25); // re-widened, not tightened
    EXPECT_GT(a.funnel_width(), 1.5);          // strictly ABOVE the old width
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
//   = 9.9999999999 ⇒ ACCEPT as F-type ⇒ register_accepted_step is called with
//   h_type = false ⇒ width stays 1.5.
TEST(FunnelAcceptance, FTypeAcceptLeavesWidthUnchanged) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/1.0, /*trial_outside=*/5.0); // τ = 1.5
    const bool ok =
        a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 9.9999, 0.0),
                                pm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);                         // F-type accept
    EXPECT_DOUBLE_EQ(a.funnel_width(), 1.5); // untouched by the F-type step
}

// ===========================================================================
// Membership gates EVERY trial, F-type included (the behavior that changed).
// ===========================================================================

// An f-type trial (switching holds, Armijo would pass) whose infeasibility sits
// OUTSIDE the funnel (θ_trial > τ) is now REJECTED at membership — Uno wraps its
// entire is_iterate_acceptable body in "if (funnel.acceptable(trial))". Same
// init and f-type arithmetic as FTypeAcceptLeavesWidthUnchanged, but θ_trial is
// raised above τ.
//   θ₀ = 1.0 ⇒ τ = 1.5, θ_min = 1e-4, θ_max = 1e4.
//   θ_trial = 2.0 > τ = 1.5 (but ≤ θ_max) ⇒ membership fails ⇒ REJECT before
//   the switching/Armijo tests. Armijo would otherwise pass (φ_trial = 9.9999 ≤
//   φ_current(10) − 1e-8·0.01), so the rejection is due to membership alone.
//   A rejected trial performs no width update ⇒ τ stays 1.5.
TEST(FunnelAcceptance, MembershipRejectsFTypeTrialOutsideFunnel) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, /*theta0=*/1.0, /*trial_outside=*/5.0); // τ = 1.5
    const bool ok =
        a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(2.0, 9.9999, 0.0),
                                pm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);                        // rejected at membership (θ_trial > τ)
    EXPECT_DOUBLE_EQ(a.funnel_width(), 1.5); // untouched by the rejected trial
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
// Width is UNCONDITIONALLY monotone non-increasing across a strategy-accepted
// sequence: the base gates every accept on membership (θ_trial ≤ τ), so every
// accepted iterate is inside the funnel and each update strictly tightens τ.
// ===========================================================================

// τ₀ = 150.0. Each step: current θ = 10.0, trial θ = 5.0 (membership 5 ≤ τ and
// H-type progress 5 ≤ β·τ hold throughout since τ stays ≈ 150) ⇒ accepted;
// convex = 0.5·10+0.5·5 = 7.5 ≪ β·τ ⇒ the floor dominates ⇒ τ⁺ = β·τ =
// 0.9999·τ. Hence τ shrinks by a constant factor 0.9999 each step: strictly
// decreasing, always > 0.
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

// ===========================================================================
// Restoration-exit test (Uno FunnelMethod): funnel-membership(θ_trial) AND
// θ_trial ≤ β·θ_ref. β = kFunnelBeta = 0.9999. The two halves reject
// independently — this section walks both.
// ===========================================================================

// Both halves pass. Prime θ₀ = 4.0 ⇒ τ = 6.0. reference θ = 10.0 ⇒ relative
// threshold 0.9999·10 = 9.999. trial θ = 5.0: membership 5 ≤ 6 AND relative
// 5 ≤ 9.999 ⇒ ACCEPT-exit.
TEST(FunnelAcceptance, ExitTestBothHalvesPass) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(pm(10.0, 0.0, 0.0),
                                                        pm(5.0, 0.0, 0.0)));
}

// Membership half rejects independently: trial θ = 7.0 > τ = 6.0 (outside the
// funnel) even though the relative half would pass (7 ≤ 0.9999·10 = 9.999).
TEST(FunnelAcceptance, ExitTestMembershipHalfRejects) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(pm(10.0, 0.0, 0.0),
                                                         pm(7.0, 0.0, 0.0)));
}

// Relative half rejects independently: reference θ = 5.0 ⇒ threshold 4.9995;
// trial θ = 5.5 passes membership (5.5 ≤ 6.0) but fails 5.5 ≤ 4.9995.
TEST(FunnelAcceptance, ExitTestRelativeHalfRejects) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(pm(5.0, 0.0, 0.0),
                                                         pm(5.5, 0.0, 0.0)));
}

// Relative-half boundary is inclusive. τ = 150 (membership never binds here);
// reference θ = 10 ⇒ threshold 0.9999·10 = 9.999. trial exactly at threshold ⇒
// pass; just above ⇒ fail.
TEST(FunnelAcceptance, ExitTestRelativeBoundaryInclusive) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 100.0, 200.0); // τ = 150.0
    const double threshold = kFunnelBeta * 10.0;
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(pm(10.0, 0.0, 0.0),
                                                        pm(threshold, 0.0, 0.0)));
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(pm(10.0, 0.0, 0.0),
                                                         pm(threshold * 1.0001, 0.0, 0.0)));
}

// ===========================================================================
// Restoration switch notifications with state isolation. The reference solver
// runs a SEPARATE optimality funnel instance across the feasibility phase, so
// its width is structurally frozen; the single-instance design STASHES the
// optimality width at entry (reinitializing a fresh feasibility-phase width) and
// RESTORES it at exit before applying the re-base κ·τ + (1−κ)·θ_exit,
// κ = kFunnelKappa = 0.5. Entry/exit are throw-guarded against mis-ordering.
// ===========================================================================

// Entry stashes the optimality width and reinitializes the working width to the
// uninitialized sentinel (the fresh feasibility-phase width is lazily re-derived
// from the feasibility-phase θ₀ on the next is_iterate_acceptable call).
TEST(FunnelAcceptance, NotifySwitchToFeasibilityStashesWidth) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    a.notify_switch_to_feasibility(pm(3.0, 1.0, 0.0));
    EXPECT_TRUE(a.in_feasibility_phase());
    EXPECT_TRUE(std::isinf(a.funnel_width()));       // working reset to sentinel
    EXPECT_DOUBLE_EQ(a.stashed_funnel_width(), 6.0); // optimality width frozen
}

// Exit restores the stashed optimality width and re-bases THAT restored width:
// τ_stashed = 6.0, θ_exit = 2.0 ⇒ τ⁺ = 0.5·6 + 0.5·2 = 4.0. The re-base applies
// to the restored 6.0, not the working +∞ sentinel left by entry (which would
// give +∞) — this pins re-base-on-restored.
TEST(FunnelAcceptance, NotifySwitchToOptimalityReBasesRestoredWidth) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);                         // τ = 6.0
    a.notify_switch_to_feasibility(pm(5.0, 0.0, 0.0)); // stash 6.0, working → +∞
    a.notify_switch_to_optimality(pm(2.0, 0.0, 0.0));
    EXPECT_FALSE(a.in_feasibility_phase());
    EXPECT_DOUBLE_EQ(a.funnel_width(), kFunnelKappa * 6.0 + (1.0 - kFunnelKappa) * 2.0); // 4.0
}

// Verbatim convex combination with no guard: an exit infeasibility ABOVE the
// restored width re-widens it (κ·6 + (1−κ)·10 = 8.0). Uno's update_restoration
// adds no floor; this pins that behavior on the restored width.
TEST(FunnelAcceptance, NotifySwitchToOptimalityAllowsReWidening) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0); // τ = 6.0
    a.notify_switch_to_feasibility(pm(5.0, 0.0, 0.0));
    a.notify_switch_to_optimality(pm(10.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(a.funnel_width(), kFunnelKappa * 6.0 + (1.0 - kFunnelKappa) * 10.0); // 8.0
    EXPECT_GT(a.funnel_width(), 6.0);
}

// While in the feasibility phase, the exit test's membership half tests against
// the STASHED width, not the live feasibility-phase width. Stash 6.0, drive the
// working width down to 1.0, then a trial θ=5.0 passes membership against 6.0
// but would fail against 1.0.
TEST(FunnelAcceptance, ExitTestReadsStashedWidthInPhase) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);                         // τ = 6.0
    a.notify_switch_to_feasibility(pm(4.0, 0.0, 0.0)); // stash 6.0, working → +∞
    ASSERT_TRUE(a.in_feasibility_phase());
    FunnelPrimeRejecting(a, /*theta0=*/0.1, /*trial_outside=*/2.0); // working width → 1.0
    ASSERT_DOUBLE_EQ(a.funnel_width(), 1.0);
    ASSERT_DOUBLE_EQ(a.stashed_funnel_width(), 6.0);

    // trial θ=5.0: membership against the stashed 6.0 passes (5 ≤ 6); against the
    // live working width 1.0 it would fail. Relative half: 5 ≤ 0.9999·100.
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(pm(100.0, 0.0, 0.0),
                                                        pm(5.0, 0.0, 0.0)));
}

// reset() mid-phase (μ-event) clears the WORKING width only; the stash + flag
// survive so the exit test still consults the frozen optimality width.
TEST(FunnelAcceptance, ResetMidPhasePreservesStashedWidth) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);                         // τ = 6.0
    a.notify_switch_to_feasibility(pm(4.0, 0.0, 0.0)); // stash 6.0
    FunnelPrimeRejecting(a, 0.1, 2.0);                          // working width → 1.0
    a.reset();                                                 // μ-event mid-restoration
    EXPECT_TRUE(a.in_feasibility_phase());            // flag survives
    EXPECT_TRUE(std::isinf(a.funnel_width()));         // working cleared to sentinel
    EXPECT_DOUBLE_EQ(a.stashed_funnel_width(), 6.0);  // stash survives
}

// reset() OUTSIDE the phase drops the stash defensively.
TEST(FunnelAcceptance, ResetOutsidePhaseDropsStashedWidth) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);
    a.notify_switch_to_feasibility(pm(4.0, 0.0, 0.0));
    a.notify_switch_to_optimality(pm(2.0, 0.0, 0.0)); // flag now false
    ASSERT_FALSE(a.in_feasibility_phase());
    a.reset();
    EXPECT_TRUE(std::isinf(a.stashed_funnel_width())); // dropped
}

// +∞ edge (FP-inert, reference behavior): restoration entered before the first
// is_iterate_acceptable call stashes the +∞ sentinel; the exit re-base is
// κ·∞ + (1−κ)·θ_exit = ∞, so the funnel simply stays wide (no NaN).
TEST(FunnelAcceptance, EntryBeforeFirstCallStashesInfinityExitStaysWide) {
    FunnelAcceptance a;
    ASSERT_TRUE(std::isinf(a.funnel_width())); // sentinel, no call yet
    a.notify_switch_to_feasibility(pm(4.0, 0.0, 0.0));
    EXPECT_TRUE(std::isinf(a.stashed_funnel_width()));
    a.notify_switch_to_optimality(pm(2.0, 0.0, 0.0));
    EXPECT_TRUE(std::isinf(a.funnel_width()));
}

// Throw guards on mis-ordered transitions (T6): a second entry without an
// intervening exit, and an exit without a preceding entry, both throw.
TEST(FunnelAcceptance, DoubleEntryThrows) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);
    a.notify_switch_to_feasibility(pm(4.0, 0.0, 0.0));
    EXPECT_THROW(a.notify_switch_to_feasibility(pm(4.0, 0.0, 0.0)), std::logic_error);
}

TEST(FunnelAcceptance, ExitWithoutEntryThrows) {
    FunnelAcceptance a;
    FunnelPrimeRejecting(a, 4.0, 10.0);
    EXPECT_THROW(a.notify_switch_to_optimality(pm(2.0, 0.0, 0.0)), std::logic_error);
}

} // namespace
