///////////////////////////////////////////////////////////////////////////////
// Unit tests for the modernized merit penalty rules (ModernMeritAcceptance).
//
// ModernMeritAcceptance is the opt-in modernized merit family
// (Settings::acceptance_strategy_ == merit), the first consumer of the GENERIC
// AcceptanceStrategy surface. Its judgment (is_iterate_acceptable) is pure
// arithmetic on a ProgressMeasures triple plus the per-solve penalty state, so
// it truth-tables in isolation — no solver, no NLP eval (the mechanism owns
// trial-point evaluation; that numeric path is exercised end-to-end by the
// solver corpus).
//
// Every scenario's accept/reject verdict and penalty trajectory is computed BY
// HAND in the comments from the paper equations and asserted exactly:
//   WMNO  — Waltz, Morales, Nocedal & Orban, Math. Program. 107 (2006), §3.1:
//           merit Eq. (3.1), ν_TRIAL Eq. (3.5, σ=0), update Eq. (3.6),
//           Armijo Eq. (3.9). ρ = 0.1, η = 1e-8, bump = +1.
//   Flex  — Curtis & Nocedal, IMA J. Numer. Anal. 28(4) (2008): interval merit
//           Eq. (2.1), χ Eq. (3.8, ω=0), π_u update Eq. (3.9), endpoint
//           acceptance (remark after Alg. 3.1), π_l update Eqs. (3.10)-(3.11).
//
// ProgressMeasures mapping (see modern_merit.h):
//   merit(pt, π)  = pt.objective + pt.auxiliary + π·pt.infeasibility  (ϕ_μ + π‖c‖)
//   pred_π        = pred.objective + π·pred.infeasibility             (m_f + π·m_θ)
//   accept (π)    ⇔ merit(current,π) − merit(trial,π) ≥ η·pred_π       (ared ≥ η·pred)
//   threshold τ/χ = −pred.objective / ((1−ρ)·pred.infeasibility)      (σ=0)
///////////////////////////////////////////////////////////////////////////////

#include "progress_measures_test_utils.h"

#include "tycho/detail/solvers/globalization/modern_merit.h"
#include "tycho/detail/solvers/psiopt_fwd.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

using tycho::solvers::kFlexInitPiL;
using tycho::solvers::kFlexInitPiU;
using tycho::solvers::kSufficientInfeasibilityDecreaseRatio;
using tycho::solvers::kWmnoInitPenalty;
using tycho::solvers::MeritPenaltyRules;
using tycho::solvers::ModernMeritAcceptance;
using tycho::solvers::ProgressMeasures;
using TychoTest::pm;

// ===========================================================================
// WMNO — Math. Program. 107 (2006), §3.1.
// ===========================================================================

// Descent step (∇ϕ_μᵀd < 0 ⇒ τ < 0): penalty NOT increased; merit decreases ⇒ accept.
//   pred = {m_θ=2, m_f=3}: τ = −3/(0.9·2) = −1.667 < ν₀=1 ⇒ ν unchanged.
//   merit(cur,1)=10+2=12; merit(tri,1)=5+1=6; ared=6; pred_π=3+1·2=5; η·pred=5e-8.
//   6 ≥ 5e-8 ⇒ ACCEPT.
TEST(WmnoRule, DescentAcceptsPenaltyUnchanged) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    const bool ok =
        a.is_iterate_acceptable(pm(2.0, 10.0, 0.0), pm(1.0, 5.0, 0.0),
                                pm(2.0, 3.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 1.0);
}

// Objective rises toward feasibility (τ > 0): penalty bumped, step still accepted.
//   pred = {m_θ=1, m_f=−9}: τ = 9/0.9 = 10; ν₀=1 < 10 ⇒ ν = 10+1 = 11.  (Eq. 3.6)
//   merit(cur,11)=10+11=21; merit(tri,11)=12+5.5=17.5; ared=3.5; pred_π=−9+11=2.
//   3.5 ≥ 2e-8 ⇒ ACCEPT.
TEST(WmnoRule, PenaltyBumpThenAccept) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    const bool ok = a.is_iterate_acceptable(pm(1.0, 10.0, 0.0),
                                            pm(0.5, 12.0, 0.0),
                                            pm(1.0, -9.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 11.0);
}

// Merit increases (infeasibility worse, objective flat), penalty unchanged ⇒ reject.
//   pred = {m_θ=2, m_f=3}: τ = −1.667 ⇒ ν stays 1.
//   merit(cur,1)=11; merit(tri,1)=13; ared=−2; pred_π=5; η·pred=5e-8.
//   −2 ≥ 5e-8 is false ⇒ REJECT.
TEST(WmnoRule, MeritIncreaseRejects) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    const bool ok =
        a.is_iterate_acceptable(pm(1.0, 10.0, 0.0),
                                pm(1.0, 12.0, 0.0),
                                pm(2.0, 3.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 1.0);
}

// Feasible current (m_θ = 0 ⇒ τ undefined): WMNO "c(z)=0 ⇒ ν⁺=ν" — no update.
//   pred_π = m_f + ν·0 = −9; ared = 10−8 = 2 ≥ η·(−9) ⇒ ACCEPT; ν stays 1.
TEST(WmnoRule, FeasibleCurrentNoPenaltyUpdate) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    const bool ok =
        a.is_iterate_acceptable(pm(0.0, 10.0, 0.0), pm(0.0, 8.0, 0.0),
                                pm(0.0, -9.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 1.0);
}

// Penalty trajectory across iterations is monotone non-decreasing (Eq. 3.6):
//   ν₀=1;  τ=10 ⇒ 11;  τ=5 (< 11) ⇒ 11 (no decrease);  τ=20 ⇒ 21.
TEST(WmnoRule, PenaltyTrajectoryMonotone) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    const ProgressMeasures cur = pm(1.0, 1.0, 0.0);
    const ProgressMeasures tri = pm(1.0, 1.0, 0.0);

    a.is_iterate_acceptable(cur, tri, pm(1.0, -9.0, 0.0), 1.0, 1.0); // τ=10
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 11.0);
    a.is_iterate_acceptable(cur, tri, pm(1.0, -4.5, 0.0), 1.0, 1.0); // τ=5
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 11.0);
    a.is_iterate_acceptable(cur, tri, pm(1.0, -18.0, 0.0), 1.0, 1.0); // τ=20
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 21.0);
}

// ===========================================================================
// Flexible interval — IMA J. Numer. Anal. 28(4) (2008).
// ===========================================================================
// Default interval: π_l₀ = 1e-8 (tiny), π_u₀ = 1e8 (large). With this envelope,
// the endpoint tests reduce to: accept at π_l ⇔ ϕ_μ decreases; accept at π_u ⇔
// infeasibility decreases (the 1e8 term dominates). This is exactly the
// filter-like acceptance region of Fig. 6.

// Region II (infeasibility ↓, objective ↑): rejected at π_l, accepted at π_u,
// so the step is accepted AND π_l is raised toward ν(step) (Eqs. 3.10-3.11).
//   pred = {m_θ=2, m_f=3}: χ = −1.667 < π_u ⇒ π_u unchanged.
//   accept_l: ared_l = 10.00000002−12.00000001 = −2.0 < η·pred_l ⇒ NO.
//   accept_u: ared_u ≈ 1e8 ≥ η·pred_u ≈ 2.0 ⇒ YES ⇒ accept.
//   π_l update: denom = 2−1 = 1 > 0; r = (12−10)/1 = 2;
//     bump = max(0.1·(2 − 1e-8), 1e-8) = 0.2; π_l = min(1e8, 1e-8 + 0.2) ≈ 0.2.
TEST(FlexRule, RegionIIAcceptsAndRaisesPiL) {
    ModernMeritAcceptance a(MeritPenaltyRules::flexible);
    const bool ok = a.is_iterate_acceptable(pm(2.0, 10.0, 0.0),
                                            pm(1.0, 12.0, 0.0),
                                            pm(2.0, 3.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_NEAR(a.flex_pi_l(), 0.2, 1e-6);
    EXPECT_DOUBLE_EQ(a.flex_pi_u(), kFlexInitPiU);
}

// Region III (objective ↓, infeasibility ↑): accepted at π_l ⇒ π_l KEPT (Eq. 3.10).
//   accept_l: ared_l = 10.00000001−5.00000002 = 5.0 ≥ η·pred_l ⇒ YES ⇒ keep π_l.
TEST(FlexRule, ObjectiveDecreaseAcceptsKeepsPiL) {
    ModernMeritAcceptance a(MeritPenaltyRules::flexible);
    const bool ok = a.is_iterate_acceptable(pm(1.0, 10.0, 0.0),
                                            pm(2.0, 5.0, 0.0),
                                            pm(1.0, 3.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_DOUBLE_EQ(a.flex_pi_l(), kFlexInitPiL);
    EXPECT_DOUBLE_EQ(a.flex_pi_u(), kFlexInitPiU);
}

// Region I (both objective ↑ and infeasibility ↑): rejected at both endpoints.
//   accept_l: ared_l = −2.0 < η·pred_l ⇒ NO;  accept_u: ared_u ≈ −1e8 ⇒ NO ⇒ REJECT.
TEST(FlexRule, BothWorseRejects) {
    ModernMeritAcceptance a(MeritPenaltyRules::flexible);
    const bool ok = a.is_iterate_acceptable(pm(1.0, 10.0, 0.0),
                                            pm(2.0, 12.0, 0.0),
                                            pm(1.0, 3.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(a.flex_pi_l(), kFlexInitPiL);
    EXPECT_DOUBLE_EQ(a.flex_pi_u(), kFlexInitPiU);
}

// π_u update branch (Eq. 3.9): when χ > π_u, raise π_u to χ + ε.
//   pred = {m_θ=1, m_f=−1.8e8}: χ = 1.8e8/(0.9·1) = 2e8 > π_u₀=1e8 ⇒ π_u = 2e8 + ε.
TEST(FlexRule, PiUpperRaisedWhenChiExceeds) {
    ModernMeritAcceptance a(MeritPenaltyRules::flexible);
    a.is_iterate_acceptable(pm(1.0, 10.0, 0.0), pm(0.5, 10.0, 0.0),
                            pm(1.0, -1.8e8, 0.0), 1.0, 1.0);
    EXPECT_NEAR(a.flex_pi_u(), 2.0e8, 10.0);
}

// ===========================================================================
// reset() clears per-solve penalty state (μ-event / phase-change hook).
// ===========================================================================
TEST(ModernMerit, ResetClearsPenaltyState) {
    // WMNO: bump ν to 11, then reset back to ν₀.
    ModernMeritAcceptance w(MeritPenaltyRules::wmno);
    w.is_iterate_acceptable(pm(1.0, 10.0, 0.0), pm(0.5, 12.0, 0.0),
                            pm(1.0, -9.0, 0.0), 1.0, 1.0);
    ASSERT_DOUBLE_EQ(w.wmno_penalty(), 11.0);
    w.reset();
    EXPECT_DOUBLE_EQ(w.wmno_penalty(), kWmnoInitPenalty);

    // Flexible: raise π_l to ≈0.2, then reset the whole interval.
    ModernMeritAcceptance f(MeritPenaltyRules::flexible);
    f.is_iterate_acceptable(pm(2.0, 10.0, 0.0), pm(1.0, 12.0, 0.0),
                            pm(2.0, 3.0, 0.0), 1.0, 1.0);
    ASSERT_NEAR(f.flex_pi_l(), 0.2, 1e-6);
    f.reset();
    EXPECT_DOUBLE_EQ(f.flex_pi_l(), kFlexInitPiL);
    EXPECT_DOUBLE_EQ(f.flex_pi_u(), kFlexInitPiU);
}

// ===========================================================================
// Restoration-exit test (Uno MeritFunction): θ_trial ≤ ratio · smallest-known
// infeasibility, ratio = 0.9. The tracker is +∞-initialized, updated by min()
// ONLY in the accept branch of is_iterate_acceptable, and cleared by reset().
// The `reference` argument is ignored (Uno reads only the trial + tracker).
// ===========================================================================

// Two accepts drive the tracker down; a reject does NOT update it. Each accept
// below is a plain descent step (merit decreases, penalty unchanged).
//   accept 1: cur(θ=8,f=10), tri(θ=5,f=4), pred(m_θ=3,m_f=6): τ<1 ⇒ ν=1;
//     ared=18−9=9 ≥ η·pred ⇒ ACCEPT ⇒ smallest = min(∞,5) = 5.
//   accept 2: cur(θ=6,f=10), tri(θ=3,f=4), pred(m_θ=3,m_f=6): ACCEPT ⇒
//     smallest = min(5,3) = 3.
//   reject:   cur(θ=10,f=10), tri(θ=1,f=100), pred(m_θ=2,m_f=3): ared=20−101=−81
//     < η·pred ⇒ REJECT ⇒ tracker UNCHANGED at 3 (the min() runs only on accept).
TEST(ModernMeritRestoration, SmallestKnownTracksAcceptsNotRejects) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    EXPECT_TRUE(std::isinf(a.smallest_known_infeasibility()));

    ASSERT_TRUE(a.is_iterate_acceptable(pm(8.0, 10.0, 0.0),
                                        pm(5.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0));
    EXPECT_DOUBLE_EQ(a.smallest_known_infeasibility(), 5.0);

    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0));
    EXPECT_DOUBLE_EQ(a.smallest_known_infeasibility(), 3.0);

    ASSERT_FALSE(a.is_iterate_acceptable(pm(10.0, 10.0, 0.0),
                                         pm(1.0, 100.0, 0.0),
                                         pm(2.0, 3.0, 0.0), 1.0, 1.0));
    EXPECT_DOUBLE_EQ(a.smallest_known_infeasibility(), 3.0); // reject did NOT lower it
}

// Exit boundary at 0.9·smallest. Prime the tracker to 3.0 (one descent accept
// with θ_trial = 3.0), so threshold = 0.9·3 = 2.7.
TEST(ModernMeritRestoration, ExitBoundaryAtRatioTimesSmallestKnown) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0));
    ASSERT_DOUBLE_EQ(a.smallest_known_infeasibility(), 3.0);
    const double threshold = kSufficientInfeasibilityDecreaseRatio * 3.0; // 2.7
    const ProgressMeasures ref = pm(100.0, 0.0, 0.0);      // ignored

    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(threshold, 0.0, 0.0)));
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(2.6, 0.0, 0.0)));
    EXPECT_FALSE(
        a.is_infeasibility_sufficiently_reduced(ref, pm(threshold * 1.0001, 0.0, 0.0)));
}

// The `reference` argument is ignored: two very different references give the
// same verdict for the same trial (only the trial + tracker matter).
TEST(ModernMeritRestoration, ReferenceArgumentIgnored) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0)); // smallest=3
    const ProgressMeasures trial = pm(2.6, 0.0, 0.0);                  // ≤ 2.7 ⇒ pass
    const bool with_tiny_ref =
        a.is_infeasibility_sufficiently_reduced(pm(1.0e-12, 0.0, 0.0), trial);
    const bool with_huge_ref =
        a.is_infeasibility_sufficiently_reduced(pm(1.0e12, 0.0, 0.0), trial);
    EXPECT_TRUE(with_tiny_ref);
    EXPECT_EQ(with_tiny_ref, with_huge_ref);
}

// reset() re-bases the tracker to +∞ (a μ-event mid-restoration): the exit test
// then trivially passes any finite trial until the next accept re-seeds it.
TEST(ModernMeritRestoration, ResetReBasesTrackerToInfinity) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0));
    ASSERT_DOUBLE_EQ(a.smallest_known_infeasibility(), 3.0);

    a.reset();
    EXPECT_TRUE(std::isinf(a.smallest_known_infeasibility()));
    // 0.9·∞ = ∞, so any finite trial passes.
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(pm(1.0, 0.0, 0.0),
                                                        pm(1.0e6, 0.0, 0.0)));
}

// The modern strategy drives the GENERIC compute_step path, not the fused
// classic one.
TEST(ModernMerit, DrivesGenericPath) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    EXPECT_FALSE(a.drives_classic_path());
}

// ===========================================================================
// Feasibility-restoration state isolation (reproducing the reference solver's
// two-instance freeze inside one object). Entry stashes ALL persistent state
// (penalties + smallest-known tracker) and reinitializes fresh; feasibility-
// phase accepts evolve only the working copy; the exit test reduces against the
// STASHED (frozen) tracker; exit restores the stash. reset() is phase-aware.
// ===========================================================================

// Drive one penalty-bump accept: cur(θ=1,f=10), tri(θ=0.5,f=12), pred(m_θ=1,
// m_f=−9) ⇒ τ=9/0.9=10, ν=11; ared=21−17.5=3.5 ≥ η·pred ⇒ ACCEPT ⇒ smallest =
// min(∞,0.5) = 0.5. Used to give the optimality phase non-trivial state.
void ModernMeritBumpAccept(ModernMeritAcceptance &a) {
    ASSERT_TRUE(a.is_iterate_acceptable(pm(1.0, 10.0, 0.0),
                                        pm(0.5, 12.0, 0.0),
                                        pm(1.0, -9.0, 0.0), 1.0, 1.0));
}

// Entry stashes the optimality-phase state and reinitializes fresh working
// state; feasibility-phase accepts touch only the working copy; exit restores.
TEST(ModernMeritRestoration, StashFreezeRestore) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ModernMeritBumpAccept(a); // ν=11, smallest=0.5
    ASSERT_DOUBLE_EQ(a.wmno_penalty(), 11.0);
    ASSERT_DOUBLE_EQ(a.smallest_known_infeasibility(), 0.5);

    // Entry: stash (11, 0.5), working reinitialized to fresh construction.
    a.notify_switch_to_feasibility(pm(9.0, 1.0, 0.0));
    EXPECT_TRUE(a.in_feasibility_phase());
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), kWmnoInitPenalty);              // working reset
    EXPECT_TRUE(std::isinf(a.smallest_known_infeasibility()));         // working reset
    EXPECT_DOUBLE_EQ(a.stashed_wmno_penalty(), 11.0);                  // frozen
    EXPECT_DOUBLE_EQ(a.stashed_smallest_known_infeasibility(), 0.5);   // frozen

    // A feasibility-phase accept: cur(θ=8,f=10), tri(θ=2,f=4), pred(m_θ=3,m_f=6)
    // ⇒ τ<1 ⇒ ν stays 1; ACCEPT ⇒ working smallest = 2. The STASH is untouched.
    ASSERT_TRUE(a.is_iterate_acceptable(pm(8.0, 10.0, 0.0),
                                        pm(2.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0));
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 1.0);                          // working evolved
    EXPECT_DOUBLE_EQ(a.smallest_known_infeasibility(), 2.0);         // working evolved
    EXPECT_DOUBLE_EQ(a.stashed_wmno_penalty(), 11.0);               // still frozen
    EXPECT_DOUBLE_EQ(a.stashed_smallest_known_infeasibility(), 0.5); // still frozen

    // Exit: restore the frozen optimality-phase state.
    a.notify_switch_to_optimality(pm(1.0, 5.0, 0.0));
    EXPECT_FALSE(a.in_feasibility_phase());
    EXPECT_DOUBLE_EQ(a.wmno_penalty(), 11.0);
    EXPECT_DOUBLE_EQ(a.smallest_known_infeasibility(), 0.5);
}

// While in the feasibility phase, the exit test reduces against the STASHED
// tracker, not the live one. Prime optimality to smallest=3; enter; drive the
// working tracker down to 1; the exit threshold must be 0.9·3 = 2.7 (stashed),
// not 0.9·1 = 0.9 (live). A trial θ=2.6 passes against 2.7 but would fail 0.9.
TEST(ModernMeritRestoration, ExitTestReadsStashedTrackerInPhase) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0)); // smallest=3
    a.notify_switch_to_feasibility(pm(6.0, 10.0, 0.0));               // stash 3
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(1.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0)); // working=1
    ASSERT_DOUBLE_EQ(a.smallest_known_infeasibility(), 1.0);
    ASSERT_DOUBLE_EQ(a.stashed_smallest_known_infeasibility(), 3.0);

    const ProgressMeasures ref = pm(100.0, 0.0, 0.0); // ignored
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(2.6, 0.0, 0.0)));
    // If it read the live tracker (1.0 ⇒ threshold 0.9), 2.6 would fail; it passes
    // ⇒ it read the stashed 3.0. A trial above 2.7 fails against the stash.
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(ref, pm(2.8, 0.0, 0.0)));
}

// reset() mid-phase (μ-event) clears WORKING state only; the stash + flag
// survive so the exit test still consults the frozen tracker.
TEST(ModernMeritRestoration, ResetMidPhasePreservesStash) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0)); // smallest=3
    a.notify_switch_to_feasibility(pm(6.0, 10.0, 0.0));               // stash 3

    a.reset(); // μ-event mid-restoration
    EXPECT_TRUE(a.in_feasibility_phase());                            // flag survives
    EXPECT_TRUE(std::isinf(a.smallest_known_infeasibility()));        // working cleared
    EXPECT_DOUBLE_EQ(a.stashed_smallest_known_infeasibility(), 3.0);  // stash survives

    const ProgressMeasures ref = pm(100.0, 0.0, 0.0);
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(2.6, 0.0, 0.0)));
}

// reset() OUTSIDE the phase drops the stash defensively (full per-phase clear).
TEST(ModernMeritRestoration, ResetOutsidePhaseDropsStash) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    ASSERT_TRUE(a.is_iterate_acceptable(pm(6.0, 10.0, 0.0),
                                        pm(3.0, 4.0, 0.0),
                                        pm(3.0, 6.0, 0.0), 1.0, 1.0));
    a.notify_switch_to_feasibility(pm(6.0, 10.0, 0.0));
    a.notify_switch_to_optimality(pm(1.0, 5.0, 0.0)); // flag now false
    ASSERT_FALSE(a.in_feasibility_phase());

    a.reset();
    EXPECT_TRUE(std::isinf(a.stashed_smallest_known_infeasibility())); // stash dropped
    EXPECT_DOUBLE_EQ(a.stashed_wmno_penalty(), kWmnoInitPenalty);
}

// The +∞ edge (reference behavior, retained): restoration entered before any
// optimality-phase accept leaves the stash at +∞, so the exit test's
// 0.9·(+∞) = +∞ passes any finite trial at the first check.
TEST(ModernMeritRestoration, EntryBeforeAnyAcceptStashesInfinity) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    a.notify_switch_to_feasibility(pm(9.0, 1.0, 0.0));
    EXPECT_TRUE(std::isinf(a.stashed_smallest_known_infeasibility()));
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(pm(1.0, 0.0, 0.0),
                                                        pm(1.0e6, 0.0, 0.0)));
}

// Throw guards on mis-ordered transitions (T6): a second entry without an
// intervening exit, and an exit without a preceding entry, both throw.
TEST(ModernMeritRestoration, DoubleEntryThrows) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    a.notify_switch_to_feasibility(pm(9.0, 1.0, 0.0));
    EXPECT_THROW(a.notify_switch_to_feasibility(pm(9.0, 1.0, 0.0)),
                 std::logic_error);
}

TEST(ModernMeritRestoration, ExitWithoutEntryThrows) {
    ModernMeritAcceptance a(MeritPenaltyRules::wmno);
    EXPECT_THROW(a.notify_switch_to_optimality(pm(1.0, 5.0, 0.0)), std::logic_error);
}

} // namespace
