///////////////////////////////////////////////////////////////////////////////
// Unit tests for FilterAcceptance — the Wächter–Biegler (θ, φ)-pair filter
// H-type acceptance strategy layered on the switching skeleton
// (SwitchingAcceptance).
//
// FilterAcceptance is concrete: it is exercised through the base's public
// template method is_iterate_acceptable(), observing the accept/reject verdict
// and the public accessors filter_size() / filter_entry() /
// successive_filter_rejections() / filter_resets(). To drive the filter's own
// (H-type) rules deterministically, every "H-type" call below passes
// predicted_reduction.objective = 0: the base's switching condition requires a
// strict descent direction (m_f > 0), so with m_f = 0 the switching test never
// fires and the trial is always routed to the H-type delegate — regardless of
// θ_min. F-type is driven separately with a descent-satisfying predicted
// reduction (see FilterFTypeAcceptDoesNotAugment).
//
// Every scenario's arithmetic is computed BY HAND (shown in the comments) from
// the rules documented in filter_acceptance.h:
//   (1a) acceptable-to-current: barrier ceiling
//          log10(φ_t − φ_c) > 5 + basval  (basval = 1 if |φ_c| ≤ 10, else
//          log10|φ_c|; only when φ_t > φ_c) ⇒ reject; then
//        θ_t ≤ (1−γ_θ)·θ_c  OR  φ_t ≤ φ_c − γ_φ·θ_c   (γ_θ = 1e-5, γ_φ = 1e-8)
//   (1b) acceptable-to-filter: an entry (φ_j, θ_j) blocks iff φ_t > φ_j AND
//        θ_t > θ_j (per-coordinate ≤ dominance)
//   (2)  augment (accepted H-type): store (φ_c − γ_φ·θ_c, (1−γ_θ)·θ_c), pruning
//        entries dominated by the new pair
//   (4)  reset heuristic (per-iteration): the counter advances once per ACCEPT
//        whose line search's last rejection was filter-caused; 5 such successive
//        iterations clear the filter, ≤ 5 times; an accept whose last rejection
//        was non-filter zeroes the run
//
// UNITY RULE: anonymous namespace does not protect names against the unity
// build — every helper here is prefixed Filter* to stay globally unique across
// tests/cpp/ (grep-confirmed no other "Filter"-prefixed free helper exists).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/filter_acceptance.h"

#include "tycho/detail/solvers/globalization/progress_measures.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace {

using tycho::solvers::FilterAcceptance;
using tycho::solvers::kFilterGammaPhi;
using tycho::solvers::kFilterGammaTheta;
using tycho::solvers::kFilterMaxResets;
using tycho::solvers::kFilterResetTrigger;
using tycho::solvers::ProgressMeasures;

// File-unique helper (see UNITY RULE above).
ProgressMeasures FilterMakePm(double infeasibility, double objective, double auxiliary) {
    ProgressMeasures pm;
    pm.infeasibility = infeasibility;
    pm.objective = objective;
    pm.auxiliary = auxiliary;
    return pm;
}

// Priming: run the FIRST is_iterate_acceptable() so the base captures θ₀ and
// calls initialize_bounds(), WITHOUT the priming call touching the filter or
// the counters. The priming trial exceeds θ_max = 1e4·max(1, θ₀), so the base
// rejects it at the ceiling BEFORE the H-type delegate ever runs — the filter
// stays empty and the counters stay zero. Asserts the clean post-prime state.
void FilterPrimeCeilingReject(FilterAcceptance &a, double theta0) {
    const double above_theta_max = 1.0e4 * std::max(1.0, theta0) * 2.0;
    const bool ok = a.is_iterate_acceptable(FilterMakePm(theta0, 0.0, 0.0),
                                            FilterMakePm(above_theta_max, 0.0, 0.0),
                                            FilterMakePm(0.0, 0.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok); // above θ_max ⇒ base rejects at the ceiling
    EXPECT_EQ(a.filter_size(), 0u);
    EXPECT_EQ(a.successive_filter_rejections(), 0);
    EXPECT_EQ(a.filter_resets(), 0);
}

// Convenience: an H-type call (m_f = 0) with explicit current/trial (θ, φ). φ is
// carried entirely in `objective` (auxiliary = 0), so φ = objective.
bool FilterHType(FilterAcceptance &a, double theta_current, double phi_current, double theta_trial,
                 double phi_trial) {
    return a.is_iterate_acceptable(FilterMakePm(theta_current, phi_current, 0.0),
                                   FilterMakePm(theta_trial, phi_trial, 0.0),
                                   FilterMakePm(0.0, 0.0, 0.0), 1.0, 1.0);
}

// --- Reset-heuristic helpers (all assume the seed entry E = augment(θ_c=4.0,
// φ_c=20.0) ⇒ E = (φ_E ≈ 20, θ_E = (1−1e-5)·4 = 3.99996) is present, and that
// θ₀ was primed to 4.0 ⇒ θ_min = 1e-4·4 = 4e-4, θ_max = 4e4). The reset
// heuristic runs ONCE PER ACCEPT (reading the last rejection's cause), so these
// helpers separate the two events the truth table depends on.

// A MEMBERSHIP rejection: trial (θ_t = 50, φ_t = 30) is dominated by E
// (50 > 3.99996 AND 30 > ≈20) ⇒ rejected at membership (kMembership) ⇒ the last
// rejection is filter-caused. The counter is NOT touched (only accepts touch it).
void FilterMembershipReject(FilterAcceptance &a) {
    EXPECT_FALSE(FilterHType(a, 100.0, 100.0, 50.0, 30.0));
}

// An F-type ACCEPT (θ_c = 1e-5 ≤ θ_min, m_f = 0.01 ⇒ switching holds; Armijo
// passes; membership passes since (φ_t = 9.9999, θ_t = 1e-6) is not dominated by
// E). Runs the reset heuristic (advancing/zeroing the counter per the last
// rejection) but does NOT augment — the filter is left unchanged.
void FilterFTypeAccept(FilterAcceptance &a) {
    EXPECT_TRUE(a.is_iterate_acceptable(FilterMakePm(1.0e-5, 10.0, 0.0),
                                        FilterMakePm(1.0e-6, 9.9999, 0.0),
                                        FilterMakePm(0.01, 0.01, 0.0), 1.0, 1.0));
}

// An F-type ARMIJO rejection (switching holds; membership passes; φ_t = 20 fails
// Armijo) ⇒ non-filter rejection (kArmijo) ⇒ last rejection is NOT filter-caused.
void FilterFTypeArmijoReject(FilterAcceptance &a) {
    EXPECT_FALSE(a.is_iterate_acceptable(FilterMakePm(1.0e-5, 10.0, 0.0),
                                         FilterMakePm(1.0e-6, 20.0, 0.0),
                                         FilterMakePm(0.01, 0.01, 0.0), 1.0, 1.0));
}

// ===========================================================================
// Empty filter before the first accepted H-type step.
// ===========================================================================

TEST(FilterAcceptance, FilterEmptyBeforeFirstAccept) {
    FilterAcceptance a;
    EXPECT_EQ(a.filter_size(), 0u);
    FilterPrimeCeilingReject(a, /*theta0=*/4.0); // θ_max = 4e4; still empty
    EXPECT_EQ(a.filter_size(), 0u);
}

// ===========================================================================
// (1a) acceptable-to-current truth table (empty filter ⇒ (1b) always passes,
// so the verdict is exactly the (1a) result).
// ===========================================================================

// First margin (θ): θ_t ≤ (1−γ_θ)·θ_c. θ_c = 1.0 ⇒ (1−1e-5)·1 = 0.99999.
//   • θ_t = 0.5 ≤ 0.99999, φ increases but only mildly ⇒ ACCEPT via θ margin.
// φ_t = φ_c ⇒ no barrier-ceiling trip (φ_t > φ_c is false).
TEST(FilterAcceptance, AcceptableToCurrentThetaMarginAccepts) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    EXPECT_TRUE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/10.0, /*θ_t=*/0.5, /*φ_t=*/10.0));
}

// Second margin (φ): φ_t ≤ φ_c − γ_φ·θ_c, with θ_t FAILING the first margin.
// θ_c = 1.0 ⇒ threshold θ = 0.99999; take θ_t = 2.0 (> 0.99999 ⇒ first fails).
// φ_c = 10.0, γ_φ·θ_c = 1e-8·1 = 1e-8 ⇒ φ threshold = 10 − 1e-8 = 9.99999999.
//   • φ_t = 9.0 ≤ 9.99999999 ⇒ ACCEPT via φ margin (φ_t < φ_c ⇒ no ceiling).
TEST(FilterAcceptance, AcceptableToCurrentPhiMarginAccepts) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    EXPECT_TRUE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/10.0, /*θ_t=*/2.0, /*φ_t=*/9.0));
}

// Both margins FAIL ⇒ REJECT. θ_c = 1.0 ⇒ θ threshold = 0.99999,
// φ threshold = 9.99999999. Trial θ_t = 2.0 (> 0.99999) AND φ_t = 10.0
// (φ_t − φ_c = 0 > −1e-8 ⇒ φ margin fails). φ_t = φ_c ⇒ no ceiling trip.
TEST(FilterAcceptance, AcceptableToCurrentBothMarginsFailRejects) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    EXPECT_FALSE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/10.0, /*θ_t=*/2.0, /*φ_t=*/10.0));
}

// θ-margin boundary is inclusive (≤). θ_c = 1.0 ⇒ threshold = (1−1e-5)·1 =
// 0.99999. Trial θ_t EXACTLY at threshold ⇒ ACCEPT (φ_t = φ_c ⇒ no ceiling).
TEST(FilterAcceptance, AcceptableToCurrentThetaBoundaryInclusive) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    const double threshold = (1.0 - kFilterGammaTheta) * 1.0; // 0.99999
    EXPECT_TRUE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/10.0, /*θ_t=*/threshold, /*φ_t=*/10.0));
}

// ===========================================================================
// (1a) barrier-objective ceiling (obj_max_inc) truth table.
// ===========================================================================

// Ceiling trips (|φ_c| ≤ 10 ⇒ basval = 1, threshold = 5 + 1 = 6). φ_c = 0,
// φ_t = 1e7 ⇒ log10(1e7 − 0) = 7 > 6 ⇒ REJECT — even though θ_t = 0 would
// otherwise satisfy the θ margin (proving the ceiling is checked FIRST).
TEST(FilterAcceptance, BarrierCeilingRejectsLargeIncrease) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    EXPECT_FALSE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/0.0, /*θ_t=*/0.0, /*φ_t=*/1.0e7));
}

// Just under the ceiling: φ_t = 1e6 ⇒ log10(1e6) = 6, NOT > 6 ⇒ ceiling passes;
// then θ_t = 0 ≤ (1−γ_θ)·1 = 0.99999 ⇒ ACCEPT. Contrast with the case above
// (same θ, one order of magnitude less φ growth).
TEST(FilterAcceptance, BarrierCeilingPassesModerateIncrease) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    EXPECT_TRUE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/0.0, /*θ_t=*/0.0, /*φ_t=*/1.0e6));
}

// basval scaling: |φ_c| = 1000 > 10 ⇒ basval = log10(1000) = 3, threshold =
// 5 + 3 = 8. φ_c = −1000, φ_t = 1e9 ⇒ φ_t − φ_c = 1e9 + 1000 ≈ 1.000000001e9,
// log10 ≈ 9.0 > 8 ⇒ REJECT (θ_t = 0 would otherwise pass the θ margin).
TEST(FilterAcceptance, BarrierCeilingBasvalScalesWithReference) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    EXPECT_FALSE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/-1000.0, /*θ_t=*/0.0, /*φ_t=*/1.0e9));
}

// ===========================================================================
// (2) augmentation stores the exact margined pair; (1b) dominance truth table.
// ===========================================================================

// One accepted H-type step augments with the CURRENT iterate's margined pair.
// current (θ_c = 4.0, φ_c = 20.0), trial (θ_t = 1.0, φ_t = 20.0):
//   (1a): θ_t = 1.0 ≤ (1−1e-5)·4 = 3.99996 ⇒ acceptable-to-current;
//   (1b): empty filter ⇒ acceptable ⇒ ACCEPT ⇒ augment.
// Stored pair (exact):
//   φ_add = φ_c − γ_φ·θ_c = 20 − 1e-8·4 = 20 − 4e-8
//   θ_add = (1−γ_θ)·θ_c   = (1−1e-5)·4 = 3.99996
TEST(FilterAcceptance, AugmentStoresExactMarginedPair) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/4.0, /*φ_c=*/20.0, /*θ_t=*/1.0, /*φ_t=*/20.0));
    ASSERT_EQ(a.filter_size(), 1u);
    const auto [phi_add, theta_add] = a.filter_entry(0);
    EXPECT_DOUBLE_EQ(phi_add, 20.0 - kFilterGammaPhi * 4.0);
    EXPECT_DOUBLE_EQ(theta_add, (1.0 - kFilterGammaTheta) * 4.0);
}

// Dominance verdict against a stored entry. After the augment above the filter
// holds one entry E = (φ_E = 20 − 4e-8, θ_E = 3.99996).
//   • A trial with φ_t > φ_E AND θ_t > θ_E is blocked by E ⇒ (1b) fails.
//     Use current (θ_c = 100, φ_c = 100) so (1a) passes via the θ margin
//     (θ_t = 50 ≤ (1−1e-5)·100 = 99.999) — isolating the (1b) rejection.
//     Trial (θ_t = 50 > 3.99996, φ_t = 30 > 20−4e-8) ⇒ dominated ⇒ REJECT.
TEST(FilterAcceptance, FilterEntryBlocksDominatedTrial) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E
    ASSERT_EQ(a.filter_size(), 1u);
    // φ_t = 30 > 20−4e-8 AND θ_t = 50 > 3.99996 ⇒ blocked by E.
    EXPECT_FALSE(FilterHType(a, /*θ_c=*/100.0, /*φ_c=*/100.0, /*θ_t=*/50.0, /*φ_t=*/30.0));
}

// Not dominated: a trial with θ_t ≤ θ_E (one coordinate ≤) is acceptable w.r.t.
// E even if φ_t > φ_E. E = (≈20, 3.99996). Trial θ_t = 2.0 ≤ 3.99996 ⇒ passes
// (1b). (1a): current (θ_c = 100, φ_c = 100), θ_t = 2.0 ≤ 99.999 ⇒ passes.
// ⇒ ACCEPT (and augments again — filter grows).
TEST(FilterAcceptance, FilterEntryAllowsTrialBelowInOneCoordinate) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E
    EXPECT_TRUE(FilterHType(a, /*θ_c=*/100.0, /*φ_c=*/100.0, /*θ_t=*/2.0, /*φ_t=*/30.0));
    EXPECT_EQ(a.filter_size(), 2u); // second accepted step augmented again
}

// Per-coordinate ≤ boundary: a trial EQUAL to E in the θ coordinate is
// acceptable (≤, not <). E = (≈20, θ_E = 3.99996). Trial θ_t = θ_E exactly,
// φ_t > φ_E ⇒ acceptable w.r.t. E (θ_t ≤ θ_E holds with equality).
// (1a): current (θ_c = 100, φ_c = 100), θ_t = 3.99996 ≤ 99.999 ⇒ passes.
TEST(FilterAcceptance, FilterDominanceBoundaryInclusive) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E
    const double theta_E = (1.0 - kFilterGammaTheta) * 4.0;
    EXPECT_TRUE(FilterHType(a, /*θ_c=*/100.0, /*φ_c=*/100.0, /*θ_t=*/theta_E, /*φ_t=*/30.0));
}

// ===========================================================================
// (2) augmentation prunes entries the new pair dominates.
// ===========================================================================

// Two augments where the second pair dominates the first ⇒ filter stays size 1
// and holds only the second. Augment order:
//   step 1: current (θ_c = 10, φ_c = 50) ⇒ E1 = (50 − 1e-8·10, (1−1e-5)·10)
//                                             = (50 − 1e-7, 9.9999)
//   step 2: current (θ_c = 5, φ_c = 20)  ⇒ E2 = (20 − 1e-8·5, (1−1e-5)·5)
//                                             = (20 − 5e-8, 4.99995)
// E2 dominates E1: φ_add(E2) = 20−5e-8 ≤ φ(E1) = 50−1e-7 AND
//   θ_add(E2) = 4.99995 ≤ θ(E1) = 9.9999 ⇒ E1 pruned. Filter = {E2}.
// Both steps must ACCEPT: use fresh current iterates that pass (1a) and are not
// blocked by the current filter contents.
//   step 1 trial (θ_t = 1, φ_t = 50): θ_t ≤ (1−1e-5)·10 = 9.9999 ⇒ (1a) ok;
//     empty filter ⇒ (1b) ok.
//   step 2 trial (θ_t = 1, φ_t = 20): θ_t ≤ (1−1e-5)·5 = 4.99995 ⇒ (1a) ok;
//     E1 = (≈50, 9.9999): θ_t = 1 ≤ 9.9999 ⇒ acceptable w.r.t. E1 ⇒ (1b) ok.
TEST(FilterAcceptance, AugmentPrunesDominatedEntries) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 10.0);
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/10.0, /*φ_c=*/50.0, /*θ_t=*/1.0, /*φ_t=*/50.0));
    ASSERT_EQ(a.filter_size(), 1u);
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/5.0, /*φ_c=*/20.0, /*θ_t=*/1.0, /*φ_t=*/20.0));
    ASSERT_EQ(a.filter_size(), 1u); // E1 pruned by E2
    const auto [phi_add, theta_add] = a.filter_entry(0);
    EXPECT_DOUBLE_EQ(phi_add, 20.0 - kFilterGammaPhi * 5.0);
    EXPECT_DOUBLE_EQ(theta_add, (1.0 - kFilterGammaTheta) * 5.0);
}

// Non-dominating second augment keeps both entries. E1 from (θ_c=5, φ_c=10);
// E2 from (θ_c=10, φ_c=5): E2 has SMALLER φ but LARGER θ than E1 ⇒ neither
// dominates the other ⇒ filter size grows to 2.
//   E1 = (10 − 5e-8, 4.99995), E2 = (5 − 1e-7, 9.9999).
//   E2 dominates E1? φ(E2)=5−1e-7 ≤ φ(E1)=10−5e-8 (yes) BUT
//     θ(E2)=9.9999 ≤ θ(E1)=4.99995 (no) ⇒ not dominated ⇒ E1 kept.
// step 2 trial (θ_t = 1, φ_t = 5): θ_t ≤ (1−1e-5)·10 = 9.9999 ⇒ (1a) ok;
//   E1: θ_t = 1 ≤ 4.99995 ⇒ acceptable w.r.t. E1 ⇒ (1b) ok.
TEST(FilterAcceptance, AugmentKeepsNonDominatedEntries) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 10.0);
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/5.0, /*φ_c=*/10.0, /*θ_t=*/1.0, /*φ_t=*/10.0));
    ASSERT_EQ(a.filter_size(), 1u);
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/10.0, /*φ_c=*/5.0, /*θ_t=*/1.0, /*φ_t=*/5.0));
    EXPECT_EQ(a.filter_size(), 2u); // neither dominates the other
}

// ===========================================================================
// (3) F-type accepted step does NOT augment.
// ===========================================================================

// θ₀ = 1.0 ⇒ θ_min = 1e-4·max(1,1) = 1e-4, θ_max = 1e4. First seed the filter
// with one entry via an accepted H-type step (filter_size 1). Then drive an
// F-type accept:
//   current θ = 1e-5 ≤ θ_min, m_f = 0.01 > 0 ⇒ switching tested:
//     lhs = 1·(0.01/1)^2.3 = 0.01^2.3 ≈ 2.511886e-05
//     rhs = 1·(1e-5)^1.1        ≈ 3.162278e-06 ⇒ lhs > rhs ⇒ switching HOLDS.
//   Armijo on φ = obj + aux: φ_trial = 9.9999 ≤ φ_current(10) − 1e-8·0.01
//     = 9.9999999999 ⇒ ACCEPT as F-type ⇒ register_accepted_step is called with
//     h_type = false (no augment) ⇒ filter_size stays 1.
TEST(FilterAcceptance, FTypeAcceptDoesNotAugment) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 1.0);
    // Seed one filter entry (H-type accept): θ_t = 0.5 ≤ (1−1e-5)·1 = 0.99999.
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/1.0, /*φ_c=*/10.0, /*θ_t=*/0.5, /*φ_t=*/10.0));
    ASSERT_EQ(a.filter_size(), 1u);

    const bool ok =
        a.is_iterate_acceptable(FilterMakePm(1.0e-5, 10.0, 0.0), FilterMakePm(1.0e-6, 9.9999, 0.0),
                                FilterMakePm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);                  // F-type accept
    EXPECT_EQ(a.filter_size(), 1u);   // untouched by the F-type step
}

// ===========================================================================
// θ_max ceiling rejects before the filter (or the counters) is consulted.
// ===========================================================================

// θ₀ = 4.0 ⇒ θ_max = 1e4·4 = 4e4. Seed a filter entry, then present a trial
// above θ_max that WOULD be acceptable to the filter (θ_t small in the entry
// coordinate). The base rejects at the ceiling ⇒ verdict false, filter_size
// unchanged, and the successive-rejection counter untouched (the H-type
// delegate never runs).
TEST(FilterAcceptance, ThetaMaxCeilingRejectsBeforeFilter) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed one entry
    ASSERT_EQ(a.filter_size(), 1u);
    ASSERT_EQ(a.successive_filter_rejections(), 0);

    const double over_max = 5.0e4; // > θ_max = 4e4
    EXPECT_FALSE(FilterHType(a, /*θ_c=*/100.0, /*φ_c=*/100.0, /*θ_t=*/over_max, /*φ_t=*/1.0));
    EXPECT_EQ(a.filter_size(), 1u);                 // ceiling reject ⇒ no augment
    EXPECT_EQ(a.successive_filter_rejections(), 0); // delegate never consulted
}

// ===========================================================================
// (4) reset heuristic — PER-ITERATION counter sequences (Ipopt-faithful). The
// heuristic runs once per ACCEPT, reading the cause of the last rejection in the
// line search that produced the accept: a filter-caused last rejection advances
// the successive-iteration counter, any other zeroes it.
// ===========================================================================

// Multiple filter-caused rejections within ONE line search (before the accept
// that ends it) count as ONE increment, not N — the counter is untouched by
// rejections and advances only on the accept.
TEST(FilterAcceptance, MultipleFilterRejectionsBeforeAcceptCountOnce) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E
    ASSERT_EQ(a.filter_size(), 1u);

    // Three membership rejections within one search (no accept between them).
    for (int i = 0; i < 3; ++i)
        FilterMembershipReject(a);
    EXPECT_EQ(a.successive_filter_rejections(), 0); // untouched until the accept

    // The single accept that ends the search advances the counter by ONE.
    FilterFTypeAccept(a);
    EXPECT_EQ(a.successive_filter_rejections(), 1);
    EXPECT_EQ(a.filter_size(), 1u); // F-type accept did not augment; E intact
}

// An iteration whose LAST rejection is a non-filter (F-type Armijo) rejection
// zeroes the streak on the next accept.
TEST(FilterAcceptance, ArmijoLastRejectionZeroesStreakOnAccept) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E

    // Three filter-caused iterations build the streak to 3.
    for (int i = 1; i <= 3; ++i) {
        FilterMembershipReject(a);
        FilterFTypeAccept(a);
        EXPECT_EQ(a.successive_filter_rejections(), i);
    }
    // An F-type Armijo rejection (non-filter) followed by an accept: the accept
    // reads last_rejection = false ⇒ streak zeroed.
    FilterFTypeArmijoReject(a);
    FilterFTypeAccept(a);
    EXPECT_EQ(a.successive_filter_rejections(), 0);
    EXPECT_EQ(a.filter_size(), 1u); // untouched (F-type accepts do not augment)
}

// Five consecutive filter-caused ITERATIONS (each: a filter-caused last
// rejection then an accept) clear the filter on the fifth accept and count one
// reset.
TEST(FilterAcceptance, FiveFilterCausedIterationsClearFilter) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E
    ASSERT_EQ(a.filter_size(), 1u);

    for (int iter = 1; iter <= kFilterResetTrigger; ++iter) {
        FilterMembershipReject(a); // filter-caused last rejection
        FilterFTypeAccept(a);      // accept advances / triggers the heuristic
        if (iter < kFilterResetTrigger) {
            EXPECT_EQ(a.successive_filter_rejections(), iter); // one per iteration
            EXPECT_EQ(a.filter_size(), 1u);                    // not cleared yet
            EXPECT_EQ(a.filter_resets(), 0);
        } else {
            EXPECT_EQ(a.filter_resets(), 1);                // cleared on the 5th accept
            EXPECT_EQ(a.successive_filter_rejections(), 0); // run zeroed
            EXPECT_EQ(a.filter_size(), 0u); // cleared (F-type accept did not re-augment)
        }
    }
}

// The cap holds: after kFilterMaxResets clears no further clear happens (a 6th
// reset never occurs). Each clear leaves an empty filter and the F-type accepts
// do not re-augment, so E is re-seeded before each fresh run of 5 iterations.
// After the cap, a further run of filter-caused iterations neither increments
// the reset count nor clears the re-seeded entry.
TEST(FilterAcceptance, ResetCapNeverExceeded) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);

    for (int r = 0; r < kFilterMaxResets; ++r) {
        ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // (re-)seed E
        ASSERT_EQ(a.filter_size(), 1u);
        for (int i = 1; i <= kFilterResetTrigger; ++i) {
            FilterMembershipReject(a);
            FilterFTypeAccept(a);
        }
        EXPECT_EQ(a.filter_size(), 0u); // cleared on the 5th accept
        EXPECT_EQ(a.filter_resets(), r + 1);
    }
    ASSERT_EQ(a.filter_resets(), kFilterMaxResets);

    // One more run of filter-caused iterations: the cap blocks any further clear.
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // re-seed E
    ASSERT_EQ(a.filter_size(), 1u);
    for (int i = 1; i <= kFilterResetTrigger + 3; ++i) {
        FilterMembershipReject(a);
        FilterFTypeAccept(a);
    }
    EXPECT_EQ(a.filter_resets(), kFilterMaxResets); // 6th reset never happened
    EXPECT_EQ(a.filter_size(), 1u);                 // entry survives (no clear)
}

// ===========================================================================
// reset() empties the filter and zeroes the counters; the next θ₀ re-arms.
// ===========================================================================

TEST(FilterAcceptance, ResetClearsEverythingAndReArms) {
    FilterAcceptance a;
    FilterPrimeCeilingReject(a, 4.0);
    ASSERT_TRUE(FilterHType(a, 4.0, 20.0, 1.0, 20.0)); // seed E
    ASSERT_EQ(a.filter_size(), 1u);
    // Two filter-caused iterations so the successive-iteration counter is > 0.
    for (int i = 1; i <= 2; ++i) {
        FilterMembershipReject(a);
        FilterFTypeAccept(a);
    }
    ASSERT_EQ(a.successive_filter_rejections(), 2);

    a.reset();
    EXPECT_EQ(a.filter_size(), 0u);
    EXPECT_EQ(a.successive_filter_rejections(), 0);
    EXPECT_EQ(a.filter_resets(), 0);

    // Re-arm on a new phase: prime with a new θ₀, then a fresh augment works.
    FilterPrimeCeilingReject(a, 2.0);
    ASSERT_TRUE(FilterHType(a, /*θ_c=*/2.0, /*φ_c=*/8.0, /*θ_t=*/1.0, /*φ_t=*/8.0));
    EXPECT_EQ(a.filter_size(), 1u);
    const auto [phi_add, theta_add] = a.filter_entry(0);
    EXPECT_DOUBLE_EQ(phi_add, 8.0 - kFilterGammaPhi * 2.0);
    EXPECT_DOUBLE_EQ(theta_add, (1.0 - kFilterGammaTheta) * 2.0);
}

} // namespace
