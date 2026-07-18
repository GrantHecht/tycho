///////////////////////////////////////////////////////////////////////////////
// Unit tests for SwitchingAcceptance — the Wächter–Biegler switching-condition
// skeleton shared by the (not-yet-implemented) filter and funnel acceptance
// strategies.
//
// SwitchingAcceptance is abstract: it delegates bound initialization, bound
// reset, the H-TYPE infeasibility verdict, and H-TYPE bookkeeping to a
// subclass. These tests exercise the template method
// (is_iterate_acceptable/reset) against a minimal concrete subclass
// (SwitchingFakeAcceptance) that records every hook call so the truth table
// below can assert exactly which branch ran.
//
// Every scenario's arithmetic is computed BY HAND (or via a one-line python
// check, reproduced in the comment) from the equations documented in
// switching_acceptance.h:
//   θ_min = kThetaMinFact · max(1, θ₀),  θ_max = kThetaMaxFact · max(1, θ₀)
//   switching (θ_k ≤ θ_min only): α·(m_f/α)^{s_φ} > δ·θ_k^{s_θ}   (Eq. 19)
//   F-type Armijo: φ(trial) ≤ φ(current) − η_φ·m_f                (Eq. 20)
//   φ(pt) = pt.objective + pt.auxiliary
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/switching_acceptance.h"

#include "tycho/detail/solvers/globalization/progress_measures.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using tycho::solvers::kThetaMaxFact;
using tycho::solvers::kThetaMinFact;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::SwitchingAcceptance;

// File-unique helper (UNITY RULE: anonymous namespace does not protect names;
// prefixed Switching* to stay unique across tests/cpp/).
ProgressMeasures SwitchingMakePm(double infeasibility, double objective, double auxiliary) {
    ProgressMeasures pm;
    pm.infeasibility = infeasibility;
    pm.objective = objective;
    pm.auxiliary = auxiliary;
    return pm;
}

// Minimal concrete subclass: records every hook call so a test can assert
// which branch of the template method ran, and lets a test script the
// H-TYPE verdict.
class SwitchingFakeAcceptance : public SwitchingAcceptance {
  public:
    bool h_verdict = true;
    int initialize_bounds_calls = 0;
    int reset_bounds_calls = 0;
    int is_infeasibility_acceptable_calls = 0;
    int register_accepted_h_type_calls = 0;
    double last_theta0 = 0.0;

  protected:
    void initialize_bounds(double theta_0) override {
        ++initialize_bounds_calls;
        last_theta0 = theta_0;
    }
    void reset_bounds() override { ++reset_bounds_calls; }
    bool is_infeasibility_acceptable(const ProgressMeasures &current,
                                     const ProgressMeasures &trial) override {
        (void)current;
        (void)trial;
        ++is_infeasibility_acceptable_calls;
        return h_verdict;
    }
    void register_accepted_h_type(const ProgressMeasures &current,
                                  const ProgressMeasures &trial) override {
        (void)current;
        (void)trial;
        ++register_accepted_h_type_calls;
    }
};

// ===========================================================================
// Lazy initialization (θ₀ capture + subclass hook dispatch).
// ===========================================================================

// First call captures θ₀ = current.infeasibility and derives θ_min/θ_max;
// initialize_bounds(θ₀) fires exactly once.
//   θ₀ = 2.0 ⇒ θ_min = 1e-4·2 = 2e-4, θ_max = 1e4·2 = 2e4.
// current.infeasibility (2.0) > θ_min (2e-4) ⇒ H-type path regardless of pred.
TEST(SwitchingAcceptance, LazyInitCapturesThetaZero) {
    SwitchingFakeAcceptance a;
    a.h_verdict = true;
    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(2.0, 10.0, 0.0),
                                            SwitchingMakePm(0.1, 9.0, 0.0),
                                            SwitchingMakePm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.initialize_bounds_calls, 1);
    EXPECT_DOUBLE_EQ(a.last_theta0, 2.0);
    EXPECT_DOUBLE_EQ(a.theta_min(), kThetaMinFact * 2.0);
    EXPECT_DOUBLE_EQ(a.theta_max(), kThetaMaxFact * 2.0);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 1);
    EXPECT_EQ(a.register_accepted_h_type_calls, 1);
}

// A second call does NOT re-derive θ_min/θ_max from its own current: the
// bounds stay pinned to the FIRST call's θ₀ until reset().
TEST(SwitchingAcceptance, LazyInitOnlyOnceUntilReset) {
    SwitchingFakeAcceptance a;
    a.is_iterate_acceptable(SwitchingMakePm(2.0, 10.0, 0.0), SwitchingMakePm(0.1, 9.0, 0.0),
                            SwitchingMakePm(1.0, 1.0, 0.0), 1.0, 1.0);
    ASSERT_EQ(a.initialize_bounds_calls, 1);
    const double theta_min_after_first = a.theta_min();
    const double theta_max_after_first = a.theta_max();

    a.is_iterate_acceptable(SwitchingMakePm(5.0, 10.0, 0.0), SwitchingMakePm(0.1, 9.0, 0.0),
                            SwitchingMakePm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_EQ(a.initialize_bounds_calls, 1); // still 1 — not re-armed
    EXPECT_DOUBLE_EQ(a.theta_min(), theta_min_after_first);
    EXPECT_DOUBLE_EQ(a.theta_max(), theta_max_after_first);
}

// reset() clears the lazy-init flag (re-arming it) and calls reset_bounds();
// the next is_iterate_acceptable() call captures a NEW θ₀.
//   θ₀ (2nd arm) = 10.0 ⇒ θ_min = 1e-4·10 = 1e-3, θ_max = 1e4·10 = 1e5.
TEST(SwitchingAcceptance, ResetRearmsLazyInit) {
    SwitchingFakeAcceptance a;
    a.is_iterate_acceptable(SwitchingMakePm(2.0, 10.0, 0.0), SwitchingMakePm(0.1, 9.0, 0.0),
                            SwitchingMakePm(1.0, 1.0, 0.0), 1.0, 1.0);
    ASSERT_EQ(a.initialize_bounds_calls, 1);

    a.reset();
    EXPECT_EQ(a.reset_bounds_calls, 1);

    a.is_iterate_acceptable(SwitchingMakePm(10.0, 10.0, 0.0), SwitchingMakePm(0.1, 9.0, 0.0),
                            SwitchingMakePm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_EQ(a.initialize_bounds_calls, 2);
    EXPECT_DOUBLE_EQ(a.last_theta0, 10.0);
    EXPECT_DOUBLE_EQ(a.theta_min(), kThetaMinFact * 10.0);
    EXPECT_DOUBLE_EQ(a.theta_max(), kThetaMaxFact * 10.0);
}

// ===========================================================================
// θ_max hard ceiling (Eq. 21) — rejects outright, before switching or H-type.
// ===========================================================================

// θ₀ = 1.0 ⇒ θ_max = 1e4. trial.infeasibility = 10100 > θ_max ⇒ reject, even
// though the fake's H-verdict is scripted true: the ceiling short-circuits
// before either the switching test or the H-type delegate ever runs.
TEST(SwitchingAcceptance, ThetaMaxCeilingRejectsRegardless) {
    SwitchingFakeAcceptance a;
    a.h_verdict = true;
    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(1.0, 10.0, 0.0),
                                            SwitchingMakePm(10100.0, 9.0, 0.0),
                                            SwitchingMakePm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(a.theta_max(), kThetaMaxFact * 1.0);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 0);
    EXPECT_EQ(a.register_accepted_h_type_calls, 0);
}

// ===========================================================================
// Switching condition (Eq. 19), tested only when θ_k ≤ θ_min.
// ===========================================================================

// Priming call establishes θ₀ = 1.0 ⇒ θ_min = 1e-4, θ_max = 1e4 (see the
// θ_max ceiling test above for the same derivation). All switching-condition
// scenarios below reuse this priming step so θ_k = 1e-5 sits below θ_min.

// Switching holds AND the Armijo condition on φ holds ⇒ F-type ACCEPT, and
// the H-type delegate is never called.
//   θ_k=1e-5, m_f=0.01, α=1: lhs = 1·(0.01/1)^2.3 = 2.511886e-05;
//     rhs = 1·(1e-5)^1.1 = 3.162278e-06; lhs > rhs ⇒ switching holds.
//   φ(current) = 10+0 = 10; φ(trial) = 9.9999+0 = 9.9999;
//     Armijo RHS = 10 − 1e-8·0.01 = 9.9999999999; 9.9999 ≤ 9.9999999999 ⇒ ACCEPT.
TEST(SwitchingAcceptance, SwitchingHoldsFTypeArmijoAccepts) {
    SwitchingFakeAcceptance a;
    a.h_verdict = false; // would reject if (wrongly) delegated to H-type
    a.is_iterate_acceptable(SwitchingMakePm(1.0, 0.0, 0.0), SwitchingMakePm(0.5, 0.0, 0.0),
                            SwitchingMakePm(0.5, 1.0, 0.0), 1.0, 1.0); // priming call, θ₀=1.0
    ASSERT_DOUBLE_EQ(a.theta_min(), 1.0e-4);
    // the priming call itself routes h-type (θ = 1.0 > θ_min); zero the
    // hook counters so the assertions below see only the call under test.
    a.is_infeasibility_acceptable_calls = 0;
    a.register_accepted_h_type_calls = 0;

    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(1.0e-5, 10.0, 0.0),
                                            SwitchingMakePm(1.0e-6, 9.9999, 0.0),
                                            SwitchingMakePm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 0);
    EXPECT_EQ(a.register_accepted_h_type_calls, 0);
}

// Switching holds but the Armijo condition on φ fails ⇒ F-type REJECT, still
// without ever delegating to H-type.
//   Same switching arithmetic as above (θ_k=1e-5, m_f=0.01 ⇒ holds).
//   φ(trial) = 20+0 = 20 > Armijo RHS = 9.9999999999 ⇒ REJECT.
TEST(SwitchingAcceptance, SwitchingHoldsFTypeArmijoRejects) {
    SwitchingFakeAcceptance a;
    a.h_verdict = true; // would accept if (wrongly) delegated to H-type
    a.is_iterate_acceptable(SwitchingMakePm(1.0, 0.0, 0.0), SwitchingMakePm(0.5, 0.0, 0.0),
                            SwitchingMakePm(0.5, 1.0, 0.0), 1.0, 1.0); // priming call, θ₀=1.0
    // the priming call itself routes h-type (θ = 1.0 > θ_min); zero the
    // hook counters so the assertions below see only the call under test.
    a.is_infeasibility_acceptable_calls = 0;
    a.register_accepted_h_type_calls = 0;

    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(1.0e-5, 10.0, 0.0),
                                            SwitchingMakePm(1.0e-6, 20.0, 0.0),
                                            SwitchingMakePm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 0);
    EXPECT_EQ(a.register_accepted_h_type_calls, 0);
}

// θ_k ≤ θ_min but predicted_reduction.objective ≤ 0 (not a descent direction
// for φ) ⇒ the descent guard forces switching_holds = false regardless of the
// inequality ⇒ H-type, delegated to the subclass.
TEST(SwitchingAcceptance, NonDescentForcesHType) {
    SwitchingFakeAcceptance a;
    a.h_verdict = true;
    a.is_iterate_acceptable(SwitchingMakePm(1.0, 0.0, 0.0), SwitchingMakePm(0.5, 0.0, 0.0),
                            SwitchingMakePm(0.5, 1.0, 0.0), 1.0, 1.0); // priming call, θ₀=1.0
    // the priming call itself routes h-type (θ = 1.0 > θ_min); zero the
    // hook counters so the assertions below see only the call under test.
    a.is_infeasibility_acceptable_calls = 0;
    a.register_accepted_h_type_calls = 0;

    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(1.0e-5, 10.0, 0.0),
                                            SwitchingMakePm(1.0e-6, 9.0, 0.0),
                                            SwitchingMakePm(0.5, 0.0, 0.0), // m_f = 0, not descent
                                            1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 1);
    EXPECT_EQ(a.register_accepted_h_type_calls, 1);
}

// θ_k ≤ θ_min, descent holds, but the switching inequality itself fails
// (m_f too small relative to θ_k) ⇒ H-type, delegated to the subclass.
//   θ_k=1e-5, m_f=1e-9, α=1: lhs = (1e-9)^2.3 ≈ 1.995262e-21;
//     rhs = (1e-5)^1.1 ≈ 3.162278e-06; lhs ≪ rhs ⇒ switching fails.
TEST(SwitchingAcceptance, InequalityFailsForcesHType) {
    SwitchingFakeAcceptance a;
    a.h_verdict = false;
    a.is_iterate_acceptable(SwitchingMakePm(1.0, 0.0, 0.0), SwitchingMakePm(0.5, 0.0, 0.0),
                            SwitchingMakePm(0.5, 1.0, 0.0), 1.0, 1.0); // priming call, θ₀=1.0
    // the priming call itself routes h-type (θ = 1.0 > θ_min); zero the
    // hook counters so the assertions below see only the call under test.
    a.is_infeasibility_acceptable_calls = 0;
    a.register_accepted_h_type_calls = 0;

    const bool ok = a.is_iterate_acceptable(
        SwitchingMakePm(1.0e-5, 10.0, 0.0), SwitchingMakePm(1.0e-6, 9.0, 0.0),
        SwitchingMakePm(0.5, 1.0e-9, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 1);
    EXPECT_EQ(a.register_accepted_h_type_calls, 0);
}

// θ_k > θ_min ⇒ the switching condition is never even evaluated (skipped
// entirely, not just failed) ⇒ H-type, even with a predicted reduction that
// would trivially satisfy the Eq. (19) inequality if it WERE tested.
//   θ_k=0.5 > θ_min=1e-4; m_f=1e6 (huge, would satisfy Eq. 19 for any
//   reasonable θ_k) — still routed to H-type because θ_k > θ_min.
TEST(SwitchingAcceptance, ThetaAboveMinAlwaysHType) {
    SwitchingFakeAcceptance a;
    a.h_verdict = true;
    a.is_iterate_acceptable(SwitchingMakePm(1.0, 0.0, 0.0), SwitchingMakePm(0.5, 0.0, 0.0),
                            SwitchingMakePm(0.5, 1.0, 0.0), 1.0, 1.0); // priming call, θ₀=1.0
    // the priming call itself routes h-type (θ = 1.0 > θ_min); zero the
    // hook counters so the assertions below see only the call under test.
    a.is_infeasibility_acceptable_calls = 0;
    a.register_accepted_h_type_calls = 0;

    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(0.5, 10.0, 0.0),
                                            SwitchingMakePm(0.4, 9.0, 0.0),
                                            SwitchingMakePm(0.1, 1.0e6, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.is_infeasibility_acceptable_calls, 1);
    EXPECT_EQ(a.register_accepted_h_type_calls, 1);
}

// ===========================================================================
// H-type bookkeeping: register_accepted_h_type() fires iff the verdict is true.
// ===========================================================================

TEST(SwitchingAcceptance, HTypeAcceptedRegistersBookkeeping) {
    SwitchingFakeAcceptance a;
    a.h_verdict = true;
    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(1.0, 10.0, 0.0),
                                            SwitchingMakePm(0.9, 9.0, 0.0),
                                            SwitchingMakePm(0.1, 1.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.register_accepted_h_type_calls, 1);
}

TEST(SwitchingAcceptance, HTypeRejectedSkipsBookkeeping) {
    SwitchingFakeAcceptance a;
    a.h_verdict = false;
    const bool ok = a.is_iterate_acceptable(SwitchingMakePm(1.0, 10.0, 0.0),
                                            SwitchingMakePm(0.9, 9.0, 0.0),
                                            SwitchingMakePm(0.1, 1.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.register_accepted_h_type_calls, 0);
}

// ===========================================================================
// Interface posture: generic driving path + restoration-hook T6 throw.
// ===========================================================================

TEST(SwitchingAcceptance, DrivesGenericPath) {
    SwitchingFakeAcceptance a;
    EXPECT_FALSE(a.drives_classic_path());
}

TEST(SwitchingAcceptance, RestorationHookThrows) {
    SwitchingFakeAcceptance a;
    EXPECT_THROW(a.is_infeasibility_sufficiently_reduced(SwitchingMakePm(1.0, 0.0, 0.0),
                                                         SwitchingMakePm(1.0, 0.0, 0.0)),
                 std::logic_error);
}

} // namespace
