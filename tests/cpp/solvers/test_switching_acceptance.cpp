///////////////////////////////////////////////////////////////////////////////
// Unit tests for SwitchingAcceptance — the Wächter–Biegler switching-condition
// skeleton shared by the filter and funnel acceptance strategies.
//
// SwitchingAcceptance is abstract: it delegates bound initialization, bound
// reset, the strategy MEMBERSHIP verdict (checked on every trial), the H-TYPE
// sufficient-progress verdict, accepted-step bookkeeping, and rejection
// notification to a subclass. These tests exercise the template method
// (is_iterate_acceptable/reset) against a minimal concrete subclass
// (SwitchingFakeAcceptance) that records every hook call so the truth table
// below can assert exactly which branch ran and which rejection cause fired.
//
// Acceptance order (see switching_acceptance.h): θ_max ceiling → membership
// (every trial) → switching selects F-type (Armijo) vs H-type (progress
// verdict) → register on accept; every rejection notifies its cause. A
// membership rejection additionally notifies whether the trial would have
// passed the type-appropriate progress test, evaluated speculatively
// (Ipopt's T1) — see the membership section below.
//
// Every scenario's arithmetic is computed BY HAND (or via a one-line python
// check, reproduced in the comment) from the equations documented in
// switching_acceptance.h:
//   θ_min = kThetaMinFact · max(1, θ₀),  θ_max = kThetaMaxFact · max(1, θ₀)
//   switching (θ_k ≤ θ_min only): α·(m_f/α)^{s_φ} > δ·θ_k^{s_θ}   (Eq. 19)
//   F-type Armijo: φ(trial) ≤ φ(current) − η_φ·m_f                (Eq. 20)
//   φ(pt) = pt.objective + pt.auxiliary
///////////////////////////////////////////////////////////////////////////////

#include "progress_measures_test_utils.h"

#include "tycho/detail/solvers/globalization/switching_acceptance.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using tycho::solvers::kThetaMaxFact;
using tycho::solvers::kThetaMinFact;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::RejectionCause;
using tycho::solvers::SwitchingAcceptance;
using TychoTest::pm;

// Minimal concrete subclass: records every hook call so a test can assert which
// branch of the template method ran, lets a test script the membership and
// H-type verdicts, and captures the last rejection cause.
class SwitchingFakeAcceptance : public SwitchingAcceptance {
  public:
    bool membership_verdict = true;
    bool h_progress_verdict = true;
    int initialize_bounds_calls = 0;
    int reset_bounds_calls = 0;
    int membership_calls = 0;
    int h_progress_calls = 0;
    int register_calls = 0;
    bool last_register_h_type = false;
    int rejected_calls = 0;
    RejectionCause last_cause = RejectionCause::kCeiling;
    bool last_trial_passed_progress_test = false;
    double last_theta0 = 0.0;

    void clear_hook_counters() {
        membership_calls = 0;
        h_progress_calls = 0;
        register_calls = 0;
        rejected_calls = 0;
    }

  protected:
    void initialize_bounds(double theta_0) override {
        ++initialize_bounds_calls;
        last_theta0 = theta_0;
    }
    void reset_bounds() override { ++reset_bounds_calls; }
    bool is_trial_acceptable_to_strategy(const ProgressMeasures &current,
                                         const ProgressMeasures &trial) override {
        (void)current;
        (void)trial;
        ++membership_calls;
        return membership_verdict;
    }
    bool is_h_type_progress_acceptable(const ProgressMeasures &current,
                                       const ProgressMeasures &trial) override {
        (void)current;
        (void)trial;
        ++h_progress_calls;
        return h_progress_verdict;
    }
    void register_accepted_step(const ProgressMeasures &current, const ProgressMeasures &trial,
                                bool h_type) override {
        (void)current;
        (void)trial;
        ++register_calls;
        last_register_h_type = h_type;
    }
    void notify_trial_rejected(RejectionCause cause, bool trial_passed_progress_test) override {
        ++rejected_calls;
        last_cause = cause;
        last_trial_passed_progress_test = trial_passed_progress_test;
    }
};

// ===========================================================================
// Lazy initialization (θ₀ capture + subclass hook dispatch).
// ===========================================================================

// First call captures θ₀ = current.infeasibility and derives θ_min/θ_max;
// initialize_bounds(θ₀) fires exactly once.
//   θ₀ = 2.0 ⇒ θ_min = 1e-4·2 = 2e-4, θ_max = 1e4·2 = 2e4.
// current.infeasibility (2.0) > θ_min (2e-4) ⇒ H-type path regardless of pred;
// membership passes, H-type progress passes ⇒ accept, register(h_type = true).
TEST(SwitchingAcceptance, LazyInitCapturesThetaZero) {
    SwitchingFakeAcceptance a;
    const bool ok =
        a.is_iterate_acceptable(pm(2.0, 10.0, 0.0), pm(0.1, 9.0, 0.0), pm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.initialize_bounds_calls, 1);
    EXPECT_DOUBLE_EQ(a.last_theta0, 2.0);
    EXPECT_DOUBLE_EQ(a.theta_min(), kThetaMinFact * 2.0);
    EXPECT_DOUBLE_EQ(a.theta_max(), kThetaMaxFact * 2.0);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 1);
    EXPECT_EQ(a.register_calls, 1);
    EXPECT_TRUE(a.last_register_h_type);
}

// A second call does NOT re-derive θ_min/θ_max from its own current: the
// bounds stay pinned to the FIRST call's θ₀ until reset().
TEST(SwitchingAcceptance, LazyInitOnlyOnceUntilReset) {
    SwitchingFakeAcceptance a;
    a.is_iterate_acceptable(pm(2.0, 10.0, 0.0), pm(0.1, 9.0, 0.0), pm(1.0, 1.0, 0.0), 1.0, 1.0);
    ASSERT_EQ(a.initialize_bounds_calls, 1);
    const double theta_min_after_first = a.theta_min();
    const double theta_max_after_first = a.theta_max();

    a.is_iterate_acceptable(pm(5.0, 10.0, 0.0), pm(0.1, 9.0, 0.0), pm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_EQ(a.initialize_bounds_calls, 1); // still 1 — not re-armed
    EXPECT_DOUBLE_EQ(a.theta_min(), theta_min_after_first);
    EXPECT_DOUBLE_EQ(a.theta_max(), theta_max_after_first);
}

// reset() clears the lazy-init flag (re-arming it) and calls reset_bounds();
// the next is_iterate_acceptable() call captures a NEW θ₀.
//   θ₀ (2nd arm) = 10.0 ⇒ θ_min = 1e-4·10 = 1e-3, θ_max = 1e4·10 = 1e5.
TEST(SwitchingAcceptance, ResetRearmsLazyInit) {
    SwitchingFakeAcceptance a;
    a.is_iterate_acceptable(pm(2.0, 10.0, 0.0), pm(0.1, 9.0, 0.0), pm(1.0, 1.0, 0.0), 1.0, 1.0);
    ASSERT_EQ(a.initialize_bounds_calls, 1);

    a.reset();
    EXPECT_EQ(a.reset_bounds_calls, 1);

    a.is_iterate_acceptable(pm(10.0, 10.0, 0.0), pm(0.1, 9.0, 0.0), pm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_EQ(a.initialize_bounds_calls, 2);
    EXPECT_DOUBLE_EQ(a.last_theta0, 10.0);
    EXPECT_DOUBLE_EQ(a.theta_min(), kThetaMinFact * 10.0);
    EXPECT_DOUBLE_EQ(a.theta_max(), kThetaMaxFact * 10.0);
}

// ===========================================================================
// θ_max hard ceiling (Eq. 21) — rejects outright, before membership/switching.
// ===========================================================================

// θ₀ = 1.0 ⇒ θ_max = 1e4. trial.infeasibility = 10100 > θ_max ⇒ reject, even
// though both scripted verdicts are true: the ceiling short-circuits before the
// membership test or either delegate runs, and notifies cause kCeiling.
TEST(SwitchingAcceptance, ThetaMaxCeilingRejectsRegardless) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = true;
    const bool ok = a.is_iterate_acceptable(pm(1.0, 10.0, 0.0), pm(10100.0, 9.0, 0.0),
                                            pm(1.0, 1.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(a.theta_max(), kThetaMaxFact * 1.0);
    EXPECT_EQ(a.membership_calls, 0);
    EXPECT_EQ(a.h_progress_calls, 0);
    EXPECT_EQ(a.register_calls, 0);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kCeiling);
    // trial_passed_progress_test is a don't-care for kCeiling (Ipopt leaves
    // its attribution flag untouched here); the base passes false.
    EXPECT_FALSE(a.last_trial_passed_progress_test);
}

// ===========================================================================
// Membership — checked for EVERY trial, before switching/Armijo (the behavior
// that changed). A trial that WOULD pass switching + Armijo (F-type) is
// rejected at membership when the strategy blocks it.
//
// On a membership rejection the base additionally evaluates, SPECULATIVELY,
// the type-appropriate T1 (Armijo for what would have been an f-type trial,
// the H-type progress delegate otherwise) and hands the verdict to
// notify_trial_rejected() as trial_passed_progress_test — see the file-top
// ordering note (5) and switching_acceptance.h.
// ===========================================================================

// Same trial as SwitchingHoldsFTypeArmijoAccepts below (θ_k = 1e-5 ≤ θ_min =
// 1e-4, m_f = 0.01 ⇒ switching holds, Armijo would pass), but membership is
// scripted false. It is checked FIRST, so the trial is rejected at membership
// before the switching/Armijo tests run at all ⇒ cause kMembership. Because
// switching WOULD have held, T1 is the Armijo test, evaluated speculatively
// (not delegated to is_h_type_progress_acceptable) ⇒ h_progress_calls stays 0
// and, since Armijo would have passed (9.9999 ≤ 9.9999999999),
// trial_passed_progress_test = true.
//   θ₀ = current.infeasibility = 1e-5 ⇒ θ_min = 1e-4·max(1, 1e-5) = 1e-4.
TEST(SwitchingAcceptance, MembershipRejectsFTypeTrial) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = false; // strategy blocks the trial
    a.h_progress_verdict = true;  // would accept if (wrongly) reached
    const bool ok = a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 9.9999, 0.0),
                                            pm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 0); // switching/Armijo/H-type never reached
    EXPECT_EQ(a.register_calls, 0);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kMembership);
    EXPECT_TRUE(a.last_trial_passed_progress_test); // speculative Armijo passed
}

// Same setup as above but φ_trial = 20 fails the speculative Armijo test
// (Armijo RHS = 9.9999999999, matching SwitchingHoldsFTypeArmijoRejects
// below) ⇒ trial_passed_progress_test = false even though the cause is still
// kMembership (membership is checked first and fails regardless of Armijo).
TEST(SwitchingAcceptance, MembershipRejectsFTypeTrialArmijoWouldFail) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = false;
    a.h_progress_verdict = true; // would accept if (wrongly) reached
    const bool ok = a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 20.0, 0.0),
                                            pm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 0); // speculative T1 is Armijo, not H-type
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kMembership);
    EXPECT_FALSE(a.last_trial_passed_progress_test); // speculative Armijo failed
}

// θ_current = 0.5 > θ_min = 1e-4 ⇒ switching is never evaluated (H-type
// regardless of m_f, as in ThetaAboveMinAlwaysHType below), so the speculative
// T1 for a membership rejection is is_h_type_progress_acceptable() itself —
// called once (h_progress_calls = 1) with its scripted verdict forwarded
// verbatim as trial_passed_progress_test.
TEST(SwitchingAcceptance, MembershipRejectsHTypeTrialProgressWouldPass) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = false;
    a.h_progress_verdict = true;
    const bool ok = a.is_iterate_acceptable(pm(0.5, 10.0, 0.0), pm(0.4, 9.0, 0.0),
                                            pm(0.1, 1.0e6, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 1); // speculative T1 delegates to H-type here
    EXPECT_EQ(a.register_calls, 0);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kMembership);
    EXPECT_TRUE(a.last_trial_passed_progress_test);
}

// Same routing as above but the scripted H-type verdict is false ⇒
// trial_passed_progress_test = false.
TEST(SwitchingAcceptance, MembershipRejectsHTypeTrialProgressWouldFail) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = false;
    a.h_progress_verdict = false;
    const bool ok = a.is_iterate_acceptable(pm(0.5, 10.0, 0.0), pm(0.4, 9.0, 0.0),
                                            pm(0.1, 1.0e6, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 1);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kMembership);
    EXPECT_FALSE(a.last_trial_passed_progress_test);
}

// ===========================================================================
// Switching condition (Eq. 19), tested only when θ_k ≤ θ_min (and only after
// membership passes).
// ===========================================================================

// Priming call establishes θ₀ = 1.0 ⇒ θ_min = 1e-4, θ_max = 1e4. All
// switching-condition scenarios below reuse this priming step so θ_k = 1e-5
// sits below θ_min. Membership is scripted true for the priming so it does not
// reject before bounds are captured (bounds are captured by the lazy init,
// which runs before membership either way).

// Switching holds AND the Armijo condition on φ holds ⇒ F-type ACCEPT
// (register with h_type = false), and the H-type delegate is never called.
//   θ_k=1e-5, m_f=0.01, α=1: lhs = 1·(0.01/1)^2.3 = 2.511886e-05;
//     rhs = 1·(1e-5)^1.1 = 3.162278e-06; lhs > rhs ⇒ switching holds.
//   φ(current) = 10+0 = 10; φ(trial) = 9.9999+0 = 9.9999;
//     Armijo RHS = 10 − 1e-8·0.01 = 9.9999999999; 9.9999 ≤ 9.9999999999 ⇒ ACCEPT.
TEST(SwitchingAcceptance, SwitchingHoldsFTypeArmijoAccepts) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = false; // would reject if (wrongly) delegated to H-type
    a.is_iterate_acceptable(pm(1.0, 0.0, 0.0), pm(0.5, 0.0, 0.0), pm(0.5, 1.0, 0.0), 1.0,
                            1.0); // priming call, θ₀=1.0
    ASSERT_DOUBLE_EQ(a.theta_min(), 1.0e-4);
    a.clear_hook_counters(); // isolate the call under test

    const bool ok = a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 9.9999, 0.0),
                                            pm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 0);
    EXPECT_EQ(a.register_calls, 1);
    EXPECT_FALSE(a.last_register_h_type); // F-type accept
}

// Switching holds but the Armijo condition on φ fails ⇒ F-type REJECT (cause
// kArmijo), still without ever delegating to H-type.
//   Same switching arithmetic as above (θ_k=1e-5, m_f=0.01 ⇒ holds).
//   φ(trial) = 20+0 = 20 > Armijo RHS = 9.9999999999 ⇒ REJECT.
TEST(SwitchingAcceptance, SwitchingHoldsFTypeArmijoRejects) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = true; // would accept if (wrongly) delegated to H-type
    a.is_iterate_acceptable(pm(1.0, 0.0, 0.0), pm(0.5, 0.0, 0.0), pm(0.5, 1.0, 0.0), 1.0,
                            1.0); // priming call, θ₀=1.0
    a.clear_hook_counters();

    const bool ok = a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 20.0, 0.0),
                                            pm(0.01, 0.01, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 0);
    EXPECT_EQ(a.register_calls, 0);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kArmijo);
    EXPECT_FALSE(a.last_trial_passed_progress_test); // T1 failed by definition here
}

// θ_k ≤ θ_min but predicted_reduction.objective ≤ 0 (not a descent direction
// for φ) ⇒ the descent guard forces switching_holds = false regardless of the
// inequality ⇒ H-type, delegated to the subclass ⇒ accept, register(h_type=true).
TEST(SwitchingAcceptance, NonDescentForcesHType) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = true;
    a.is_iterate_acceptable(pm(1.0, 0.0, 0.0), pm(0.5, 0.0, 0.0), pm(0.5, 1.0, 0.0), 1.0,
                            1.0); // priming call, θ₀=1.0
    a.clear_hook_counters();

    const bool ok = a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 9.0, 0.0),
                                            pm(0.5, 0.0, 0.0), // m_f = 0, not descent
                                            1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 1);
    EXPECT_EQ(a.register_calls, 1);
    EXPECT_TRUE(a.last_register_h_type);
}

// θ_k ≤ θ_min, descent holds, but the switching inequality itself fails
// (m_f too small relative to θ_k) ⇒ H-type; the scripted progress verdict is
// false ⇒ REJECT with cause kHTypeProgress.
//   θ_k=1e-5, m_f=1e-9, α=1: lhs = (1e-9)^2.3 ≈ 1.995262e-21;
//     rhs = (1e-5)^1.1 ≈ 3.162278e-06; lhs ≪ rhs ⇒ switching fails.
TEST(SwitchingAcceptance, InequalityFailsForcesHType) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = false;
    a.is_iterate_acceptable(pm(1.0, 0.0, 0.0), pm(0.5, 0.0, 0.0), pm(0.5, 1.0, 0.0), 1.0,
                            1.0); // priming call, θ₀=1.0
    a.clear_hook_counters();

    const bool ok = a.is_iterate_acceptable(pm(1.0e-5, 10.0, 0.0), pm(1.0e-6, 9.0, 0.0),
                                            pm(0.5, 1.0e-9, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 1);
    EXPECT_EQ(a.register_calls, 0);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kHTypeProgress);
    EXPECT_FALSE(a.last_trial_passed_progress_test); // T1 failed by definition here
}

// θ_k > θ_min ⇒ the switching condition is never even evaluated (skipped
// entirely, not just failed) ⇒ H-type, even with a predicted reduction that
// would trivially satisfy the Eq. (19) inequality if it WERE tested.
//   θ_k=0.5 > θ_min=1e-4; m_f=1e6 (huge, would satisfy Eq. 19 for any
//   reasonable θ_k) — still routed to H-type because θ_k > θ_min.
TEST(SwitchingAcceptance, ThetaAboveMinAlwaysHType) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = true;
    a.is_iterate_acceptable(pm(1.0, 0.0, 0.0), pm(0.5, 0.0, 0.0), pm(0.5, 1.0, 0.0), 1.0,
                            1.0); // priming call, θ₀=1.0
    a.clear_hook_counters();

    const bool ok = a.is_iterate_acceptable(pm(0.5, 10.0, 0.0), pm(0.4, 9.0, 0.0),
                                            pm(0.1, 1.0e6, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.membership_calls, 1);
    EXPECT_EQ(a.h_progress_calls, 1);
    EXPECT_EQ(a.register_calls, 1);
    EXPECT_TRUE(a.last_register_h_type);
}

// ===========================================================================
// Accept bookkeeping: register_accepted_step() fires iff a verdict accepts,
// carrying the correct h_type flag.
// ===========================================================================

TEST(SwitchingAcceptance, HTypeAcceptedRegistersBookkeeping) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = true;
    const bool ok =
        a.is_iterate_acceptable(pm(1.0, 10.0, 0.0), pm(0.9, 9.0, 0.0), pm(0.1, 1.0, 0.0), 1.0, 1.0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(a.register_calls, 1);
    EXPECT_TRUE(a.last_register_h_type);
}

TEST(SwitchingAcceptance, HTypeRejectedSkipsBookkeeping) {
    SwitchingFakeAcceptance a;
    a.membership_verdict = true;
    a.h_progress_verdict = false;
    const bool ok =
        a.is_iterate_acceptable(pm(1.0, 10.0, 0.0), pm(0.9, 9.0, 0.0), pm(0.1, 1.0, 0.0), 1.0, 1.0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(a.register_calls, 0);
    EXPECT_EQ(a.rejected_calls, 1);
    EXPECT_EQ(a.last_cause, RejectionCause::kHTypeProgress);
    EXPECT_FALSE(a.last_trial_passed_progress_test); // T1 failed by definition here
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
    EXPECT_THROW(a.is_infeasibility_sufficiently_reduced(pm(1.0, 0.0, 0.0), pm(1.0, 0.0, 0.0)),
                 std::logic_error);
}

} // namespace
