///////////////////////////////////////////////////////////////////////////////
// Unit tests for MonitoredBarrierGovernor — the free<->monotone monitored
// barrier governor.
//
// The monitor / handoff / Fiacco–McCormick / re-entry / mu-event / reset logic
// is driven directly through the testable state-machine seam (decide() and the
// pure/reference-window members) WITHOUT a real KKT solve: every IterateInfo is
// hand-built with the residual scalars the monitor reads (kkt_inf_ / econ_inf_ /
// icon_inf_ / barr_inf_). Free-mode delegation is exercised separately through
// update_barrier with a recording fake governor injected as the delegate.
//
// Every scenario's arithmetic is computed BY HAND in the comments from the
// rules and constants documented in monitored_governor.h:
//   kAdaptiveMuKktErrorRedIters   = 4
//   kAdaptiveMuKktErrorRedFact    = 0.9999
//   kAdaptiveMuMonotoneInitFactor = 0.8
//   kBarrierKappaMu               = 0.2
//   kBarrierThetaMu               = 1.5
//   kBarrierTolFactor             = 10.0
//
// UNITY RULE: the unity build defeats anonymous namespaces for ODR, so every
// file-local helper here is prefixed MonGov* to stay globally unique across
// tests/cpp/ (grep-confirmed no other "MonGov" symbol exists).
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/globalization/monitored_governor.h"

#include "tycho/detail/solvers/globalization/globalization_mechanism.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::BarrierGovernor;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kAdaptiveMuKktErrorRedFact;
using tycho::solvers::kBarrierTolFactor;
using tycho::solvers::MonitoredBarrierGovernor;
using tycho::solvers::PSIOPT;
using tycho::solvers::SolverContext;
using TychoTest::InertSolverContext;

// Build an IterateInfo carrying only the residual scalars the monitor reads.
IterateInfo MonGovIterate(double kkt_inf, double econ_inf, double icon_inf, double barr_inf) {
    IterateInfo it;
    it.kkt_inf_ = kkt_inf;
    it.econ_inf_ = econ_inf;
    it.icon_inf_ = icon_inf;
    it.barr_inf_ = barr_inf;
    return it;
}

// An IterateInfo (the `current` argument decide()/update_barrier() now take
// directly) with every residual equal to `v`, so monitor_error = 4 v² and
// barrier_subproblem_error = v.
IterateInfo MonGovUniform(double v) { return MonGovIterate(v, v, v, v); }

// Recording fake free-mode delegate: never consulted unless the governor is in
// free mode. Records the call and echoes a sentinel mu + barr_obj so the test
// can prove the pass-through is verbatim.
class MonGovFakeDelegate : public BarrierGovernor {
  public:
    int calls = 0;
    double seen_mu_in = 0.0;
    double seen_avgcomp = 0.0;
    double seen_mincomp = 0.0;
    double return_mu = 12345.0;
    double set_barr_obj = 678.0;

    double update_barrier(PSIOPT::BarrierModes, double mu_in, double avgcomp, double mincomp,
                          Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                          Eigen::VectorXd &, GlobalizationMechanism &, SolverContext &,
                          double &barr_obj, const IterateInfo &, bool &mu_event) override {
        ++calls;
        seen_mu_in = mu_in;
        seen_avgcomp = avgcomp;
        seen_mincomp = mincomp;
        barr_obj = set_barr_obj;
        mu_event = true; // deliberately set: must NOT leak past the free path.
        return return_mu;
    }
    void reset() override {}
};

// Inert mechanism — the fake delegate ignores it, so its body must never run.
class MonGovUnusedMechanism : public GlobalizationMechanism {
  public:
    double compute_step(PSIOPT::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
                        Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                        AcceptanceStrategy &, double &, double &, IterateInfo &,
                        const std::vector<IterateInfo> &, SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached";
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached";
    }
    void reset() override {}
};

// -----------------------------------------------------------------------------
// (1) monitor_error / barrier_subproblem_error composition.
// -----------------------------------------------------------------------------

TEST(MonGovErrorNorm, MonitorErrorIsSumOfSquaredResiduals) {
    // kkt=1, econ=2, icon=3, barr=4 -> 1 + 4 + 9 + 16 = 30.
    const IterateInfo it = MonGovIterate(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(MonitoredBarrierGovernor::monitor_error(it), 30.0);
    // subproblem error is the max part = 4.
    EXPECT_DOUBLE_EQ(MonitoredBarrierGovernor::barrier_subproblem_error(it), 4.0);
}

// -----------------------------------------------------------------------------
// (2) Monitor window truth table.
// -----------------------------------------------------------------------------

TEST(MonGovMonitorWindow, FewerThanFourRefsAlwaysSufficient) {
    MonitoredBarrierGovernor g;
    // 0..3 references: sufficient regardless of a huge current error.
    for (int n = 0; n < 4; ++n) {
        EXPECT_TRUE(g.check_sufficient_progress(1e30)) << "n=" << n;
        g.remember_accepted(1.0);
    }
    // Now 4 references, all == 1.0: 1e30 <= 0.9999*1.0 is false -> not sufficient.
    EXPECT_EQ(g.reference_values().size(), 4u);
    EXPECT_FALSE(g.check_sufficient_progress(1e30));
}

TEST(MonGovMonitorWindow, SufficientDecreaseBoundary) {
    MonitoredBarrierGovernor g;
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1.0); // window = {1,1,1,1}
    }
    // factor * ref = 0.9999 * 1.0 = 0.9999.
    //   curr just below (0.9998) -> sufficient.
    //   curr exactly at the boundary (0.9999) -> sufficient (<=).
    //   curr just above (1.0) -> NOT sufficient.
    EXPECT_TRUE(g.check_sufficient_progress(0.9998));
    EXPECT_TRUE(g.check_sufficient_progress(kAdaptiveMuKktErrorRedFact * 1.0));
    EXPECT_FALSE(g.check_sufficient_progress(1.0));
}

TEST(MonGovMonitorWindow, NonmonotoneAnyReferenceSatisfies) {
    MonitoredBarrierGovernor g;
    // window = {0.5, 0.5, 0.5, 10.0}. curr = 5.0.
    //   against 0.5: 5.0 <= 0.49995? no.
    //   against 10.0: 5.0 <= 9.999?  yes -> sufficient (any reference suffices).
    g.remember_accepted(0.5);
    g.remember_accepted(0.5);
    g.remember_accepted(0.5);
    g.remember_accepted(10.0);
    EXPECT_TRUE(g.check_sufficient_progress(5.0));
    // curr = 9.9999 > 0.9999*10 = 9.999 and > all others -> not sufficient.
    EXPECT_FALSE(g.check_sufficient_progress(9.9999));
}

TEST(MonGovMonitorWindow, ReferenceListFifoCapAtFour) {
    MonitoredBarrierGovernor g;
    g.remember_accepted(1.0);
    g.remember_accepted(2.0);
    g.remember_accepted(3.0);
    g.remember_accepted(4.0); // window = {1,2,3,4}
    g.remember_accepted(5.0); // pop 1 -> window = {2,3,4,5}
    ASSERT_EQ(g.reference_values().size(), 4u);
    EXPECT_DOUBLE_EQ(g.reference_values().front(), 2.0);
    EXPECT_DOUBLE_EQ(g.reference_values().back(), 5.0);
}

// -----------------------------------------------------------------------------
// (4) Handoff arithmetic.
// -----------------------------------------------------------------------------

TEST(MonGovHandoff, MuIsPointEightTimesAvgComp) {
    // 0.8 * 1e-3 = 8e-4, inside [1e-12, 100].
    EXPECT_DOUBLE_EQ(MonitoredBarrierGovernor::handoff_mu(1e-3, 1e-12, 100.0), 8e-4);
    // clamp to max: 0.8 * 1000 = 800 -> clamped to 100.
    EXPECT_DOUBLE_EQ(MonitoredBarrierGovernor::handoff_mu(1000.0, 1e-12, 100.0), 100.0);
    // clamp to min: 0.8 * 1e-15 = 8e-16 -> clamped to 1e-12.
    EXPECT_DOUBLE_EQ(MonitoredBarrierGovernor::handoff_mu(1e-15, 1e-12, 100.0), 1e-12);
}

TEST(MonGovHandoff, DecideSwitchesToMonotoneOnMonitorFailure) {
    MonitoredBarrierGovernor g;
    // Prime a full, tight reference window so the monitor fails on a large error.
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1.0);
    }
    // current monitor_error = 4 * 100^2 = 40000 >> 0.9999 -> monitor fails.
    const auto current = MonGovUniform(100.0);
    const auto d = g.decide(current, /*mu_in=*/0.01, /*avgcomp=*/1e-3, /*bar_tol=*/1e-6,
                            /*kkt_tol=*/1e-6, /*min_mu=*/1e-12, /*max_mu=*/100.0);
    EXPECT_TRUE(d.monotone);
    EXPECT_TRUE(d.mu_event); // handoff begins a new barrier subproblem.
    EXPECT_DOUBLE_EQ(d.mu, 8e-4); // 0.8 * 1e-3.
    EXPECT_TRUE(g.in_monotone_mode());
    EXPECT_DOUBLE_EQ(g.monotone_mu(), 8e-4);
    EXPECT_EQ(g.last_monotone_switches(), 1);
}

// -----------------------------------------------------------------------------
// (6) Fiacco–McCormick sequence, floor, and subproblem-convergence gate.
// -----------------------------------------------------------------------------

TEST(MonGovFiaccoMcCormick, SequenceHandComputed) {
    // bar_tol = kkt_tol = 1e-6 -> floor = min(1e-6,1e-6)/(10+1) = 1e-6/11 ≈ 9.09e-8.
    // min_mu = 1e-12, max_mu = 100 (both inert here).
    const double bt = 1e-6, kt = 1e-6, lo = 1e-12, hi = 100.0;
    // mu0 = 0.1: min(0.2*0.1=0.02, 0.1^1.5=0.0316227766) = 0.02.
    const double mu1 = MonitoredBarrierGovernor::fiacco_mccormick_mu(0.1, bt, kt, lo, hi);
    EXPECT_DOUBLE_EQ(mu1, 0.02);
    // mu1 = 0.02: min(0.2*0.02=0.004, 0.02^1.5) ; 0.02^1.5 = 0.02*sqrt(0.02).
    const double expect2 = std::min(0.2 * 0.02, std::pow(0.02, 1.5));
    const double mu2 = MonitoredBarrierGovernor::fiacco_mccormick_mu(0.02, bt, kt, lo, hi);
    EXPECT_DOUBLE_EQ(mu2, expect2); // 0.02^1.5 ≈ 0.0028284 < 0.004 -> superlinear wins.
    EXPECT_LT(mu2, 0.004);
}

TEST(MonGovFiaccoMcCormick, FloorClampsTinyMu) {
    // mu = 1e-7: linear 0.2e-7 = 2e-8, superlinear (1e-7)^1.5 ≈ 3.162e-11.
    // min = 3.162e-11 < floor (1e-6/11 ≈ 9.09e-8) -> clamped up to the floor.
    const double floor = 1e-6 / (kBarrierTolFactor + 1.0);
    const double mu = MonitoredBarrierGovernor::fiacco_mccormick_mu(1e-7, 1e-6, 1e-6, 1e-12, 100.0);
    EXPECT_DOUBLE_EQ(mu, floor);
}

TEST(MonGovFiaccoMcCormick, GateBlocksAdvanceUntilSubproblemConverged) {
    MonitoredBarrierGovernor g;
    // Force into monotone mode at monotone_mu_ = 0.8 * 0.0125 = 0.01.
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1e-30); // tight refs -> any real error fails the monitor
    }
    const auto trigger = MonGovUniform(1.0); // monitor_error = 4 >> refs
    g.decide(trigger, 0.01, /*avgcomp=*/0.0125, 1e-6, 1e-6, 1e-12, 100.0);
    ASSERT_TRUE(g.in_monotone_mode());
    ASSERT_DOUBLE_EQ(g.monotone_mu(), 0.01);
    const double mu_before = g.monotone_mu();

    // Gate threshold = kBarrierTolFactor * mu = 10 * 0.01 = 0.1.
    // (a) sub_problem_error = 0.2 (barr_inf_) > 0.1 -> NO advance, NO mu_event.
    //     Keep the monitor "failing" (monitor_error large) so we stay monotone:
    //     residuals {0.05,0.05,0.05,0.2}: monitor_error = 3*0.0025 + 0.04 large vs 1e-30.
    const IterateInfo not_converged = MonGovIterate(0.05, 0.05, 0.05, 0.2);
    auto d_hold = g.decide(not_converged, mu_before, 0.0125, 1e-6, 1e-6, 1e-12, 100.0);
    EXPECT_TRUE(d_hold.monotone);
    EXPECT_FALSE(d_hold.mu_event);
    EXPECT_DOUBLE_EQ(g.monotone_mu(), mu_before); // held.

    // (b) sub_problem_error = 0.05 (all parts) <= 0.1 -> advance, mu_event.
    //     monitor_error = 4*0.05^2 = 0.01 still >> 1e-30 refs -> stays monotone.
    const IterateInfo converged = MonGovIterate(0.05, 0.05, 0.05, 0.05);
    auto d_adv = g.decide(converged, g.monotone_mu(), 0.0125, 1e-6, 1e-6, 1e-12, 100.0);
    EXPECT_TRUE(d_adv.monotone);
    EXPECT_TRUE(d_adv.mu_event);
    // advanced: 0.01 -> min(0.2*0.01=0.002, 0.01^1.5=0.001) = 0.001.
    EXPECT_DOUBLE_EQ(g.monotone_mu(), 0.001);
    EXPECT_DOUBLE_EQ(d_adv.mu, 0.001);
    EXPECT_EQ(g.last_monotone_iters(), 2); // two monotone-mode decide() calls.
}

// -----------------------------------------------------------------------------
// (5) Re-entry to free mode when the KKT error re-enters the reference band.
// -----------------------------------------------------------------------------

TEST(MonGovReentry, ReturnsToFreeWhenErrorReentersBand) {
    MonitoredBarrierGovernor g;
    // Reference band frozen at monitor_error = 4*1^2 = 4 (residuals all 1.0)
    // captured four times, then a handoff to monotone.
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(4.0);
    }
    const auto fail = MonGovUniform(100.0); // monitor_error 40000 -> fails
    g.decide(fail, 0.01, 1e-3, 1e-6, 1e-6, 1e-12, 100.0);
    ASSERT_TRUE(g.in_monotone_mode());

    // Now the error re-enters the band: residuals all 0.5 -> monitor_error = 1.0.
    // 1.0 <= 0.9999 * 4.0 = 3.9996 -> sufficient -> re-enter free.
    const auto recover = MonGovUniform(0.5);
    const auto d = g.decide(recover, 0.01, 1e-3, 1e-6, 1e-6, 1e-12, 100.0);
    EXPECT_FALSE(d.monotone);
    EXPECT_FALSE(d.mu_event); // re-entry does not fire a mu_event.
    EXPECT_FALSE(g.in_monotone_mode());
    // The recovering error was pushed onto the reference window (FIFO): back = 1.0.
    EXPECT_DOUBLE_EQ(g.reference_values().back(), 1.0);
}

// -----------------------------------------------------------------------------
// mu_event exactness — fires ONLY on handoff and on strict monotone advances.
// -----------------------------------------------------------------------------

TEST(MonGovMuEvent, StayingFreeDoesNotFire) {
    MonitoredBarrierGovernor g;
    // Empty window (<4 refs) -> monitor sufficient -> stay free.
    const auto current = MonGovUniform(1.0);
    const auto d = g.decide(current, 0.01, 1e-3, 1e-6, 1e-6, 1e-12, 100.0);
    EXPECT_FALSE(d.monotone);
    EXPECT_FALSE(d.mu_event);
    EXPECT_EQ(g.last_monotone_switches(), 0);
}

TEST(MonGovMuEvent, HoldingMonotoneDoesNotFire) {
    MonitoredBarrierGovernor g;
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1e-30);
    }
    g.decide(MonGovUniform(1.0), 0.01, 0.0125, 1e-6, 1e-6, 1e-12, 100.0); // handoff
    ASSERT_TRUE(g.in_monotone_mode());
    // sub_problem_error = 100 (barr_inf_) > 10*0.01 = 0.1 -> gate blocks -> hold.
    const IterateInfo big = MonGovIterate(1.0, 1.0, 1.0, 100.0);
    const auto d = g.decide(big, g.monotone_mu(), 0.0125, 1e-6, 1e-6, 1e-12, 100.0);
    EXPECT_TRUE(d.monotone);
    EXPECT_FALSE(d.mu_event);
}

// -----------------------------------------------------------------------------
// reset() truth table.
// -----------------------------------------------------------------------------

TEST(MonGovReset, ClearsAllState) {
    MonitoredBarrierGovernor g;
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1e-30);
    }
    g.decide(MonGovUniform(1.0), 0.01, 0.0125, 1e-6, 1e-6, 1e-12, 100.0); // -> monotone
    ASSERT_TRUE(g.in_monotone_mode());
    ASSERT_GT(g.reference_values().size(), 0u);
    ASSERT_EQ(g.last_monotone_switches(), 1);

    g.reset();
    EXPECT_FALSE(g.in_monotone_mode());
    EXPECT_EQ(g.reference_values().size(), 0u);
    EXPECT_DOUBLE_EQ(g.monotone_mu(), 0.0);
    EXPECT_EQ(g.last_monotone_switches(), 0);
    EXPECT_EQ(g.last_monotone_iters(), 0);
}

// -----------------------------------------------------------------------------
// Free-mode delegation pass-through (through the full update_barrier plumbing).
// -----------------------------------------------------------------------------

TEST(MonGovDelegation, FreeModeForwardsToDelegateVerbatim) {
    auto fake_owned = std::make_unique<MonGovFakeDelegate>();
    MonGovFakeDelegate *fake = fake_owned.get();
    MonitoredBarrierGovernor g(std::move(fake_owned));

    // Minimal all-zero-dimension context: the fake delegate ignores it.
    InertSolverContext inert;
    SolverContext ctx = inert.ctx();
    MonGovUnusedMechanism mechanism;
    Eigen::VectorXd XSL, RHS, DXSL, Temp; // empty (dims all zero).

    // Fresh governor, empty window -> monitor sufficient -> stays free -> delegates.
    const auto current = MonGovUniform(1.0);
    double barr_obj = -1.0;
    bool mu_event = false;
    const double mu = g.update_barrier(PSIOPT::BarrierModes::LOQO, /*mu_in=*/0.007,
                                       /*avgcomp=*/0.55, /*mincomp=*/0.11, XSL, RHS, DXSL, Temp,
                                       mechanism, ctx, barr_obj, current, mu_event);
    EXPECT_EQ(fake->calls, 1);
    EXPECT_DOUBLE_EQ(fake->seen_mu_in, 0.007);   // mu_in forwarded verbatim.
    EXPECT_DOUBLE_EQ(fake->seen_avgcomp, 0.55);  // avgcomp forwarded verbatim.
    EXPECT_DOUBLE_EQ(fake->seen_mincomp, 0.11);  // mincomp forwarded verbatim.
    EXPECT_DOUBLE_EQ(mu, fake->return_mu);       // delegate's mu returned.
    EXPECT_DOUBLE_EQ(barr_obj, fake->set_barr_obj); // delegate's barr_obj propagated.
    EXPECT_FALSE(mu_event); // delegate's inner event must NOT leak past the free path.
    EXPECT_FALSE(g.in_monotone_mode());
}

// -----------------------------------------------------------------------------
// Multi-call sequences driven THROUGH update_barrier (not decide() directly),
// each call passing a DIFFERENT `current`. This exercises the fix for the
// governor reading the wrong iterate: with the old `iters` (history) parameter,
// a caller that popped the current iterate before calling update_barrier (as
// alg_impl does) would make every one of these calls see the PREVIOUS current
// instead of the one just passed. `current` is now the argument itself, so
// each call below must react to exactly the value passed to it.
// -----------------------------------------------------------------------------

TEST(MonGovSequence, SufficientProgressVerdictFlipsWithPassedCurrent) {
    auto fake_owned = std::make_unique<MonGovFakeDelegate>();
    MonGovFakeDelegate *fake = fake_owned.get();
    MonitoredBarrierGovernor g(std::move(fake_owned));

    // Minimal all-zero-dimension context (as MonGovDelegation above): the fake
    // delegate ignores it, and with inequal_cons_ == 0 the monotone-mode
    // barrier tail (unused here in the free-mode call, but exercised by the
    // next test) is also a no-op over empty XSL/RHS.
    InertSolverContext inert;
    SolverContext ctx = inert.ctx();
    MonGovUnusedMechanism mechanism;
    Eigen::VectorXd XSL, RHS, DXSL, Temp;

    // Prime a full reference window of four 1.0's.
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1.0);
    }

    // Call 1: current has icon_inf_ = 0.9, everything else 0 -> monitor_error =
    // 0.9^2 = 0.81. 0.81 <= 0.9999 * 1.0 -> sufficient -> stays free -> delegates.
    const IterateInfo current1 = MonGovIterate(0.0, 0.0, 0.9, 0.0);
    double barr_obj1 = 0.0;
    bool event1 = false;
    g.update_barrier(PSIOPT::BarrierModes::LOQO, 0.01, 0.0125, 0.0, XSL, RHS, DXSL, Temp, mechanism,
                     ctx, barr_obj1, current1, event1);
    EXPECT_FALSE(event1);
    EXPECT_FALSE(g.in_monotone_mode());
    EXPECT_EQ(fake->calls, 1);
    // remember_accepted(0.81) pops the oldest 1.0 -> window = {1,1,1,0.81}.
    ASSERT_EQ(g.reference_values().size(), 4u);
    EXPECT_DOUBLE_EQ(g.reference_values().back(), 0.81);

    // Call 2: current has icon_inf_ = 1.0, everything else 0 -> monitor_error =
    // 1.0^2 = 1.0. Checked against window {1,1,1,0.81}: 1.0 <= 0.9999*1.0 =
    // 0.9999? no. 1.0 <= 0.9999*0.81 = 0.809919? no. Fails every reference ->
    // handoff to monotone. This is the SAME kind of residual shape as call 1
    // (only icon_inf_ differs, 0.9 -> 1.0): the verdict flip is driven purely
    // by the `current` passed to THIS call, not by any stale value.
    const IterateInfo current2 = MonGovIterate(0.0, 0.0, 1.0, 0.0);
    double barr_obj2 = 0.0;
    bool event2 = false;
    g.update_barrier(PSIOPT::BarrierModes::LOQO, 0.01, 0.0125, 0.0, XSL, RHS, DXSL, Temp, mechanism,
                     ctx, barr_obj2, current2, event2);
    EXPECT_TRUE(event2);
    EXPECT_TRUE(g.in_monotone_mode());
    EXPECT_DOUBLE_EQ(g.monotone_mu(), 0.01); // handoff_mu(0.0125, ...) = 0.8*0.0125.
    // The free delegate is never consulted again once monotone mode is entered.
    EXPECT_EQ(fake->calls, 1);
}

TEST(MonGovSequence, FiaccoMcCormickGateAdvancesOnSatisfyingCall) {
    // Default governor (real ClassicAdaptiveGovernor free delegate) -- never
    // reached, since every call below forces/keeps monotone mode.
    MonitoredBarrierGovernor g;

    InertSolverContext inert;
    SolverContext ctx = inert.ctx();
    MonGovUnusedMechanism mechanism;
    Eigen::VectorXd XSL, RHS, DXSL, Temp;

    // Prime a tight reference window so the monitor fails for the rest of this
    // test regardless of the (small) currents passed below -- isolates the FM
    // gate behavior from the monitor's free<->monotone re-entry decision.
    for (int n = 0; n < 4; ++n) {
        g.remember_accepted(1e-30);
    }

    // Call 1 (handoff): current = uniform 1.0 -> monitor_error = 4 >> refs.
    // monotone_mu_ = handoff_mu(0.0125, ...) = 0.8*0.0125 = 0.01.
    double barr_obj = 0.0;
    bool event = false;
    g.update_barrier(PSIOPT::BarrierModes::LOQO, 0.01, /*avgcomp=*/0.0125, 0.0, XSL, RHS, DXSL,
                     Temp, mechanism, ctx, barr_obj, MonGovUniform(1.0), event);
    ASSERT_TRUE(g.in_monotone_mode());
    ASSERT_DOUBLE_EQ(g.monotone_mu(), 0.01);
    ASSERT_TRUE(event);

    // Call 2 (hold): current = {0.05,0.05,0.05,0.2}. Gate threshold =
    // kBarrierTolFactor*mu = 10*0.01 = 0.1; sub_problem_error = max(...) = 0.2
    // > 0.1 -> gate blocks -> NO advance, NO mu_event.
    event = false;
    const double mu_after_hold = g.update_barrier(
        PSIOPT::BarrierModes::LOQO, g.monotone_mu(), 0.0125, 0.0, XSL, RHS, DXSL, Temp, mechanism,
        ctx, barr_obj, MonGovIterate(0.05, 0.05, 0.05, 0.2), event);
    EXPECT_FALSE(event);
    EXPECT_DOUBLE_EQ(g.monotone_mu(), 0.01);
    EXPECT_DOUBLE_EQ(mu_after_hold, 0.01);

    // Call 3 (advance): current = {0.05,0.05,0.05,0.05} -> sub_problem_error =
    // 0.05 <= 0.1 -> gate passes -> advance. PSIOPT::Settings defaults
    // bar_tol_ = kkt_tol_ = 1e-6, so floor = min(1e-6,1e-6)/(10+1) ≈ 9.09e-8,
    // well below the candidate, so
    // fiacco_mccormick_mu(0.01, 1e-6, 1e-6, min_mu_, max_mu_) =
    // max(9.09e-8, min(0.2*0.01=0.002, 0.01^1.5=0.001)) = 0.001 (superlinear
    // wins, matching MonGovFiaccoMcCormick.SequenceHandComputed's mu1->mu2 step),
    // then clamped to [1e-12, 100] (inert).
    event = false;
    const double mu_after_advance = g.update_barrier(
        PSIOPT::BarrierModes::LOQO, g.monotone_mu(), 0.0125, 0.0, XSL, RHS, DXSL, Temp, mechanism,
        ctx, barr_obj, MonGovIterate(0.05, 0.05, 0.05, 0.05), event);
    EXPECT_TRUE(event); // advance on exactly this call, not the previous hold.
    EXPECT_DOUBLE_EQ(g.monotone_mu(), 0.001);
    EXPECT_DOUBLE_EQ(mu_after_advance, 0.001);
    EXPECT_EQ(g.last_monotone_iters(), 2); // two "remain monotone" calls (hold + advance).
}

} // namespace
