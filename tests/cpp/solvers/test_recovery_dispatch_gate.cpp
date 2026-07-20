///////////////////////////////////////////////////////////////////////////////
// Unit test for the recovery-dispatch gate (PSIOPT globalization).
//
// The merit line search records its accept/reject verdict on the per-iteration
// IterateInfo (accepted_), and alg_impl drives the RecoveryChain hook only when
// that verdict is "rejected" AND the KKT step direction was usable (GoodStep).
// The gate condition is factored into should_dispatch_recovery() so it has a
// single definition; this test exercises that predicate directly and then
// drives it end-to-end with:
//   - a stub AcceptanceStrategy, whose classic_line_search stamps the accepted_
//     signal onto Citer exactly as the real merit variants do at the merit
//     test, and
//   - a recording RecoveryChain, which counts on_step_rejected invocations,
// asserting the hook fires on a rejected step, stays silent on an accepted
// step, and stays silent when the step direction was non-finite (no line
// search runs, so Citer keeps its fresh default and GoodStep is false).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/classic_adaptive_governor.h"
#include "tycho/detail/solvers/globalization/filter_acceptance.h"
#include "tycho/detail/solvers/globalization/funnel_acceptance.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/monitored_governor.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"
#include "tycho/detail/solvers/psiopt_fwd.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

namespace {

using tycho::solvers::AcceptanceStrategies;
using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::BarrierGovernors;
using tycho::solvers::FilterAcceptance;
using tycho::solvers::FunnelAcceptance;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kRecoveryDepthUnresolved;
using tycho::solvers::KktSolverType;
using tycho::solvers::MonitoredBarrierGovernor;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::RecoveryChain;
using tycho::solvers::should_dispatch_recovery;
using tycho::solvers::SolverContext;

// Stub AcceptanceStrategy: classic_line_search stamps the configured
// accept/reject verdict onto Citer.accepted_ (as the real merit variants do at
// the merit test) and returns a unit step. The generic filter/funnel hooks are
// unused here.
class StubAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    explicit StubAcceptance(bool accept) : accept_(accept) {}

    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}

    double classic_line_search(PSIOPT::LineSearchModes, double, double, double, double,
                               Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                               Eigen::VectorXd &, Eigen::VectorXd &, IterateInfo &Citer,
                               const std::vector<IterateInfo> &) override {
        Citer.accepted_ = accept_;
        return 1.0;
    }

  private:
    bool accept_;
};

// Recording RecoveryChain: counts hook invocations so the test can assert the
// gate fires it exactly when expected. Returns kAcceptAsIs and touches none of
// its arguments (this test exercises the dispatch GATE, not the correction
// itself — see test_soc.cpp for the SOC policy).
class RecordingRecovery : public RecoveryChain {
  public:
    Action on_step_rejected(IterateInfo &, const std::vector<IterateInfo> &, SolverContext &,
                            AcceptanceStrategy &, GlobalizationMechanism &,
                            PSIOPT::LineSearchModes, double, double, double, double,
                            Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                            Eigen::VectorXd &, Eigen::VectorXd &, double &, double &, double &,
                            int &, int &, int &) override {
        ++calls_;
        return Action::kAcceptAsIs;
    }
    void reset() override {}

    int calls_ = 0;
};

// Inert GlobalizationMechanism: RecordingRecovery ignores it, so its bodies are
// never reached; present only to satisfy the on_step_rejected signature.
class UnusedMechanism : public GlobalizationMechanism {
  public:
    double compute_step(PSIOPT::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
                        Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                        AcceptanceStrategy &, double &, double &, IterateInfo &,
                        const std::vector<IterateInfo> &, SolverContext &) override {
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {}
    void reset() override {}
};

// Faithful replica of alg_impl's recovery-hook wiring: dispatch only when the
// shared gate predicate says so. The extra working-set arguments are inert here
// (RecordingRecovery ignores them) and bound to local dummies.
void drive_gate(bool good_step, IterateInfo &citer, RecoveryChain &recovery,
                AcceptanceStrategy &acceptance, const std::vector<IterateInfo> &iters,
                SolverContext &ctx) {
    if (should_dispatch_recovery(good_step, citer)) {
        UnusedMechanism mechanism;
        Eigen::VectorXd v;
        double alpha = 1.0, alphap = 1.0, alphad = 1.0;
        int soc_steps = 0;
        int resolved_depth = kRecoveryDepthUnresolved;
        int watchdog_activations = 0;
        recovery.on_step_rejected(citer, iters, ctx, acceptance, mechanism,
                                  PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0, 0.0, v, v, v, v,
                                  v, alpha, alphap, alphad, soc_steps, resolved_depth,
                                  watchdog_activations);
    }
}

// Minimal SolverContext for the recovery signature. RecordingRecovery never
// dereferences it (the hook only bumps its counter), so every member binds to
// an inert dummy: a null NLP, a default-constructed (never-factorized) KKT
// solver, default Settings, a shared zero dimension, and a shared empty vector.
SolverContext make_dummy_context(KktSolverType &solver, PSIOPT::Settings &settings, int &zero,
                                 Eigen::VectorXd &scratch) {
    return SolverContext{nullptr, solver,  settings, zero,    zero,    zero,
                         zero,    zero,    scratch,  scratch, scratch, scratch};
}

TEST(RecoveryDispatchGate, PredicateTruthTable) {
    IterateInfo rejected;
    rejected.accepted_ = false;
    IterateInfo accepted;
    accepted.accepted_ = true;

    EXPECT_TRUE(should_dispatch_recovery(/*good_step=*/true, rejected));
    EXPECT_FALSE(should_dispatch_recovery(/*good_step=*/true, accepted));
    EXPECT_FALSE(should_dispatch_recovery(/*good_step=*/false, rejected));
    EXPECT_FALSE(should_dispatch_recovery(/*good_step=*/false, accepted));
}

TEST(RecoveryDispatchGate, StubAcceptanceDrivesHook) {
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = make_dummy_context(solver, settings, zero, scratch);
    const std::vector<IterateInfo> iters;
    Eigen::VectorXd v; // inert working vectors for the stub line search

    // Rejected step (GoodStep): the stub stamps accepted_ = false; gate fires.
    {
        RecordingRecovery recovery;
        StubAcceptance acceptance(/*accept=*/false);
        IterateInfo citer;
        acceptance.classic_line_search(PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0, 0.0, v, v,
                                       v, v, v, citer, iters);
        EXPECT_FALSE(citer.accepted_);
        drive_gate(/*good_step=*/true, citer, recovery, acceptance, iters, ctx);
        EXPECT_EQ(recovery.calls_, 1);
    }

    // Accepted step (GoodStep): the stub stamps accepted_ = true; gate silent.
    {
        RecordingRecovery recovery;
        StubAcceptance acceptance(/*accept=*/true);
        IterateInfo citer;
        acceptance.classic_line_search(PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0, 0.0, v, v,
                                       v, v, v, citer, iters);
        EXPECT_TRUE(citer.accepted_);
        drive_gate(/*good_step=*/true, citer, recovery, acceptance, iters, ctx);
        EXPECT_EQ(recovery.calls_, 0);
    }

    // Non-finite step direction (!GoodStep): alg_impl runs no line search, so
    // Citer keeps its fresh default (accepted_ == false); the gate stays silent
    // because GoodStep is false.
    {
        RecordingRecovery recovery;
        StubAcceptance acceptance(/*accept=*/false); // never reached (gate stays silent)
        IterateInfo citer;                           // no line search ran
        EXPECT_FALSE(citer.accepted_);
        drive_gate(/*good_step=*/false, citer, recovery, acceptance, iters, ctx);
        EXPECT_EQ(recovery.calls_, 0);
    }
}

// Settings::validate()'s strategy-combination guard: originally scoped to
// acceptance_strategy_ == merit, now generalized to every non-classic_merit
// strategy (see the guard's comment in psiopt.cpp). Exercise both new
// strategies against both offending knobs.
TEST(RecoveryDispatchGate, ValidateRejectsFunnelWithMaxSoc) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::funnel;
    settings.max_soc_ = 1;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(RecoveryDispatchGate, ValidateRejectsFunnelWithLsExtendedIters) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::funnel;
    settings.ls_extended_iters_ = 1;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(RecoveryDispatchGate, ValidateRejectsFilterWithMaxSoc) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::filter;
    settings.max_soc_ = 1;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(RecoveryDispatchGate, ValidateRejectsFilterWithLsExtendedIters) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::filter;
    settings.ls_extended_iters_ = 1;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

// classic_merit is unaffected by the widened guard: max_soc_/ls_extended_iters_
// combine with it exactly as before.
TEST(RecoveryDispatchGate, ValidateAcceptsClassicMeritWithMaxSoc) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::classic_merit;
    settings.max_soc_ = 1;
    settings.ls_extended_iters_ = 1;
    EXPECT_NO_THROW(settings.validate());
}

// merit's pre-existing rejection still fires under the generalized guard.
TEST(RecoveryDispatchGate, ValidateStillRejectsMeritWithMaxSoc) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::merit;
    settings.max_soc_ = 1;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

// The recovery-link guard covers only max_soc_/ls_extended_iters_; watchdog_
// combines freely with every acceptance strategy, including the two newly
// wired non-classic strategies.
TEST(RecoveryDispatchGate, ValidateAcceptsFunnelWithWatchdog) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::funnel;
    settings.barrier_governor_ = BarrierGovernors::monitored;
    settings.watchdog_ = true;
    EXPECT_NO_THROW(settings.validate());
}

TEST(RecoveryDispatchGate, ValidateAcceptsFilterWithWatchdog) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::filter;
    settings.never_monotone_ = true;
    settings.watchdog_ = true;
    EXPECT_NO_THROW(settings.validate());
}

// Settings::validate()'s barrier_governor/never_monotone truth table (see the
// guard's comment in psiopt.cpp): funnel/filter with barrier_governor=
// classic_adaptive (the default) and never_monotone=false rejects; either
// opt-in (barrier_governor=monitored, or never_monotone=true) accepts; the two
// opt-ins together are a direct contradiction and reject; classic_merit/merit
// are unaffected in every combination.

TEST(RecoveryDispatchGate, ValidateRejectsFunnelWithClassicAdaptiveGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::funnel;
    settings.barrier_governor_ = BarrierGovernors::classic_adaptive;
    settings.never_monotone_ = false;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(RecoveryDispatchGate, ValidateRejectsFilterWithClassicAdaptiveGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::filter;
    settings.barrier_governor_ = BarrierGovernors::classic_adaptive;
    settings.never_monotone_ = false;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(RecoveryDispatchGate, ValidateAcceptsFunnelWithMonitoredGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::funnel;
    settings.barrier_governor_ = BarrierGovernors::monitored;
    EXPECT_NO_THROW(settings.validate());
}

TEST(RecoveryDispatchGate, ValidateAcceptsFilterWithMonitoredGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::filter;
    settings.barrier_governor_ = BarrierGovernors::monitored;
    EXPECT_NO_THROW(settings.validate());
}

TEST(RecoveryDispatchGate, ValidateAcceptsFunnelWithNeverMonotone) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::funnel;
    settings.barrier_governor_ = BarrierGovernors::classic_adaptive;
    settings.never_monotone_ = true;
    EXPECT_NO_THROW(settings.validate());
}

TEST(RecoveryDispatchGate, ValidateAcceptsFilterWithNeverMonotone) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::filter;
    settings.barrier_governor_ = BarrierGovernors::classic_adaptive;
    settings.never_monotone_ = true;
    EXPECT_NO_THROW(settings.validate());
}

TEST(RecoveryDispatchGate, ValidateRejectsNeverMonotoneWithMonitoredGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::classic_merit;
    settings.barrier_governor_ = BarrierGovernors::monitored;
    settings.never_monotone_ = true;
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST(RecoveryDispatchGate, ValidateAcceptsClassicMeritWithClassicAdaptiveGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::classic_merit;
    settings.barrier_governor_ = BarrierGovernors::classic_adaptive;
    settings.never_monotone_ = false;
    EXPECT_NO_THROW(settings.validate());
}

// merit is a generic-path strategy like funnel/filter but is explicitly
// unaffected by the monotone-safeguard guard -- only funnel/filter are gated.
TEST(RecoveryDispatchGate, ValidateAcceptsMeritWithClassicAdaptiveGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::merit;
    settings.barrier_governor_ = BarrierGovernors::classic_adaptive;
    settings.never_monotone_ = false;
    EXPECT_NO_THROW(settings.validate());
}

// classic_merit + monitored is allowed opt-in (bit-identity is about the
// DEFAULT governor selection, not about excluding classic_merit from pairing
// with the monitored governor).
TEST(RecoveryDispatchGate, ValidateAcceptsClassicMeritWithMonitoredGovernor) {
    PSIOPT::Settings settings;
    settings.acceptance_strategy_ = AcceptanceStrategies::classic_merit;
    settings.barrier_governor_ = BarrierGovernors::monitored;
    EXPECT_NO_THROW(settings.validate());
}

} // namespace

// Settings::acceptance_strategy_ = funnel/filter must construct the matching
// concrete strategy and drive the generic AcceptanceStrategy path, exactly
// like the modern merit family (see test_merit_rules.cpp's DrivesGenericPath).
//
// Test access: these two cases call the private
// PSIOPT::rebuild_globalization_components() and read the private
// acceptance_ member directly, so they are declared as friends in psiopt.h
// (narrowly, by their gtest-generated class names) rather than PSIOPT
// exposing a public rebuild hook for this alone. gtest TEST() macros expand
// to a class at the enclosing scope, and a friend declaration cannot name a
// class inside an anonymous namespace from a production header, so these two
// cases live at global scope instead of inside the anonymous namespace above.

TEST(RecoveryDispatchGate, FunnelSelectionConstructsFunnelAcceptance) {
    tycho::solvers::PSIOPT solver;
    solver.settings().acceptance_strategy_ = tycho::solvers::AcceptanceStrategies::funnel;
    solver.rebuild_globalization_components();
    tycho::solvers::AcceptanceStrategy *acceptance = solver.acceptance_.get();
    ASSERT_NE(dynamic_cast<tycho::solvers::FunnelAcceptance *>(acceptance), nullptr);
    EXPECT_FALSE(acceptance->drives_classic_path());
}

TEST(RecoveryDispatchGate, FilterSelectionConstructsFilterAcceptance) {
    tycho::solvers::PSIOPT solver;
    solver.settings().acceptance_strategy_ = tycho::solvers::AcceptanceStrategies::filter;
    solver.rebuild_globalization_components();
    tycho::solvers::AcceptanceStrategy *acceptance = solver.acceptance_.get();
    ASSERT_NE(dynamic_cast<tycho::solvers::FilterAcceptance *>(acceptance), nullptr);
    EXPECT_FALSE(acceptance->drives_classic_path());
}

// Settings::barrier_governor_ = monitored must construct MonitoredBarrierGovernor
// (the default, classic_adaptive, constructs ClassicAdaptiveGovernor -- covered
// implicitly by every other test in this file, which never touches
// barrier_governor_ and still solves/validates against the classic governor).
//
// Test access: same pattern as the funnel/filter construction tests above --
// calls the private rebuild_globalization_components() and reads the private
// governor_ member, so it is declared as a friend in psiopt.h and lives at
// global scope (see the comment above for why).
TEST(RecoveryDispatchGate, MonitoredSelectionConstructsMonitoredGovernor) {
    tycho::solvers::PSIOPT solver;
    solver.settings().barrier_governor_ = tycho::solvers::BarrierGovernors::monitored;
    solver.rebuild_globalization_components();
    tycho::solvers::BarrierGovernor *governor = solver.governor_.get();
    ASSERT_NE(dynamic_cast<tycho::solvers::MonitoredBarrierGovernor *>(governor), nullptr);
    EXPECT_FALSE(governor->in_monotone_mode());
}
