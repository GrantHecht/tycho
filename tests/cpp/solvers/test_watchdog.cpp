///////////////////////////////////////////////////////////////////////////////
// Unit tests for the second batch of live RecoveryChain links: extended
// backtracking, the watchdog, and the ChainedRecovery composition that ties
// them (and SOC) together.
//
// Three pieces are tested, each truth-tabled against pure/machinery-free
// policy where possible:
//   - WatchdogState — the arm/trial/revert/reset state machine (Chamberlain,
//     Powell, Lemaréchal & Pedersen 1982; constants per Wächter & Biegler
//     2006), driven with scripted (mu, merit) sequences — no Eigen/solver
//     types involved.
//   - ExtendedBacktrackRecovery — the ladder-continuation arithmetic, driven
//     with a scripted AcceptanceStrategy that records the scaled direction
//     each call receives (no real KKT/merit machinery).
//   - ChainedRecovery / WatchdogRecovery — the composition wiring, driven
//     with spy RecoveryChain links that record invocation counts/order (no
//     real SOC/extended-backtrack policy).
//
// See test_soc.cpp for the SOC policy tests and test_recovery_dispatch_gate.
// cpp for the alg_impl dispatch-gate test; this file does not repeat either.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/globalization/watchdog.h"
#include "tycho/detail/solvers/iterate_info.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include <Eigen/Core>

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::ChainedRecovery;
using tycho::solvers::ExtendedBacktrackRecovery;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kRecoveryDepthExtended;
using tycho::solvers::kRecoveryDepthSoc;
using tycho::solvers::kRecoveryDepthUnresolved;
using tycho::solvers::kRecoveryDepthWatchdog;
using tycho::solvers::KktSolverType;
using tycho::solvers::kWatchdogShortenedIterTrigger;
using tycho::solvers::kWatchdogTrialIterMax;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::RecoveryChain;
using tycho::solvers::SolverContext;
using tycho::solvers::WatchdogRecovery;
using tycho::solvers::WatchdogState;

using Action = RecoveryChain::Action;
using Outcome = WatchdogState::Outcome;

///////////////////////////////////////////////////////////////////////////////
// WatchdogState — pure state-machine truth table.
///////////////////////////////////////////////////////////////////////////////

// Arming threshold, exact boundary: the first kWatchdogShortenedIterTrigger-1
// calls accumulate without arming; the kWatchdogShortenedIterTrigger-th call
// arms (and IS trial #1 of the window).
TEST(WatchdogStateArm, ExactBoundary) {
    WatchdogState s;
    for (int i = 1; i < kWatchdogShortenedIterTrigger; ++i) {
        EXPECT_EQ(s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0), Outcome::kAccumulate);
        EXPECT_FALSE(s.armed());
        EXPECT_EQ(s.consecutive_shortened(), i);
    }
    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0), Outcome::kArmed);
    EXPECT_TRUE(s.armed());
    EXPECT_EQ(s.trial_count(), 1);
}

// A mu change before arming resets the consecutive-shortened count (this
// call becomes the first of a fresh count, not a continuation).
TEST(WatchdogStateArm, MuChangeResetsBeforeArming) {
    WatchdogState s;
    for (int i = 0; i < 5; ++i)
        s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0);
    ASSERT_EQ(s.consecutive_shortened(), 5);

    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/2.0, /*merit=*/10.0), Outcome::kAccumulate);
    EXPECT_EQ(s.consecutive_shortened(), 1);
    EXPECT_FALSE(s.armed());
}

// A genuinely accepted iteration in between rejections (record_accepted_
// iteration -- see notify_step_accepted() on RecoveryChain) resets the
// consecutive-shortened count exactly like a mu change does: [9 rejections,
// accepted, 1 rejection] must NOT arm (this is the sequence the watchdog.h
// docstring's now-corrected "cannot mis-arm" note pins). The control case --
// 10 STRAIGHT rejections, no accepted iteration in between -- DOES arm.
TEST(WatchdogStateArm, AcceptedIterationResetsConsecutiveCount) {
    static_assert(kWatchdogShortenedIterTrigger == 10,
                 "test assumes the paper's 10-rejection trigger");

    WatchdogState not_armed;
    for (int i = 0; i < kWatchdogShortenedIterTrigger - 1; ++i)
        not_armed.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0);
    ASSERT_EQ(not_armed.consecutive_shortened(), kWatchdogShortenedIterTrigger - 1);

    not_armed.record_accepted_iteration();
    EXPECT_EQ(not_armed.consecutive_shortened(), 0);
    EXPECT_FALSE(not_armed.armed());

    EXPECT_EQ(not_armed.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0),
              Outcome::kAccumulate); // 1st of a fresh count, not the 10th overall
    EXPECT_EQ(not_armed.consecutive_shortened(), 1);
    EXPECT_FALSE(not_armed.armed());

    // Control: the same kWatchdogShortenedIterTrigger rejections with no
    // accepted iteration in between DOES arm.
    WatchdogState armed;
    for (int i = 0; i < kWatchdogShortenedIterTrigger - 1; ++i)
        armed.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0);
    EXPECT_EQ(armed.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0), Outcome::kArmed);
    EXPECT_TRUE(armed.armed());
}

// Once armed, a trial iterate that beats the snapshot's merit reference
// disarms (kTrialProgress) and clears the count entirely.
TEST(WatchdogStateTrial, ProgressDisarms) {
    WatchdogState s;
    for (int i = 0; i < kWatchdogShortenedIterTrigger; ++i)
        s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0);
    ASSERT_TRUE(s.armed());

    // Trial #2: no improvement yet (merit unchanged) -> still relaxed.
    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0), Outcome::kTrialRelax);
    EXPECT_TRUE(s.armed());

    // Trial #3 would be the window's last slot, but progress arrives first:
    // merit strictly beats the snapshot reference (10.0) -> disarm.
    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/5.0), Outcome::kTrialProgress);
    EXPECT_FALSE(s.armed());
    EXPECT_EQ(s.consecutive_shortened(), 0);
}

// Equal merit does not count as "beats the reference" (strict <): the window
// still exhausts and reverts.
TEST(WatchdogStateTrial, WindowExhaustionReverts) {
    WatchdogState s;
    for (int i = 0; i < kWatchdogShortenedIterTrigger; ++i)
        s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0); // trial #1 (arms)
    ASSERT_TRUE(s.armed());

    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0),
              Outcome::kTrialRelax); // trial #2, no improvement
    ASSERT_TRUE(s.armed());

    static_assert(kWatchdogTrialIterMax == 3, "test assumes the paper's 3-trial window");
    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0),
              Outcome::kTrialRevert); // trial #3, window exhausted
    EXPECT_FALSE(s.armed());
    EXPECT_EQ(s.consecutive_shortened(), 0);
}

// A mu change while armed resets the WHOLE state (not a revert -- see
// WatchdogState's class doc) and re-processes the call as the first of a
// fresh count.
TEST(WatchdogStateTrial, MuChangeWhileArmedResetsEntirely) {
    WatchdogState s;
    for (int i = 0; i < kWatchdogShortenedIterTrigger; ++i)
        s.record_rejected_iteration(/*mu=*/1.0, /*merit=*/10.0);
    ASSERT_TRUE(s.armed());

    EXPECT_EQ(s.record_rejected_iteration(/*mu=*/2.0, /*merit=*/10.0), Outcome::kAccumulate);
    EXPECT_FALSE(s.armed());
    EXPECT_EQ(s.consecutive_shortened(), 1);
    EXPECT_EQ(s.trial_count(), 0);
}

///////////////////////////////////////////////////////////////////////////////
// ExtendedBacktrackRecovery — ladder-continuation arithmetic.
///////////////////////////////////////////////////////////////////////////////

// Inert AcceptanceStrategy: used on the off (cap == 0) early-exit path, which
// must never reach it.
class ExtBtUnusedAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "acceptance must not be reached when ls_extended_iters_ == 0";
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
};

// Pass-through GlobalizationMechanism: ExtendedBacktrackRecovery re-drives the
// acceptance backtrack through the mechanism's run_acceptance_backtrack seam
// (not by calling the acceptance directly), so this double forwards it verbatim
// to the (classic) acceptance's classic_line_search — reproducing the routing
// BacktrackingLineSearch performs for a classic strategy, so the scripted
// acceptance below still sees each trial's DXSL and stamps the verdict. The
// fraction-to-boundary halves (compute_step / max_primal_dual_step) are NOT part
// of the extended-backtracking path and must never be reached.
class ExtBtPassThroughMechanism : public GlobalizationMechanism {
  public:
    double compute_step(PSIOPT::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
                        Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                        AcceptanceStrategy &, double &, double &, IterateInfo &,
                        const std::vector<IterateInfo> &, SolverContext &) override {
        ADD_FAILURE() << "compute_step must never be reached by ExtendedBacktrackRecovery";
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {
        ADD_FAILURE() << "max_primal_dual_step must never be reached by ExtendedBacktrackRecovery";
    }
    double run_acceptance_backtrack(PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                                    double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                                    Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                    Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                                    AcceptanceStrategy &acceptance, IterateInfo &Citer,
                                    const std::vector<IterateInfo> &iters, SolverContext &) override {
        return acceptance.classic_line_search(lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, DXSL,
                                              XSL2, RHS, RHS2, Citer, iters);
    }
    void reset() override {}
};

// Scripted AcceptanceStrategy: on the i-th call to classic_line_search,
// stamps Citer.accepted_ from outcomes_[i].first and returns
// outcomes_[i].second, recording a copy of the DXSL argument it was handed
// (the scaled direction ExtendedBacktrackRecovery built for that trial).
class ExtBtScriptedAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    explicit ExtBtScriptedAcceptance(std::vector<std::pair<bool, double>> outcomes)
        : outcomes_(std::move(outcomes)) {}

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
                               Eigen::VectorXd &, Eigen::VectorXd &DXSL, Eigen::VectorXd &,
                               Eigen::VectorXd &, Eigen::VectorXd &, IterateInfo &Citer,
                               const std::vector<IterateInfo> &) override {
        recorded_dxsl_.push_back(DXSL);
        const auto &[accept, alpha_result] = outcomes_.at(static_cast<std::size_t>(calls_));
        ++calls_;
        Citer.accepted_ = accept;
        if (accept) {
            Citer.ls_iters_ = 7;
            Citer.merit_val_ = 42.0;
        }
        return alpha_result;
    }

    int calls_ = 0;
    std::vector<Eigen::VectorXd> recorded_dxsl_;

  private:
    std::vector<std::pair<bool, double>> outcomes_;
};

SolverContext extbt_dummy_context(KktSolverType &solver, PSIOPT::Settings &settings, int &zero,
                                  Eigen::VectorXd &scratch) {
    return SolverContext{nullptr, solver,  settings, zero,    zero,    zero,
                         zero,    zero,    scratch};
}

// ls_extended_iters_ == 0 (off): decline immediately, touching neither the
// acceptance nor the mechanism.
TEST(ExtendedBacktrackGuards, DisabledDeclines) {
    PSIOPT::Settings settings;
    settings.ls_extended_iters_ = 0;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = extbt_dummy_context(solver, settings, zero, scratch);
    ExtBtUnusedAcceptance acceptance;
    ExtBtPassThroughMechanism mechanism;
    IterateInfo citer;
    const std::vector<IterateInfo> iters;
    Eigen::VectorXd XSL(1), DXSL(1), XSL2(1), RHS(1), RHS2(1);
    DXSL << 1.0;
    double alpha = 0.5, alphap = 1.0, alphad = 1.0;
    int soc_steps = 0, resolved_depth = kRecoveryDepthUnresolved, watchdog_activations = 0;

    ExtendedBacktrackRecovery ext;
    const Action action = ext.on_step_rejected(
        citer, iters, ctx, acceptance, mechanism, PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0,
        0.0, XSL, DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
        watchdog_activations);
    EXPECT_EQ(action, Action::kAcceptAsIs);
    EXPECT_DOUBLE_EQ(alpha, 0.5); // untouched
}

// Ladder arithmetic: the first trial's scale is the LIVE alpha (continues
// the ladder -- does not restart at 1.0), and each subsequent trial's scale
// is the previous call's returned alpha times the previous scale. The cap
// (ls_extended_iters_) bounds the number of external calls.
TEST(ExtendedBacktrackLadder, ContinuesFromLiveAlphaAndHonorsCap) {
    PSIOPT::Settings settings;
    settings.ls_extended_iters_ = 3;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = extbt_dummy_context(solver, settings, zero, scratch);
    ExtBtPassThroughMechanism mechanism;
    IterateInfo citer;
    const std::vector<IterateInfo> iters;

    Eigen::VectorXd DXSL(3);
    DXSL << 1.0, 2.0, 3.0;
    Eigen::VectorXd XSL(3), XSL2(3), RHS(3), RHS2(3);
    const double live_alpha = 0.5;
    double alpha = live_alpha, alphap = 1.0, alphad = 1.0;
    int soc_steps = 0, resolved_depth = kRecoveryDepthUnresolved, watchdog_activations = 0;

    // Three rejections in a row (cap == 3): never accepted.
    ExtBtScriptedAcceptance acceptance({{false, 0.25}, {false, 0.5}, {false, 0.1}});

    ExtendedBacktrackRecovery ext;
    const Action action = ext.on_step_rejected(
        citer, iters, ctx, acceptance, mechanism, PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0,
        0.0, XSL, DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
        watchdog_activations);

    EXPECT_EQ(action, Action::kAcceptAsIs); // budget exhausted, no acceptance
    EXPECT_EQ(acceptance.calls_, 3);        // cap honored: exactly 3 external trials
    EXPECT_DOUBLE_EQ(alpha, live_alpha);    // untouched on kAcceptAsIs

    ASSERT_EQ(acceptance.recorded_dxsl_.size(), 3u);
    // Trial 1 scale == live_alpha (continues the ladder; does not restart at 1.0).
    EXPECT_DOUBLE_EQ(acceptance.recorded_dxsl_[0][0], live_alpha * 1.0);
    // Trial 2 scale == (trial 1's returned alpha) * (trial 1's scale).
    const double scale2 = 0.25 * live_alpha;
    EXPECT_DOUBLE_EQ(acceptance.recorded_dxsl_[1][0], scale2 * 1.0);
    // Trial 3 scale == (trial 2's returned alpha) * (trial 2's scale).
    const double scale3 = 0.5 * scale2;
    EXPECT_DOUBLE_EQ(acceptance.recorded_dxsl_[2][0], scale3 * 1.0);
}

// Acceptance mid-ladder stops immediately (does not spend the remaining
// budget) and commits the accepted scaled direction/alpha.
TEST(ExtendedBacktrackLadder, AcceptsAndStopsEarly) {
    PSIOPT::Settings settings;
    settings.ls_extended_iters_ = 5; // budget bigger than needed
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = extbt_dummy_context(solver, settings, zero, scratch);
    ExtBtPassThroughMechanism mechanism;
    IterateInfo citer;
    const std::vector<IterateInfo> iters;

    Eigen::VectorXd DXSL(2);
    DXSL << 4.0, 0.0;
    Eigen::VectorXd XSL(2), XSL2(2), RHS(2), RHS2(2);
    const double live_alpha = 0.8;
    double alpha = live_alpha, alphap = 1.0, alphad = 1.0;
    int soc_steps = 0, resolved_depth = kRecoveryDepthUnresolved, watchdog_activations = 0;

    ExtBtScriptedAcceptance acceptance({{false, 0.3}, {true, 0.2}});

    ExtendedBacktrackRecovery ext;
    const Action action = ext.on_step_rejected(
        citer, iters, ctx, acceptance, mechanism, PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0,
        0.0, XSL, DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
        watchdog_activations);

    EXPECT_EQ(action, Action::kRetry);
    EXPECT_EQ(acceptance.calls_, 2); // stopped as soon as accepted
    EXPECT_TRUE(citer.accepted_);
    EXPECT_EQ(citer.ls_iters_, 7);
    EXPECT_DOUBLE_EQ(citer.merit_val_, 42.0);
    EXPECT_DOUBLE_EQ(alpha, 0.2); // the accepted call's returned alpha

    const double scale2 = 0.3 * live_alpha;
    EXPECT_DOUBLE_EQ(DXSL[0], scale2 * 4.0); // committed direction == scale2 * original DXSL
    EXPECT_DOUBLE_EQ(DXSL[1], 0.0);
}

///////////////////////////////////////////////////////////////////////////////
// ChainedRecovery / WatchdogRecovery — composition wiring (spy links).
///////////////////////////////////////////////////////////////////////////////

// Inert AcceptanceStrategy / GlobalizationMechanism: the spy links below never
// touch them, so any call is a wiring bug.
class WatchdogUnusedAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "acceptance must not be reached by a spy recovery link";
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
};

class WatchdogUnusedMechanism : public GlobalizationMechanism {
  public:
    double compute_step(PSIOPT::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
                        Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                        AcceptanceStrategy &, double &, double &, IterateInfo &,
                        const std::vector<IterateInfo> &, SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached by a spy recovery link";
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached by a spy recovery link";
    }
    void reset() override {}
};

// Records invocation/reset counts and returns a configured Action, touching
// none of the working-set arguments (this suite exercises composition
// wiring, not any individual link's numeric policy).
class WatchdogSpyRecovery : public RecoveryChain {
  public:
    explicit WatchdogSpyRecovery(Action action) : action_(action) {}

    Action on_step_rejected(IterateInfo &, const std::vector<IterateInfo> &, SolverContext &,
                            AcceptanceStrategy &, GlobalizationMechanism &,
                            PSIOPT::LineSearchModes, double, double, double, double,
                            Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                            Eigen::VectorXd &, Eigen::VectorXd &, double &, double &, double &,
                            int &, int &, int &) override {
        ++calls_;
        return action_;
    }
    void notify_step_accepted() override { ++notify_accepted_calls_; }
    void reset() override { ++resets_; }

    int calls_ = 0;
    int resets_ = 0;
    int notify_accepted_calls_ = 0;

  private:
    Action action_;
};

SolverContext watchdog_dummy_context(KktSolverType &solver, PSIOPT::Settings &settings, int &zero,
                                     Eigen::VectorXd &scratch) {
    return SolverContext{nullptr, solver,  settings, zero,    zero,    zero,
                         zero,    zero,    scratch};
}

struct ChainDrive {
    IterateInfo citer;
    std::vector<IterateInfo> iters;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(1);
    double alpha = 1.0, alphap = 1.0, alphad = 1.0;
    int soc_steps = 0;
    int resolved_depth = kRecoveryDepthUnresolved;
    int watchdog_activations = 0;
};

Action drive_chain(RecoveryChain &chain, ChainDrive &d, SolverContext &ctx,
                   AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism) {
    return chain.on_step_rejected(d.citer, d.iters, ctx, acceptance, mechanism,
                                  PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0, 0.0, d.v, d.v,
                                  d.v, d.v, d.v, d.alpha, d.alphap, d.alphad, d.soc_steps,
                                  d.resolved_depth, d.watchdog_activations);
}

// SOC tried first: if the SOC-position link resolves (returns non-kAcceptAsIs),
// the extended-position link is never consulted, and resolved_depth == SOC.
TEST(ChainedRecoveryOrdering, SocResolvesFirst) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    auto soc_spy = std::make_unique<WatchdogSpyRecovery>(Action::kRetry);
    auto ext_spy = std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs);
    WatchdogSpyRecovery *soc_ptr = soc_spy.get();
    WatchdogSpyRecovery *ext_ptr = ext_spy.get();
    ChainedRecovery chain(std::move(soc_spy), std::move(ext_spy));

    ChainDrive d;
    const Action action = drive_chain(chain, d, ctx, acceptance, mechanism);

    EXPECT_EQ(action, Action::kRetry);
    EXPECT_EQ(soc_ptr->calls_, 1);
    EXPECT_EQ(ext_ptr->calls_, 0); // never consulted
    EXPECT_EQ(d.resolved_depth, kRecoveryDepthSoc);
}

// SOC declines (kAcceptAsIs): the extended-position link IS consulted, and if
// it resolves, resolved_depth == extended.
TEST(ChainedRecoveryOrdering, ExtendedResolvesWhenSocDeclines) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    auto soc_spy = std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs);
    auto ext_spy = std::make_unique<WatchdogSpyRecovery>(Action::kRetry);
    WatchdogSpyRecovery *soc_ptr = soc_spy.get();
    WatchdogSpyRecovery *ext_ptr = ext_spy.get();
    ChainedRecovery chain(std::move(soc_spy), std::move(ext_spy));

    ChainDrive d;
    const Action action = drive_chain(chain, d, ctx, acceptance, mechanism);

    EXPECT_EQ(action, Action::kRetry);
    EXPECT_EQ(soc_ptr->calls_, 1);
    EXPECT_EQ(ext_ptr->calls_, 1);
    EXPECT_EQ(d.resolved_depth, kRecoveryDepthExtended);
}

// Both links decline: unresolved (today's classic give-up).
TEST(ChainedRecoveryOrdering, BothDeclineIsUnresolved) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    ChainedRecovery chain(std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs),
                         std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs));

    ChainDrive d;
    const Action action = drive_chain(chain, d, ctx, acceptance, mechanism);

    EXPECT_EQ(action, Action::kAcceptAsIs);
    EXPECT_EQ(d.resolved_depth, kRecoveryDepthUnresolved);
}

// notify_step_accepted() propagates to BOTH links unconditionally (unlike
// on_step_rejected's short-circuit dispatch order) -- every link with
// per-solve state gets a chance to reset on real progress.
TEST(ChainedRecoveryOrdering, NotifyStepAcceptedPropagatesToBothLinks) {
    auto soc_spy = std::make_unique<WatchdogSpyRecovery>(Action::kRetry);
    auto ext_spy = std::make_unique<WatchdogSpyRecovery>(Action::kRetry);
    WatchdogSpyRecovery *soc_ptr = soc_spy.get();
    WatchdogSpyRecovery *ext_ptr = ext_spy.get();
    ChainedRecovery chain(std::move(soc_spy), std::move(ext_spy));

    chain.notify_step_accepted();

    EXPECT_EQ(soc_ptr->notify_accepted_calls_, 1);
    EXPECT_EQ(ext_ptr->notify_accepted_calls_, 1);
}

// Drives WatchdogRecovery once with a fixed mu (kept constant across the
// whole test so only `merit` = prim_obj + barr_obj varies) and a fresh
// per-call working vector `v` bound to all five KKT-layout positions (this
// suite exercises the decorator's own state machine / snapshot handling, not
// any real KKT layout). `citer`/`resolved_depth`/`watchdog_activations` are
// caller-owned so the test can inspect them after the call.
Action drive_watchdog(WatchdogRecovery &watchdog, SolverContext &ctx,
                     AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism,
                     Eigen::VectorXd &v, IterateInfo &citer, double mu, double prim_obj,
                     double &alpha, int &resolved_depth, int &watchdog_activations) {
    std::vector<IterateInfo> iters;
    double alphap = 1.0, alphad = 1.0;
    int soc_steps = 0;
    return watchdog.on_step_rejected(citer, iters, ctx, acceptance, mechanism,
                                     PSIOPT::LineSearchModes::AUGLANG, /*obj_scale=*/1.0, mu,
                                     prim_obj, /*barr_obj=*/0.0, v, v, v, v, v, alpha, alphap,
                                     alphad, soc_steps, resolved_depth, watchdog_activations);
}

// WatchdogRecovery as an outer decorator: while not armed, delegates every
// rejection through to the wrapped chain unchanged.
TEST(WatchdogRecoveryDecorator, DelegatesWhileNotArmed) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    auto inner = std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs);
    WatchdogSpyRecovery *inner_ptr = inner.get();
    WatchdogRecovery watchdog(std::move(inner));

    int watchdog_activations = 0;
    for (int i = 0; i < kWatchdogShortenedIterTrigger - 1; ++i) {
        Eigen::VectorXd v(1);
        v << 1.0;
        IterateInfo citer;
        double alpha = 1.0;
        int resolved_depth = kRecoveryDepthUnresolved;
        const Action action = drive_watchdog(watchdog, ctx, acceptance, mechanism, v, citer,
                                             /*mu=*/1.0, /*prim_obj=*/10.0, alpha, resolved_depth,
                                             watchdog_activations);
        EXPECT_EQ(action, Action::kAcceptAsIs);
    }
    EXPECT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1);
    EXPECT_EQ(watchdog_activations, 0);
}

// Once armed, the watchdog overrides with a relaxed accept and never
// consults the wrapped chain; on window exhaustion it reverts XSL to the
// snapshot in place and increments watchdog_activations exactly once (at the
// arm event, not on every subsequent trial).
TEST(WatchdogRecoveryDecorator, ArmsRelaxesThenReverts) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    auto inner = std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs);
    WatchdogSpyRecovery *inner_ptr = inner.get();
    WatchdogRecovery watchdog(std::move(inner));

    constexpr double kMu = 1.0;
    constexpr double kSnapshotValue = 100.0;
    int watchdog_activations = 0;

    // kWatchdogShortenedIterTrigger - 1 plain rejections (not armed yet;
    // consulted through to inner each time). merit == prim_obj (barr_obj ==
    // 0) stays constant throughout so only the arming COUNT drives outcomes.
    for (int i = 0; i < kWatchdogShortenedIterTrigger - 1; ++i) {
        Eigen::VectorXd v(1);
        v << kSnapshotValue;
        IterateInfo citer;
        double alpha = 1.0;
        int resolved_depth = kRecoveryDepthUnresolved;
        drive_watchdog(watchdog, ctx, acceptance, mechanism, v, citer, kMu, /*prim_obj=*/10.0,
                      alpha, resolved_depth, watchdog_activations);
    }
    ASSERT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1);

    // Arming call (trial #1): XSL snapshotted at its CURRENT value (kSnapshotValue);
    // relaxed accept, inner not consulted, one activation recorded.
    Eigen::VectorXd v1(1);
    v1 << kSnapshotValue;
    IterateInfo citer1;
    double alpha1 = 1.0;
    int resolved_depth1 = kRecoveryDepthUnresolved;
    Action action = drive_watchdog(watchdog, ctx, acceptance, mechanism, v1, citer1, kMu,
                                   /*prim_obj=*/10.0, alpha1, resolved_depth1,
                                   watchdog_activations);
    EXPECT_EQ(action, Action::kAcceptAsIs);
    EXPECT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1); // still not consulted
    EXPECT_EQ(resolved_depth1, kRecoveryDepthWatchdog);
    EXPECT_EQ(watchdog_activations, 1);
    EXPECT_TRUE(citer1.accepted_);

    // Trial #2: merit unchanged (no progress) -> still relaxed, no 2nd activation.
    Eigen::VectorXd v2(1);
    v2 << kSnapshotValue;
    IterateInfo citer2;
    double alpha2 = 1.0;
    int resolved_depth2 = kRecoveryDepthUnresolved;
    action = drive_watchdog(watchdog, ctx, acceptance, mechanism, v2, citer2, kMu,
                            /*prim_obj=*/10.0, alpha2, resolved_depth2, watchdog_activations);
    EXPECT_EQ(action, Action::kAcceptAsIs);
    EXPECT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1); // still bypassed
    EXPECT_EQ(watchdog_activations, 1);                              // unchanged

    // Trial #3: window exhausted, still no progress -> revert.
    static_assert(kWatchdogTrialIterMax == 3, "test assumes the paper's 3-trial window");
    Eigen::VectorXd v3(1);
    v3 << 999.0; // the "about to be applied" value the revert must discard
    IterateInfo citer3;
    double alpha3 = 1.0;
    int resolved_depth3 = kRecoveryDepthUnresolved;
    action = drive_watchdog(watchdog, ctx, acceptance, mechanism, v3, citer3, kMu,
                            /*prim_obj=*/10.0, alpha3, resolved_depth3, watchdog_activations);
    EXPECT_EQ(action, Action::kRetry);
    EXPECT_DOUBLE_EQ(v3[0], kSnapshotValue); // XSL reverted to the arm-time snapshot
    EXPECT_DOUBLE_EQ(alpha3, 0.0);           // no-op commit
    EXPECT_EQ(resolved_depth3, kRecoveryDepthWatchdog);
    EXPECT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1); // never consulted
    EXPECT_EQ(watchdog_activations, 1); // one activation for the whole episode
}

// A trial iterate whose merit beats the snapshot reference disarms and hands
// the rejection back to the wrapped chain — the "emergency is over" path.
TEST(WatchdogRecoveryDecorator, ProgressDisarmsAndDelegatesBack) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    auto inner = std::make_unique<WatchdogSpyRecovery>(Action::kRetry);
    WatchdogSpyRecovery *inner_ptr = inner.get();
    WatchdogRecovery watchdog(std::move(inner));

    constexpr double kMu = 1.0;
    int watchdog_activations = 0;

    for (int i = 0; i < kWatchdogShortenedIterTrigger; ++i) { // arms on the last of these
        Eigen::VectorXd v(1);
        v << 1.0;
        IterateInfo citer;
        double alpha = 1.0;
        int resolved_depth = kRecoveryDepthUnresolved;
        drive_watchdog(watchdog, ctx, acceptance, mechanism, v, citer, kMu, /*prim_obj=*/10.0,
                      alpha, resolved_depth, watchdog_activations);
    }
    ASSERT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1); // arming call bypassed inner
    ASSERT_EQ(watchdog_activations, 1);

    // Progress: prim_obj (== merit) strictly below the snapshot's 10.0 ->
    // disarm and delegate to inner (which now DOES get consulted).
    Eigen::VectorXd v(1);
    v << 1.0;
    IterateInfo citer;
    double alpha = 1.0;
    int resolved_depth = kRecoveryDepthUnresolved;
    const Action action = drive_watchdog(watchdog, ctx, acceptance, mechanism, v, citer, kMu,
                                         /*prim_obj=*/5.0, alpha, resolved_depth,
                                         watchdog_activations);
    EXPECT_EQ(action, Action::kRetry); // inner's configured return value
    EXPECT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger); // consulted this time
    EXPECT_EQ(watchdog_activations, 1);                          // no re-arm from this call
}

// notify_step_accepted() resets the arming counter AND threads through to
// inner_: [9 rejections, notify accepted, 1 rejection] must NOT arm (mirrors
// the WatchdogStateArm-level pin, driven this time through the decorator so
// the inner-delegation wiring is covered too).
TEST(WatchdogRecoveryDecorator, NotifyStepAcceptedResetsArmingCounter) {
    PSIOPT::Settings settings;
    KktSolverType solver;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = watchdog_dummy_context(solver, settings, zero, scratch);
    WatchdogUnusedAcceptance acceptance;
    WatchdogUnusedMechanism mechanism;

    auto inner = std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs);
    WatchdogSpyRecovery *inner_ptr = inner.get();
    WatchdogRecovery watchdog(std::move(inner));

    constexpr double kMu = 1.0;
    int watchdog_activations = 0;

    static_assert(kWatchdogShortenedIterTrigger == 10,
                 "test assumes the paper's 10-rejection trigger");
    for (int i = 0; i < kWatchdogShortenedIterTrigger - 1; ++i) {
        Eigen::VectorXd v(1);
        v << 1.0;
        IterateInfo citer;
        double alpha = 1.0;
        int resolved_depth = kRecoveryDepthUnresolved;
        drive_watchdog(watchdog, ctx, acceptance, mechanism, v, citer, kMu, /*prim_obj=*/10.0,
                      alpha, resolved_depth, watchdog_activations);
    }
    ASSERT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger - 1);
    ASSERT_EQ(watchdog_activations, 0); // not yet armed

    watchdog.notify_step_accepted();
    EXPECT_EQ(inner_ptr->notify_accepted_calls_, 1); // threaded through to inner_

    // One more rejection: with the counter reset by the notify above, this is
    // the 1st of a fresh count (kAccumulate, delegates to inner), NOT the
    // 10th overall (kArmed, which would bypass inner and revert on the
    // NEXT two calls instead).
    Eigen::VectorXd v(1);
    v << 1.0;
    IterateInfo citer;
    double alpha = 1.0;
    int resolved_depth = kRecoveryDepthUnresolved;
    const Action action = drive_watchdog(watchdog, ctx, acceptance, mechanism, v, citer, kMu,
                                         /*prim_obj=*/10.0, alpha, resolved_depth,
                                         watchdog_activations);
    EXPECT_EQ(action, Action::kAcceptAsIs); // inner's configured return value
    EXPECT_EQ(inner_ptr->calls_, kWatchdogShortenedIterTrigger); // consulted, not bypassed
    EXPECT_EQ(watchdog_activations, 0);                          // still not armed
}

// reset() propagates to the wrapped chain (behavior-neutral for a stateless
// inner, but observable through the spy's reset counter).
TEST(WatchdogRecoveryDecorator, ResetPropagatesToInner) {
    auto inner = std::make_unique<WatchdogSpyRecovery>(Action::kAcceptAsIs);
    WatchdogSpyRecovery *inner_ptr = inner.get();
    WatchdogRecovery watchdog(std::move(inner));
    watchdog.reset();
    EXPECT_EQ(inner_ptr->resets_, 1);
}

// The decorator requires a wrapped chain: constructing it without one is a
// caller bug and must throw rather than defer a null dereference to solve time.
TEST(WatchdogRecoveryDecorator, NullInnerThrowsAtConstruction) {
    EXPECT_THROW(WatchdogRecovery(nullptr), std::invalid_argument);
}

} // namespace
