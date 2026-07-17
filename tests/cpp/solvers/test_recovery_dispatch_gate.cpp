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
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::IterateInfo;
using tycho::solvers::KktSolverType;
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
    explicit StubAcceptance(bool accept) : accept_(accept) {}

    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double) override {
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
// gate fires it exactly when expected. Returns kAcceptAsIs (today's only wired
// behavior) and touches none of its arguments.
class RecordingRecovery : public RecoveryChain {
  public:
    Action on_step_rejected(IterateInfo &, const std::vector<IterateInfo> &,
                            SolverContext &) override {
        ++calls_;
        return Action::kAcceptAsIs;
    }
    void reset() override {}

    int calls_ = 0;
};

// Faithful replica of alg_impl's recovery-hook wiring: dispatch only when the
// shared gate predicate says so.
void drive_gate(bool good_step, IterateInfo &citer, RecoveryChain &recovery,
                const std::vector<IterateInfo> &iters, SolverContext &ctx) {
    if (should_dispatch_recovery(good_step, citer)) {
        recovery.on_step_rejected(citer, iters, ctx);
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
        drive_gate(/*good_step=*/true, citer, recovery, iters, ctx);
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
        drive_gate(/*good_step=*/true, citer, recovery, iters, ctx);
        EXPECT_EQ(recovery.calls_, 0);
    }

    // Non-finite step direction (!GoodStep): alg_impl runs no line search, so
    // Citer keeps its fresh default (accepted_ == false); the gate stays silent
    // because GoodStep is false.
    {
        RecordingRecovery recovery;
        IterateInfo citer; // no line search ran
        EXPECT_FALSE(citer.accepted_);
        drive_gate(/*good_step=*/false, citer, recovery, iters, ctx);
        EXPECT_EQ(recovery.calls_, 0);
    }
}

} // namespace
