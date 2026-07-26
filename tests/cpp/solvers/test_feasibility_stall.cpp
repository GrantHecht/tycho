///////////////////////////////////////////////////////////////////////////////
// FeasibilityStallDetector truth table: the windowed no-progress signal that
// lets a zero-objective feasibility stage dispatch restoration.
//
// Two layers:
//   1. The detector's own truth table (isolated).
//   2. Through-the-public-API solves of a feasibility-only (SOE) stage: a
//      stalled stage dispatches restoration when a strategy is configured, the
//      default (restoration off) path leaves the diagnostics at their
//      sentinels, and a healthy stage never trips the detector.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/feasibility_stall.h"

#include "solver_test_utils.h"

#include "tycho/detail/solvers/optimization_problem.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <memory>

using namespace TychoTest;

namespace ts = tycho::solvers;

TEST(FeasStallDetector, FirstObservationSetsBaselineAndNeverFires) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    EXPECT_EQ(d.best_theta_, 1.0);
    EXPECT_EQ(d.iters_without_improvement_, 0);
}

TEST(FeasStallDetector, FiresAfterWindowWithoutImprovement) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.0)) << "premature fire at " << i;
    EXPECT_TRUE(d.observe(1.0)); // the kFeasStallWindow-th stalled observation
}

TEST(FeasStallDetector, GrowingViolationCountsAsStalled) {
    ts::FeasibilityStallDetector d;
    double theta = 1.0;
    EXPECT_FALSE(d.observe(theta));
    bool fired = false;
    for (int i = 0; i < ts::kFeasStallWindow; ++i) {
        theta *= 1.05; // worsening, as in the recorded stall trace
        fired = d.observe(theta);
    }
    EXPECT_TRUE(fired);
    EXPECT_EQ(d.best_theta_, 1.0); // best-seen never moved
}

TEST(FeasStallDetector, SufficientImprovementRestartsWindow) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.0));
    // One >1% improvement of the best resets the counter entirely.
    EXPECT_FALSE(d.observe(0.9));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(0.9)) << "window did not restart at " << i;
    EXPECT_TRUE(d.observe(0.9));
}

TEST(FeasStallDetector, SubThresholdImprovementDoesNotRestartWindow) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    bool fired = false;
    for (int i = 0; i < ts::kFeasStallWindow; ++i)
        fired = d.observe(0.995); // 0.5% better: below the 1% threshold
    EXPECT_TRUE(fired);
}

TEST(FeasStallDetector, ResetReArmsCompletely) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 0; i < ts::kFeasStallWindow; ++i)
        d.observe(1.0);
    d.reset();
    EXPECT_EQ(d.iters_without_improvement_, 0);
    EXPECT_FALSE(d.observe(5.0)); // any value is a fresh baseline after reset
    EXPECT_EQ(d.best_theta_, 5.0);
}

// -----------------------------------------------------------------------------
// Composition: the detector wired into alg_impl's feasibility-only stage.
// -----------------------------------------------------------------------------

namespace {

// Inconsistent equalities (x - 1 = 0 AND x - 2 = 0): the feasibility system
// has no root, so a solve() stage settles near the least-squares point and
// plateaus — every step accepted, no progress, exactly the stall the detector
// exists for. One variable, start at 0.
std::unique_ptr<ts::OptimizationProblem> feas_stall_build_nlp(bool inconsistent) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<ts::OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 0.0));
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
    }
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
    }
    if (inconsistent) {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - 2.0), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3); // fully silent
    return prob;
}

} // namespace

// A stalled feasibility stage with restoration configured dispatches at least
// one restoration episode instead of silently burning the iteration budget.
TEST_F(SolverTest, FeasStallStageDispatchesProximalRestoration) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/true);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::proximal_switch;
    prob->optimizer_->set_max_iters(200);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1);
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED); // genuinely infeasible
}

// Same dispatch, but under the filter acceptance strategy: the entry call
// now runs a live notify_switch_to_feasibility handshake (a no-op under the
// default classic_merit) instead of the trivial base-class stub. Pins that
// the stall-dispatch seam feeds FilterAcceptance a consistent phase state
// (a throw here would mean the new entry path double-enters the feasibility
// phase or otherwise desyncs from the filter's own bookkeeping).
TEST_F(SolverTest, FeasStallDispatchUnderFilterAcceptanceHandshakes) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/true);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::proximal_switch;
    prob->optimizer_->settings().acceptance_strategy_ = ts::AcceptanceStrategies::filter;
    prob->optimizer_->settings().barrier_governor_ = ts::BarrierGovernors::monitored;
    prob->optimizer_->set_max_iters(200);
    ASSERT_NO_THROW(prob->solve());
    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1);
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED); // genuinely infeasible
}

// Once the per-phase restoration entry budget is spent, a still-stalled
// feasibility stage has nothing left to consult, so it must stop rather than
// burn the remaining iteration budget. The stage ends early with the honest
// verdict; in a multi-phase sequence the next phase resumes from this point.
//
// Uses the nested l1 mode because its episodes run to completion and hand the
// stage back, which is what lets the stage spend both entries and then stall a
// third time with the budget gone. (Under proximal_switch this same problem
// never gets that far: the first episode is entered and, on a zero-objective
// stage, never satisfies its own exit test, so the budget is never exhausted.)
TEST_F(SolverTest, FeasStallStageStopsBurningAfterBudgetExhaustion) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/true);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::l1_nested;
    prob->optimizer_->set_max_iters(400);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, 2); // default max_feas_rest_ fully used
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
    // Two stall windows plus two short restoration episodes plus slack is well
    // under this bound; the pre-change behaviour ran out the full 400.
    EXPECT_LT(r.iter_num_, 200);
}

TEST_F(SolverTest, FeasStallStageDispatchesNestedRestoration) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/true);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::l1_nested;
    prob->optimizer_->set_max_iters(200);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1);
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
}

// Default path untouched: restoration off leaves the sentinel diagnostics and
// the stage burns its budget exactly as before.
TEST_F(SolverTest, FeasStallStageOffModeKeepsSentinels) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/true);
    prob->optimizer_->set_max_iters(200);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, -1);
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
}

// A healthy feasibility stage must never trip the detector: a consistent
// problem converges with restoration configured and zero entries.
TEST_F(SolverTest, FeasStallHealthyStageNeverDispatches) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/false);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::proximal_switch;
    prob->optimizer_->set_max_iters(200);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
    EXPECT_EQ(r.last_feas_rest_entries_, 0); // strategy built, never entered
}
