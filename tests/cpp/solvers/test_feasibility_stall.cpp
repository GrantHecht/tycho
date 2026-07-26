///////////////////////////////////////////////////////////////////////////////
// FeasibilityStallDetector truth table: the windowed sustained-worsening signal
// that lets a zero-objective feasibility stage dispatch restoration.
//
// Two layers:
//   1. The detector's own truth table (isolated).
//   2. Through-the-public-API solves of a feasibility-only (SOE) stage: a
//      worsening stage dispatches restoration when a strategy is configured, a
//      plateaued stage never does, the default (restoration off) path leaves
//      the diagnostics at their sentinels, and a healthy stage never trips the
//      detector.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/feasibility_stall.h"

#include "solver_test_utils.h"

#include "tycho/detail/solvers/optimization_problem.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <limits>
#include <memory>

using namespace TychoTest;

namespace ts = tycho::solvers;

TEST(FeasStallDetector, FirstObservationSetsBaselineAndNeverFires) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    EXPECT_EQ(d.best_theta_, 1.0);
    EXPECT_EQ(d.iters_elevated_, 0);
}

// A full window of consecutive observations at or above the elevation mark —
// and nothing shorter — declares the stage worsening.
TEST(FeasStallDetector, FiresAfterAWindowOfSustainedElevation) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0)); // baseline: best-seen violation
    const double elevated = 1.3;  // above 1.25x the best
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(elevated)) << "premature fire at " << i;
    EXPECT_TRUE(d.observe(elevated)); // the kFeasStallWindow-th elevated observation
    EXPECT_EQ(d.best_theta_, 1.0);    // best-seen never moved
}

// Compounding growth, as in the recorded stall trace: the violation crosses the
// elevation mark part-way through and then holds a full window above it.
TEST(FeasStallDetector, GrowingViolationCountsAsStalled) {
    ts::FeasibilityStallDetector d;
    double theta = 1.0;
    EXPECT_FALSE(d.observe(theta));
    // 1.05^n first reaches 1.25 at n = 5, so the run of elevated observations
    // starts there and the window closes kFeasStallWindow observations later.
    const int first_elevated = 5;
    const int fire_at = first_elevated + ts::kFeasStallWindow - 1;
    for (int i = 1; i <= fire_at; ++i) {
        theta *= 1.05;
        EXPECT_EQ(d.observe(theta), i == fire_at) << "wrong verdict at " << i;
    }
    EXPECT_EQ(d.best_theta_, 1.0); // best-seen never moved
}

// One observation back at or below the elevation mark breaks the run, and the
// count starts over from the next elevated one.
TEST(FeasStallDetector, DipBelowTheElevationMarkRestartsWindow) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.3));
    EXPECT_FALSE(d.observe(0.9)); // a new low, one observation short of firing
    EXPECT_EQ(d.best_theta_, 0.9);
    // The mark now sits at 1.25 * 0.9, and a fresh full window is required.
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.3)) << "window did not restart at " << i;
    EXPECT_TRUE(d.observe(1.3));
}

// Elevation is what accumulates the window, not failure to improve: an
// observation WORSE than the best but under the mark still breaks the run.
TEST(FeasStallDetector, NonImprovingObservationUnderTheMarkRestartsWindow) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.3));
    EXPECT_FALSE(d.observe(1.1)); // worse than the best, but below 1.25x it
    EXPECT_EQ(d.best_theta_, 1.0);
    EXPECT_EQ(d.iters_elevated_, 0);
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.3)) << "window did not restart at " << i;
    EXPECT_TRUE(d.observe(1.3));
}

// A stage sitting exactly at its best is a plateau, not a worsening stage, and
// must never dispatch however long it sits there.
TEST(FeasStallDetector, ExactlyFlatPlateauNeverFires) {
    ts::FeasibilityStallDetector d;
    for (int i = 0; i < 200; ++i)
        EXPECT_FALSE(d.observe(1.0)) << "fired on a flat plateau at " << i;
    EXPECT_EQ(d.best_theta_, 1.0);
}

// A productive crawl, however slow, keeps setting a new best, so it is never
// elevated against it and the detector never fires.
TEST(FeasStallDetector, SustainedProductiveCrawlNeverFires) {
    ts::FeasibilityStallDetector d;
    double theta = 1.0;
    for (int i = 0; i < 200; ++i) {
        theta *= 1.0 - 1.0e-6;
        EXPECT_FALSE(d.observe(theta)) << "fired on a productive crawl at " << i;
    }
}

// Re-arming after a dispatched episode must not forget where recovery last
// handed the stage back: that value is the reference the caller compares
// against to decide whether the stage has gained any ground since.
TEST(FeasStallDetector, ResetWindowPreservesLastDispatchViolation) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    d.note_dispatch(1.0);
    for (int i = 0; i < ts::kFeasStallWindow / 2; ++i)
        d.observe(1.3); // part-way through a window of elevated observations
    ASSERT_GT(d.iters_elevated_, 0);
    d.note_dispatch(0.4);
    d.reset_window();
    EXPECT_EQ(d.iters_elevated_, 0);
    EXPECT_EQ(d.best_theta_, std::numeric_limits<double>::infinity());
    EXPECT_EQ(d.theta_at_last_dispatch_, 0.4);
}

// The reference tracks the MOST RECENT dispatch: the caller's question is
// whether the stage has gained anything since recovery last handed it back.
TEST(FeasStallDetector, NoteDispatchRecordsTheLatestEntry) {
    ts::FeasibilityStallDetector d;
    EXPECT_EQ(d.theta_at_last_dispatch_, std::numeric_limits<double>::infinity());
    d.note_dispatch(2.0);
    EXPECT_EQ(d.theta_at_last_dispatch_, 2.0);
    d.note_dispatch(0.5); // every later episode moves the reference
    EXPECT_EQ(d.theta_at_last_dispatch_, 0.5);
    d.note_dispatch(9.0);
    EXPECT_EQ(d.theta_at_last_dispatch_, 9.0);
}

// -----------------------------------------------------------------------------
// Composition: the detector wired into alg_impl's feasibility-only stage.
// -----------------------------------------------------------------------------

namespace {

// Inconsistent equalities (x - 1 = 0 AND x - 2 = 0): the feasibility system
// has no root, so a solve() stage walks its violation DOWN to the least-squares
// point and plateaus there — every step accepted, and from then on no progress
// and no worsening either. That plateau is the case the detector deliberately
// leaves alone. One variable, start at 0.
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
    // Single-threaded factorization, belt and braces: the 25% elevation margin
    // already puts the dispatch schedule some eleven orders of magnitude clear
    // of threaded-Pardiso FP jitter, so this only removes run-to-run drift from
    // the iteration counts the assertions below quote.
    prob->optimizer_->set_qp_threads(1);
    return prob;
}

// A feasibility system with no solution whose stage worsens monotonically. The
// single equality is (x^2 + 1)^(1/4) = 0, whose left side never drops below 1,
// so the stage can never be satisfied. With one variable and one equality the
// stage's step is the exact Newton step for that residual,
// x <- x - c/c' = -(x^2 + 2)/x, which raises x^2 by more than 4 every
// iteration and the violation with it: a deterministic, monotonically growing
// violation that stays comfortably inside double range for thousands of
// iterations (it grows like the fourth root of the iteration count). Starting
// at x = 1 the violation is 1.189, and the very first step takes it to 1.778 —
// past the 1.25x elevation mark, which it then clears forever.
std::unique_ptr<ts::OptimizationProblem> feas_stall_build_worsening_nlp() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<ts::OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 1.0));
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
    }
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(sqrt(sqrt(x * x + 1.0))),
                            (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3); // fully silent
    prob->optimizer_->set_qp_threads(1);  // as above
    return prob;
}

} // namespace

// A worsening feasibility stage with restoration configured dispatches at least
// one restoration episode instead of silently burning the iteration budget.
TEST_F(SolverTest, FeasStallStageDispatchesProximalRestoration) {
    auto prob = feas_stall_build_worsening_nlp();
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
    auto prob = feas_stall_build_worsening_nlp();
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::proximal_switch;
    prob->optimizer_->settings().acceptance_strategy_ = ts::AcceptanceStrategies::filter;
    prob->optimizer_->settings().barrier_governor_ = ts::BarrierGovernors::monitored;
    prob->optimizer_->set_max_iters(200);
    ASSERT_NO_THROW(prob->solve());
    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1);
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED); // genuinely infeasible
}

// Once the per-phase restoration entry budget is spent, a still-worsening
// feasibility stage that has gained no ground since its LAST restoration entry
// has nothing left to consult, so it must stop rather than burn the remaining
// iteration budget. This fixture's violation only ever grows, so each episode
// hands the stage back to a violation that climbs straight past the value
// recorded at that entry: there is no net progress to protect and the stage
// ends early with the honest verdict. In a multi-phase sequence the next phase
// resumes from this point.
//
// Uses the nested l1 mode because its episodes run to completion and hand the
// stage back, which is what lets the stage spend both entries and then worsen a
// third time with the budget gone. (Under proximal_switch an episode on a
// zero-objective stage need never satisfy its own exit test, in which case the
// budget is never exhausted.)
TEST_F(SolverTest, FeasStallStageStopsBurningAfterBudgetExhaustion) {
    auto prob = feas_stall_build_worsening_nlp();
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::l1_nested;
    prob->optimizer_->set_max_iters(400);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, 2); // default max_feas_rest_ fully used
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
    // Three elevation windows plus two restoration episodes plus generous slack
    // is well under this bound; the stage ends around 165 iterations, where
    // without the exit it would run out the full 400.
    EXPECT_LT(r.iter_num_, 300);
}

TEST_F(SolverTest, FeasStallStageDispatchesNestedRestoration) {
    auto prob = feas_stall_build_worsening_nlp();
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::l1_nested;
    prob->optimizer_->set_max_iters(200);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1);
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
}

// The deliberate narrowing: a stage that plateaus at its own best is not
// worsening, so it never dispatches an episode however long it sits there. It
// runs its iteration budget out exactly as it would with restoration off —
// which is the point, since the only measured value of a dispatched episode is
// on a stage that is getting worse.
TEST_F(SolverTest, FeasStallPlateauedStageNeverDispatches) {
    auto prob = feas_stall_build_nlp(/*inconsistent=*/true);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::l1_nested;
    prob->optimizer_->set_max_iters(200);
    prob->solve();
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, 0); // strategy built, never entered
    EXPECT_NE(r.converge_flag_, ts::PSIOPT::ConvergenceFlags::CONVERGED);
    EXPECT_GE(r.iter_num_, 190); // ran to its cap
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
