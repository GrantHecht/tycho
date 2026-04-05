///////////////////////////////////////////////////////////////////////////////
// PSIOPT convergence tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>
#include <limits>

using namespace tycho;
using namespace TychoTest;

TEST_F(SolverTest, BrachistochroneEndToEnd) {
    auto phase = make_brach_solver_phase(32);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    auto result = phase->return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(SolverTest, BrachistochroneSolveOnly) {
    auto phase = make_brach_solver_phase(16);
    // solve() only finds feasibility
    auto status = phase->solve();
    // Should converge (feasible) -- Brachistochrone is well-posed.
    // ConvergenceFlags enum is ordered by severity: CONVERGED < ACCEPTABLE < NOTCONVERGED <
    // DIVERGING, so <= ACCEPTABLE accepts either CONVERGED or ACCEPTABLE.
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
}

TEST_F(SolverTest, ConvergenceFlagOrdering) {
    // Verify the severity ordering of convergence flags
    EXPECT_LT(PSIOPT::ConvergenceFlags::CONVERGED, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    EXPECT_LT(PSIOPT::ConvergenceFlags::ACCEPTABLE, PSIOPT::ConvergenceFlags::NOTCONVERGED);
    EXPECT_LT(PSIOPT::ConvergenceFlags::NOTCONVERGED, PSIOPT::ConvergenceFlags::DIVERGING);
}

TEST_F(SolverTest, PrintLevelZeroSilent) {
    auto phase = make_brach_solver_phase(16);
    phase->optimizer_->set_print_level(0);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);
}

// =============================================================================
// Setter validation tests
// =============================================================================

TEST_F(SolverTest, ToleranceSetterRejectsInvalid) {
    PSIOPT opt;
    // Negative
    EXPECT_THROW(opt.set_kkt_tol(-1.0), std::invalid_argument);
    // Zero
    EXPECT_THROW(opt.set_kkt_tol(0.0), std::invalid_argument);
    // NaN
    EXPECT_THROW(opt.set_kkt_tol(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    // Inf
    EXPECT_THROW(opt.set_kkt_tol(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    // Valid
    EXPECT_NO_THROW(opt.set_kkt_tol(1e-8));

    // One representative from each group
    EXPECT_THROW(opt.set_acc_econ_tol(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_acc_econ_tol(1e-4));
    EXPECT_THROW(opt.set_div_bar_tol(0.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_div_bar_tol(1e6));
}

TEST_F(SolverTest, MuSettersRejectInvalid) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_init_mu(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_init_mu(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_init_mu(0.01));
    EXPECT_THROW(opt.set_min_mu(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_max_mu(-1.0), std::invalid_argument);
}

TEST_F(SolverTest, QpThreadsRejectsInvalid) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_qp_threads(0), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_threads(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_threads(2));
}

TEST_F(SolverTest, ObjScaleRejectsZero) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_obj_scale(0.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_obj_scale(-1.0)); // negative = maximization
    EXPECT_NO_THROW(opt.set_obj_scale(2.0));
}

TEST_F(SolverTest, BoundFractionRejectsOutOfRange) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_bound_fraction(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_bound_fraction(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_bound_fraction(0.5));
}

TEST_F(SolverTest, AlphaRedRejectsBelowOne) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_alpha_red(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_alpha_red(2.0));
}

TEST_F(SolverTest, HpertParamsValidation) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_delta_h(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_incr_h(1.0), std::invalid_argument);
    EXPECT_THROW(opt.set_decr_h(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_decr_h(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_hpert_params(1e-5, 8.0, 0.333));
}

// =============================================================================
// run_phase_sequence guard tests
// =============================================================================

TEST_F(SolverTest, OptimizeThrowsWithoutNlp) {
    PSIOPT opt;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(10);
    EXPECT_THROW(opt.optimize(x), std::runtime_error);
}

TEST_F(SolverTest, OptimizeThrowsOnSizeMismatch) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false); // set up NLP without solving
    Eigen::VectorXd wrong_size = Eigen::VectorXd::Zero(3);
    EXPECT_THROW(phase->optimizer_->optimize(wrong_size), std::invalid_argument);
}

// =============================================================================
// Result accessor tests
// =============================================================================

TEST_F(SolverTest, ResultAccessorPopulatedAfterSolve) {
    auto phase = make_brach_solver_phase(32);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    const auto &r = phase->optimizer_->result();
    EXPECT_GT(r.iter_num_, 0);
    EXPECT_GT(r.obj_val_, 0.0);
    EXPECT_GT(r.total_time_, 0.0);
    EXPECT_GT(r.primals_.size(), 0);
}
