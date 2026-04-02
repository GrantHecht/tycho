///////////////////////////////////////////////////////////////////////////////
// PSIOPT convergence tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

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
    phase->optimizer_->print_level_ = 0;
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);
}
