///////////////////////////////////////////////////////////////////////////////
// PSIOPT convergence tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

TEST_F(SolverTest, BrachistochroneEndToEnd) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf_val = 10.0, yf_val = 5.0;

    int n_pts = 100;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf_val - x0) * s;
        pt[1] = y0 + (yf_val - y0) * s;
        pt[2] = g * s * 1.0 * std::cos(1.0);
        pt[3] = t0 + 1.0 * s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, 32);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf_val, yf_val;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);
    phase->optimizer->PrintLevel = 0;

    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    auto result = phase->returnTraj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(SolverTest, BrachistochroneSolveOnly) {
    // Solve (feasibility) without optimization
    constexpr double g = 9.81;

    int n_pts = 50;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = 10.0 * s;
        pt[1] = 10.0 - 5.0 * s;
        pt[2] = g * s * 1.0 * std::cos(1.0);
        pt[3] = 1.0 * s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, 16);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << 0.0, 10.0, 0.0, 0.0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << 10.0, 5.0;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);
    phase->optimizer->PrintLevel = 0;

    // solve() only finds feasibility
    auto status = phase->solve();
    // Should converge (feasible) -- Brachistochrone is well-posed
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
    phase->optimizer->PrintLevel = 0;
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);
}
