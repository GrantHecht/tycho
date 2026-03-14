///////////////////////////////////////////////////////////////////////////////
// Multi-phase OCP and solver integration tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

TEST_F(OptimalControlTest, TwoPhaseOCPConstruct) {
    auto phase1 = make_brach_phase(50, 16);
    auto phase2 = make_brach_phase(50, 16);

    OptimalControlProblem ocp;
    ocp.addPhase(phase1);
    ocp.addPhase(phase2);
    // Should construct without error
    SUCCEED();
}

TEST_F(OptimalControlTest, BrachistochroneSolveOptimize) {
    auto phase = make_brach_phase(100, 32);
    phase->optimizer->PrintLevel = 0;

    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED)
        << "Brachistochrone should converge to optimal";

    auto result = phase->returnTraj();
    double tf = result.back()[3];
    // Known optimal time ~1.8013 s
    EXPECT_NEAR(tf, 1.8013, 0.01) << "Optimal time should be near 1.8013 s, got " << tf;
}

TEST_F(OptimalControlTest, BrachistochroneFinalBoundary) {
    auto phase = make_brach_phase(100, 32);
    phase->optimizer->PrintLevel = 0;

    auto status = phase->solve_optimize();
    ASSERT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    auto result = phase->returnTraj();
    // Check final boundary conditions: x(tf)=10, y(tf)=5
    EXPECT_NEAR(result.back()[0], 10.0, 1e-4);
    EXPECT_NEAR(result.back()[1], 5.0, 1e-4);
    // Check initial boundary conditions
    EXPECT_NEAR(result.front()[0], 0.0, 1e-4);
    EXPECT_NEAR(result.front()[1], 10.0, 1e-4);
    EXPECT_NEAR(result.front()[2], 0.0, 1e-4);
    EXPECT_NEAR(result.front()[3], 0.0, 1e-4);
}

TEST_F(OptimalControlTest, KeplerPhaseConstruct) {
    double mu = 398600.4418;
    Kepler kep(mu);

    // Circular orbit initial state
    double r0 = 7000.0;
    double v_circ = std::sqrt(mu / r0);
    double T = 2.0 * M_PI * std::sqrt(r0 * r0 * r0 / mu);

    int n_pts = 50;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        double t = s * T / 4.0;
        double angle = s * M_PI / 2.0;
        Eigen::VectorXd pt(7);
        pt << r0 * std::cos(angle), r0 * std::sin(angle), 0.0, -v_circ * std::sin(angle),
            v_circ * std::cos(angle), 0.0, t;
        traj.push_back(pt);
    }

    auto phase = std::make_shared<ODEPhase<Kepler>>(kep, TranscriptionModes::LGL3, traj, 16);
    // Should construct without error
    EXPECT_GT(traj.size(), 0u);
}
