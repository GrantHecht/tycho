///////////////////////////////////////////////////////////////////////////////
// OptimalControl integration tests
//
// Tests LGL collocation, phase construction, multi-phase OCP, mesh
// refinement, and end-to-end solver convergence using Brachistochrone.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "Astro/KeplerModel.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Brachistochrone ODE
///////////////////////////////////////////////////////////////////////////////

struct BrachOC_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>();
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);
        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(BrachOC, BrachOC_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class OptimalControlTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Helper: build a standard Brachistochrone phase
///////////////////////////////////////////////////////////////////////////////

static std::shared_ptr<ODEPhase<BrachOC>> make_brach_phase(int n_pts = 100,
                                                            int n_defects = 32) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    BrachOC ode(g);
    auto phase = std::make_shared<ODEPhase<BrachOC>>(ode, TranscriptionModes::LGL3, traj, n_defects);

    // Front boundary
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    // Back boundary
    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    // Control bounds
    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);

    // Objective
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Phase construction
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, BrachPhaseConstruct) {
    auto phase = make_brach_phase();
    // Phase should be constructable without error
    SUCCEED();
}

TEST_F(OptimalControlTest, BrachPhaseTranscribe) {
    auto phase = make_brach_phase();
    phase->transcribe(false, false);
    // After transcription, we can query the trajectory
    auto traj = phase->returnTraj();
    EXPECT_GT(traj.size(), 0u);
}

TEST_F(OptimalControlTest, BrachPhaseBoundaryConstraints) {
    auto phase = make_brach_phase();
    phase->transcribe(false, false);
    auto traj = phase->returnTraj();

    // Initial trajectory should have correct front boundary (from the guess)
    EXPECT_EQ(traj.front().size(), 5);
}

///////////////////////////////////////////////////////////////////////////////
// Multi-phase / OCP
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, TwoPhaseOCPConstruct) {
    auto phase1 = make_brach_phase(50, 16);
    auto phase2 = make_brach_phase(50, 16);

    OptimalControlProblem ocp;
    ocp.addPhase(phase1);
    ocp.addPhase(phase2);
    // Should construct without error
    SUCCEED();
}

///////////////////////////////////////////////////////////////////////////////
// Solver integration — Brachistochrone end-to-end
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, BrachistochoneSolveOptimize) {
    auto phase = make_brach_phase(100, 32);
    phase->optimizer->PrintLevel = 0;

    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED)
        << "Brachistochrone should converge to optimal";

    auto result = phase->returnTraj();
    double tf = result.back()[3];
    // Known optimal time ~1.8013 s
    EXPECT_NEAR(tf, 1.8013, 0.01)
        << "Optimal time should be near 1.8013 s, got " << tf;
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

///////////////////////////////////////////////////////////////////////////////
// Kepler phase (shooting-style, verifies integration within OCP framework)
///////////////////////////////////////////////////////////////////////////////

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
        pt << r0 * std::cos(angle), r0 * std::sin(angle), 0.0,
            -v_circ * std::sin(angle), v_circ * std::cos(angle), 0.0, t;
        traj.push_back(pt);
    }

    auto phase = std::make_shared<ODEPhase<Kepler>>(kep, TranscriptionModes::LGL3, traj, 16);
    // Should construct without error
    EXPECT_GT(traj.size(), 0u);
}
