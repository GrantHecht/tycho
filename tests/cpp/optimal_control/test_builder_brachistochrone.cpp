///////////////////////////////////////////////////////////////////////////////
// End-to-end Brachistochrone via the Builder API
//
// Defines the ODE with ODEBuilder, sets up the phase with named variables,
// solves, and verifies convergence to the known optimal time (~1.8013 s).
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/ode_builder.h>
#include <tycho/detail/phase_wrapper.h>
#include <tycho/detail/runtime_ode.h>
#include <tycho/tycho.h>

using namespace Tycho;
using namespace TychoTest;

class BuilderAPITest : public OptimalControlTest {};

TEST_F(BuilderAPITest, BrachistochroneConverges) {
    constexpr double g = 9.81;

    // Define ODE via ODEBuilder with named variables
    auto ode = ODEBuilder(3, 1)
                   .define([&](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
                   .build();

    // Build initial guess
    constexpr int n_pts = 100;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = 10.0 * s;                // x: 0 -> 10
        pt[1] = 10.0 + (5.0 - 10.0) * s; // y: 10 -> 5
        pt[2] = g * s * std::cos(1.0);   // v
        pt[3] = s;                       // t: 0 -> 1
        pt[4] = 1.0;                     // theta guess
        traj.push_back(pt);
    }

    // Construct phase with named variables
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Front boundary: fix x, y, v, t
    phase.addBoundaryValue(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Back boundary: fix x, y
    phase.addBoundaryValue(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(10, 5));

    // Control bounds: theta in [-0.1, 2.0]
    phase.addLUVarBound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    // Objective: minimize time
    phase.addDeltaTimeObjective(1.0);

    // Solve
    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    // Check optimal time
    auto result = phase.returnTraj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(BuilderAPITest, BrachistochroneMixedAPI) {
    // Same problem but mixing named and index-based APIs
    constexpr double g = 9.81;

    auto ode = ODEBuilder(3, 1)
                   .define([&](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
                   .build();

    constexpr int n_pts = 100;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = 10.0 * s;
        pt[1] = 10.0 + (5.0 - 10.0) * s;
        pt[2] = g * s * std::cos(1.0);
        pt[3] = s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named front boundary
    phase.addBoundaryValue(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Index-based back boundary
    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    phase.addBoundaryValue(PhaseRegionFlags::Back, back_idx, Eigen::Vector2d(10, 5));

    // Index-based control bounds
    phase.addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0);

    // Minimize final time
    phase.addDeltaTimeObjective(1.0);

    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.returnTraj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}
