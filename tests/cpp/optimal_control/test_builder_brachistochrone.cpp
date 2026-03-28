///////////////////////////////////////////////////////////////////////////////
// End-to-end Brachistochrone via the Builder API
//
// Defines the ODE with ODEBuilder, sets up the phase with named variables,
// solves, and verifies convergence to the known optimal time (~1.8013 s).
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/ocp_wrapper.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/detail/optimal_control/builder/runtime_ode.h>
#include <tycho/tycho.h>

using namespace tycho;
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
    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Back boundary: fix x, y
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(10, 5));

    // Control bounds: theta in [-0.1, 2.0]
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    // Objective: minimize time
    phase.add_delta_time_objective(1.0);

    // Solve
    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    // Check optimal time
    auto result = phase.return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(BuilderAPITest, OCPWrapperAddPhaseAndLink) {
    // Two identical Brachistochrone phases linked via OCP wrapper with named vars
    constexpr double g = 9.81;

    auto make_ode = [g]() {
        return ODEBuilder(3, 1)
            .define([g](auto &args) {
                auto v = args.XVar(2);
                auto theta = args.UVar(0);
                return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
            })
            .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
            .build();
    };

    auto ode1 = make_ode();
    auto ode2 = make_ode();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, g * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase1 = ode1.phase(TranscriptionModes::LGL3, traj, 16);
    auto phase2 = ode2.phase(TranscriptionModes::LGL3, traj, 16);

    // Use OCP wrapper — no base_ptr() needed
    OCP ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);

    // Named-variable forward link
    ocp.add_forward_link_equal_con(phase1, phase2, {"x", "y", "v", "t"});

    // Verify the phases were added (base() escape hatch)
    EXPECT_EQ(ocp.base().phases.size(), 2u);
    SUCCEED();
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
    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Index-based back boundary
    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, Eigen::Vector2d(10, 5));

    // Index-based control bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0);

    // Minimize final time
    phase.add_delta_time_objective(1.0);

    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(BuilderAPITest, OCPLinkThrowsWhenP2HasNoRegistry) {
    constexpr double g = 9.81;

    // Phase 1: has registry
    auto ode1 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();

    // Phase 2: no registry
    auto ode2 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .build();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, g * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase1 = ode1.phase(TranscriptionModes::LGL3, traj, 16);
    auto phase2 = ode2.phase(TranscriptionModes::LGL3, traj, 16);

    OCP ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);

    EXPECT_THROW(
        ocp.add_forward_link_equal_con(phase1, phase2, {"x", "y", "v", "t"}),
        std::invalid_argument);
}

TEST_F(BuilderAPITest, OCPLinkThrowsOnMismatchedRegistries) {
    constexpr double g = 9.81;

    // Phase 1: x=0, y=1, v=2
    auto ode1 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();

    // Phase 2: same names but x=1, y=0 (swapped)
    auto ode2 = ODEBuilder(3, 1)
        .define([g](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
        })
        .var_names({{"x", 1}, {"y", 0}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, g * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase1 = ode1.phase(TranscriptionModes::LGL3, traj, 16);
    auto phase2 = ode2.phase(TranscriptionModes::LGL3, traj, 16);

    OCP ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);

    EXPECT_THROW(
        ocp.add_forward_link_equal_con(phase1, phase2, {"x", "y"}),
        std::invalid_argument);
}

TEST_F(BuilderAPITest, OCPIndexBasedLinkConstraint) {
    // Verifies the index-based overload — the escape hatch recommended by
    // error messages when named variables resolve to different indices.
    constexpr double g = 9.81;

    auto make_ode = [g]() {
        return ODEBuilder(3, 1)
            .define([g](auto &args) {
                auto v = args.XVar(2);
                auto theta = args.UVar(0);
                return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
            })
            .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
            .build();
    };

    auto ode1 = make_ode();
    auto ode2 = make_ode();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, g * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase1 = ode1.phase(TranscriptionModes::LGL3, traj, 16);
    auto phase2 = ode2.phase(TranscriptionModes::LGL3, traj, 16);

    OCP ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);

    // Index-based link — the escape hatch for heterogeneous phase layouts
    Eigen::VectorXi link_vars(4);
    link_vars << 0, 1, 2, 3;  // x, y, v, t
    ocp.add_forward_link_equal_con(phase1, phase2, link_vars);

    EXPECT_EQ(ocp.base().phases.size(), 2u);
}

TEST_F(BuilderAPITest, OCPSolveThrowsWithNoPhases) {
    OCP ocp;
    EXPECT_THROW(ocp.solve(), std::invalid_argument);
    EXPECT_THROW(ocp.optimize(), std::invalid_argument);
    EXPECT_THROW(ocp.solve_optimize(), std::invalid_argument);
    EXPECT_THROW(ocp.optimize_solve(), std::invalid_argument);
}
