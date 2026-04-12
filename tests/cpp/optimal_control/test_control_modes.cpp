///////////////////////////////////////////////////////////////////////////////
// Control mode tests
//
// Verifies that all ControlModes produce convergent solutions on the
// Brachistochrone problem.  BlockConstant mode exercises
// Blocked_ODE_Wrapper whose override methods must match the base-class
// snake_case names; a mismatch causes silent dimension errors that
// manifest as SIGBUS / SIGSEGV at runtime (see commit c83a4c6).
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/detail/optimal_control/builder/runtime_ode.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class ControlModeTest : public OptimalControlTest {};

namespace {

ODE make_control_modes_brach_ode() {
    return ODEBuilder(3, 1)
        .define([](auto &args) {
            auto v = args.x_var(2);
            auto theta = args.u_var(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
        })
        .build();
}

std::vector<Eigen::VectorXd> make_control_modes_brach_guess() {
    constexpr double g = 9.81;
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
    return traj;
}

Phase make_brach_phase_with_mode(ODE &ode, ControlModes mode) {
    auto traj = make_control_modes_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.set_control_mode(mode);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    phase.add_boundary_value(PhaseRegionFlags::Front, front_idx, Eigen::Vector4d(0, 10, 0, 0));

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, Eigen::Vector2d(10, 5));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0);
    phase.add_delta_time_objective(1.0);

    phase.optimizer().set_print_level(0);

    return phase;
}

} // namespace

TEST_F(ControlModeTest, HighestOrderSplineConverges) {
    auto ode = make_control_modes_brach_ode();
    auto phase = make_brach_phase_with_mode(ode, ControlModes::HighestOrderSpline);

    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.02);
}

TEST_F(ControlModeTest, FirstOrderSplineConverges) {
    auto ode = make_control_modes_brach_ode();
    auto phase = make_brach_phase_with_mode(ode, ControlModes::FirstOrderSpline);

    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.02);
}

TEST_F(ControlModeTest, NoSplineConverges) {
    auto ode = make_control_modes_brach_ode();
    auto phase = make_brach_phase_with_mode(ode, ControlModes::NoSpline);

    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.02);
}

TEST_F(ControlModeTest, BlockConstantConverges) {
    auto ode = make_control_modes_brach_ode();
    auto phase = make_brach_phase_with_mode(ode, ControlModes::BlockConstant);

    auto status = phase.solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase.return_traj();
    double tf = result.back()[3];
    // BlockConstant has coarser control resolution, allow wider tolerance
    EXPECT_NEAR(tf, 1.8013, 0.05);
}

TEST_F(ControlModeTest, BlockConstantStaticAPI) {
    // Also verify BlockConstant via the static (non-builder) API
    auto phase = make_brach_phase();
    phase->set_control_mode(ControlModes::BlockConstant);
    phase->optimizer_->set_print_level(0);

    auto status = phase->solve_optimize();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto result = phase->return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.05);
}
