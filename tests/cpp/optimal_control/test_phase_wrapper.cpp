///////////////////////////////////////////////////////////////////////////////
// Phase wrapper unit tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/ode_builder.h>
#include <tycho/detail/phase_wrapper.h>
#include <tycho/detail/runtime_ode.h>
#include <tycho/tycho.h>

using namespace Tycho;
using namespace TychoTest;

class PhaseWrapperTest : public OptimalControlTest {};

namespace {

RuntimeODE make_brach_runtime_ode() {
    return ODEBuilder(3, 1)
        .define([](auto &args) {
            auto v = args.XVar(2);
            auto theta = args.UVar(0);
            return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
        })
        .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
        .build();
}

std::vector<Eigen::VectorXd> make_brach_guess() {
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

} // namespace

TEST_F(PhaseWrapperTest, NamedBoundaryValue) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named overload
    phase.addBoundaryValue(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Named overload for back
    phase.addBoundaryValue(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(10, 5));

    SUCCEED(); // Construction and constraint addition didn't throw
}

TEST_F(PhaseWrapperTest, IndexBasedPassthrough) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Index overload (direct passthrough)
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    phase.addBoundaryValue(PhaseRegionFlags::Front, front_idx, Eigen::Vector4d(0, 10, 0, 0));

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    phase.addBoundaryValue(PhaseRegionFlags::Back, back_idx, Eigen::Vector2d(10, 5));

    SUCCEED();
}

TEST_F(PhaseWrapperTest, NamedLUVarBound) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named
    phase.addLUVarBound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    // Index passthrough
    phase.addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0);

    SUCCEED();
}

TEST_F(PhaseWrapperTest, DeltaTimeObjective) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.addDeltaTimeObjective(1.0);

    SUCCEED();
}

TEST_F(PhaseWrapperTest, SetUnitsNamed) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.setUnits({{"x", 10.0}, {"y", 10.0}, {"v", 5.0}, {"t", 1.0}, {"theta", 1.0}});

    SUCCEED();
}

TEST_F(PhaseWrapperTest, AccessBase) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    EXPECT_EQ(phase.base().XVars(), 3);
    EXPECT_EQ(phase.base().UVars(), 1);
    EXPECT_EQ(phase.base().PVars(), 0);
}
