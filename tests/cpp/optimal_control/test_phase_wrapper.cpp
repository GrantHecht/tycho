///////////////////////////////////////////////////////////////////////////////
// Phase wrapper unit tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/phase_wrapper.h>
#include <tycho/detail/optimal_control/builder/runtime_ode.h>
#include <tycho/tycho.h>

using namespace tycho;
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
    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Named overload for back
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(10, 5));

    SUCCEED(); // Construction and constraint addition didn't throw
}

TEST_F(PhaseWrapperTest, IndexBasedPassthrough) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Index overload (direct passthrough)
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    phase.add_boundary_value(PhaseRegionFlags::Front, front_idx, Eigen::Vector4d(0, 10, 0, 0));

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, Eigen::Vector2d(10, 5));

    SUCCEED();
}

TEST_F(PhaseWrapperTest, NamedLUVarBound) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    // Index passthrough
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0);

    SUCCEED();
}

TEST_F(PhaseWrapperTest, DeltaTimeObjective) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.add_delta_time_objective(1.0);

    SUCCEED();
}

TEST_F(PhaseWrapperTest, SetUnitsNamed) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.set_units({{"x", 10.0}, {"y", 10.0}, {"v", 5.0}, {"t", 1.0}, {"theta", 1.0}});

    SUCCEED();
}

TEST_F(PhaseWrapperTest, BasePhaseResolvesNames) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // After FlatMap bridge: base phase resolves names directly
    auto idx = phase.base().idx("theta");
    EXPECT_EQ(idx.size(), 1);
    EXPECT_EQ(idx[0], 4);
}

TEST_F(PhaseWrapperTest, LUVarBoundMultiIndexThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // "pos" maps to 2 indices — add_lu_var_bound requires exactly 1
    EXPECT_THROW(phase.add_lu_var_bound(PhaseRegionFlags::Path, "pos", -1.0, 1.0),
                 std::invalid_argument);
}

TEST_F(PhaseWrapperTest, AccessBase) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    EXPECT_EQ(phase.base().XVars(), 3);
    EXPECT_EQ(phase.base().UVars(), 1);
    EXPECT_EQ(phase.base().PVars(), 0);
}

TEST_F(PhaseWrapperTest, NullPhaseThrows) {
    EXPECT_THROW(Phase(nullptr, VarRegistry(3, 1, 0)), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, ScalarBoundaryValueGroupThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // "pos" maps to 2 indices — scalar add_boundary_value requires exactly 1
    EXPECT_THROW(phase.add_boundary_value(PhaseRegionFlags::Front, "pos", 5.0),
                 std::invalid_argument);
}

TEST_F(PhaseWrapperTest, DeltaVarObjectiveGroupThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // "pos" maps to 2 indices — add_delta_var_objective requires exactly 1
    EXPECT_THROW(phase.add_delta_var_objective("pos", 1.0), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, LowerVarBoundGroupThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // "pos" maps to 2 indices — add_lower_var_bound requires exactly 1
    EXPECT_THROW(phase.add_lower_var_bound(PhaseRegionFlags::Path, "pos", -1.0),
                 std::invalid_argument);
}

TEST_F(PhaseWrapperTest, UpperVarBoundGroupThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // "pos" maps to 2 indices — add_upper_var_bound requires exactly 1
    EXPECT_THROW(phase.add_upper_var_bound(PhaseRegionFlags::Path, "pos", 1.0),
                 std::invalid_argument);
}

TEST_F(PhaseWrapperTest, ValueObjectiveGroupThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // "pos" maps to 2 indices — add_value_objective requires exactly 1
    EXPECT_THROW(phase.add_value_objective(PhaseRegionFlags::Front, "pos", 1.0),
                 std::invalid_argument);
}

TEST_F(PhaseWrapperTest, NamedLowerVarBound) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.add_lower_var_bound(PhaseRegionFlags::Path, "theta", -0.1);
    SUCCEED();
}

TEST_F(PhaseWrapperTest, NamedUpperVarBound) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.add_upper_var_bound(PhaseRegionFlags::Path, "theta", 2.0);
    SUCCEED();
}

TEST_F(PhaseWrapperTest, NamedValueObjective) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.add_value_objective(PhaseRegionFlags::Front, "v", 1.0);
    SUCCEED();
}

TEST_F(PhaseWrapperTest, BoundaryValueSizeMismatchThrows) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // 3 names but 2 values
    EXPECT_THROW(
        phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v"}, Eigen::Vector2d(0, 10)),
        std::invalid_argument);
}

TEST_F(PhaseWrapperTest, NamedDeltaVarEqualCon) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.add_delta_var_equal_con("v", 5.0);
    SUCCEED();
}

TEST_F(PhaseWrapperTest, NamedLowerDeltaVarBound) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.add_lower_delta_var_bound("v", 0.0);
    SUCCEED();
}

TEST_F(PhaseWrapperTest, NamedEqualCon) {
    auto ode = make_brach_runtime_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Constraint: identity function on "x" at the front
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);
    phase.add_equal_con(PhaseRegionFlags::Front, identity, {"x"});

    SUCCEED();
}
