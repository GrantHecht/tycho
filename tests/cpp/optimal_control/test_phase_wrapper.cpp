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

ODE make_brach_ode() {
    return ODEBuilder(3, 1)
        .define([](auto &args) {
            auto v = args.x_var(2);
            auto theta = args.u_var(0);
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
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named overload
    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(0, 10, 0, 0));

    // Named overload for back
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(10, 5));

    // Pin the underlying name->index map so a future registry refactor that
    // breaks resolution but still type-checks fails this test instead of
    // silently mis-routing constraints.
    EXPECT_EQ(phase.base().idx("x")[0], 0);
    EXPECT_EQ(phase.base().idx("y")[0], 1);
    EXPECT_EQ(phase.base().idx("v")[0], 2);
    EXPECT_EQ(phase.base().idx("t")[0], 3);
}

TEST_F(PhaseWrapperTest, IndexBasedPassthrough) {
    auto ode = make_brach_ode();
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
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Named
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    // Index passthrough
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0);

    EXPECT_EQ(phase.base().idx("theta")[0], 4);
}

TEST_F(PhaseWrapperTest, DeltaTimeObjective) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.add_delta_time_objective(1.0);

    SUCCEED();
}

TEST_F(PhaseWrapperTest, SetUnitsNamed) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.set_units({{"x", 10.0}, {"y", 10.0}, {"v", 5.0}, {"t", 1.0}, {"theta", 1.0}});

    SUCCEED();
}

TEST_F(PhaseWrapperTest, BasePhaseResolvesNames) {
    auto ode = make_brach_ode();
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
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
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
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    EXPECT_EQ(phase.base().x_vars(), 3);
    EXPECT_EQ(phase.base().u_vars(), 1);
    EXPECT_EQ(phase.base().p_vars(), 0);
}

TEST_F(PhaseWrapperTest, NullPhaseThrows) {
    EXPECT_THROW(Phase(nullptr, VarRegistry(3, 1, 0)), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, ScalarBoundaryValueGroupThrows) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
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
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
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
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
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
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
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
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
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
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.add_lower_var_bound(PhaseRegionFlags::Path, "theta", -0.1);
    EXPECT_EQ(phase.base().idx("theta")[0], 4);
}

TEST_F(PhaseWrapperTest, NamedUpperVarBound) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.add_upper_var_bound(PhaseRegionFlags::Path, "theta", 2.0);
    EXPECT_EQ(phase.base().idx("theta")[0], 4);
}

TEST_F(PhaseWrapperTest, NamedValueObjective) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    phase.add_value_objective(PhaseRegionFlags::Front, "v", 1.0);
    EXPECT_EQ(phase.base().idx("v")[0], 2);
}

TEST_F(PhaseWrapperTest, BoundaryValueSizeMismatchThrows) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // 3 names but 2 values
    EXPECT_THROW(
        phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v"}, Eigen::Vector2d(0, 10)),
        std::invalid_argument);
}

TEST_F(PhaseWrapperTest, NamedDeltaVarEqualCon) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.add_delta_var_equal_con("v", 5.0);
    EXPECT_EQ(phase.base().idx("v")[0], 2);
}

TEST_F(PhaseWrapperTest, NamedLowerDeltaVarBound) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);
    phase.add_lower_delta_var_bound("v", 0.0);
    EXPECT_EQ(phase.base().idx("v")[0], 2);
}

TEST_F(PhaseWrapperTest, NamedEqualCon) {
    auto ode = make_brach_ode();
    auto traj = make_brach_guess();
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 32);

    // Constraint: identity function on "x" at the front
    auto args = Arguments<1>();
    GenericFunction<-1, -1> identity(args);
    phase.add_equal_con(PhaseRegionFlags::Front, identity, {"x"});

    EXPECT_EQ(phase.base().idx("x")[0], 0);
}

TEST_F(PhaseWrapperTest, AddStaticParamNameRejectsEmptyName) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);
    EXPECT_THROW(phase.add_static_param_name("", 0), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, AddStaticParamNameRejectsDuplicate) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);
    phase.add_static_param_name("mass", 0);
    EXPECT_THROW(phase.add_static_param_name("mass", 1), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, AddStaticParamGroupRejectsEmptyName) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);
    EXPECT_THROW(phase.add_static_param_group("", 0, 3), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, AddStaticParamGroupRejectsNonPositiveCount) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);
    EXPECT_THROW(phase.add_static_param_group("g", 0, 0), std::invalid_argument);
    EXPECT_THROW(phase.add_static_param_group("h", 0, -1), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, AddStaticParamGroupRejectsDuplicate) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);
    phase.add_static_param_group("v", 0, 3);
    EXPECT_THROW(phase.add_static_param_group("v", 3, 3), std::invalid_argument);
}

TEST_F(PhaseWrapperTest, SetTrajLerpIgOverloadAccepted) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);

    // 3-arg overload with lerp_ig = true — pin the wrapper forwards it.
    // Pass a coarser grid to force the initial-guess lerp path to fire.
    auto coarse = make_brach_guess();
    coarse.resize(50);
    EXPECT_NO_THROW(phase.set_traj(coarse, 8, /*lerp_ig=*/true));
}

// resolve_for_region(ODEParams, ...) with an X-block name must throw via
// the Batch A unification of the multi-name overload. Prior to the fix, the
// multi-name path silently returned the raw XtUP index for non-ODEParams
// regions, so a caller asking for an X-block var in the ODEParams region
// would have silently installed a nonsense index instead of erroring.
TEST_F(PhaseWrapperTest, ResolveForRegionODEParamsRejectsXBlockName) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 16);
    phase.add_static_param_group("alpha", 0, 1);

    // Multi-name overload routes through the unified resolve_for_region.
    EXPECT_THROW(
        phase.add_boundary_value(PhaseRegionFlags::ODEParams, {"x"}, Eigen::VectorXd::Zero(1)),
        std::invalid_argument);
    // Single-name overload — symmetric behavior.
    EXPECT_THROW(phase.add_boundary_value(PhaseRegionFlags::ODEParams, "x", 0.0),
                 std::invalid_argument);
}

// Pins ODEPhaseBase::add_func_impl's validate-before-mutate ordering on the
// phase-side equality-constraint map. Pre-fix, map[index] = func ran first and
// check_function_size threw afterwards, leaving a bad entry in
// user_equalities_ that would duplicate on retry. Mirrors the OCP-side test
// AddLinkEqualConPartialMutationOnVarsSizeMismatch in test_ocp_direct_link.cpp.
//
// user_equalities_ is protected on ODEPhaseBase, so this test pins the fix
// indirectly: the index returned by a follow-up successful add_equal_con
// must be 0 if the prior failed add did not mutate the map. Pre-fix it would
// have been 1 (the failed insert sat at index 0).
TEST_F(PhaseWrapperTest, AddEqualConPartialMutationOnFuncSizeMismatch) {
    auto ode = make_brach_ode();
    auto phase = ode.phase(TranscriptionModes::LGL3, make_brach_guess(), 32);

    // Brach phase has xtu = 5 (3 states + 1 control + 1 time). Build a function
    // with IRows = 4 and pass vars of size 2; check_function_size compares
    // irows=4 against isize=vars.size()=2 in the Front-region branch and throws.
    auto args = Arguments<-1>(4);
    auto bad_expr = args.head<-1>(2);
    GenericFunction<-1, -1> bad_func(bad_expr);

    Eigen::VectorXi vars(2);
    vars << 0, 1;

    EXPECT_THROW(phase.base().add_equal_con(PhaseRegionFlags::Front, bad_func, vars,
                                            ScaleModes::AUTO),
                 std::invalid_argument);

    // Retry must also throw cleanly (pre-fix, the second insert would have
    // landed at index 1 next to the orphaned index 0, duplicating).
    EXPECT_THROW(phase.base().add_equal_con(PhaseRegionFlags::Front, bad_func, vars,
                                            ScaleModes::AUTO),
                 std::invalid_argument);

    // A valid follow-up add must land at index 0. Pre-fix it would have landed
    // at index 1 or 2 because the failed inserts left orphans at 0 (and 1).
    auto good_args = Arguments<-1>(2);
    auto good_expr = good_args.head<-1>(1);
    GenericFunction<-1, -1> good_func(good_expr);

    Eigen::VectorXi good_vars(2);
    good_vars << 0, 1;

    int idx = phase.base().add_equal_con(PhaseRegionFlags::Front, good_func, good_vars,
                                         ScaleModes::AUTO);
    EXPECT_EQ(idx, 0)
        << "First valid add after failed adds must land at index 0; pre-fix it "
           "would have been bumped to 1 or 2 because the failed inserts orphaned "
           "entries at 0 and 1.";
}
