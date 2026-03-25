///////////////////////////////////////////////////////////////////////////////
// RuntimeODE unit tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/ode_builder.h>
#include <tycho/detail/phase_wrapper.h>
#include <tycho/detail/runtime_ode.h>
#include <tycho/tycho.h>

using namespace Tycho;
using namespace TychoTest;

class RuntimeODETest : public OptimalControlTest {};

TEST_F(RuntimeODETest, ConstructFromExpression) {
    // Build Brachistochrone dynamics using the VF DSL
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);     // XVar(2)
    auto theta = args.segment(4, 1).coeff(0); // UVar(0)

    auto xdot = sin(theta) * v;
    auto ydot = cos(theta) * v * (-1.0);
    auto vdot = 9.81 * cos(theta);

    auto ode_expr = stack(xdot, ydot, vdot);

    RuntimeODE ode(ode_expr, 3, 1, 0);
    EXPECT_EQ(ode.xvars(), 3);
    EXPECT_EQ(ode.uvars(), 1);
    EXPECT_EQ(ode.pvars(), 0);
    EXPECT_EQ(ode.xtup_size(), 5);
}

TEST_F(RuntimeODETest, VarNameRegistration) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);

    auto ode_expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    RuntimeODE ode(ode_expr, 3, 1, 0);
    ode.var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}});

    EXPECT_TRUE(ode.has_registry());
    auto input = ode.make_input({{"x", 0}, {"y", 10}, {"v", 0}, {"theta", 1.0}});
    EXPECT_EQ(input.size(), 5);
    EXPECT_DOUBLE_EQ(input[0], 0.0);
    EXPECT_DOUBLE_EQ(input[1], 10.0);
    EXPECT_DOUBLE_EQ(input[4], 1.0);
}

TEST_F(RuntimeODETest, GenericODEConstruction) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);

    auto ode_expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    RuntimeODE ode(ode_expr, 3, 1, 0);
    auto gen_ode = ode.generic_ode();

    EXPECT_EQ(gen_ode.XVars(), 3);
    EXPECT_EQ(gen_ode.UVars(), 1);
    EXPECT_EQ(gen_ode.PVars(), 0);
}

TEST_F(RuntimeODETest, SizeMismatchThrows) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);

    // Expression has 3 outputs but we claim xvars=2
    auto ode_expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    EXPECT_THROW(RuntimeODE(ode_expr, 2, 1, 0), std::invalid_argument);
}

TEST_F(RuntimeODETest, InvalidSizesThrow) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);
    auto ode_expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    EXPECT_THROW(RuntimeODE(ode_expr, 0, 1, 0), std::invalid_argument);
    EXPECT_THROW(RuntimeODE(ode_expr, -1, 1, 0), std::invalid_argument);
    EXPECT_THROW(RuntimeODE(ode_expr, 3, -1, 0), std::invalid_argument);
}

TEST_F(RuntimeODETest, InputSizeMismatchThrows) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);
    auto ode_expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    // Expression built for (3,1,0) but we claim (3,2,0) — input size mismatch
    EXPECT_THROW(RuntimeODE(ode_expr, 3, 2, 0), std::invalid_argument);
}

TEST_F(RuntimeODETest, FlatMapPopulatedOnPhase) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
                   .build();

    std::vector<Eigen::VectorXd> traj;
    for (int i = 0; i < 50; ++i) {
        double s = static_cast<double>(i) / 49.0;
        Eigen::VectorXd pt(5);
        pt << 10.0 * s, 10.0 - 5.0 * s, 9.81 * s * std::cos(1.0), s, 1.0;
        traj.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, 16);

    // Verify FlatMap was populated: base phase resolves names via idx()
    auto x_idx = phase.base().idx("x");
    EXPECT_EQ(x_idx.size(), 1);
    EXPECT_EQ(x_idx[0], 0);

    auto theta_idx = phase.base().idx("theta");
    EXPECT_EQ(theta_idx.size(), 1);
    EXPECT_EQ(theta_idx[0], 4);
}

TEST_F(RuntimeODETest, FluentVarGroupRegistration) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);

    auto ode_expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    RuntimeODE ode(ode_expr, 3, 1, 0);
    ode.var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}}).var_group("pos", 0, 2);

    auto idx = ode.registry().resolve("pos");
    EXPECT_EQ(idx.size(), 2);
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 1);
}
