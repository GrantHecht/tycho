///////////////////////////////////////////////////////////////////////////////
// Multi-parameter ODE expression tests
//
// Verifies BUILD_ODE_FROM_EXPRESSION works correctly with multiple
// constructor parameters and zero parameters.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Multi-parameter ODE — drag brachistochrone with g and drag coefficient
///////////////////////////////////////////////////////////////////////////////

struct DragBrach_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g, double drag) {
        auto args = Arguments<5>(); // [x, y, v, t, theta]
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        return StackedOutputs{
            sin(theta) * v,
            cos(theta) * v * (-1.0),
            g * cos(theta) - drag * v};
    }
};
BUILD_ODE_FROM_EXPRESSION(DragBrach, DragBrach_Impl, double, double);

///////////////////////////////////////////////////////////////////////////////
// Three-parameter ODE
///////////////////////////////////////////////////////////////////////////////

struct ThreeParam_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax, double wVel, double wAng) {
        auto args = Arguments<4>();
        auto theta = args.coeff<3>();
        return StackedOutputs{
            vMax * cos(theta) + wVel * std::cos(wAng),
            vMax * sin(theta) + wVel * std::sin(wAng)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ThreeParamODE, ThreeParam_Impl, double, double, double);

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, MultiParamODE_TwoParam_Constructs) {
    DragBrach ode(9.81, 0.1);
    EXPECT_EQ(ode.input_rows(), 5);
    EXPECT_EQ(ode.output_rows(), 3);
    EXPECT_EQ(ode.x_vars(), 3);
    EXPECT_EQ(ode.u_vars(), 1);
    EXPECT_EQ(ode.p_vars(), 0);
}

TEST_F(VFCompositionTest, MultiParamODE_TwoParam_Compute) {
    DragBrach ode(9.81, 0.1);
    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    ode.compute(x, fx);

    double v = 5.0, theta = std::numbers::pi / 4.0;
    EXPECT_NEAR(fx[0], std::sin(theta) * v, 1e-12);
    EXPECT_NEAR(fx[1], -std::cos(theta) * v, 1e-12);
    EXPECT_NEAR(fx[2], 9.81 * std::cos(theta) - 0.1 * v, 1e-12);
}

TEST_F(VFCompositionTest, MultiParamODE_TwoParam_AdjointConsistency) {
    DragBrach ode(9.81, 0.1);
    Eigen::VectorXd x = deterministic_random_vector(5, 300, 0.1, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 301, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, MultiParamODE_ThreeParam_Constructs) {
    ThreeParamODE ode(1.0, 0.5, 0.3);
    EXPECT_EQ(ode.input_rows(), 4);
    EXPECT_EQ(ode.output_rows(), 2);
}

TEST_F(VFCompositionTest, MultiParamODE_ThreeParam_AdjointConsistency) {
    ThreeParamODE ode(1.0, 0.5, 0.3);
    Eigen::VectorXd x = deterministic_random_vector(4, 302, 0.1, 3.0);
    Eigen::VectorXd lm = deterministic_random_vector(2, 303, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, MultiParamODE_TwoParam_GenericErasure) {
    DragBrach ode(9.81, 0.1);
    GenericFunction<-1, -1> gf(ode);
    EXPECT_EQ(gf.input_rows(), 5);
    EXPECT_EQ(gf.output_rows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx_orig(3), fx_gf(3);
    fx_orig.setZero();
    fx_gf.setZero();
    ode.compute(x, fx_orig);
    gf.compute(x, fx_gf);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(fx_orig[i], fx_gf[i], 1e-14);
}

TEST_F(VFCompositionTest, MultiParamODE_TwoParam_Integrator) {
    DragBrach ode(9.81, 0.1);
    auto integ = ode.integrator(0.01);
    // Integrator input = XtUP + step_size = 5 + 1 = 6
    EXPECT_EQ(integ.input_rows(), 6);
    EXPECT_EQ(integ.output_rows(), 5);
}

///////////////////////////////////////////////////////////////////////////////
// Zero-parameter ODE — exercises __VA_OPT__ fix
///////////////////////////////////////////////////////////////////////////////

struct SHO_ZeroParam_Impl : ODESize<2, 0, 0> {
    static auto Definition() {
        auto args = Arguments<3>(); // [x, v, t]
        auto x = args.coeff<0>();
        auto v = args.coeff<1>();
        return StackedOutputs{v, -1.0 * x}; // xdot = v, vdot = -x
    }
};
BUILD_ODE_FROM_EXPRESSION(SHO_ZeroParam, SHO_ZeroParam_Impl);

TEST_F(VFCompositionTest, ZeroParamODE_Constructs) {
    SHO_ZeroParam ode;
    EXPECT_EQ(ode.input_rows(), 3);
    EXPECT_EQ(ode.output_rows(), 2);
    EXPECT_EQ(ode.x_vars(), 2);
    EXPECT_EQ(ode.u_vars(), 0);
    EXPECT_EQ(ode.p_vars(), 0);
}

TEST_F(VFCompositionTest, ZeroParamODE_Compute) {
    SHO_ZeroParam ode;
    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 0.0; // [x=1, v=2, t=0]
    Eigen::VectorXd fx(2);
    fx.setZero();
    ode.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 2.0);  // xdot = v = 2
    EXPECT_DOUBLE_EQ(fx[1], -1.0); // vdot = -x = -1
}

TEST_F(VFCompositionTest, ZeroParamODE_AdjointConsistency) {
    SHO_ZeroParam ode;
    Eigen::VectorXd x = deterministic_random_vector(3, 500, 0.1, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(2, 501, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

///////////////////////////////////////////////////////////////////////////////
// Zero-parameter non-ODE expression — exercises __VA_OPT__ in BUILD_FROM_EXPRESSION
///////////////////////////////////////////////////////////////////////////////

struct UnitCircle_Impl {
    static auto Definition() {
        auto args = Arguments<1>(); // [theta]
        auto theta = args.coeff<0>();
        return StackedOutputs{cos(theta), sin(theta)};
    }
};
BUILD_FROM_EXPRESSION(UnitCircle, UnitCircle_Impl);

TEST_F(VFCompositionTest, ZeroParamExpression_Constructs) {
    UnitCircle expr;
    EXPECT_EQ(expr.input_rows(), 1);
    EXPECT_EQ(expr.output_rows(), 2);
}

TEST_F(VFCompositionTest, ZeroParamExpression_Compute) {
    UnitCircle expr;
    Eigen::VectorXd x(1);
    x << 0.0;
    Eigen::VectorXd fx(2);
    fx.setZero();
    expr.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 1.0); // cos(0)
    EXPECT_DOUBLE_EQ(fx[1], 0.0); // sin(0)
}

TEST_F(VFCompositionTest, ZeroParamExpression_JacobianFD) {
    UnitCircle expr;
    Eigen::VectorXd x(1);
    x << std::numbers::pi / 3.0;
    verify_jacobian_fd(expr, x);
}
