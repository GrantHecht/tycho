///////////////////////////////////////////////////////////////////////////////
// Compile-time named variable tags — XVar, UVar, PVar
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>
#include <type_traits>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Basic indexing — XVar<I> maps to coeff<I>() on Arguments
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_XVar_IndexAccess) {
    auto args = Arguments<4>();
    auto x0_tag = args[XVar<0>];
    auto x0_coeff = args.coeff<0>();

    Eigen::VectorXd input(4);
    input << 1.0, 2.0, 3.0, 0.5;

    Eigen::VectorXd out_tag(1), out_coeff(1);
    out_tag.setZero();
    out_coeff.setZero();
    x0_tag.compute(input, out_tag);
    x0_coeff.compute(input, out_coeff);
    EXPECT_DOUBLE_EQ(out_tag(0), out_coeff(0));
    EXPECT_DOUBLE_EQ(out_tag(0), 1.0);
}

TEST_F(CommonFunctionsTest, VarTag_MultipleVars) {
    auto args = Arguments<5>();
    auto x = args[XVar<0>];
    auto y = args[XVar<1>];
    auto v = args[XVar<2>];

    Eigen::VectorXd input(5);
    input << 10.0, 20.0, 30.0, 0.0, 0.5;

    Eigen::VectorXd fx(1);

    fx.setZero();
    x.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 10.0);

    fx.setZero();
    y.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 20.0);

    fx.setZero();
    v.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 30.0);
}

///////////////////////////////////////////////////////////////////////////////
// UVar and PVar on ODEArguments — offset-aware access
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_UVar_ODEArguments) {
    // ODE layout: [x0, x1, x2, t, u0] — XV=3, UV=1, PV=0
    auto args = ODEArguments<3, 1, 0>();
    auto u0 = args[UVar<0>]; // should resolve to coeff<4>() (XV+1+0 = 4)

    Eigen::VectorXd input(5);
    input << 1.0, 2.0, 3.0, 0.0, 42.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    u0.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 42.0);
}

TEST_F(CommonFunctionsTest, VarTag_PVar_ODEArguments) {
    // ODE layout: [x0, x1, t, u0, p0, p1] — XV=2, UV=1, PV=2
    auto args = ODEArguments<2, 1, 2>();
    auto p0 = args[PVar<0>]; // coeff<4>() (XV+1+UV+0 = 4)
    auto p1 = args[PVar<1>]; // coeff<5>() (XV+1+UV+1 = 5)

    Eigen::VectorXd input(6);
    input << 1.0, 2.0, 0.0, 3.0, 99.0, 77.0;
    Eigen::VectorXd fx(1);

    fx.setZero();
    p0.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 99.0);

    fx.setZero();
    p1.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 77.0);
}

TEST_F(CommonFunctionsTest, VarTag_XVar_ODEArguments) {
    // Verify XVar works on ODEArguments too
    auto args = ODEArguments<3, 1, 0>();
    auto x2 = args[XVar<2>];

    Eigen::VectorXd input(5);
    input << 10.0, 20.0, 30.0, 0.0, 0.5;
    Eigen::VectorXd fx(1);
    fx.setZero();
    x2.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 30.0);
}

TEST_F(CommonFunctionsTest, VarTag_ODEArguments_InExpression) {
    // Build an ODE-like expression using offset-aware tags
    auto args = ODEArguments<3, 1, 0>();
    auto v = args[XVar<2>];     // index 2
    auto theta = args[UVar<0>]; // index 4 (XV+1+0)

    auto expr = sin(theta) * v;

    Eigen::VectorXd input(5);
    input << 0, 0, 5.0, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    expr.compute(input, fx);
    EXPECT_NEAR(fx(0), std::sin(std::numbers::pi / 4.0) * 5.0, 1e-12);
}

///////////////////////////////////////////////////////////////////////////////
// Used in expressions — composable with DSL operators
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_InExpression) {
    auto args = Arguments<5>();
    auto v = args[XVar<2>];
    auto theta = args[XVar<4>];

    auto expr = sin(theta) * v;

    Eigen::VectorXd input(5);
    input << 0, 0, 5.0, 0, std::numbers::pi / 4.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    expr.compute(input, fx);
    EXPECT_NEAR(fx(0), std::sin(std::numbers::pi / 4.0) * 5.0, 1e-12);
}

TEST_F(CommonFunctionsTest, VarTag_InExpression_Jacobian) {
    auto args = Arguments<3>();
    auto x = args[XVar<0>];
    auto y = args[XVar<1>];
    auto expr = 2.0 * x + 3.0 * y;

    Eigen::VectorXd input(3);
    input << 1.0, 2.0, 3.0;

    // Jacobian should be [2, 3, 0]
    Eigen::MatrixXd expected(1, 3);
    expected << 2.0, 3.0, 0.0;
    verify_jacobian_analytical(expr, input, expected);
}

///////////////////////////////////////////////////////////////////////////////
// Type identity — VarTag access produces same type as coeff<>
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_SameTypeAsCoeff) {
    auto args = Arguments<3>();
    auto via_tag = args[XVar<1>];
    auto via_coeff = args.coeff<1>();
    static_assert(std::is_same_v<decltype(via_tag), decltype(via_coeff)>);
}
