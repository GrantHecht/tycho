///////////////////////////////////////////////////////////////////////////////
// Compile-time named variable tags — XVar, UVar, PVar, TVar, XVec, UVec,
// PVec, XSeg, USeg, PSeg. All tags are ODEArguments-only.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>
#include <type_traits>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Scalar tags — XVar, UVar, PVar, TVar on ODEArguments
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_XVar_ODEArguments) {
    auto args = ODEArguments<3, 1, 0>();
    auto x2 = args[XVar<2>];

    Eigen::VectorXd input(5);
    input << 10.0, 20.0, 30.0, 0.0, 0.5;
    Eigen::VectorXd fx(1);
    fx.setZero();
    x2.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 30.0);
}

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

TEST_F(CommonFunctionsTest, VarTag_TVar_ODEArguments) {
    // ODE layout: [x0, x1, x2, t, u0] — XV=3, UV=1, PV=0
    auto args = ODEArguments<3, 1, 0>();
    auto t = args[TVar]; // should resolve to coeff<3>() (XV = 3)

    Eigen::VectorXd input(5);
    input << 1.0, 2.0, 3.0, 7.5, 0.5;
    Eigen::VectorXd fx(1);
    fx.setZero();
    t.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 7.5);
}

TEST_F(CommonFunctionsTest, VarTag_SameTypeAsCoeff) {
    auto args = ODEArguments<3, 1, 0>();
    auto via_xvar = args[XVar<1>];
    auto via_coeff = args.coeff<1>();
    static_assert(std::is_same_v<decltype(via_xvar), decltype(via_coeff)>);

    auto via_tvar = args[TVar];
    auto via_tcoeff = args.coeff<3>(); // XV = 3
    static_assert(std::is_same_v<decltype(via_tvar), decltype(via_tcoeff)>);
}

///////////////////////////////////////////////////////////////////////////////
// Full-vector tags — XVec, UVec, PVec on ODEArguments
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_XVec_FullStateVector) {
    auto args = ODEArguments<3, 1, 0>();
    auto x_all = args[XVec]; // segment<3, 0>()

    Eigen::VectorXd input(5);
    input << 10.0, 20.0, 30.0, 0.0, 0.5;
    Eigen::VectorXd fx(3);
    fx.setZero();
    x_all.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 10.0);
    EXPECT_DOUBLE_EQ(fx(1), 20.0);
    EXPECT_DOUBLE_EQ(fx(2), 30.0);
}

TEST_F(CommonFunctionsTest, VarTag_UVec_FullControlVector) {
    // ODE layout: [x0, x1, t, u0, u1, u2] — XV=2, UV=3, PV=0
    auto args = ODEArguments<2, 3, 0>();
    auto u_all = args[UVec]; // segment<3, 3>() (XV+1 = 3)

    Eigen::VectorXd input(6);
    input << 1.0, 2.0, 0.0, 40.0, 50.0, 60.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    u_all.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 40.0);
    EXPECT_DOUBLE_EQ(fx(1), 50.0);
    EXPECT_DOUBLE_EQ(fx(2), 60.0);
}

TEST_F(CommonFunctionsTest, VarTag_PVec_FullParameterVector) {
    // ODE layout: [x0, x1, t, u0, p0, p1] — XV=2, UV=1, PV=2
    auto args = ODEArguments<2, 1, 2>();
    auto p_all = args[PVec]; // segment<2, 4>() (XV+1+UV = 4)

    Eigen::VectorXd input(6);
    input << 1.0, 2.0, 0.0, 3.0, 99.0, 77.0;
    Eigen::VectorXd fx(2);
    fx.setZero();
    p_all.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 99.0);
    EXPECT_DOUBLE_EQ(fx(1), 77.0);
}

TEST_F(CommonFunctionsTest, VarTag_XVec_SameTypeAsSegment) {
    auto args = ODEArguments<3, 1, 0>();
    auto via_tag = args[XVec];
    auto via_seg = args.segment<3, 0>();
    static_assert(std::is_same_v<decltype(via_tag), decltype(via_seg)>);
}

///////////////////////////////////////////////////////////////////////////////
// Sub-vector tags — XSeg, USeg, PSeg on ODEArguments
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_XSeg_StateSubVector) {
    // ODE layout: [x0..x5, t, u0, u1] — XV=6, UV=2, PV=0
    auto args = ODEArguments<6, 2, 0>();
    auto pos = args[XSeg<0, 3>]; // first 3 states (position)
    auto vel = args[XSeg<3, 3>]; // next 3 states (velocity)

    Eigen::VectorXd input(9);
    input << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 10.0, 20.0;

    Eigen::VectorXd fx_pos(3), fx_vel(3);
    fx_pos.setZero();
    fx_vel.setZero();
    pos.compute(input, fx_pos);
    vel.compute(input, fx_vel);

    EXPECT_DOUBLE_EQ(fx_pos(0), 1.0);
    EXPECT_DOUBLE_EQ(fx_pos(1), 2.0);
    EXPECT_DOUBLE_EQ(fx_pos(2), 3.0);
    EXPECT_DOUBLE_EQ(fx_vel(0), 4.0);
    EXPECT_DOUBLE_EQ(fx_vel(1), 5.0);
    EXPECT_DOUBLE_EQ(fx_vel(2), 6.0);
}

TEST_F(CommonFunctionsTest, VarTag_USeg_ControlSubVector) {
    // ODE layout: [x0, x1, t, u0, u1, u2, u3] — XV=2, UV=4, PV=0
    auto args = ODEArguments<2, 4, 0>();
    auto u_first2 = args[USeg<0, 2>]; // first 2 controls
    auto u_last2 = args[USeg<2, 2>];  // last 2 controls

    Eigen::VectorXd input(7);
    input << 1.0, 2.0, 0.0, 10.0, 20.0, 30.0, 40.0;

    Eigen::VectorXd fx1(2), fx2(2);
    fx1.setZero();
    fx2.setZero();
    u_first2.compute(input, fx1);
    u_last2.compute(input, fx2);

    EXPECT_DOUBLE_EQ(fx1(0), 10.0);
    EXPECT_DOUBLE_EQ(fx1(1), 20.0);
    EXPECT_DOUBLE_EQ(fx2(0), 30.0);
    EXPECT_DOUBLE_EQ(fx2(1), 40.0);
}

TEST_F(CommonFunctionsTest, VarTag_PSeg_ParameterSubVector) {
    // ODE layout: [x0, t, u0, p0, p1, p2] — XV=1, UV=1, PV=3
    auto args = ODEArguments<1, 1, 3>();
    auto p_first = args[PSeg<0, 1>];
    auto p_last2 = args[PSeg<1, 2>];

    Eigen::VectorXd input(6);
    input << 1.0, 0.0, 5.0, 100.0, 200.0, 300.0;

    Eigen::VectorXd fx1(1), fx2(2);
    fx1.setZero();
    fx2.setZero();
    p_first.compute(input, fx1);
    p_last2.compute(input, fx2);

    EXPECT_DOUBLE_EQ(fx1(0), 100.0);
    EXPECT_DOUBLE_EQ(fx2(0), 200.0);
    EXPECT_DOUBLE_EQ(fx2(1), 300.0);
}

TEST_F(CommonFunctionsTest, VarTag_XSeg_SameTypeAsSegment) {
    auto args = ODEArguments<6, 2, 0>();
    auto via_tag = args[XSeg<3, 3>];
    auto via_seg = args.segment<3, 3>();
    static_assert(std::is_same_v<decltype(via_tag), decltype(via_seg)>);
}

///////////////////////////////////////////////////////////////////////////////
// Expression composition — tags work in DSL expressions
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_ODEArguments_InExpression) {
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

TEST_F(CommonFunctionsTest, VarTag_XVec_InExpression_Jacobian) {
    // Dot product of state vector with itself: x^T * x = sum of squares
    auto args = ODEArguments<3, 1, 0>();
    auto x_all = args[XVec];
    auto expr = x_all.squared_norm();

    Eigen::VectorXd input(5);
    input << 1.0, 2.0, 3.0, 0.0, 0.5;
    Eigen::VectorXd fx(1);
    fx.setZero();
    expr.compute(input, fx);
    EXPECT_NEAR(fx(0), 14.0, 1e-12); // 1 + 4 + 9

    verify_jacobian_fd(expr, input);
}

TEST_F(CommonFunctionsTest, VarTag_XSeg_InExpression_Jacobian) {
    // Use sub-vectors in an expression: pos.norm() + vel.norm()
    auto args = ODEArguments<6, 1, 0>();
    auto pos = args[XSeg<0, 3>];
    auto vel = args[XSeg<3, 3>];

    auto expr = pos.norm() + vel.norm();

    Eigen::VectorXd input(8);
    input << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.5;

    verify_jacobian_fd(expr, input);
}

///////////////////////////////////////////////////////////////////////////////
// Boundary segment tests — Start + Size == UV/PV (exact end of range)
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, VarTag_USeg_BoundaryFullRange) {
    // USeg covering the full control range should match UVec
    auto args = ODEArguments<3, 2, 1>();
    auto u_seg = args[USeg<0, 2>]; // Start=0, Size=UV=2 → exact boundary
    auto u_vec = args[UVec];
    static_assert(std::is_same_v<decltype(u_seg), decltype(u_vec)>);

    Eigen::VectorXd input(7);
    input << 1.0, 2.0, 3.0, 0.0, 40.0, 50.0, 99.0;
    Eigen::VectorXd fx(2);
    fx.setZero();
    u_seg.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 40.0);
    EXPECT_DOUBLE_EQ(fx(1), 50.0);
}

TEST_F(CommonFunctionsTest, VarTag_PSeg_BoundaryFullRange) {
    // PSeg covering the full parameter range should match PVec
    auto args = ODEArguments<3, 2, 1>();
    auto p_seg = args[PSeg<0, 1>]; // Start=0, Size=PV=1 �� exact boundary
    auto p_vec = args[PVec];
    static_assert(std::is_same_v<decltype(p_seg), decltype(p_vec)>);

    Eigen::VectorXd input(7);
    input << 1.0, 2.0, 3.0, 0.0, 40.0, 50.0, 99.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    p_seg.compute(input, fx);
    EXPECT_DOUBLE_EQ(fx(0), 99.0);
}

TEST_F(CommonFunctionsTest, VarTag_XSeg_BoundaryLastElement) {
    // XSeg accessing the very last state element
    auto args = ODEArguments<6, 2, 0>();
    auto x_last = args[XSeg<5, 1>]; // Start=5, Size=1 → Start+Size == XV
    auto x_coeff = args[XVar<5>];

    Eigen::VectorXd input(9);
    input << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 10.0, 20.0;

    Eigen::VectorXd fx1(1), fx2(1);
    fx1.setZero();
    fx2.setZero();
    x_last.compute(input, fx1);
    x_coeff.compute(input, fx2);
    EXPECT_DOUBLE_EQ(fx1(0), 6.0);
    EXPECT_DOUBLE_EQ(fx1(0), fx2(0));
}
