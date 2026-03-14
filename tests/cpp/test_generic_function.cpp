///////////////////////////////////////////////////////////////////////////////
// GenericFunction type erasure tests
//
// Tests GenericFunction<IR,OR>, GFStorage SBO/heap, copy/move semantics,
// cross-type erasure, solver interface packing, GenericConditional/Comparative.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class GenericFunctionTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Helper ODE for testing
///////////////////////////////////////////////////////////////////////////////

struct TestODE_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>();
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);
        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(TestODE, TestODE_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Basic erasure
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, FixedSizeErase) {
    auto args = Arguments<3>();
    auto scaled = 2.0 * args;
    GenericFunction<3, 3> gf(scaled);
    EXPECT_EQ(gf.IRows(), 3);
    EXPECT_EQ(gf.ORows(), 3);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    gf.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 2.0);
    EXPECT_DOUBLE_EQ(fx[1], 4.0);
    EXPECT_DOUBLE_EQ(fx[2], 6.0);
}

TEST_F(GenericFunctionTest, DynamicErase) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf(ode);
    EXPECT_EQ(gf.IRows(), 5);
    EXPECT_EQ(gf.ORows(), 3);
}

TEST_F(GenericFunctionTest, ComputeMatchesOriginal) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf(ode);
    Eigen::VectorXd x(5);
    x << 1.0, 8.0, 3.0, 0.5, 0.7;

    Eigen::VectorXd fx_orig(3), fx_gf(3);
    fx_orig.setZero();
    fx_gf.setZero();
    ode.compute(x, fx_orig);
    gf.compute(x, fx_gf);

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(fx_orig[i], fx_gf[i]) << "Output mismatch at " << i;
    }
}

TEST_F(GenericFunctionTest, JacobianMatchesOriginal) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf(ode);
    Eigen::VectorXd x(5);
    x << 1.0, 8.0, 3.0, 0.5, 0.7;

    Eigen::VectorXd fx1(3), fx2(3);
    Eigen::MatrixXd jx1(3, 5), jx2(3, 5);
    fx1.setZero();
    fx2.setZero();
    jx1.setZero();
    jx2.setZero();
    ode.compute_jacobian(x, fx1, jx1);
    gf.compute_jacobian(x, fx2, jx2);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_DOUBLE_EQ(jx1(i, j), jx2(i, j))
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Copy / Move
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, CopyConstruct) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf1(ode);
    GenericFunction<-1, -1> gf2(gf1);

    Eigen::VectorXd x(5);
    x << 2.0, 7.0, 4.0, 0.3, 1.2;
    Eigen::VectorXd fx1(3), fx2(3);
    fx1.setZero();
    fx2.setZero();
    gf1.compute(x, fx1);
    gf2.compute(x, fx2);

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(fx1[i], fx2[i]);
    }
}

TEST_F(GenericFunctionTest, MoveConstruct) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf1(ode);
    GenericFunction<-1, -1> gf2(std::move(gf1));
    EXPECT_EQ(gf2.IRows(), 5);
    EXPECT_EQ(gf2.ORows(), 3);

    Eigen::VectorXd x(5);
    x << 1, 2, 3, 0, 0.5;
    Eigen::VectorXd fx(3);
    fx.setZero();
    gf2.compute(x, fx);
    // Should produce finite output
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(fx[i]));
    }
}

TEST_F(GenericFunctionTest, CopyAssign) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf1(ode);
    auto args = Arguments<3>();
    GenericFunction<-1, -1> gf2(args);
    EXPECT_EQ(gf2.IRows(), 3);

    gf2 = gf1;
    EXPECT_EQ(gf2.IRows(), 5);
    EXPECT_EQ(gf2.ORows(), 3);
}

TEST_F(GenericFunctionTest, MoveAssign) {
    TestODE ode(9.81);
    GenericFunction<-1, -1> gf1(ode);
    auto args = Arguments<3>();
    GenericFunction<-1, -1> gf2(args);

    gf2 = std::move(gf1);
    EXPECT_EQ(gf2.IRows(), 5);
    EXPECT_EQ(gf2.ORows(), 3);
}

TEST_F(GenericFunctionTest, CrossTypeErasure) {
    auto args = Arguments<5>();
    auto scaled = 2.0 * args;
    GenericFunction<5, 5> gf_fixed(scaled);
    GenericFunction<-1, -1> gf_dynamic(gf_fixed);
    EXPECT_EQ(gf_dynamic.IRows(), 5);
    EXPECT_EQ(gf_dynamic.ORows(), 5);

    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;
    Eigen::VectorXd fx_fixed(5), fx_dyn(5);
    fx_fixed.setZero();
    fx_dyn.setZero();
    gf_fixed.compute(x, fx_fixed);
    gf_dynamic.compute(x, fx_dyn);

    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(fx_fixed[i], fx_dyn[i]);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Solver interfaces
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, PackIntoConstraintInterface) {
    auto args = Arguments<5>();
    auto scaled = 2.0 * args;
    GenericFunction<-1, -1> gf(scaled);
    ConstraintInterface ci(gf);

    EXPECT_EQ(ci.IRows(), 5);
    EXPECT_EQ(ci.ORows(), 5);
}

TEST_F(GenericFunctionTest, PackIntoObjectiveInterface) {
    auto args = Arguments<3>();
    auto n = args.norm();
    GenericFunction<-1, 1> gf(n);
    ObjectiveInterface oi(gf);

    EXPECT_EQ(oi.IRows(), 3);
    EXPECT_EQ(oi.ORows(), 1);
}

TEST_F(GenericFunctionTest, CopyConstraintInterface) {
    auto args = Arguments<4>();
    auto sq = args.squared_norm();
    GenericFunction<-1, 1> gf(sq);
    ConstraintInterface ci1(gf);
    ConstraintInterface ci2(ci1);

    EXPECT_EQ(ci2.IRows(), 4);
    EXPECT_EQ(ci2.ORows(), 1);
}

///////////////////////////////////////////////////////////////////////////////
// GenericConditional / Comparative
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, ConditionalGreaterThan) {
    auto args = Arguments<2>();
    auto lhs = args.coeff<0>(); // scalar: x[0]
    auto rhs = args.coeff<1>(); // scalar: x[1]

    ConditionalStatement cond(lhs, ConditionalFlags::GreaterThanFlag, rhs);
    GenericConditional<2> gc(cond);

    Eigen::Vector2d x1(5.0, 3.0); // 5 > 3 → true
    EXPECT_TRUE(gc.compute(x1));

    Eigen::Vector2d x2(2.0, 7.0); // 2 > 7 → false
    EXPECT_FALSE(gc.compute(x2));

    Eigen::Vector2d x3(4.0, 4.0); // 4 > 4 → false (strict)
    EXPECT_FALSE(gc.compute(x3));
}

TEST_F(GenericFunctionTest, ConditionalCompoundAND) {
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();

    // Create a zero constant for comparison
    Eigen::VectorXd zero_val(1);
    zero_val << 0.0;
    auto zero = Constant<2, 1>(2, zero_val);

    // c1: x[0] > 0,  c2: x[1] > 0
    ConditionalStatement c1(x0, ConditionalFlags::GreaterThanFlag, zero);
    ConditionalStatement c2(x1, ConditionalFlags::GreaterThanFlag, zero);

    // AND: both must be true
    ConditionalStatement both(c1, ConditionalFlags::ANDFlag, c2);
    GenericConditional<2> gc(both);

    Eigen::Vector2d tt(1.0, 1.0); // T, T → T
    EXPECT_TRUE(gc.compute(tt));

    Eigen::Vector2d tf(1.0, -1.0); // T, F → F
    EXPECT_FALSE(gc.compute(tf));

    Eigen::Vector2d ft(-1.0, 1.0); // F, T → F
    EXPECT_FALSE(gc.compute(ft));
}

TEST_F(GenericFunctionTest, ComparativeBasic) {
    // GenericComparative has the same interface, just a distinct type
    auto args = Arguments<2>();
    auto lhs = args.coeff<0>();
    auto rhs = args.coeff<1>();

    ConditionalStatement cond(lhs, ConditionalFlags::LessThanFlag, rhs);
    GenericComparative<2> gc(cond);

    Eigen::Vector2d x1(3.0, 5.0); // 3 < 5 → true
    EXPECT_TRUE(gc.compute(x1));

    Eigen::Vector2d x2(7.0, 2.0); // 7 < 2 → false
    EXPECT_FALSE(gc.compute(x2));
}

///////////////////////////////////////////////////////////////////////////////
// SBO vs Heap behavior (indirect — kind_ is private)
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, SmallFunctionInline) {
    // Arguments<3> is small (< 128 bytes), should fit in SBO
    auto args = Arguments<3>();
    GenericFunction<3, 3> gf(args);
    EXPECT_EQ(gf.IRows(), 3);
    EXPECT_EQ(gf.ORows(), 3);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    gf.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 1.0);
    EXPECT_DOUBLE_EQ(fx[1], 2.0);
    EXPECT_DOUBLE_EQ(fx[2], 3.0);

    // Verify jacobian is identity
    Eigen::MatrixXd jx(3, 3);
    jx.setZero();
    fx.setZero();
    gf.compute_jacobian(x, fx, jx);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(jx(i, j), (i == j) ? 1.0 : 0.0);
}

TEST_F(GenericFunctionTest, LargeFunctionHeap) {
    // Build a deeply nested expression that should exceed 128-byte SBO buffer
    auto args = Arguments<3>();
    auto f = args.Sin().Square().Sin().Square().Sin().Square().norm();
    GenericFunction<-1, -1> gf(f);
    EXPECT_EQ(gf.IRows(), 3);
    EXPECT_EQ(gf.ORows(), 1);

    Eigen::VectorXd x(3);
    x << 0.5, 1.0, 1.5;
    Eigen::VectorXd fx_orig(1), fx_gf(1);
    fx_orig.setZero();
    fx_gf.setZero();
    f.compute(x, fx_orig);
    gf.compute(x, fx_gf);
    EXPECT_DOUBLE_EQ(fx_orig[0], fx_gf[0]);
}

TEST_F(GenericFunctionTest, CopyLargeFunction) {
    auto args = Arguments<3>();
    auto f = args.Sin().Square().Sin().Square().Sin().Square().norm();
    GenericFunction<-1, -1> gf1(f);
    GenericFunction<-1, -1> gf2(gf1); // copy

    Eigen::VectorXd x(3);
    x << 0.5, 1.0, 1.5;
    Eigen::VectorXd fx1(1), fx2(1);
    fx1.setZero();
    fx2.setZero();
    gf1.compute(x, fx1);
    gf2.compute(x, fx2);
    EXPECT_DOUBLE_EQ(fx1[0], fx2[0]);
}

TEST_F(GenericFunctionTest, MoveLargeFunction) {
    auto args = Arguments<3>();
    auto f = args.Sin().Square().Sin().Square().Sin().Square().norm();
    GenericFunction<-1, -1> gf1(f);

    // Compute reference before move
    Eigen::VectorXd x(3);
    x << 0.5, 1.0, 1.5;
    Eigen::VectorXd fx_ref(1);
    fx_ref.setZero();
    gf1.compute(x, fx_ref);

    GenericFunction<-1, -1> gf2(std::move(gf1)); // move
    Eigen::VectorXd fx(1);
    fx.setZero();
    gf2.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], fx_ref[0]);
}
