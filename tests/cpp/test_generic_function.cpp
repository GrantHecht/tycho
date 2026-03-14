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
