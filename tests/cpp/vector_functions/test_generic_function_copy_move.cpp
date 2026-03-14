///////////////////////////////////////////////////////////////////////////////
// GenericFunction type erasure tests — Copy / Move / CrossType / SBO vs Heap
//
// Tests copy/move constructors and assignment, cross-type erasure,
// SBO vs heap behavior.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Copy / Move
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, CopyConstruct) {
    BrachODE ode(9.81);
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
    BrachODE ode(9.81);
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
    BrachODE ode(9.81);
    GenericFunction<-1, -1> gf1(ode);
    auto args = Arguments<3>();
    GenericFunction<-1, -1> gf2(args);
    EXPECT_EQ(gf2.IRows(), 3);

    gf2 = gf1;
    EXPECT_EQ(gf2.IRows(), 5);
    EXPECT_EQ(gf2.ORows(), 3);
}

TEST_F(GenericFunctionTest, MoveAssign) {
    BrachODE ode(9.81);
    GenericFunction<-1, -1> gf1(ode);
    auto args = Arguments<3>();
    GenericFunction<-1, -1> gf2(args);

    gf2 = std::move(gf1);
    EXPECT_EQ(gf2.IRows(), 5);
    EXPECT_EQ(gf2.ORows(), 3);
}

///////////////////////////////////////////////////////////////////////////////
// CrossTypeErasure
///////////////////////////////////////////////////////////////////////////////

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
