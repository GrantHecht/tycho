///////////////////////////////////////////////////////////////////////////////
// GenericFunction basic erasure tests
//
// Extracted from test_generic_function.cpp — BrachODE struct,
// BUILD_ODE_FROM_EXPRESSION macro, and Basic erasure section.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

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
    BrachODE ode(9.81);
    GenericFunction<-1, -1> gf(ode);
    EXPECT_EQ(gf.IRows(), 5);
    EXPECT_EQ(gf.ORows(), 3);
}

TEST_F(GenericFunctionTest, ComputeMatchesOriginal) {
    BrachODE ode(9.81);
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
    BrachODE ode(9.81);
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
