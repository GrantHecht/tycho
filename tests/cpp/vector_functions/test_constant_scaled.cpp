///////////////////////////////////////////////////////////////////////////////
// CommonFunctions unit tests — Constant / Scaled
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Constant
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ConstantOutput) {
    Eigen::VectorXd val(3);
    val << 7.0, 8.0, 9.0;
    auto c = Constant<-1, -1>(4, val);
    EXPECT_EQ(c.input_rows(), 4);
    EXPECT_EQ(c.output_rows(), 3);

    Eigen::VectorXd x(4);
    x << 1, 2, 3, 4;
    Eigen::VectorXd fx(3);
    fx.setZero();
    c.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 7.0);
    EXPECT_DOUBLE_EQ(fx[1], 8.0);
    EXPECT_DOUBLE_EQ(fx[2], 9.0);
}

TEST_F(CommonFunctionsTest, ConstantZeroJacobian) {
    Eigen::VectorXd val(2);
    val << 1.0, 2.0;
    auto c = Constant<-1, -1>(3, val);
    Eigen::VectorXd x(3);
    x << 1, 2, 3;
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(2, 3);
    verify_jacobian_analytical(c, x, expected);
}

///////////////////////////////////////////////////////////////////////////////
// Scaled
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ScaledJacobian) {
    auto args = Arguments<3>();
    auto scaled = 2.5 * args;
    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    // Jacobian of 2.5*I is 2.5*I
    Eigen::MatrixXd expected = 2.5 * Eigen::MatrixXd::Identity(3, 3);
    verify_jacobian_analytical(scaled, x, expected);
}

TEST_F(CommonFunctionsTest, NegativeScale) {
    auto args = Arguments<3>();
    auto neg = (-1.0) * args;
    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    neg.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], -1.0);
    EXPECT_DOUBLE_EQ(fx[1], -2.0);
    EXPECT_DOUBLE_EQ(fx[2], -3.0);
}
