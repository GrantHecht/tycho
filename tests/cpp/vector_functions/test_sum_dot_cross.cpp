///////////////////////////////////////////////////////////////////////////////
// CommonFunctions unit tests — Sum / Difference, DotProduct, CrossProduct
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

class CommonFunctionsTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Sum / Difference
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, SumAdjointConsistency) {
    // operator+ on same-sized VFs is element-wise sum (TwoFunctionSum)
    auto args = Arguments<4>();
    auto a = args.template head<2>();
    auto b = args.template tail<2>();
    auto sum = a + b; // element-wise sum: head(2) + tail(2), both OR=2
    EXPECT_EQ(sum.ORows(), 2);

    Eigen::VectorXd x = deterministic_random_vector(4, 42, 1.0, 10.0);
    Eigen::VectorXd lm = deterministic_random_vector(2, 43, -1.0, 1.0);
    verify_adjoint_consistency(sum, x, lm);
}

TEST_F(CommonFunctionsTest, ScaledSumAdjointConsistency) {
    // 2*args + 3*args = 5*args (element-wise sum of same-sized VFs)
    auto args = Arguments<3>();
    auto a = 2.0 * args;
    auto b = 3.0 * args;
    auto result = a + b;
    EXPECT_EQ(result.ORows(), 3);

    Eigen::VectorXd x = deterministic_random_vector(3, 44, 1.0, 10.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 45, -1.0, 1.0);
    verify_adjoint_consistency(result, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// DotProduct
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, DotProductDimensions) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto dp = a.dot(b);
    EXPECT_EQ(dp.IRows(), 6);
    EXPECT_EQ(dp.ORows(), 1);
}

TEST_F(CommonFunctionsTest, DotProductKnownValue) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto dp = a.dot(b);
    Eigen::VectorXd x(6);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    dp.compute(x, fx);
    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_DOUBLE_EQ(fx[0], 32.0);
}

TEST_F(CommonFunctionsTest, DotProductAnalyticalJacobian) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto dp = a.dot(b);
    Eigen::VectorXd x(6);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    // J = [b^T, a^T] = [4 5 6 1 2 3]
    Eigen::MatrixXd expected(1, 6);
    expected << 4.0, 5.0, 6.0, 1.0, 2.0, 3.0;
    verify_jacobian_analytical(dp, x, expected);
}

///////////////////////////////////////////////////////////////////////////////
// CrossProduct
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, CrossProductDimensions) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto cp = a.cross(b);
    EXPECT_EQ(cp.IRows(), 6);
    EXPECT_EQ(cp.ORows(), 3);
}

TEST_F(CommonFunctionsTest, CrossProductKnownValue) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto cp = a.cross(b);
    Eigen::VectorXd x(6);
    x << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0; // e1 x e2 = e3
    Eigen::VectorXd fx(3);
    fx.setZero();
    cp.compute(x, fx);
    EXPECT_NEAR(fx[0], 0.0, 1e-14);
    EXPECT_NEAR(fx[1], 0.0, 1e-14);
    EXPECT_NEAR(fx[2], 1.0, 1e-14);
}

TEST_F(CommonFunctionsTest, CrossProductAnalyticalJacobian) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto cp = a.cross(b);
    Eigen::VectorXd x(6);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    // Cross product: c = a x b
    // dc/da = [a]_x^T = skew(b)^T = -skew(b) applied to left
    // J = [-skew(b), skew(a)]
    // skew(v) = [0 -v3 v2; v3 0 -v1; -v2 v1 0]
    // dc/da = [0 b3 -b2; -b3 0 b1; b2 -b1 0]
    // dc/db = [0 -a3 a2; a3 0 -a1; -a2 a1 0]
    double a1 = 1, a2 = 2, a3 = 3, b1 = 4, b2 = 5, b3 = 6;
    Eigen::MatrixXd expected(3, 6);
    // clang-format off
    expected << 0,  b3, -b2,  0, -a3,  a2,
               -b3,  0,  b1, a3,   0, -a1,
                b2, -b1,  0, -a2,  a1,  0;
    // clang-format on
    verify_jacobian_analytical(cp, x, expected, 1e-11);
}
