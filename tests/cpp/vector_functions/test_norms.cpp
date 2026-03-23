///////////////////////////////////////////////////////////////////////////////
// Norms unit tests
//
// Extracted from test_common_functions.cpp — Norms section.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Norms
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, NormValue) {
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    n.compute(x, fx);
    EXPECT_NEAR(fx[0], 5.0, 1e-14);
}

TEST_F(CommonFunctionsTest, NormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    // J = x^T / ||x|| = [3/5, 4/5, 0]
    double norm_val = 5.0;
    Eigen::MatrixXd expected(1, 3);
    expected << x[0] / norm_val, x[1] / norm_val, x[2] / norm_val;
    verify_jacobian_analytical(n, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, SquaredNormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    // d(x^T x)/dx = 2*x^T
    Eigen::MatrixXd expected(1, 3);
    expected << 2.0, 4.0, 6.0;
    verify_jacobian_analytical(sn, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, InverseNormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto inv_n = args.inverse_norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    double norm_val = 5.0;
    // d(1/||x||)/dx = -x^T / ||x||^3
    Eigen::MatrixXd expected(1, 3);
    double n3 = norm_val * norm_val * norm_val;
    expected << -x[0] / n3, -x[1] / n3, -x[2] / n3;
    verify_jacobian_analytical(inv_n, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, InverseSquaredNormAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto isn = args.inverse_squared_norm();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    double n2 = x.squaredNorm();
    // d(1/||x||^2)/dx = -2*x^T / ||x||^4
    Eigen::MatrixXd expected(1, 3);
    double n4 = n2 * n2;
    expected << -2.0 * x[0] / n4, -2.0 * x[1] / n4, -2.0 * x[2] / n4;
    verify_jacobian_analytical(isn, x, expected, 1e-12);
}

TEST_F(CommonFunctionsTest, NormPowerValue) {
    auto args = Arguments<3>();
    auto np = args.norm_power<3>();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    np.compute(x, fx);
    EXPECT_NEAR(fx[0], 125.0, 1e-10); // 5^3 = 125
}

TEST_F(CommonFunctionsTest, NormPowerAdjointConsistency) {
    auto args = Arguments<3>();
    auto np = args.norm_power<3>();
    Eigen::VectorXd x = deterministic_random_vector(3, 50, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(np, x, lm);
}

TEST_F(CommonFunctionsTest, NormAdjointConsistencyRandomInput) {
    auto args = Arguments<5>();
    auto n = args.norm();
    Eigen::VectorXd x = deterministic_random_vector(5, 51, 1.0, 10.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(n, x, lm);
}
