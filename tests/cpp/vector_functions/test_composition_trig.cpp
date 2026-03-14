///////////////////////////////////////////////////////////////////////////////
// Nested / Composition and Trig / Math function unit tests
//
// Extracted from test_common_functions.cpp — Nested / Composition and
// Trig / Math functions sections.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class CommonFunctionsTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Nested / Composition
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, NestedComposition) {
    auto args = Arguments<3>();
    auto scaled = 2.0 * args;
    auto normed = scaled.norm();
    EXPECT_EQ(normed.IRows(), 3);
    EXPECT_EQ(normed.ORows(), 1);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 2.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    normed.compute(x, fx);
    // ||2x|| = 2*||x|| = 2*3 = 6
    EXPECT_NEAR(fx[0], 6.0, 1e-12);
}

TEST_F(CommonFunctionsTest, NestedAdjointConsistency) {
    auto args = Arguments<4>();
    auto scaled = 3.0 * args;
    auto normed = scaled.norm();
    Eigen::VectorXd x = deterministic_random_vector(4, 70, 1.0, 10.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(normed, x, lm);
}

TEST_F(CommonFunctionsTest, TripleCompositionAdjoint) {
    auto args = Arguments<3>();
    auto result = 5.0 * args;
    auto final_expr = result.norm();
    Eigen::VectorXd x = deterministic_random_vector(3, 71, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(final_expr, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Trig / Math functions
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, SinAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto s = args.Sin();
    Eigen::VectorXd x(3);
    x << 0.5, 1.0, 1.5;
    // J = diag(cos(x))
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 0) = std::cos(x[0]);
    expected(1, 1) = std::cos(x[1]);
    expected(2, 2) = std::cos(x[2]);
    verify_jacobian_analytical(s, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, CosAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto c = args.Cos();
    Eigen::VectorXd x(3);
    x << 0.5, 1.0, 1.5;
    // J = diag(-sin(x))
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 0) = -std::sin(x[0]);
    expected(1, 1) = -std::sin(x[1]);
    expected(2, 2) = -std::sin(x[2]);
    verify_jacobian_analytical(c, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, SqrtAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto sq = args.Sqrt();
    Eigen::VectorXd x(3);
    x << 4.0, 9.0, 16.0;
    // J = diag(1 / (2*sqrt(x)))
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 0) = 1.0 / (2.0 * std::sqrt(x[0]));
    expected(1, 1) = 1.0 / (2.0 * std::sqrt(x[1]));
    expected(2, 2) = 1.0 / (2.0 * std::sqrt(x[2]));
    verify_jacobian_analytical(sq, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, ExpAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto ex = args.Exp();
    Eigen::VectorXd x(3);
    x << 0.0, 1.0, -1.0;
    // J = diag(exp(x))
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 0) = std::exp(x[0]);
    expected(1, 1) = std::exp(x[1]);
    expected(2, 2) = std::exp(x[2]);
    verify_jacobian_analytical(ex, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, SquareAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto sq = args.Square();
    Eigen::VectorXd x(3);
    x << 2.0, 3.0, -1.0;
    // J = diag(2*x)
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 0) = 2.0 * x[0];
    expected(1, 1) = 2.0 * x[1];
    expected(2, 2) = 2.0 * x[2];
    verify_jacobian_analytical(sq, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, CwiseProductAdjointConsistency) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto cw = a.cwiseProduct(b);
    Eigen::VectorXd x = deterministic_random_vector(6, 80, 1.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 81, -1.0, 1.0);
    verify_adjoint_consistency(cw, x, lm);
}
