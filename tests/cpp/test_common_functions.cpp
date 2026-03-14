///////////////////////////////////////////////////////////////////////////////
// CommonFunctions unit tests
//
// Tests the ~40 VectorFunction building blocks: Arguments, Segment,
// Constant, Scaled, DotProduct, CrossProduct, Norms, Normalized,
// Stacked, Padded, Nested, trig/math functions, etc.
//
// Validation approach:
//  - Simple functions: compare jacobian against analytical closed form
//  - Complex compositions: adjoint consistency (gx == J^T * lm)
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
// Arguments / Segment
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ArgumentsDimensions) {
    auto args = Arguments<5>();
    EXPECT_EQ(args.IRows(), 5);
    EXPECT_EQ(args.ORows(), 5);
}

TEST_F(CommonFunctionsTest, ArgumentsIdentityJacobian) {
    auto args = Arguments<4>();
    Eigen::VectorXd x(4);
    x << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd expected = Eigen::MatrixXd::Identity(4, 4);
    verify_jacobian_analytical(args, x, expected);
}

TEST_F(CommonFunctionsTest, SegmentDimensions) {
    auto args = Arguments<6>();
    auto seg = args.template segment<3, 2>(); // 3 elements starting at index 2
    EXPECT_EQ(seg.IRows(), 6);
    EXPECT_EQ(seg.ORows(), 3);
}

TEST_F(CommonFunctionsTest, SegmentValues) {
    auto args = Arguments<6>();
    auto seg = args.template segment<3, 2>();
    Eigen::VectorXd x(6);
    x << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    seg.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 30.0);
    EXPECT_DOUBLE_EQ(fx[1], 40.0);
    EXPECT_DOUBLE_EQ(fx[2], 50.0);
}

TEST_F(CommonFunctionsTest, SegmentJacobian) {
    auto args = Arguments<5>();
    auto seg = args.template segment<2, 1>(); // 2 elements starting at index 1
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;
    // Jacobian should be [0 1 0 0 0; 0 0 1 0 0]
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(2, 5);
    expected(0, 1) = 1.0;
    expected(1, 2) = 1.0;
    verify_jacobian_analytical(seg, x, expected);
}

///////////////////////////////////////////////////////////////////////////////
// Constant
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ConstantOutput) {
    Eigen::VectorXd val(3);
    val << 7.0, 8.0, 9.0;
    auto c = Constant<-1, -1>(4, val);
    EXPECT_EQ(c.IRows(), 4);
    EXPECT_EQ(c.ORows(), 3);

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

///////////////////////////////////////////////////////////////////////////////
// Normalized
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, NormalizedUnitVector) {
    auto args = Arguments<3>();
    auto nrm = args.normalized();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    nrm.compute(x, fx);
    EXPECT_NEAR(fx[0], 0.6, 1e-14);
    EXPECT_NEAR(fx[1], 0.8, 1e-14);
    EXPECT_NEAR(fx[2], 0.0, 1e-14);
    // Output norm should be 1
    EXPECT_NEAR(fx.norm(), 1.0, 1e-14);
}

TEST_F(CommonFunctionsTest, NormalizedAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto nrm = args.normalized();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    double n = x.norm();
    // J = (I - x_hat * x_hat^T) / ||x||
    Eigen::VectorXd x_hat = x / n;
    Eigen::MatrixXd expected = (Eigen::MatrixXd::Identity(3, 3) - x_hat * x_hat.transpose()) / n;
    verify_jacobian_analytical(nrm, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, NormalizedPowerAdjointConsistency) {
    auto args = Arguments<3>();
    auto np = args.normalized_power<2>();
    Eigen::VectorXd x = deterministic_random_vector(3, 52, 1.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 53, -1.0, 1.0);
    verify_adjoint_consistency(np, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Stacked
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, StackedDimensions) {
    auto args = Arguments<5>();
    auto a = args.template head<2>();
    auto b = args.template segment<3, 2>();
    auto stacked = StackedOutputs{a, b};
    EXPECT_EQ(stacked.IRows(), 5);
    EXPECT_EQ(stacked.ORows(), 5); // 2 + 3
}

TEST_F(CommonFunctionsTest, StackedValues) {
    auto args = Arguments<5>();
    auto a = args.template head<2>();
    auto b = args.template tail<3>();
    auto stacked = StackedOutputs{a, b};
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;
    Eigen::VectorXd fx(5);
    fx.setZero();
    stacked.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 1.0);
    EXPECT_DOUBLE_EQ(fx[1], 2.0);
    EXPECT_DOUBLE_EQ(fx[2], 3.0);
    EXPECT_DOUBLE_EQ(fx[3], 4.0);
    EXPECT_DOUBLE_EQ(fx[4], 5.0);
}

TEST_F(CommonFunctionsTest, StackedAdjointConsistency) {
    auto args = Arguments<6>();
    auto a = args.template head<2>();
    auto b = args.template segment<2, 2>();
    auto c = args.template tail<2>();
    auto stacked = StackedOutputs{a, b, c};
    EXPECT_EQ(stacked.ORows(), 6);

    Eigen::VectorXd x = deterministic_random_vector(6, 60, 1.0, 10.0);
    Eigen::VectorXd lm = deterministic_random_vector(6, 61, -1.0, 1.0);
    verify_adjoint_consistency(stacked, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Padded
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, PaddedUpperDimensions) {
    auto args = Arguments<3>();
    auto padded = args.padded_upper<2>();
    EXPECT_EQ(padded.IRows(), 3);
    EXPECT_EQ(padded.ORows(), 5); // 3 + 2 zeros on top
}

TEST_F(CommonFunctionsTest, PaddedLowerDimensions) {
    auto args = Arguments<3>();
    auto padded = args.padded_lower<2>();
    EXPECT_EQ(padded.IRows(), 3);
    EXPECT_EQ(padded.ORows(), 5);
}

TEST_F(CommonFunctionsTest, PaddedAdjointConsistency) {
    auto args = Arguments<3>();
    auto padded = args.template padded<2, 3>();
    EXPECT_EQ(padded.ORows(), 8); // 2 + 3 + 3
    Eigen::VectorXd x = deterministic_random_vector(3, 62, 1.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(8, 63, -1.0, 1.0);
    verify_adjoint_consistency(padded, x, lm);
}

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

///////////////////////////////////////////////////////////////////////////////
// Elements
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ElementsExtraction) {
    auto args = Arguments<5>();
    auto elems = args.template elements<0, 2, 4>();
    EXPECT_EQ(elems.ORows(), 3);
    Eigen::VectorXd x(5);
    x << 10.0, 20.0, 30.0, 40.0, 50.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    elems.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 10.0);
    EXPECT_DOUBLE_EQ(fx[1], 30.0);
    EXPECT_DOUBLE_EQ(fx[2], 50.0);
}

TEST_F(CommonFunctionsTest, ElementsJacobian) {
    auto args = Arguments<5>();
    auto elems = args.template elements<1, 3>();
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;
    // Jacobian: permutation rows of identity
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(2, 5);
    expected(0, 1) = 1.0;
    expected(1, 3) = 1.0;
    verify_jacobian_analytical(elems, x, expected);
}

///////////////////////////////////////////////////////////////////////////////
// CwiseSum
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, CwiseSumValue) {
    auto args = Arguments<4>();
    auto s = args.sum();
    EXPECT_EQ(s.ORows(), 1);
    Eigen::VectorXd x(4);
    x << 1.0, 2.0, 3.0, 4.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    s.compute(x, fx);
    EXPECT_NEAR(fx[0], 10.0, 1e-14);
}

TEST_F(CommonFunctionsTest, CwiseSumJacobian) {
    auto args = Arguments<4>();
    auto s = args.sum();
    Eigen::VectorXd x(4);
    x << 1, 2, 3, 4;
    // J = [1 1 1 1]
    Eigen::MatrixXd expected(1, 4);
    expected << 1.0, 1.0, 1.0, 1.0;
    verify_jacobian_analytical(s, x, expected);
}

///////////////////////////////////////////////////////////////////////////////
// CwiseInverse
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, CwiseInverseAdjoint) {
    auto args = Arguments<3>();
    auto inv = args.cwiseInverse();
    Eigen::VectorXd x(3);
    x << 2.0, 4.0, 5.0;
    Eigen::VectorXd lm = deterministic_random_vector(3, 90, -1.0, 1.0);
    verify_adjoint_consistency(inv, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Arithmetic chains
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ChainedArithmeticAdjointConsistency) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    // Scale + stack: [2*a; 3*b]
    auto result = StackedOutputs{2.0 * a, 3.0 * b}; // stacked, OR=6

    Eigen::VectorXd x = deterministic_random_vector(6, 100, 1.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(6, 101, -1.0, 1.0);
    verify_adjoint_consistency(result, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Complex compositions for adjoint/hessian cross-check
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, NormOfCompositionAdjoint) {
    auto args = Arguments<4>();
    auto sq = args.Square();
    auto n = sq.norm();
    Eigen::VectorXd x = deterministic_random_vector(4, 110, 0.5, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(n, x, lm);
}

TEST_F(CommonFunctionsTest, DotOfCompositionsAdjoint) {
    auto args = Arguments<6>();
    auto a = args.template head<3>().Sin();
    auto b = args.template tail<3>().Cos();
    auto dp = a.dot(b);
    Eigen::VectorXd x = deterministic_random_vector(6, 111, 0.1, 3.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(dp, x, lm);
}

TEST_F(CommonFunctionsTest, FiveNestedLevels) {
    auto args = Arguments<3>();
    auto step1 = 2.0 * args;
    auto step2 = step1.Sin();
    auto step3 = step2.Square();
    auto step4 = step3.sum();
    auto step5 = 3.0 * step3;

    Eigen::VectorXd x = deterministic_random_vector(3, 112, 0.5, 2.0);
    Eigen::VectorXd lm4(1);
    lm4 << 1.0;
    verify_adjoint_consistency(step4, x, lm4);

    Eigen::VectorXd lm5 = deterministic_random_vector(3, 113, -1.0, 1.0);
    verify_adjoint_consistency(step5, x, lm5);
}
