///////////////////////////////////////////////////////////////////////////////
// CwiseFunctionOperator Jacobian stress tests
//
// All CwiseFunctionOperator types share the same compute_jacobian_impl
// which scales the inner Jacobian in-place via right_jacobian_product
// with Aliased=true.  These tests exercise every cwise op against a
// variety of inner function structures — segment sums, scaled segments,
// differences, products, nested compositions, GenericFunction wrappers,
// and larger input sizes — to ensure the aliased scaling is correct.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// FD Jacobian helper
///////////////////////////////////////////////////////////////////////////////

template <class Func>
Eigen::MatrixXd fd_jacobian(Func &f, const Eigen::VectorXd &x, double eps = 1e-7) {
    int nr = f.ORows(), nc = f.IRows();
    Eigen::MatrixXd jac(nr, nc);
    Eigen::VectorXd xp = x, fp(nr), fm(nr);
    for (int j = 0; j < nc; ++j) {
        xp[j] = x[j] + eps;
        f.compute(xp, fp);
        xp[j] = x[j] - eps;
        f.compute(xp, fm);
        jac.col(j) = (fp - fm) / (2.0 * eps);
        xp[j] = x[j];
    }
    return jac;
}

///////////////////////////////////////////////////////////////////////////////
// Verify Jacobian matches FD (direct and through GenericFunction)
///////////////////////////////////////////////////////////////////////////////

template <class Func>
void check_jac_fd(Func &f, const Eigen::VectorXd &x, double tol = 1e-5) {
    int nr = f.ORows(), nc = f.IRows();
    Eigen::VectorXd fx(nr);
    Eigen::MatrixXd jx(nr, nc);
    fx.setZero();
    jx.setZero();
    f.compute_jacobian(x, fx, jx);
    Eigen::MatrixXd jx_fd = fd_jacobian(f, x);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), tol)
                << "at (" << i << "," << j << ")";
}

template <int IR, int OR, class Func>
void check_gf_jac_fd(Func &f, const Eigen::VectorXd &x, double tol = 1e-5) {
    GenericFunction<IR, OR> gf(f);
    Eigen::MatrixXd jx = gf.jacobian(x);
    Eigen::MatrixXd jx_fd = fd_jacobian(gf, x);
    for (int i = 0; i < gf.ORows(); ++i)
        for (int j = 0; j < gf.IRows(); ++j)
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), tol)
                << "GF at (" << i << "," << j << ")";
}

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class CwiseJacobianStressTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Section 1: Every CwiseOp on 2-segment sum  (x+y)
//
// All go through CwiseFunctionOperator::compute_jacobian_impl with the
// same Aliased=true self-scaling.  One test per op family.
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Sin_SegmentSum) {
    auto a = Arguments<2>();
    auto f = sin(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.7, -0.3;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Cos_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.7, -0.3;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Tan_SegmentSum) {
    auto a = Arguments<2>();
    auto f = tan(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.3, -0.1;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, ArcSin_SegmentSum) {
    auto a = Arguments<2>();
    // Keep argument in (-1, 1)
    auto f = asin(0.3 * (a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.2, -0.1;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, ArcCos_SegmentSum) {
    auto a = Arguments<2>();
    auto f = acos(0.3 * (a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.2, -0.1;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, ArcTan_SegmentSum) {
    auto a = Arguments<2>();
    auto f = atan(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 1.5, -0.7;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Exp_SegmentSum) {
    auto a = Arguments<2>();
    auto f = exp(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.3, -0.1;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Log_SegmentSum) {
    auto a = Arguments<2>();
    // Ensure positive argument
    auto f = log(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 2.0, 1.0;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Sqrt_SegmentSum) {
    auto a = Arguments<2>();
    auto f = sqrt(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 2.0, 1.0;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Sinh_SegmentSum) {
    auto a = Arguments<2>();
    auto f = sinh(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.5, -0.2;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Cosh_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cosh(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.5, -0.2;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Tanh_SegmentSum) {
    auto a = Arguments<2>();
    auto f = tanh(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.5, -0.2;
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 2: Scaled segment sums — ScaledSegment inside sum
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Cos_ScaledSegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(3.0 * a.coeff<0>() + (-2.0) * a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.4, 0.6;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Sin_ScaledSegmentSum) {
    auto a = Arguments<2>();
    auto f = sin(0.5 * a.coeff<0>() + 1.5 * a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 1.0, -0.5;
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 3: FunctionDifference (x - y) inside cwise
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Cos_SegmentDifference) {
    auto a = Arguments<2>();
    auto f = cos(a.coeff<0>() - a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.8, 0.3;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Exp_SegmentDifference) {
    auto a = Arguments<2>();
    auto f = exp(a.coeff<0>() - a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.5, 0.3;
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 4: MultiFunctionSum (3+ segments) inside cwise
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Sin_ThreeSegmentSum) {
    auto a = Arguments<3>();
    auto f = sin(a.coeff<0>() + a.coeff<1>() + a.coeff<2>());
    Eigen::VectorXd tp(3);
    tp << 0.1, -0.2, 0.3;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Cos_FourSegmentSum) {
    auto a = Arguments<4>();
    auto f = cos(a.coeff<0>() + a.coeff<1>() + a.coeff<2>() + a.coeff<3>());
    Eigen::VectorXd tp(4);
    tp << 0.1, -0.2, 0.3, -0.4;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Exp_FiveSegmentSum) {
    auto a = Arguments<5>();
    auto f = exp(a.coeff<0>() + a.coeff<1>() + a.coeff<2>() + a.coeff<3>() + a.coeff<4>());
    Eigen::VectorXd tp(5);
    tp << 0.05, -0.1, 0.05, -0.1, 0.1;
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 5: Nested cwise compositions — cwise(cwise(segment_sum))
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Cos_Sin_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(sin(a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.5, -0.3;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Exp_Cos_SegmentSum) {
    auto a = Arguments<2>();
    auto f = exp(cos(a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.4, -0.2;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Sin_Sqrt_SegmentSum) {
    auto a = Arguments<2>();
    // sqrt requires positive argument
    auto f = sin(sqrt(a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 1.5, 0.5;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Log_Exp_SegmentSum) {
    auto a = Arguments<2>();
    auto f = log(exp(a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.3, -0.1;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, TripleNesting) {
    auto a = Arguments<2>();
    auto f = cos(sin(exp(a.coeff<0>() + a.coeff<1>())));
    Eigen::VectorXd tp(2);
    tp << 0.1, -0.05;
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 6: Cwise applied to products and compound expressions
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Cos_ScaledSum) {
    auto a = Arguments<2>();
    auto f = cos(2.0 * (a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Sin_SumTimesScalar) {
    auto a = Arguments<2>();
    auto f = sin(5.0 * (a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.1, 0.05;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Cos_SumOfProducts) {
    // x*x + y*y is NOT a sum of segments — exercises the general path
    auto a = Arguments<2>();
    auto x = a.coeff<0>();
    auto y = a.coeff<1>();
    auto f = cos(x * x + y * y);
    Eigen::VectorXd tp(2);
    tp << 0.5, -0.3;
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 7: Through GenericFunction (virtual dispatch)
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, GF_Cos_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(a.coeff<0>() + a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.7, -0.3;
    check_gf_jac_fd<2, 1>(f, tp);
}

TEST_F(CwiseJacobianStressTest, GF_Sin_ScaledSegmentSum) {
    auto a = Arguments<2>();
    auto f = sin(3.0 * a.coeff<0>() + (-2.0) * a.coeff<1>());
    Eigen::VectorXd tp(2);
    tp << 0.4, 0.6;
    check_gf_jac_fd<2, 1>(f, tp);
}

TEST_F(CwiseJacobianStressTest, GF_Exp_ThreeSegmentSum) {
    auto a = Arguments<3>();
    auto f = exp(a.coeff<0>() + a.coeff<1>() + a.coeff<2>());
    Eigen::VectorXd tp(3);
    tp << 0.1, -0.2, 0.05;
    check_gf_jac_fd<3, 1>(f, tp);
}

TEST_F(CwiseJacobianStressTest, GF_Cos_Sin_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(sin(a.coeff<0>() + a.coeff<1>()));
    Eigen::VectorXd tp(2);
    tp << 0.5, -0.3;
    check_gf_jac_fd<2, 1>(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 8: Adjoint and Hessian consistency for cwise on segment sums
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Adjoint_Sin_SegmentSum) {
    auto a = Arguments<2>();
    auto f = sin(a.coeff<0>() + a.coeff<1>());
    GenericFunction<2, 1> gf(f);
    Eigen::VectorXd tp(2), lm(1);
    tp << 0.7, -0.3;
    lm << 1.0;
    verify_adjoint_consistency(gf, tp, lm, 1e-12);
}

TEST_F(CwiseJacobianStressTest, Adjoint_Exp_SegmentSum) {
    auto a = Arguments<2>();
    auto f = exp(a.coeff<0>() + a.coeff<1>());
    GenericFunction<2, 1> gf(f);
    Eigen::VectorXd tp(2), lm(1);
    tp << 0.3, -0.1;
    lm << 1.0;
    verify_adjoint_consistency(gf, tp, lm, 1e-12);
}

TEST_F(CwiseJacobianStressTest, Hessian_Cos_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(a.coeff<0>() + a.coeff<1>());
    GenericFunction<2, 1> gf(f);
    Eigen::VectorXd tp(2), lm(1);
    tp << 0.7, -0.3;
    lm << 1.0;
    verify_hessian_consistency(gf, tp, lm, 1e-12);
}

TEST_F(CwiseJacobianStressTest, Hessian_Sin_ScaledSegmentSum) {
    auto a = Arguments<2>();
    auto f = sin(3.0 * a.coeff<0>() + (-2.0) * a.coeff<1>());
    GenericFunction<2, 1> gf(f);
    Eigen::VectorXd tp(2), lm(1);
    tp << 0.4, 0.6;
    lm << 1.0;
    verify_hessian_consistency(gf, tp, lm, 1e-12);
}

TEST_F(CwiseJacobianStressTest, Hessian_Exp_ThreeSegmentSum) {
    auto a = Arguments<3>();
    auto f = exp(a.coeff<0>() + a.coeff<1>() + a.coeff<2>());
    GenericFunction<3, 1> gf(f);
    Eigen::VectorXd tp(3), lm(1);
    tp << 0.1, -0.2, 0.05;
    lm << 1.0;
    verify_hessian_consistency(gf, tp, lm, 1e-12);
}

TEST_F(CwiseJacobianStressTest, Hessian_Cos_Sin_SegmentSum) {
    auto a = Arguments<2>();
    auto f = cos(sin(a.coeff<0>() + a.coeff<1>()));
    GenericFunction<2, 1> gf(f);
    Eigen::VectorXd tp(2), lm(1);
    tp << 0.5, -0.3;
    lm << 1.0;
    verify_hessian_consistency(gf, tp, lm, 1e-12);
}

///////////////////////////////////////////////////////////////////////////////
// Section 9: Larger input sizes
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, Cos_TwoSegSum_LargeInput) {
    // x and y are segments of a 10-element input
    auto a = Arguments<10>();
    auto f = cos(a.coeff<3>() + a.coeff<7>());
    Eigen::VectorXd tp = deterministic_random_vector(10, 99, -1.0, 1.0);
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, Sin_ThreeSegSum_LargeInput) {
    auto a = Arguments<10>();
    auto f = sin(a.coeff<0>() + a.coeff<4>() + a.coeff<9>());
    Eigen::VectorXd tp = deterministic_random_vector(10, 100, -1.0, 1.0);
    check_jac_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Section 10: Complex ODE-like expressions combining segment sums in cwise
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseJacobianStressTest, ODE_SinTheta_Times_V) {
    // ODE-style: sin(theta) * v where theta and v are segments
    auto a = Arguments<5>();
    auto theta = a.coeff<4>();
    auto v = a.coeff<2>();
    auto f = sin(theta) * v;
    GenericFunction<5, 1> gf(f);
    Eigen::VectorXd tp(5);
    tp << 0.0, -1.0, 2.0, 0.5, 0.8;
    check_gf_jac_fd<5, 1>(f, tp);
}

TEST_F(CwiseJacobianStressTest, ODE_CosThetaPlusV) {
    // cos(theta + v) where theta and v are segments
    auto a = Arguments<5>();
    auto theta = a.coeff<4>();
    auto v = a.coeff<2>();
    auto f = cos(theta + v);
    Eigen::VectorXd tp(5);
    tp << 0.0, -1.0, 2.0, 0.5, 0.8;
    check_jac_fd(f, tp);
}

TEST_F(CwiseJacobianStressTest, ODE_StackedTrigSegmentSums) {
    // Stacked: [sin(x+y); cos(x+y); exp(x+y)]
    auto a = Arguments<2>();
    auto xy = a.coeff<0>() + a.coeff<1>();
    auto f = stack(sin(xy), cos(xy), exp(xy));
    GenericFunction<2, 3> gf(f);
    Eigen::VectorXd tp(2);
    tp << 0.3, -0.1;
    check_gf_jac_fd<2, 3>(f, tp);
}
