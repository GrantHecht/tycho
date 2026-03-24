///////////////////////////////////////////////////////////////////////////////
// CwiseFunctionOperator + segment-sum Jacobian correctness tests
//
// Verifies that CwiseFunctionOperator's in-place Jacobian scaling (which
// passes jx_ as both target and right with Aliased=true) produces correct
// results when the inner function is a sum of segments.
//
// This is a regression test for the bug where TwoFunctionSum's
// right_jacobian_product ignored the Aliased flag, using PlusEquals on
// non-zero aliased columns instead of DirectAssignment.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class CwiseSegmentSumJacobianTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// cos(2*(x+y)) — the minimal reproducer
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, Cos2XplusY_Direct) {
    auto a = Arguments<2>();
    auto x = a.coeff<0>();
    auto y = a.coeff<1>();
    auto f = cos(2.0 * (x + y));

    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    verify_jacobian_fd(f, tp);
}

TEST_F(CwiseSegmentSumJacobianTest, Cos2XplusY_GenericFunction) {
    auto a = Arguments<2>();
    auto f = cos(2.0 * (a.coeff<0>() + a.coeff<1>()));

    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    verify_gf_jacobian_fd<2, 1>(f, tp);
}

TEST_F(CwiseSegmentSumJacobianTest, Cos2XplusY_Analytical) {
    auto a = Arguments<2>();
    auto f = cos(2.0 * (a.coeff<0>() + a.coeff<1>()));

    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;

    // d/dx cos(2(x+y)) = -sin(2(x+y)) * 2, same for d/dy
    double s = -std::sin(2.0 * (0.2 + (-0.5))) * 2.0;
    Eigen::MatrixXd expected(1, 2);
    expected << s, s;
    verify_jacobian_analytical(f, tp, expected, 1e-12);
}

///////////////////////////////////////////////////////////////////////////////
// sin(x+y) — simpler nested trig on segment sum
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, SinXplusY) {
    auto a = Arguments<2>();
    auto f = sin(a.coeff<0>() + a.coeff<1>());

    Eigen::VectorXd tp(2);
    tp << 0.7, -0.3;
    verify_jacobian_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// exp(x+y) — another cwise op on segment sum
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, ExpXplusY) {
    auto a = Arguments<2>();
    auto f = exp(a.coeff<0>() + a.coeff<1>());

    Eigen::VectorXd tp(2);
    tp << 0.1, 0.2;
    verify_jacobian_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// cos(x+y+z) — MultiFunctionSum (3 segments)
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, CosXplusYplusZ) {
    auto a = Arguments<3>();
    auto f = cos(a.coeff<0>() + a.coeff<1>() + a.coeff<2>());

    Eigen::VectorXd tp(3);
    tp << 0.2, -0.5, 0.3;
    verify_jacobian_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// sin(r) * cos(2*(x+y)) — product of two cwise-segment-sum expressions
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, SinR_Times_Cos2XplusY) {
    auto a = Arguments<2>();
    auto x = a.coeff<0>();
    auto y = a.coeff<1>();
    auto r = sqrt(x * x + y * y);
    auto f = sin(r) * cos(2.0 * (x + y));

    GenericFunction<2, 1> gf(f);
    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    verify_gf_jacobian_fd<2, 1>(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Stacked output: [cos(2*(x+y)); sin(x+y)]
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, StackedCosAndSin) {
    auto a = Arguments<2>();
    auto x = a.coeff<0>();
    auto y = a.coeff<1>();
    auto f = stack(cos(2.0 * (x + y)), sin(x + y));

    GenericFunction<2, 2> gf(f);
    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    verify_gf_jacobian_fd<2, 2>(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Scaled segment sum inside sin: sin(3*(x - 2*y))
// Tests ScaledSegment inside sum inside cwise
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, SinScaledDifference) {
    auto a = Arguments<2>();
    auto x = a.coeff<0>();
    auto y = a.coeff<1>();
    auto f = sin(3.0 * (x + (-2.0) * y));

    Eigen::VectorXd tp(2);
    tp << 1.0, 0.5;
    verify_jacobian_fd(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Adjoint gradient consistency for cos(2*(x+y))
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, Cos2XplusY_AdjointConsistency) {
    auto a = Arguments<2>();
    auto f = cos(2.0 * (a.coeff<0>() + a.coeff<1>()));

    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    Eigen::VectorXd lm(1);
    lm << 1.0;

    GenericFunction<2, 1> gf(f);
    verify_adjoint_consistency(gf, tp, lm, 1e-12);
}

///////////////////////////////////////////////////////////////////////////////
// Hessian consistency for cos(2*(x+y))
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, Cos2XplusY_HessianConsistency) {
    auto a = Arguments<2>();
    auto f = cos(2.0 * (a.coeff<0>() + a.coeff<1>()));

    Eigen::VectorXd tp(2);
    tp << 0.2, -0.5;
    Eigen::VectorXd lm(1);
    lm << 1.0;

    GenericFunction<2, 1> gf(f);
    verify_hessian_consistency(gf, tp, lm, 1e-12);
}

///////////////////////////////////////////////////////////////////////////////
// FunctionDifference + Segment (exercises the func1_is_sumordiff branch)
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, Cos_DifferencePlusSegment) {
    auto a = Arguments<3>();
    auto f = cos((a.coeff<0>() - a.coeff<1>()) + a.coeff<2>());

    Eigen::VectorXd tp(3);
    tp << 0.5, 0.2, 0.3;
    verify_jacobian_fd(f, tp);
    verify_gf_jacobian_fd<3, 1>(f, tp);
}

///////////////////////////////////////////////////////////////////////////////
// Multiple test points to catch optimisation-level-dependent issues
///////////////////////////////////////////////////////////////////////////////

TEST_F(CwiseSegmentSumJacobianTest, Cos2XplusY_MultiplePoints) {
    auto a = Arguments<2>();
    auto f = cos(2.0 * (a.coeff<0>() + a.coeff<1>()));

    std::vector<std::pair<double, double>> points = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 2.0}, {3.14, -1.57}, {0.001, -0.001}};

    Eigen::VectorXd tp(2);
    for (auto &[px, py] : points) {
        tp << px, py;
        verify_jacobian_fd(f, tp);
    }
}
