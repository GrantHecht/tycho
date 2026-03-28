///////////////////////////////////////////////////////////////////////////////
// Elements, CwiseSum, CwiseInverse, Arithmetic chains, and Complex
// compositions unit tests
//
// Extracted from test_common_functions.cpp — Elements, CwiseSum,
// CwiseInverse, Arithmetic chains, and Complex compositions sections.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Elements
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, ElementsExtraction) {
    auto args = Arguments<5>();
    auto elems = args.template elements<0, 2, 4>();
    EXPECT_EQ(elems.output_rows(), 3);
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
    EXPECT_EQ(s.output_rows(), 1);
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
