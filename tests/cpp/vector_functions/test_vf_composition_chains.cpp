///////////////////////////////////////////////////////////////////////////////
// Complex composition chains tests
//
// Extracted from test_vector_function_composition.cpp — Complex composition
// chains section.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Complex composition chains
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, FiveNestingLevels) {
    auto args = Arguments<4>();
    auto l1 = 2.0 * args;
    auto l2 = l1.sin();
    auto l3 = l2.square();
    auto l4 = l3.norm();
    auto l5 = 3.0 * l3;

    Eigen::VectorXd x = deterministic_random_vector(4, 210, 0.5, 3.0);
    Eigen::VectorXd lm4(1);
    lm4 << 1.0;
    verify_adjoint_consistency(l4, x, lm4, 1e-10);

    Eigen::VectorXd lm5 = deterministic_random_vector(4, 211, -1.0, 1.0);
    verify_adjoint_consistency(l5, x, lm5, 1e-10);
}

TEST_F(VFCompositionTest, MixedArithmeticComposition) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto cw = a.cwise_product(b); // element-wise
    auto n = cw.norm();          // scalar
    auto dp = a.dot(b);          // scalar

    Eigen::VectorXd x = deterministic_random_vector(6, 220, 0.5, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(n, x, lm, 1e-10);
    verify_adjoint_consistency(dp, x, lm, 1e-10);
}

TEST_F(VFCompositionTest, NormOfComposition) {
    auto args = Arguments<3>();
    auto sq = args.square();
    auto n = sq.norm();
    // ||x^2|| at x=[3,4,0]: ||[9,16,0]|| = sqrt(81+256) = sqrt(337)
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    n.compute(x, fx);
    EXPECT_NEAR(fx[0], std::sqrt(337.0), 1e-10);
}

TEST_F(VFCompositionTest, DotOfCompositions) {
    auto args = Arguments<6>();
    auto a = args.template head<3>().sin();
    auto b = args.template tail<3>().cos();
    auto dp = a.dot(b);
    Eigen::VectorXd x = deterministic_random_vector(6, 221, 0.1, 3.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(dp, x, lm, 1e-10);
}
