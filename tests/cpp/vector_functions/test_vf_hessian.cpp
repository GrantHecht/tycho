///////////////////////////////////////////////////////////////////////////////
// Hessian / adjoint integration tests
//
// Extracted from test_vector_function_composition.cpp — Hessian / adjoint
// integration tests section.
///////////////////////////////////////////////////////////////////////////////

#include "Astro/CR3BPModel.h"
#include "Astro/KeplerModel.h"
#include "Tycho.h"
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Hessian / adjoint integration tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, NormHessianSymmetry) {
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x = deterministic_random_vector(3, 230, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(n, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, SquaredNormHessian) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    Eigen::VectorXd x = deterministic_random_vector(3, 231, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    // Hessian of ||x||^2 = 2*I
    const int ir = 3;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, ir);
    Eigen::VectorXd gx(ir);
    Eigen::MatrixXd hx(ir, ir);
    fx.setZero();
    jx.setZero();
    gx.setZero();
    hx.setZero();
    sn.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

    // Hessian should be 2*I
    for (int i = 0; i < ir; ++i) {
        for (int j = 0; j < ir; ++j) {
            double expected = (i == j) ? 2.0 : 0.0;
            EXPECT_NEAR(hx(i, j), expected, 1e-11)
                << "Hessian mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(VFCompositionTest, DotProductHessian) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto dp = a.dot(b);
    Eigen::VectorXd x = deterministic_random_vector(6, 232, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;

    // Hessian of a^T b: H = [0, I; I, 0] (6x6 block matrix)
    const int ir = 6;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, ir);
    Eigen::VectorXd gx(ir);
    Eigen::MatrixXd hx(ir, ir);
    fx.setZero();
    jx.setZero();
    gx.setZero();
    hx.setZero();
    dp.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

    // Check structure: top-left 3x3 = 0, top-right 3x3 = I, etc.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(hx(i, j), 0.0, 1e-12);         // top-left
            EXPECT_NEAR(hx(i + 3, j + 3), 0.0, 1e-12); // bottom-right
            double expected_cross = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(hx(i, j + 3), expected_cross, 1e-12); // top-right
            EXPECT_NEAR(hx(i + 3, j), expected_cross, 1e-12); // bottom-left
        }
    }
}

TEST_F(VFCompositionTest, BrachistochroneHessianSymmetry) {
    BrachODE ode(9.81);
    // Need a scalar function for hessian — use squared norm of ODE output
    auto sq = ode.squared_norm();
    Eigen::VectorXd x = deterministic_random_vector(5, 233, 0.1, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(sq, x, lm, 1e-10);
}

TEST_F(VFCompositionTest, NestedScalarHessianSymmetry) {
    auto args = Arguments<4>();
    auto sq = args.Square();
    auto n = sq.norm(); // scalar
    Eigen::VectorXd x = deterministic_random_vector(4, 234, 0.5, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(n, x, lm, 1e-10);
}
