///////////////////////////////////////////////////////////////////////////////
// LockArgs (value-lock) regression tests
//
// Pins commit 8ffc2a9: compute_impl must leave fx untouched so PSIOPT's
// zero-initialized residual stays zero. Writing fx = x leaks the substituted
// primal back into the constraint and causes divergence (see comment in
// include/tycho/detail/vf/common/value_lock.h).
///////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

#include "tycho/detail/vf/common/value_lock.h"

using namespace tycho;
using namespace TychoTest;

class LockArgsTest : public VectorFunctionFixture {};

TEST_F(LockArgsTest, ComputeLeavesResidualUntouched) {
    tycho::vf::LockArgs<3> lock(3);
    Eigen::VectorXd x = deterministic_random_vector(3, 301, -5.0, 5.0);

    // Seed fx with a sentinel value — compute_impl must NOT overwrite it.
    Eigen::VectorXd fx(3);
    fx.setConstant(1.2345);
    lock.compute(x, fx);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(fx[i], 1.2345)
            << "LockArgs::compute_impl wrote to fx; value-lock residual is no longer zero-safe";
    }
}

TEST_F(LockArgsTest, ComputeResidualFromZeroStaysZero) {
    tycho::vf::LockArgs<4> lock(4);
    Eigen::VectorXd x = deterministic_random_vector(4, 302, -3.0, 3.0);
    Eigen::VectorXd fx(4);
    fx.setZero();
    lock.compute(x, fx);
    EXPECT_TRUE(fx.isZero(0.0)) << "Expected residual to remain zero after compute";
}

TEST_F(LockArgsTest, JacobianIsIdentity) {
    tycho::vf::LockArgs<3> lock(3);
    Eigen::VectorXd x = deterministic_random_vector(3, 303, -5.0, 5.0);
    Eigen::VectorXd fx(3);
    Eigen::MatrixXd jx(3, 3);
    fx.setZero();
    jx.setZero();
    lock.compute_jacobian(x, fx, jx);

    EXPECT_TRUE(fx.isZero(0.0)) << "compute_jacobian wrote to fx";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(jx(i, j), expected)
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(LockArgsTest, AdjointHessianIsZero) {
    tycho::vf::LockArgs<3> lock(3);
    Eigen::VectorXd x = deterministic_random_vector(3, 304, -5.0, 5.0);
    Eigen::VectorXd fx(3), adjgrad(3), lm(3);
    Eigen::MatrixXd jx(3, 3), adjhess(3, 3);
    fx.setZero();
    jx.setZero();
    adjgrad.setZero();
    adjhess.setZero();
    lm << 1.0, -2.0, 3.0;

    lock.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);

    EXPECT_TRUE(fx.isZero(0.0));
    EXPECT_TRUE(adjhess.isZero(0.0)) << "Adjoint Hessian of an identity map must be zero";
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(adjgrad[i], lm[i]) << "adjgrad = J^T lm should equal lm";
    }
}
