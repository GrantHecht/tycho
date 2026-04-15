///////////////////////////////////////////////////////////////////////////////
// SubstitutedVarPlaceholder (value-lock) regression tests
//
// Contract: compute_impl, compute_jacobian_impl, and the full derivative
// overload all write fx = 0 (zero residual) and a unit-diagonal Jacobian.
// The placeholder is paired with ODEPhaseBase::sub_variables(), which
// substitutes constant values for the primal variables before the
// constraint is evaluated — the residual must therefore be identically
// zero at every iteration.
///////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

#include "tycho/detail/vf/common/value_lock.h"

using namespace tycho;
using namespace TychoTest;

class SubstitutedVarPlaceholderTest : public VectorFunctionFixture {};

TEST_F(SubstitutedVarPlaceholderTest, ComputeWritesZeroResidual) {
    tycho::vf::SubstitutedVarPlaceholder<3> lock(3);
    Eigen::VectorXd x = deterministic_random_vector(3, 301, -5.0, 5.0);

    // Seed fx with a sentinel value — compute_impl must overwrite it with zero.
    Eigen::VectorXd fx(3);
    fx.setConstant(1.2345);
    lock.compute(x, fx);
    EXPECT_TRUE(fx.isZero(0.0))
        << "SubstitutedVarPlaceholder::compute_impl did not zero fx — "
        << "value-lock residual must be identically zero";
}

TEST_F(SubstitutedVarPlaceholderTest, JacobianIsIdentity) {
    tycho::vf::SubstitutedVarPlaceholder<3> lock(3);
    Eigen::VectorXd x = deterministic_random_vector(3, 303, -5.0, 5.0);
    Eigen::VectorXd fx(3);
    Eigen::MatrixXd jx(3, 3);
    fx.setConstant(9.99); // sentinel — must be zeroed
    jx.setZero();
    lock.compute_jacobian(x, fx, jx);

    EXPECT_TRUE(fx.isZero(0.0)) << "compute_jacobian did not write fx = 0";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(jx(i, j), expected)
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(SubstitutedVarPlaceholderTest, AdjointHessianIsZero) {
    tycho::vf::SubstitutedVarPlaceholder<3> lock(3);
    Eigen::VectorXd x = deterministic_random_vector(3, 304, -5.0, 5.0);
    Eigen::VectorXd fx(3), adjgrad(3), lm(3);
    Eigen::MatrixXd jx(3, 3), adjhess(3, 3);
    fx.setConstant(7.77); // sentinel — must be zeroed
    jx.setZero();
    adjgrad.setZero();
    adjhess.setZero();
    lm << 1.0, -2.0, 3.0;

    lock.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, lm);

    EXPECT_TRUE(fx.isZero(0.0)) << "compute_jacobian_adjoint... did not write fx = 0";
    EXPECT_TRUE(adjhess.isZero(0.0)) << "Adjoint Hessian of an identity map must be zero";
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(adjgrad[i], lm[i]) << "adjgrad = J^T lm should equal lm";
    }
}
