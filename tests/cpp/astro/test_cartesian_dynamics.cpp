///////////////////////////////////////////////////////////////////////////////
// CartesianDynamics tests — value spot-check, adjoint/Hessian consistency,
// extra-acceleration partial, constructor validation, and setter equivalence.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace tycho;

class CartesianDynamicsTest : public TychoTest::VectorFunctionFixture {};

TEST_F(CartesianDynamicsTest, ValueSpotCheck) {
    // For inputs [r, v, a_extra], output is [v, -mu * r / |r|^3 + a_extra].
    double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);

    Eigen::VectorXd x(9);
    x << 7000.0, 1000.0, -500.0,  // r
         0.5, 7.5, 0.1,           // v
         1e-4, -2e-4, 5e-5;       // extra accel
    Eigen::VectorXd fx(6);
    fx.setZero();
    dyn.compute(x, fx);

    const double r = std::sqrt(7000.0 * 7000.0 + 1000.0 * 1000.0 + 500.0 * 500.0);
    const double r3 = r * r * r;
    EXPECT_DOUBLE_EQ(fx[0], 0.5);
    EXPECT_DOUBLE_EQ(fx[1], 7.5);
    EXPECT_DOUBLE_EQ(fx[2], 0.1);
    EXPECT_NEAR(fx[3], -mu * 7000.0 / r3 + 1e-4, 1e-14);
    EXPECT_NEAR(fx[4], -mu * 1000.0 / r3 - 2e-4, 1e-14);
    EXPECT_NEAR(fx[5], -mu * -500.0 / r3 + 5e-5, 1e-14);
}

TEST_F(CartesianDynamicsTest, AdjointConsistency) {
    double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 7000.0, 1000.0, -500.0, 0.5, 7.5, 0.1, 0.0, 0.0, 0.0;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 400, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(dyn, x, lm, 1e-8);
}

TEST_F(CartesianDynamicsTest, AdjointConsistencyExtraAccel) {
    // Nonzero extra-accel path — exercises the ∂fdot/∂a = I₃ block that
    // the zero-accel variant cannot catch.
    double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 7000.0, 1000.0, -500.0, 0.5, 7.5, 0.1, 1e-3, -2e-3, 5e-4;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 401, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(dyn, x, lm, 1e-8);
}

TEST_F(CartesianDynamicsTest, HessianSymmetry) {
    // Locks in the codegen Hessian two-pass fix for CartesianDynamics.
    double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 7000.0, 1000.0, -500.0, 0.5, 7.5, 0.1, 1e-3, -2e-3, 5e-4;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 402, -1.0, 1.0);
    TychoTest::verify_hessian_consistency(dyn, x, lm, 1e-6);
}

TEST_F(CartesianDynamicsTest, ExtraAccelPartialIsIdentity) {
    double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 7000.0, 1000.0, -500.0, 0.5, 7.5, 0.1, 1e-3, -2e-3, 5e-4;

    Eigen::VectorXd fx(6);
    Eigen::MatrixXd jx(6, 9);
    fx.setZero();
    jx.setZero();
    dyn.compute_jacobian(x, fx, jx);

    Eigen::Matrix3d extra_block = jx.block<3, 3>(3, 6);
    EXPECT_TRUE(extra_block.isApprox(Eigen::Matrix3d::Identity(), 1e-14))
        << "Expected I3 for ∂fdot/∂a, got:\n" << extra_block;
}

TEST_F(CartesianDynamicsTest, ConstructorRejectsInvalidMu) {
    EXPECT_THROW(astro::CartesianDynamics(-1.0), std::invalid_argument);
    EXPECT_THROW(astro::CartesianDynamics(0.0), std::invalid_argument);
    EXPECT_THROW(astro::CartesianDynamics(std::nan("")), std::invalid_argument);
    EXPECT_NO_THROW(astro::CartesianDynamics(1.0));
}

TEST_F(CartesianDynamicsTest, SetMuMatchesFreshConstruction) {
    astro::CartesianDynamics dyn_fresh(3.986e5);
    astro::CartesianDynamics dyn_set(1.0);
    dyn_set.set_mu(3.986e5);

    Eigen::VectorXd x(9);
    x << 7000.0, 1000.0, -500.0, 0.5, 7.5, 0.1, 1e-3, -2e-3, 5e-4;
    Eigen::VectorXd fx_fresh(6), fx_set(6);
    fx_fresh.setZero();
    fx_set.setZero();
    dyn_fresh.compute(x, fx_fresh);
    dyn_set.compute(x, fx_set);
    EXPECT_EQ(fx_fresh, fx_set);

    EXPECT_THROW(dyn_set.set_mu(-1.0), std::invalid_argument);
}
