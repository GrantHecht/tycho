///////////////////////////////////////////////////////////////////////////////
// CartesianDynamics tests — value spot-check, adjoint/Hessian consistency,
// extra-acceleration partial, constructor validation, and setter equivalence.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include <array>
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

namespace {

// Independent hand-coded reference for two-body Cartesian dynamics with an
// extra acceleration forcing term. Kept deliberately naive (no CSE, no
// lifted compound bases) so any sign or scaling error baked into BOTH
// ``compute`` and ``compute_jacobian`` of the generated code would still
// fail to match this. Catches the class of bugs that adjoint-consistency
// cannot: both the analytic Jacobian and adjoint gradient are derived
// from the same SymPy expression tree, so a sign flip in the source
// propagates to both and verify_adjoint_consistency passes anyway.
Eigen::VectorXd cartesian_reference_fx(double mu, const Eigen::VectorXd& x) {
    const double rx = x[0], ry = x[1], rz = x[2];
    const double vx = x[3], vy = x[4], vz = x[5];
    const double ax = x[6], ay = x[7], az = x[8];
    const double r2 = rx * rx + ry * ry + rz * rz;
    const double r = std::sqrt(r2);
    const double r3 = r2 * r;
    Eigen::VectorXd fx(6);
    fx[0] = vx;
    fx[1] = vy;
    fx[2] = vz;
    fx[3] = -mu * rx / r3 + ax;
    fx[4] = -mu * ry / r3 + ay;
    fx[5] = -mu * rz / r3 + az;
    return fx;
}

}  // namespace

TEST_F(CartesianDynamicsTest, ValueMatchesReferenceAtRandomStates) {
    const double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);
    const std::array<int, 5> seeds{500, 501, 502, 503, 504};
    for (int seed : seeds) {
        Eigen::VectorXd x = TychoTest::deterministic_random_vector(9, seed, -1.0, 1.0);
        // Rescale position to an orbital radius so r > 0 comfortably.
        x.head<3>() *= 8000.0;
        x.segment<3>(3) *= 8.0;
        x.tail<3>() *= 1e-3;

        Eigen::VectorXd fx(6);
        fx.setZero();
        dyn.compute(x, fx);
        Eigen::VectorXd fx_ref = cartesian_reference_fx(mu, x);
        EXPECT_TRUE(fx.isApprox(fx_ref, 1e-12))
            << "seed=" << seed << " fx=\n" << fx << "\nref=\n" << fx_ref;
    }
}

TEST_F(CartesianDynamicsTest, JacobianMatchesFiniteDifferenceOfCompute) {
    // Central-difference the GENERATED compute() and compare against the
    // GENERATED analytic Jacobian. Together with ValueMatchesReference
    // above, this closes the loop: compute is correct (vs. reference),
    // and the Jacobian is the actual derivative of compute (vs. fd),
    // which catches sign errors in the codegen Jacobian path even when
    // both jx and gx agree.
    const double mu = 3.986e5;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 8000.0, 1500.0, -400.0, 0.3, 7.2, -0.1, 2e-4, -1e-4, 3e-5;

    Eigen::VectorXd fx(6);
    Eigen::MatrixXd jx(6, 9);
    fx.setZero();
    jx.setZero();
    dyn.compute_jacobian(x, fx, jx);

    const double h = 1e-4;
    Eigen::MatrixXd jx_fd(6, 9);
    for (int j = 0; j < 9; ++j) {
        Eigen::VectorXd xp = x, xm = x;
        // Scale step by variable magnitude for position entries to keep
        // relative perturbation consistent across scales.
        const double step = (j < 3) ? h * 8000.0 : (j < 6 ? h * 8.0 : h * 1e-3);
        xp[j] += step;
        xm[j] -= step;
        Eigen::VectorXd fp(6), fm(6);
        fp.setZero();
        fm.setZero();
        dyn.compute(xp, fp);
        dyn.compute(xm, fm);
        jx_fd.col(j) = (fp - fm) / (2.0 * step);
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 9; ++j) {
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), 1e-5)
                << "Jacobian mismatch vs fd at (" << i << "," << j << ")";
        }
    }
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
