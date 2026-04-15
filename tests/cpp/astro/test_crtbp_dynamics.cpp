///////////////////////////////////////////////////////////////////////////////
// CRTBPDynamics tests — ported from CR3BP in the cr3bp_model.h removal refactor.
//
// ODE adjoint consistency and L1 approximate location verification.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace tycho;

class CRTBPDynamicsTest : public TychoTest::VectorFunctionFixture {};

TEST_F(CRTBPDynamicsTest, ODEAdjointConsistency) {
    double mu = 0.012150585; // Earth-Moon mass parameter
    astro::CRTBPDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 300, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(dyn, x, lm, 1e-10);
}

TEST_F(CRTBPDynamicsTest, ODEAdjointConsistencyExtraAccel) {
    // Exercises the ∂fdot/∂a = I₃ partial-derivative block by setting
    // the last three inputs (extra acceleration) nonzero. The zero-accel
    // variant above degenerates to classical CR3BP and cannot catch bugs
    // in the extra-accel Jacobian/Hessian terms.
    double mu = 0.012150585;
    astro::CRTBPDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 0.5, 0.1, -0.05, 0.05, 0.5, -0.02, 0.1, -0.2, 0.05;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 301, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(dyn, x, lm, 1e-10);

    // The extra-accel block of the Jacobian is constant and equals I₃.
    Eigen::VectorXd fx(6);
    Eigen::MatrixXd jx(6, 9);
    fx.setZero();
    jx.setZero();
    dyn.compute_jacobian(x, fx, jx);
    Eigen::Matrix3d extra_block = jx.block<3, 3>(3, 6);
    EXPECT_TRUE(extra_block.isApprox(Eigen::Matrix3d::Identity(), 1e-14))
        << "Expected I3 for ∂fdot/∂a, got:\n" << extra_block;
}

TEST_F(CRTBPDynamicsTest, HessianSymmetry) {
    // Locks in the codegen Hessian two-pass emission fix. Before the fix,
    // the lower triangle of _hx_ was read before being written and ended
    // up zero-filled (masked by EIGEN_INITIALIZE_MATRICES_BY_ZERO), and
    // PSIOPT's upper-triangle-only reader silently never noticed. If
    // someone reverts the two-pass split, this test fires immediately at
    // the off-diagonal entries.
    double mu = 0.012150585;
    astro::CRTBPDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 0.5, 0.1, -0.05, 0.05, 0.5, -0.02, 0.1, -0.2, 0.05;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 302, -1.0, 1.0);
    TychoTest::verify_hessian_consistency(dyn, x, lm, 1e-10);
}

TEST_F(CRTBPDynamicsTest, ConstructorRejectsInvalidMu) {
    EXPECT_THROW(astro::CRTBPDynamics(-0.1), std::invalid_argument);
    EXPECT_THROW(astro::CRTBPDynamics(0.0), std::invalid_argument);
    EXPECT_THROW(astro::CRTBPDynamics(1.0), std::invalid_argument);
    EXPECT_THROW(astro::CRTBPDynamics(1.5), std::invalid_argument);
    EXPECT_THROW(astro::CRTBPDynamics(std::nan("")), std::invalid_argument);
    // Sanity: a valid value does not throw.
    EXPECT_NO_THROW(astro::CRTBPDynamics(0.5));
}

TEST_F(CRTBPDynamicsTest, SetMuRecomputesPrecomputed) {
    // The ctor initializes pc0_/pc1_/pc2_ (derived from mu_); set_mu must
    // re-run that derivation. Compare compute() output after set_mu() to
    // a fresh-constructed instance — they must agree bit-for-bit.
    astro::CRTBPDynamics dyn_fresh(0.02);
    astro::CRTBPDynamics dyn_set(0.01);
    dyn_set.set_mu(0.02);

    Eigen::VectorXd x(9);
    x << 0.5, 0.1, -0.05, 0.05, 0.5, -0.02, 0.1, -0.2, 0.05;
    Eigen::VectorXd fx_fresh(6), fx_set(6);
    fx_fresh.setZero();
    fx_set.setZero();
    dyn_fresh.compute(x, fx_fresh);
    dyn_set.compute(x, fx_set);
    EXPECT_EQ(fx_fresh, fx_set);

    // set_mu must also re-run validation.
    EXPECT_THROW(dyn_set.set_mu(-1.0), std::invalid_argument);
    EXPECT_THROW(dyn_set.set_mu(1.0), std::invalid_argument);
}

TEST_F(CRTBPDynamicsTest, L1ApproximateLocation) {
    // Earth-Moon system: mu ~ 0.012150585
    // L1 is between the two bodies, on the x-axis between them.
    double mu = 0.012150585;
    astro::CRTBPDynamics dyn(mu);

    // Bisection to find L1: bracket the x-axis zero of the x-acceleration.
    // At L1, velocity = 0 and all accelerations vanish. We search for the
    // zero of a_x (deriv[3]) since a_y and a_z are identically zero on-axis.
    double lo = 0.5, hi = 1.0 - mu; // L1 lies between the two primaries
    Eigen::VectorXd state(9);
    Eigen::VectorXd deriv(6);

    auto ax_at = [&](double x) {
        state << x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        deriv.setZero();
        dyn.compute(state, deriv);
        return deriv[3]; // x-acceleration
    };

    double ax_lo = ax_at(lo);
    for (int i = 0; i < 200; ++i) {
        double mid = 0.5 * (lo + hi);
        double ax_mid = ax_at(mid);
        if ((ax_lo > 0) == (ax_mid > 0)) {
            lo = mid;
            ax_lo = ax_mid;
        } else {
            hi = mid;
        }
    }

    double x_L1 = 0.5 * (lo + hi);
    state << x_L1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    deriv.setZero();
    dyn.compute(state, deriv);
    double acc_mag = deriv.tail<3>().norm();
    EXPECT_LT(acc_mag, 1e-12) << "Acceleration at L1 should vanish";
}
