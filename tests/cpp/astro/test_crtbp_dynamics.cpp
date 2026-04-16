///////////////////////////////////////////////////////////////////////////////
// CRTBPDynamics tests — ported from CR3BP in the cr3bp_model.h removal refactor.
//
// ODE adjoint consistency and L1 approximate location verification.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include <array>
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

namespace {

// Hand-coded reference for CR3BP rotating-frame dynamics with extra
// acceleration forcing. Independent of the SymPy source the codegen
// consumed, so a sign flip baked into both compute and compute_jacobian
// of the generated header would still fail this comparison.
Eigen::VectorXd crtbp_reference_fx(double mu, const Eigen::VectorXd& s) {
    const double x = s[0], y = s[1], z = s[2];
    const double vx = s[3], vy = s[4], vz = s[5];
    const double ax = s[6], ay = s[7], az = s[8];
    const double one_minus_mu = 1.0 - mu;
    const double dx1 = x + mu;
    const double dx2 = x - one_minus_mu;
    const double r1 = std::sqrt(dx1 * dx1 + y * y + z * z);
    const double r2 = std::sqrt(dx2 * dx2 + y * y + z * z);
    const double inv_r1_3 = 1.0 / (r1 * r1 * r1);
    const double inv_r2_3 = 1.0 / (r2 * r2 * r2);

    Eigen::VectorXd fx(6);
    fx[0] = vx;
    fx[1] = vy;
    fx[2] = vz;
    fx[3] = 2.0 * vy + x - one_minus_mu * dx1 * inv_r1_3 - mu * dx2 * inv_r2_3 + ax;
    fx[4] = -2.0 * vx + y - one_minus_mu * y * inv_r1_3 - mu * y * inv_r2_3 + ay;
    fx[5] = -one_minus_mu * z * inv_r1_3 - mu * z * inv_r2_3 + az;
    return fx;
}

}  // namespace

TEST_F(CRTBPDynamicsTest, ValueMatchesReferenceAtRandomStates) {
    const double mu = 0.012150585;  // Earth-Moon
    astro::CRTBPDynamics dyn(mu);
    const std::array<int, 5> seeds{600, 601, 602, 603, 604};
    for (int seed : seeds) {
        Eigen::VectorXd s = TychoTest::deterministic_random_vector(9, seed, -1.0, 1.0);
        // Keep position scaled so r1, r2 stay well away from zero.
        s.head<3>() *= 0.8;
        s[0] += 0.3;  // bias off the primaries
        s.segment<3>(3) *= 0.5;
        s.tail<3>() *= 0.01;

        Eigen::VectorXd fx(6);
        fx.setZero();
        dyn.compute(s, fx);
        Eigen::VectorXd fx_ref = crtbp_reference_fx(mu, s);
        EXPECT_TRUE(fx.isApprox(fx_ref, 1e-12))
            << "seed=" << seed << " fx=\n" << fx << "\nref=\n" << fx_ref;
    }
}

TEST_F(CRTBPDynamicsTest, JacobianMatchesFiniteDifferenceOfCompute) {
    const double mu = 0.012150585;
    astro::CRTBPDynamics dyn(mu);
    Eigen::VectorXd s(9);
    s << 0.5, 0.1, -0.05, 0.05, 0.5, -0.02, 0.1, -0.2, 0.05;

    Eigen::VectorXd fx(6);
    Eigen::MatrixXd jx(6, 9);
    fx.setZero();
    jx.setZero();
    dyn.compute_jacobian(s, fx, jx);

    const double h = 1e-6;
    Eigen::MatrixXd jx_fd(6, 9);
    for (int j = 0; j < 9; ++j) {
        Eigen::VectorXd sp_ = s, sm = s;
        sp_[j] += h;
        sm[j] -= h;
        Eigen::VectorXd fp(6), fm(6);
        fp.setZero();
        fm.setZero();
        dyn.compute(sp_, fp);
        dyn.compute(sm, fm);
        jx_fd.col(j) = (fp - fm) / (2.0 * h);
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 9; ++j) {
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), 1e-6)
                << "Jacobian mismatch vs fd at (" << i << "," << j << ")";
        }
    }
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
