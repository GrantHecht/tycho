///////////////////////////////////////////////////////////////////////////////
// CRTBPDynamics tests — ported from CR3BP in the cr3bp_model.h removal refactor.
//
// ODE adjoint consistency and L1 approximate location verification.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

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
