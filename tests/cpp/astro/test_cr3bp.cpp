///////////////////////////////////////////////////////////////////////////////
// CR3BP (Circular Restricted Three-Body Problem) tests
//
// ODE adjoint consistency and L1 approximate location verification.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;

class CR3BPTest : public TychoTest::VectorFunctionFixture {};

TEST_F(CR3BPTest, ODEAdjointConsistency) {
    double mu = 0.012150585; // Earth-Moon mass parameter
    CR3BP cr3bp(mu);
    Eigen::VectorXd x(7);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 300, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(cr3bp, x, lm, 1e-10);
}

TEST_F(CR3BPTest, L1ApproximateLocation) {
    // Earth-Moon system: mu ~ 0.012150585
    // L1 is between the two bodies, on the x-axis between them.
    double mu = 0.012150585;
    CR3BP cr3bp(mu);

    // Bisection to find L1: bracket the x-axis zero of the x-acceleration.
    // At L1, velocity = 0 and all accelerations vanish. We search for the
    // zero of a_x (deriv[3]) since a_y and a_z are identically zero on-axis.
    double lo = 0.5, hi = 1.0 - mu; // L1 lies between the two primaries
    Eigen::VectorXd state(7);
    Eigen::VectorXd deriv(6);

    auto ax_at = [&](double x) {
        state << x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        deriv.setZero();
        cr3bp.compute(state, deriv);
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
    state << x_L1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    deriv.setZero();
    cr3bp.compute(state, deriv);
    double acc_mag = deriv.tail<3>().norm();
    EXPECT_LT(acc_mag, 1e-12) << "Acceleration at L1 should vanish";
}
