///////////////////////////////////////////////////////////////////////////////
// CR3BP (Circular Restricted Three-Body Problem) tests
//
// ODE adjoint consistency and L1 approximate location verification.
///////////////////////////////////////////////////////////////////////////////

#include "Astro/CR3BPModel.h"
#include "Tycho.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;

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
    // L1 is between the two bodies, at approximately x ~ 0.8369 (normalized)
    double mu = 0.012150585;
    CR3BP cr3bp(mu);

    // L1 approximate location (on x-axis, between bodies)
    // Use Newton iteration to find exact location is overkill for a test;
    // instead verify that acceleration is small near L1
    double x_L1 = 0.8369;
    Eigen::VectorXd state(7);
    state << x_L1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    Eigen::VectorXd deriv(6);
    deriv.setZero();
    cr3bp.compute(state, deriv);

    // At L1: velocity = 0, so dx/dt = 0. Acceleration should be small.
    // deriv[0:3] = velocity (0), deriv[3:6] = acceleration
    double acc_mag = deriv.tail<3>().norm();
    EXPECT_LT(acc_mag, 0.1) << "Acceleration at approximate L1 should be small";
}
