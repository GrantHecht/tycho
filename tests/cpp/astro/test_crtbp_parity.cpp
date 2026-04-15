// TEMPORARY: deleted together with cr3bp_model.h in the cr3bp-removal commit.
// Verifies numerical parity between the new codegen CRTBPDynamics and the
// legacy hand-written CR3BP struct.

#include <tycho/tycho.h>
#include "test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;

class CRTBPParityTest : public TychoTest::VectorFunctionFixture {};

TEST_F(CRTBPParityTest, ValueParityAgainstLegacyCR3BP) {
    const double mu = 0.012150585; // Earth-Moon
    astro::CR3BP legacy(mu);
    astro::CRTBPDynamics dyn(mu);

    // Random state, zero extra acceleration.
    Eigen::VectorXd x_legacy(7);
    x_legacy << 0.5, 0.1, 0.05, 0.02, 0.5, 0.03, 0.0;

    Eigen::VectorXd x_new(9);
    x_new.head<6>() = x_legacy.head<6>();
    x_new.tail<3>().setZero();

    Eigen::VectorXd out_legacy(6), out_new(6);
    out_legacy.setZero();
    out_new.setZero();
    legacy.compute(x_legacy, out_legacy);
    dyn.compute(x_new, out_new);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(out_legacy[i], out_new[i], 1e-12)
            << "Output mismatch at index " << i;
    }
}

TEST_F(CRTBPParityTest, AdjointConsistencyMatchesLegacy) {
    const double mu = 0.012150585;
    astro::CRTBPDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 300, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(dyn, x, lm, 1e-10);
}

TEST_F(CRTBPParityTest, CartesianDynamicsValueSpotCheck) {
    const double mu = 1.0;
    astro::CartesianDynamics dyn(mu);
    Eigen::VectorXd x(9);
    x << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd out(6);
    out.setZero();
    dyn.compute(x, out);

    EXPECT_NEAR(out[0], 0.0, 1e-15);
    EXPECT_NEAR(out[1], 1.0, 1e-15);
    EXPECT_NEAR(out[2], 0.0, 1e-15);
    EXPECT_NEAR(out[3], -1.0, 1e-15); // -mu * x / r^3 with x=r=1
    EXPECT_NEAR(out[4], 0.0, 1e-15);
    EXPECT_NEAR(out[5], 0.0, 1e-15);
}
