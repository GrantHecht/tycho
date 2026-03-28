///////////////////////////////////////////////////////////////////////////////
// Kepler propagation tests
//
// Multi-period return, hyperbolic trajectory, and KeplerPropagator adjoint
// consistency.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "astro_test_utils.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST(KeplerPropagation, MultiPeriodReturn) {
    auto oe = leoClassic();
    double a = oe[0];
    auto rv0 = classic_to_cartesian<double>(oe, MU_EARTH);
    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / MU_EARTH);

    // Propagate 5 full periods
    auto rv5 = propagate_cartesian<double>(rv0, 5.0 * T, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv0[i], rv5[i], 1e-4) << "Component " << i << " mismatch after 5 periods";
    }
}

TEST(KeplerPropagation, HyperbolicTrajectory) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    Vector6<double> rv0 = classic_to_cartesian<double>(oe, MU_EARTH);

    // Verify positive specific energy (v^2/2 - mu/r > 0)
    double r = Eigen::Vector3d(rv0.head(3)).norm();
    double v = Eigen::Vector3d(rv0.tail(3)).norm();
    double energy = 0.5 * v * v - MU_EARTH / r;
    EXPECT_GT(energy, 0.0) << "Hyperbolic orbit should have positive energy";

    // Propagate forward — should produce finite results
    Vector6<double> rv1 = propagate_cartesian<double>(rv0, 100.0, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(rv1[i]))
            << "Component " << i << " not finite in hyperbolic propagation";
    }
    // Radius should increase for hyperbolic trajectory
    double r1 = Eigen::Vector3d(rv1.head(3)).norm();
    EXPECT_GT(r1, r);
}

///////////////////////////////////////////////////////////////////////////////
// KeplerPropagator adjoint consistency (VectorFunction-based)
///////////////////////////////////////////////////////////////////////////////

class KeplerPropagationFixture : public TychoTest::VectorFunctionFixture {};

TEST_F(KeplerPropagationFixture, KeplerPropagatorAdjointConsistency) {
    // KeplerPropagator is a VectorFunction<7, 6>: input [r, v, dt], output [r', v']
    KeplerPropagator prop(MU_EARTH);
    Eigen::VectorXd x(7);
    // Circular orbit state + propagation time
    double r0 = 7000.0;
    double v0 = std::sqrt(MU_EARTH / r0);
    x << r0, 0.0, 0.0, 0.0, v0, 0.0, 100.0; // dt = 100 s
    Eigen::VectorXd lm = TychoTest::deterministic_random_vector(6, 400, -1.0, 1.0);
    TychoTest::verify_adjoint_consistency(prop, x, lm, 1e-6);
}
