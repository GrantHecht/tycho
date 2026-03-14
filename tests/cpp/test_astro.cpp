///////////////////////////////////////////////////////////////////////////////
// Astrodynamics tests
//
// Tests Kepler conversions, propagation, Lambert solver, and CR3BP model.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "Astro/KeplerUtils.h"
#include "Astro/KeplerModel.h"
#include "Astro/CR3BPModel.h"
#include "Astro/LambertSolvers.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace Tycho;

// Standard gravitational parameter for Earth (km^3/s^2)
static constexpr double MU_EARTH = 398600.4418;

// LEO classical elements: a=7000 km, e=0.01, i=28.5°, RAAN=45°, w=30°, M=60°
static Vector6<double> leoClassic() {
    Vector6<double> oe;
    oe << 7000.0, 0.01, 28.5 * M_PI / 180.0, 45.0 * M_PI / 180.0, 30.0 * M_PI / 180.0,
        60.0 * M_PI / 180.0;
    return oe;
}

TEST(KeplerConversions, ClassicToCartesianRoundTrip) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(oe[i], oe2[i], 1e-10)
            << "Element " << i << " mismatch in classic->cart->classic round trip";
    }
}

TEST(KeplerConversions, CartesianToClassicRoundTrip) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
    auto rv2 = classic_to_cartesian<double>(oe2, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv[i], rv2[i], 1e-10)
            << "Component " << i << " mismatch in cart->classic->cart round trip";
    }
}

TEST(KeplerConversions, ModifiedToCartesianRoundTrip) {
    auto oe = leoClassic();
    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    auto rv = modified_to_cartesian<double>(mee, MU_EARTH);
    auto mee2 = cartesian_to_modified<double>(rv, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(mee[i], mee2[i], 1e-10)
            << "Element " << i << " mismatch in MEE->cart->MEE round trip";
    }
}

TEST(KeplerConversions, ClassicToModifiedRoundTrip) {
    auto oe = leoClassic();
    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    auto oe2 = modified_to_classic<double>(mee, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(oe[i], oe2[i], 1e-10)
            << "Element " << i << " mismatch in classic->MEE->classic round trip";
    }
}

TEST(KeplerConversions, CircularOrbitPeriod) {
    // Circular orbit: e=0 is degenerate for w/M, use small e
    Vector6<double> oe;
    double a = 7000.0;
    oe << a, 1e-8, 0.0, 0.0, 0.0, 0.0; // near-circular, equatorial

    auto rv0 = classic_to_cartesian<double>(oe, MU_EARTH);

    // Orbital period: T = 2*pi*sqrt(a^3/mu)
    double T = 2.0 * M_PI * std::sqrt(a * a * a / MU_EARTH);

    auto rv1 = propagate_cartesian<double>(rv0, T, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv0[i], rv1[i], 1e-6)
            << "Component " << i << " mismatch after full period propagation";
    }
}

TEST(KeplerConversions, HyperbolicClassicToCartesian) {
    // Hyperbolic orbit: e=1.5, a<0 (by convention, a is negative for hyperbolic)
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * M_PI / 180.0, 0.0, 0.0, 0.3;

    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);

    // Just verify it doesn't crash and produces finite values
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(rv[i]))
            << "Component " << i << " is not finite for hyperbolic orbit";
    }

    // Round-trip check: the conversion should be consistent
    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
    auto rv2 = classic_to_cartesian<double>(oe2, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv[i], rv2[i], 1e-8)
            << "Component " << i << " mismatch in hyperbolic cart round trip";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Kepler edge cases
///////////////////////////////////////////////////////////////////////////////

TEST(KeplerEdgeCases, NearCircularEquatorial) {
    // Nearly circular, equatorial orbit (e~0, i~0)
    Vector6<double> oe;
    oe << 7000.0, 1e-10, 1e-10, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(rv[i]))
            << "Component " << i << " not finite for near-circular equatorial";
    }
    // Round-trip through MEE (which handles near-circular well)
    auto mee = cartesian_to_modified<double>(rv, MU_EARTH);
    auto rv2 = modified_to_cartesian<double>(mee, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv[i], rv2[i], 1e-6)
            << "Component " << i << " mismatch in near-circ equatorial cart->MEE->cart";
    }
}

TEST(KeplerEdgeCases, RetrogradeOrbit) {
    // Retrograde orbit: i = 150 degrees
    Vector6<double> oe;
    oe << 8000.0, 0.1, 150.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 45.0 * M_PI / 180.0,
        20.0 * M_PI / 180.0;
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
    auto rv2 = classic_to_cartesian<double>(oe2, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv[i], rv2[i], 1e-8)
            << "Component " << i << " mismatch in retrograde round trip";
    }
}

TEST(KeplerEdgeCases, FullRoundTripClassicMEECartClassic) {
    auto oe = leoClassic();
    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    auto rv = modified_to_cartesian<double>(mee, MU_EARTH);
    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(oe[i], oe2[i], 1e-9)
            << "Element " << i << " mismatch in Classic->MEE->Cart->Classic";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Kepler propagation
///////////////////////////////////////////////////////////////////////////////

TEST(KeplerPropagation, MultiPeriodReturn) {
    auto oe = leoClassic();
    double a = oe[0];
    auto rv0 = classic_to_cartesian<double>(oe, MU_EARTH);
    double T = 2.0 * M_PI * std::sqrt(a * a * a / MU_EARTH);

    // Propagate 5 full periods
    auto rv5 = propagate_cartesian<double>(rv0, 5.0 * T, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv0[i], rv5[i], 1e-4)
            << "Component " << i << " mismatch after 5 periods";
    }
}

TEST(KeplerPropagation, HyperbolicTrajectory) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * M_PI / 180.0, 0.0, 0.0, 0.0;
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
// Lambert solver
///////////////////////////////////////////////////////////////////////////////

TEST(LambertSolver, SimpleTransfer) {
    // Transfer between two points on a circular orbit
    double a = 7000.0;

    Vector3<double> R1, R2;
    R1 << a, 0.0, 0.0;
    // 60 degree transfer
    R2 << a * std::cos(M_PI / 3.0), a * std::sin(M_PI / 3.0), 0.0;

    // Flight time for 60 degree transfer on circular orbit
    double T = 2.0 * M_PI * std::sqrt(a * a * a / MU_EARTH);
    double dt = T / 6.0; // 60 degrees = 1/6 of period

    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false);
    auto &V1 = result[0];
    auto &V2 = result[1];

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(V1[i])) << "V1[" << i << "] not finite";
        EXPECT_TRUE(std::isfinite(V2[i])) << "V2[" << i << "] not finite";
    }
}

TEST(LambertSolver, ShortWayTransfer) {
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << 0.0, 8000.0, 0.0; // 90 degree transfer

    double dt = 2000.0; // seconds
    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false);

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(result[0][i]));
        EXPECT_TRUE(std::isfinite(result[1][i]));
    }
}

TEST(LambertSolver, LongWayTransfer) {
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << 0.0, 8000.0, 0.0;

    double dt = 5000.0;
    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, true);

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(result[0][i]));
        EXPECT_TRUE(std::isfinite(result[1][i]));
    }
}

TEST(LambertSolver, MultiRevolution) {
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << 0.0, 7000.0, 0.0;

    // Long enough flight time for multi-rev
    double T = 2.0 * M_PI * std::sqrt(7000.0 * 7000.0 * 7000.0 / MU_EARTH);
    double dt = T + T / 4.0; // 1.25 periods

    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false, 1, false);

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(result[0][i])) << "Multi-rev V1[" << i << "] not finite";
        EXPECT_TRUE(std::isfinite(result[1][i])) << "Multi-rev V2[" << i << "] not finite";
    }
}

TEST(LambertSolver, SelfConsistency) {
    // Solve Lambert, then propagate with V1 — should arrive at R2
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << -3500.0, 6062.18, 0.0; // ~120 degree transfer

    double dt = 3000.0;
    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false);
    auto &V1 = result[0];

    // Propagate from R1 with V1 for dt
    Vector6<double> rv0;
    rv0 << R1[0], R1[1], R1[2], V1[0], V1[1], V1[2];
    Vector6<double> rv_final = propagate_cartesian<double>(rv0, dt, MU_EARTH);

    // Final position should match R2
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(rv_final[i], R2[i], 1.0)
            << "Position " << i << " mismatch in Lambert self-consistency";
    }
}

///////////////////////////////////////////////////////////////////////////////
// CR3BP (requires MemoryManager for VF evaluation)
///////////////////////////////////////////////////////////////////////////////

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
