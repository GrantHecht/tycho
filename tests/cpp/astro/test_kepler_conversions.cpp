///////////////////////////////////////////////////////////////////////////////
// Kepler conversion tests
//
// Round-trip checks for classic, modified (MEE), and Cartesian element sets.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "astro_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

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
    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / MU_EARTH);

    auto rv1 = propagate_cartesian<double>(rv0, T, MU_EARTH);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv0[i], rv1[i], 1e-6)
            << "Component " << i << " mismatch after full period propagation";
    }
}

TEST(KeplerConversions, HyperbolicClassicToCartesian) {
    // Hyperbolic orbit: e=1.5, a<0 (by convention, a is negative for hyperbolic)
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.3;

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
