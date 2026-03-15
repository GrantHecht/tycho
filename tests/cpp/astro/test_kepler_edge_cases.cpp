///////////////////////////////////////////////////////////////////////////////
// Kepler edge-case tests
//
// Near-circular equatorial, retrograde, full multi-representation round trip,
// and near-parabolic orbits.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "astro_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

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
    oe << 8000.0, 0.1, 150.0 * std::numbers::pi / 180.0, 30.0 * std::numbers::pi / 180.0,
        45.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0;
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

TEST(KeplerEdgeCases, NearParabolic) {
    // Near-parabolic orbit: e = 0.9999
    Vector6<double> oe;
    oe << 50000.0, 0.9999, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.1;
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(rv[i]))
            << "Component " << i << " not finite for near-parabolic orbit";
    }
    // Round-trip through Cartesian
    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
    auto rv2 = classic_to_cartesian<double>(oe2, MU_EARTH);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv[i], rv2[i], 1e-4)
            << "Component " << i << " mismatch in near-parabolic round trip";
    }
}
