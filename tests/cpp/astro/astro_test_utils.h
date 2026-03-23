///////////////////////////////////////////////////////////////////////////////
// Shared astro test utilities
//
// Constants and helpers used across multiple astro test files.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>
#include <cmath>
#include <numbers>

namespace TychoTest {

// Standard gravitational parameter for Earth (km^3/s^2)
inline constexpr double MU_EARTH = 398600.4418;

// LEO classical elements: a=7000 km, e=0.01, i=28.5, RAAN=45, w=30, M=60
inline Tycho::Vector6<double> leoClassic() {
    Tycho::Vector6<double> oe;
    oe << 7000.0, 0.01, 28.5 * std::numbers::pi / 180.0, 45.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, 60.0 * std::numbers::pi / 180.0;
    return oe;
}

} // namespace TychoTest
