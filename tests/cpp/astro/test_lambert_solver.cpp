///////////////////////////////////////////////////////////////////////////////
// Lambert solver tests
//
// Simple, short-way, long-way, multi-revolution, self-consistency, and
// near-antipodal transfer tests.
///////////////////////////////////////////////////////////////////////////////

#include "Astro/KeplerUtils.h"
#include "Astro/LambertSolvers.h"
#include "Tycho.h"
#include "astro_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

TEST(LambertSolver, SimpleTransfer) {
    // Transfer between two points on a circular orbit
    double a = 7000.0;

    Vector3<double> R1, R2;
    R1 << a, 0.0, 0.0;
    // 60 degree transfer
    R2 << a * std::cos(std::numbers::pi / 3.0), a * std::sin(std::numbers::pi / 3.0), 0.0;

    // Flight time for 60 degree transfer on circular orbit
    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / MU_EARTH);
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
    double T = 2.0 * std::numbers::pi * std::sqrt(7000.0 * 7000.0 * 7000.0 / MU_EARTH);
    double dt = T + T / 4.0; // 1.25 periods

    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false, 1, false);

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(result[0][i])) << "Multi-rev V1[" << i << "] not finite";
        EXPECT_TRUE(std::isfinite(result[1][i])) << "Multi-rev V2[" << i << "] not finite";
    }
}

TEST(LambertSolver, SelfConsistency) {
    // Solve Lambert, then propagate with V1 — should arrive at R2.
    // Use the same well-conditioned 60-degree circular orbit geometry as SimpleTransfer.
    double a = 7000.0;
    Vector3<double> R1, R2;
    R1 << a, 0.0, 0.0;
    R2 << a * std::cos(std::numbers::pi / 3.0), a * std::sin(std::numbers::pi / 3.0), 0.0;

    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / MU_EARTH);
    double dt = T / 6.0; // 60 degrees = 1/6 of period
    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false);
    auto &V1 = result[0];

    // Propagate from R1 with V1 for dt
    Vector6<double> rv0;
    rv0 << R1[0], R1[1], R1[2], V1[0], V1[1], V1[2];
    Vector6<double> rv_final = propagate_cartesian<double>(rv0, dt, MU_EARTH);

    // Final position should match R2
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(rv_final[i], R2[i], 0.01)
            << "Position " << i << " mismatch in Lambert self-consistency";
    }
}

TEST(LambertSolver, NearAntipodalTransfer) {
    // Near-180 degree transfer (slightly off to avoid exact degeneracy)
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << -6999.0, 100.0, 0.0; // ~179 degrees
    double dt = 4000.0;
    auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(result[0][i]))
            << "V1[" << i << "] not finite for near-antipodal transfer";
        EXPECT_TRUE(std::isfinite(result[1][i]))
            << "V2[" << i << "] not finite for near-antipodal transfer";
    }
}
