///////////////////////////////////////////////////////////////////////////////
// Kepler Laguerre-Conway-Der iteration kernel tests
//
// Verifies kepler_lcd_iterate<double> against the existing propagate_cartesian
// reference, and exercises edge cases (dt=0 fast path, hyperbolic asymptote
// guard, multi-period round-trip).
///////////////////////////////////////////////////////////////////////////////

#include "astro_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <tycho/detail/astro/kepler_lcd_iterate.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace tycho::astro;
using namespace tycho::astro::detail;

namespace {
Vector6<double> apply_fg(const Vector6<double> &RV, const KeplerLCDResult<double> &k, double mu) {
    const double sqmu = std::sqrt(mu);
    const double aF = 1.0 - k.U2 / k.r0;
    const double aG = (k.r0 * k.U1 + k.sigma0 * k.U2) / sqmu;
    const double aFt = -sqmu / (k.r0 * k.r) * k.U1;
    const double aGt = 1.0 - k.U2 / k.r;
    Vector6<double> out;
    out.head<3>() = aF * RV.head<3>() + aG * RV.tail<3>();
    out.tail<3>() = aFt * RV.head<3>() + aGt * RV.tail<3>();
    return out;
}
} // namespace

TEST(KeplerLCDKernel, MatchesPropagateCartesianLEO) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv_ref = propagate_cartesian<double>(rv, 300.0, TychoTest::MU_EARTH);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_lcd[i], rv_ref[i], 1e-10 * std::max(1.0, std::abs(rv_ref[i])))
            << "comp " << i;
}

TEST(KeplerLCDKernel, MatchesPropagateCartesianMEO) {
    Vector6<double> oe;
    oe << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv_ref = propagate_cartesian<double>(rv, 600.0, TychoTest::MU_EARTH);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_lcd[i], rv_ref[i], 1e-10 * std::max(1.0, std::abs(rv_ref[i])));
}

TEST(KeplerLCDKernel, MatchesPropagateCartesianHyperbolic) {
    // The existing propagate_cartesian<double> does not converge on the e=1.5,
    // a=-10000 km, dt=100 s reference case (its hyperbolic seed is far from the
    // root and Newton-Raphson is capped at 19 iterations); the partially-
    // converged output disagrees with the analytic answer by tens of thousands
    // of km.  Task 5 of the LCD plan is the rewrite of propagate_cartesian on
    // top of this kernel that fixes the bug.  Until that lands, validate the
    // LCD answer against orbit invariants (energy, angular momentum) and the
    // hyperbolic asymptote (radius monotonically increasing past periapsis)
    // instead of point-comparing to the broken reference.
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH);

    const double mu = TychoTest::MU_EARTH;
    const Vector3<double> R0 = rv.head<3>();
    const Vector3<double> V0 = rv.tail<3>();
    const Vector3<double> R1 = rv_lcd.head<3>();
    const Vector3<double> V1 = rv_lcd.tail<3>();

    const double r0 = R0.norm();
    const double r1 = R1.norm();
    const double E0 = 0.5 * V0.squaredNorm() - mu / r0;
    const double E1 = 0.5 * V1.squaredNorm() - mu / r1;
    EXPECT_NEAR(E1, E0, 1e-9 * std::abs(E0));

    const Vector3<double> h0 = R0.cross(V0);
    const Vector3<double> h1 = R1.cross(V1);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(h1[i], h0[i], 1e-9 * std::max(1.0, std::abs(h0[i])));

    // Periapsis at t=0 with R perpendicular to V => radius must increase.
    EXPECT_GT(r1, r0);
}

TEST(KeplerLCDKernel, ZeroDtFastPath) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 0.0, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    EXPECT_DOUBLE_EQ(k.X, 0.0);
    EXPECT_DOUBLE_EQ(k.U0, 1.0);
    EXPECT_DOUBLE_EQ(k.U1, 0.0);
    EXPECT_DOUBLE_EQ(k.U2, 0.0);
    EXPECT_DOUBLE_EQ(k.U3, 0.0);
    auto rv_lcd = apply_fg(rv, k, TychoTest::MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_lcd[i], rv[i], 1e-13);
}

TEST(KeplerLCDKernel, HyperbolicAsymptoteGuard) {
    // X grows only logarithmically with dt on a hyperbolic orbit, so even a
    // 1e8 s propagation leaves sqma*X ~ 11 — well below the default guard of
    // 30.  Force the guard to trip by setting hyp_guard tighter than the
    // converged X for this case; this exercises the bail-out branch.
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    KeplerLCDOptions opts;
    opts.hyp_guard = 1.0;
    auto k =
        kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 1.0e8, TychoTest::MU_EARTH, opts);
    EXPECT_FALSE(k.converged);
}

TEST(KeplerLCDKernel, MultiPeriodReturn) {
    auto oe = TychoTest::leoClassic();
    double a = oe[0];
    double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / TychoTest::MU_EARTH);
    auto rv0 = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv0.head<3>(), rv0.tail<3>(), 5.0 * T, TychoTest::MU_EARTH);
    EXPECT_TRUE(k.converged);
    auto rv5 = apply_fg(rv0, k, TychoTest::MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv0[i], rv5[i], 1e-4);
}
