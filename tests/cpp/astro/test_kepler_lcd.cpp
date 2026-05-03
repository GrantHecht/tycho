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
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tycho/detail/astro/kepler/kepler_lcd_iterate.h>
#include <tycho/detail/astro/kepler/kepler_primal_vf.h>
#include <tycho/detail/astro/kepler/kepler_residual_vf.h>
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
    // propagate_cartesian uses the same LCD kernel, so equality would be
    // tautological; validate via orbit invariants (energy + angular momentum)
    // and the hyperbolic asymptote (radius monotonically increasing past
    // periapsis) instead.
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
    // Bail-out U-state recovery: the kernel must report finite last-stable
    // values, not garbage from the unstable post-loop refresh.
    EXPECT_TRUE(std::isfinite(k.X));
    EXPECT_TRUE(std::isfinite(k.U0));
    EXPECT_TRUE(std::isfinite(k.U1));
    EXPECT_TRUE(std::isfinite(k.U2));
    EXPECT_TRUE(std::isfinite(k.U3));
    EXPECT_TRUE(std::isfinite(k.r));
    EXPECT_TRUE(std::isfinite(k.sigma));
    EXPECT_GT(k.r, 0.0);
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

TEST(KeplerClosedForm, PrimalMatchesInlineFG) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH);
    ASSERT_TRUE(k.converged);

    // Pack codegen input: R0(3) + V0(3) + dt + X + U0..U2 = 11
    Eigen::Matrix<double, 11, 1> in;
    in.head<3>() = rv.head<3>();
    in.segment<3>(3) = rv.tail<3>();
    in[6] = 300.0;
    in[7] = k.X;
    in[8] = k.U0;
    in[9] = k.U1;
    in[10] = k.U2;

    KeplerPrimal_VF primal{TychoTest::MU_EARTH};
    Vector6<double> out;
    primal.compute_impl(in, out);

    Vector6<double> expected = apply_fg(rv, k, TychoTest::MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(out[i], expected[i], 1e-13 * std::max(1.0, std::abs(expected[i])))
            << "comp " << i;
}

TEST(KeplerClosedForm, ResidualVanishesAtConverged) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH);
    ASSERT_TRUE(k.converged);

    // F input: R0(3) + V0(3) + dt + U1 + U2 + U3 = 10
    Eigen::Matrix<double, 10, 1> in;
    in.head<3>() = rv.head<3>();
    in.segment<3>(3) = rv.tail<3>();
    in[6] = 300.0;
    in[7] = k.U1;
    in[8] = k.U2;
    in[9] = k.U3;

    KeplerResidual_VF residual{TychoTest::MU_EARTH};
    Eigen::Matrix<double, 1, 1> F_val;
    residual.compute_impl(in, F_val);
    EXPECT_NEAR(F_val[0], 0.0, 1e-9); // |F| at converged X* should be at noise floor
}

TEST(KeplerLCDKernelSS, MixedOrbitsFourLanes) {
    using SS = Eigen::Array<double, 4, 1>;
    Vector3<SS> R0, V0;
    SS dt;

    // Lane 0: LEO
    auto rv0 = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    // Lane 1: MEO e=0.5
    Vector6<double> oe1;
    oe1 << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    auto rv1 = classic_to_cartesian<double>(oe1, TychoTest::MU_EARTH);
    // Lane 2: hyperbolic
    Vector6<double> oe2;
    oe2 << -10000.0, 1.5, 0.2, 0.0, 0.0, 0.0;
    auto rv2 = classic_to_cartesian<double>(oe2, TychoTest::MU_EARTH);
    // Lane 3: near-parabolic ellipse — orbit-regime diversity vs lane 1.
    // a=1e7 km, e=0.99 → periapsis 1e5 km, alpha≈1e-7.  Well-conditioned
    // (periapsis well above Earth) and exercises a distinct regime.
    Vector6<double> oe3;
    oe3 << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv3 = classic_to_cartesian<double>(oe3, TychoTest::MU_EARTH);

    for (int i = 0; i < 3; ++i) {
        R0[i] << rv0[i], rv1[i], rv2[i], rv3[i];
        V0[i] << rv0[i + 3], rv1[i + 3], rv2[i + 3], rv3[i + 3];
    }
    dt << 100.0, 600.0, 50.0, 200.0;

    auto k_ss = kepler_lcd_iterate(R0, V0, dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(all_converged(k_ss.converged));

    Vector6<double> rv_arr[4] = {rv0, rv1, rv2, rv3};
    double dt_arr[4] = {100.0, 600.0, 50.0, 200.0};
    for (int lane = 0; lane < 4; ++lane) {
        auto k = kepler_lcd_iterate<double>(rv_arr[lane].head<3>(), rv_arr[lane].tail<3>(),
                                            dt_arr[lane], TychoTest::MU_EARTH);
        EXPECT_NEAR(k_ss.X[lane], k.X, 1e-12) << "X lane " << lane;
        EXPECT_NEAR(k_ss.U0[lane], k.U0, 1e-12) << "U0 lane " << lane;
        EXPECT_NEAR(k_ss.U1[lane], k.U1, 1e-12) << "U1 lane " << lane;
        EXPECT_NEAR(k_ss.U2[lane], k.U2, 1e-12) << "U2 lane " << lane;
        EXPECT_NEAR(k_ss.U3[lane], k.U3, 1e-12) << "U3 lane " << lane;
        EXPECT_NEAR(k_ss.r[lane], k.r, 1e-12) << "r lane " << lane;
        EXPECT_NEAR(k_ss.sigma[lane], k.sigma, 1e-12) << "sigma lane " << lane;
    }
}

TEST(KeplerLCDKernelSS, UniformEllipticFourLanesHitsSimdPath) {
    // Four elliptic LEO/MEO orbits — exercises the true-SIMD uniform-ellipse
    // path (kepler_lcd_iterate_simd_ellipse).  The dispatch in
    // kepler_lcd_iterate<W> routes here when every lane satisfies
    // alpha > alpha_tol and dt != 0; we still expect bit-near agreement with
    // the per-lane scalar reference.
    using SS = Eigen::Array<double, 4, 1>;
    Vector3<SS> R0, V0;
    SS dt;

    Vector6<double> oes[4];
    oes[0] << 7000.0, 0.01, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oes[1] << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    oes[2] << 9000.0, 0.2, 0.3, 0.4, 0.5, 0.6;
    oes[3] << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    const double dts[4] = {100.0, 600.0, 250.0, 200.0};

    Vector6<double> rvs[4];
    for (int lane = 0; lane < 4; ++lane) {
        rvs[lane] = classic_to_cartesian<double>(oes[lane], TychoTest::MU_EARTH);
        dt[lane] = dts[lane];
    }
    for (int i = 0; i < 3; ++i) {
        for (int lane = 0; lane < 4; ++lane) {
            R0[i][lane] = rvs[lane][i];
            V0[i][lane] = rvs[lane][i + 3];
        }
    }

    auto k_ss = kepler_lcd_iterate(R0, V0, dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(all_converged(k_ss.converged));

    // Tolerance is relative — SIMD evaluation re-orders sums vs scalar, so
    // sub-ULP rounding differences (~1e-15 relative) are expected.
    auto rel_near = [](double a, double b) {
        return std::max(1e-13, 1e-13 * std::max(std::abs(a), std::abs(b)));
    };
    for (int lane = 0; lane < 4; ++lane) {
        auto k = kepler_lcd_iterate<double>(rvs[lane].head<3>(), rvs[lane].tail<3>(), dts[lane],
                                            TychoTest::MU_EARTH);
        EXPECT_NEAR(k_ss.X[lane], k.X, rel_near(k_ss.X[lane], k.X)) << "X lane " << lane;
        EXPECT_NEAR(k_ss.U0[lane], k.U0, rel_near(k_ss.U0[lane], k.U0)) << "U0 lane " << lane;
        EXPECT_NEAR(k_ss.U1[lane], k.U1, rel_near(k_ss.U1[lane], k.U1)) << "U1 lane " << lane;
        EXPECT_NEAR(k_ss.U2[lane], k.U2, rel_near(k_ss.U2[lane], k.U2)) << "U2 lane " << lane;
        EXPECT_NEAR(k_ss.U3[lane], k.U3, rel_near(k_ss.U3[lane], k.U3)) << "U3 lane " << lane;
        EXPECT_NEAR(k_ss.r[lane], k.r, rel_near(k_ss.r[lane], k.r)) << "r lane " << lane;
        EXPECT_NEAR(k_ss.sigma[lane], k.sigma, rel_near(k_ss.sigma[lane], k.sigma))
            << "sigma lane " << lane;
    }
}

TEST(KeplerLCDKernel, NegativeDtRoundtripLEO) {
    // Backward propagation is a routine PSIOPT call pattern (e.g., shooting
    // from a final state).  Ensure +dt then -dt recovers the initial state.
    auto rv0 = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    const double dt = 300.0;
    auto k_fwd =
        kepler_lcd_iterate<double>(rv0.head<3>(), rv0.tail<3>(), dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(k_fwd.converged);
    auto rv1 = apply_fg(rv0, k_fwd, TychoTest::MU_EARTH);
    auto k_back =
        kepler_lcd_iterate<double>(rv1.head<3>(), rv1.tail<3>(), -dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(k_back.converged);
    auto rv_back = apply_fg(rv1, k_back, TychoTest::MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_back[i], rv0[i], 1e-10 * std::max(1.0, std::abs(rv0[i])));
}

TEST(KeplerLCDKernel, NegativeDtRoundtripHyperbolic) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv0 = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    const double dt = 100.0;
    auto k_fwd =
        kepler_lcd_iterate<double>(rv0.head<3>(), rv0.tail<3>(), dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(k_fwd.converged);
    auto rv1 = apply_fg(rv0, k_fwd, TychoTest::MU_EARTH);
    auto k_back =
        kepler_lcd_iterate<double>(rv1.head<3>(), rv1.tail<3>(), -dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(k_back.converged);
    auto rv_back = apply_fg(rv1, k_back, TychoTest::MU_EARTH);
    // Looser tolerance than LEO: hyperbolic round-trip is more sensitive to
    // numerical conditioning of the forward + backward universal-variable solves.
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(rv_back[i], rv0[i], 1e-8 * std::max(1.0, std::abs(rv0[i])));
}

TEST(KeplerLCDKernel, NaNInjectionFlagsNonConvergence) {
    // NaN in dt: r0 stays finite (so the input-validation throw isn't hit),
    // but the loop body produces NaN, which IEEE-754 makes |NaN| > Xtol false.
    // The post-loop finiteness gate must catch this and set converged = false.
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    const double nan_dt = std::numeric_limits<double>::quiet_NaN();
    auto k = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), nan_dt, TychoTest::MU_EARTH);
    EXPECT_FALSE(k.converged);
}

TEST(KeplerLCDKernel, RejectsZeroR0) {
    Vector3<double> R0 = Vector3<double>::Zero();
    Vector3<double> V0(0.0, 7.5, 0.0);
    EXPECT_THROW(kepler_lcd_iterate<double>(R0, V0, 100.0, TychoTest::MU_EARTH),
                 std::invalid_argument);
}

TEST(KeplerLCDKernel, RejectsNonPositiveMu) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    EXPECT_THROW(kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 100.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 100.0, -1.0),
                 std::invalid_argument);
}

TEST(KeplerLCDKernelSS, AllLanesBailoutSnapshotRecovery) {
    // SIMD-path bailout regression test.  All four lanes are uniform-elliptic
    // with nonzero dt (so the dispatcher routes to kepler_lcd_iterate_simd_ellipse),
    // but Xtol is tightened below the achievable convergence floor with a low
    // max_order budget — every lane must exit unconverged.  The kernel must:
    //   (1) report converged=false on every lane, and
    //   (2) keep the snapshot-restored X / U_n / r / sigma finite (the SIMD
    //       analog of the scalar HyperbolicAsymptoteGuard recovery).
    using SS = Eigen::Array<double, 4, 1>;
    Vector3<SS> R0, V0;
    SS dt;
    Vector6<double> oes[4];
    oes[0] << 7000.0, 0.01, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    oes[1] << 12000.0, 0.5, 0.5, 0.0, 0.0, 0.0;
    oes[2] << 9000.0, 0.2, 0.3, 0.4, 0.5, 0.6;
    oes[3] << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    Vector6<double> rvs[4];
    for (int lane = 0; lane < 4; ++lane)
        rvs[lane] = classic_to_cartesian<double>(oes[lane], TychoTest::MU_EARTH);
    for (int i = 0; i < 3; ++i) {
        for (int lane = 0; lane < 4; ++lane) {
            R0[i][lane] = rvs[lane][i];
            V0[i][lane] = rvs[lane][i + 3];
        }
    }
    dt << 100.0, 600.0, 250.0, 200.0;

    KeplerLCDOptions opts;
    opts.Xtol = 1.0e-300; // unachievable
    opts.max_order = 2;
    opts.iters_per_order = 2;

    auto k = kepler_lcd_iterate(R0, V0, dt, TychoTest::MU_EARTH, opts);
    for (int lane = 0; lane < 4; ++lane) {
        EXPECT_FALSE(k.converged[lane]) << "lane " << lane << " unexpectedly converged";
        EXPECT_TRUE(std::isfinite(k.X[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U0[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U1[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U2[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U3[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.r[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.sigma[lane])) << "lane " << lane;
        EXPECT_GT(k.r[lane], 0.0) << "lane " << lane;
    }
}

TEST(KeplerLCDKernelSS, SmallDtHitsStumpffTaylorBranch) {
    // Tiny per-lane dt drives the Newton seed X = sqmu*dt*alpha into the
    // y = alpha*X^2 < kStumpffTaylorEps regime.  Without the y-to-zero Taylor
    // fallback, the SIMD Stumpff (1 - cos(sqrt(y)))/y and (sqrt(y) - sin(sqrt(y)))/(y*sqrt(y))
    // expressions divide 0/0 and poison every output to NaN.  This test asserts
    // the kernel converges and produces finite, scalar-equivalent results.
    using SS = Eigen::Array<double, 4, 1>;
    Vector3<SS> R0, V0;
    SS dt;
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    for (int i = 0; i < 3; ++i) {
        R0[i] = SS::Constant(rv[i]);
        V0[i] = SS::Constant(rv[i + 3]);
    }
    // Lane dts span six orders of magnitude — from microseconds (where y is tiny)
    // up to a normal LEO step (where the recursion form is well-conditioned).
    dt << 1.0e-6, 1.0e-3, 1.0, 100.0;

    auto k = kepler_lcd_iterate(R0, V0, dt, TychoTest::MU_EARTH);
    EXPECT_TRUE(all_converged(k.converged));
    for (int lane = 0; lane < 4; ++lane) {
        EXPECT_TRUE(std::isfinite(k.X[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U0[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U1[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U2[lane])) << "lane " << lane;
        EXPECT_TRUE(std::isfinite(k.U3[lane])) << "lane " << lane;
    }

    // Cross-check vs scalar.
    const double dts[4] = {1.0e-6, 1.0e-3, 1.0, 100.0};
    for (int lane = 0; lane < 4; ++lane) {
        auto ks = kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), dts[lane],
                                             TychoTest::MU_EARTH);
        EXPECT_TRUE(ks.converged) << "scalar ref lane " << lane;
        EXPECT_NEAR(k.X[lane], ks.X, 1e-12 * std::max(1.0, std::abs(ks.X))) << "lane " << lane;
        EXPECT_NEAR(k.r[lane], ks.r, 1e-12 * std::max(1.0, std::abs(ks.r))) << "lane " << lane;
    }
}

TEST(KeplerLCDKernel, TrueParabolicConverges) {
    // Construct an exactly-parabolic orbit: at periapsis (R perpendicular to V)
    // with v = sqrt(2*mu/r0), so alpha = 2/r0 - v^2/mu = 0 exactly.  This drives
    // the |alpha| <= alpha_tol Taylor branch in compute_universal_functions and
    // the parabolic initial-X branch.  A working kernel produces a converged X*
    // satisfying Barker's equation X + alpha*X^3/6 = sqrt(mu)*dt with alpha = 0.
    const double mu = TychoTest::MU_EARTH;
    const double r0 = 7000.0;
    Vector3<double> R0(r0, 0.0, 0.0);
    Vector3<double> V0(0.0, std::sqrt(2.0 * mu / r0), 0.0);
    const double dt = 100.0;
    auto k = kepler_lcd_iterate<double>(R0, V0, dt, mu);
    EXPECT_TRUE(k.converged);
    EXPECT_TRUE(std::isfinite(k.X));
    // Parabolic Stumpff U_n: U0=1, U1=X, U2=X^2/2, U3=X^3/6 (no recursion-form
    // cancellation since alpha=0).  Confirm them directly.
    EXPECT_DOUBLE_EQ(k.U0, 1.0);
    EXPECT_DOUBLE_EQ(k.U1, k.X);
    EXPECT_DOUBLE_EQ(k.U2, k.X * k.X / 2.0);
    EXPECT_DOUBLE_EQ(k.U3, k.X * k.X * k.X / 6.0);
    // Barker's equation: r0*X + sigma0*X^2/2 + X^3/6 = sqrt(mu)*dt.
    // sigma0 = R.V/sqrt(mu) = 0 at periapsis.
    const double F = r0 * k.X + k.X * k.X * k.X / 6.0 - std::sqrt(mu) * dt;
    EXPECT_NEAR(F, 0.0, 1e-9);
}

TEST(KeplerLCDOptions, ValidateRejectsInvalid) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    auto call = [&](KeplerLCDOptions opts) {
        return kepler_lcd_iterate<double>(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH,
                                          opts);
    };
    {
        KeplerLCDOptions o;
        o.Xtol = 0.0;
        EXPECT_THROW(call(o), std::invalid_argument);
        o.Xtol = -1.0;
        EXPECT_THROW(call(o), std::invalid_argument);
        o.Xtol = std::nan("");
        EXPECT_THROW(call(o), std::invalid_argument);
    }
    {
        KeplerLCDOptions o;
        o.alpha_tol = -1.0;
        EXPECT_THROW(call(o), std::invalid_argument);
        o.alpha_tol = std::nan("");
        EXPECT_THROW(call(o), std::invalid_argument);
    }
    {
        KeplerLCDOptions o;
        o.max_order = 0;
        EXPECT_THROW(call(o), std::invalid_argument);
        o.max_order = -1;
        EXPECT_THROW(call(o), std::invalid_argument);
    }
    {
        KeplerLCDOptions o;
        o.iters_per_order = 0;
        EXPECT_THROW(call(o), std::invalid_argument);
    }
    {
        KeplerLCDOptions o;
        o.hyp_guard = 0.0;
        EXPECT_THROW(call(o), std::invalid_argument);
        o.hyp_guard = -1.0;
        EXPECT_THROW(call(o), std::invalid_argument);
    }
    // Default-constructed options must not throw.
    EXPECT_NO_THROW(call(KeplerLCDOptions{}));
}

TEST(KeplerPropagator, MuValidationThrows) {
    EXPECT_THROW(KeplerPropagator(0.0), std::invalid_argument);
    EXPECT_THROW(KeplerPropagator(-1.0), std::invalid_argument);
    EXPECT_THROW(KeplerPropagator(std::nan("")), std::invalid_argument);
    // Default-constructed propagator uses mu=1.0 — must not throw.
    EXPECT_NO_THROW(KeplerPropagator());
    // set_mu is the validation bottleneck for re-binding.
    KeplerPropagator kp;
    EXPECT_THROW(kp.set_mu(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(kp.set_mu(TychoTest::MU_EARTH));
    EXPECT_DOUBLE_EQ(kp.mu(), TychoTest::MU_EARTH);
}
