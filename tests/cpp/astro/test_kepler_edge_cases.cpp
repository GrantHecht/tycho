///////////////////////////////////////////////////////////////////////////////
// Kepler edge-case tests
//
// Near-circular equatorial, retrograde, full multi-representation round trip,
// and near-parabolic orbits.
///////////////////////////////////////////////////////////////////////////////

#include "astro_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

namespace {

// Independently recover the hyperbolic anomaly H from a Cartesian state via
// the same closed-form arctanh relation cartesian_to_classic uses internally
// for its (Newton-free) M computation. Used to check the hyperbolic Kepler
// equation residual e*sinh(H) - H - M against the *input* M without relying
// on the Newton loop under test to grade its own homework.
double RecoverHyperbolicAnomaly(const Vector6<double> &rv, double mu, double e) {
    auto oe_true = cartesian_to_classic_true<double>(rv, mu);
    double v = oe_true[5];
    return 2.0 * std::atanh(std::tan(v / 2.0) / std::sqrt((1.0 + e) / (e - 1.0)));
}

} // namespace

TEST(KeplerEdgeCases, NearCircularEquatorial) {
    // Nearly circular, equatorial orbit (e~0, i~0).
    // Use e=1e-6, i=1e-6 rather than e.g. 1e-10: cartesian_to_modified routes
    // through cartesian_to_classic, which suffers catastrophic cancellation in the
    // eccentricity vector (V×h/mu − R̂) when e is near double-precision noise.
    // At 1e-6 the round-trip is well above the FP noise floor on all platforms.
    Vector6<double> oe;
    oe << 7000.0, 1e-6, 1e-6, 0.0, 0.0, 0.0;
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
    // Near-parabolic orbit: e = 0.9999, sampled AWAY from periapsis (M = 1.0).
    // The elliptic Newton solve now tests convergence on the step size
    // (|dE| < 1e-12, matching kepler_lcd_iterate and the hyperbolic branch).
    // From the E = M seed this converges for e = 0.9999 at M = 1.0 (~6 iters,
    // verified via standalone probe) but NOT for small M near periapsis
    // (M <~ 0.15), where the 1 - e*cosE denominator shrinks and the seed
    // leaves the Newton basin -- those inputs are now NaN-poisoned, covered by
    // EllipticNonConvergencePoisonsOutput below.  (Pre-fix, the residual-break
    // loop silently returned a finite-but-wrong state for the small-M case.)
    Vector6<double> oe;
    oe << 50000.0, 0.9999, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 1.0;
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

// ---------------------------------------------------------------------------
// OC §1.14: hyperbolic-anomaly Newton solve seed + non-convergence signalling
//
// Both classic_to_cartesian and classic_to_modified solve the hyperbolic
// Kepler equation M = e*sinh(H) - H via Newton iteration seeded at H = M.
// For moderate |M| (>~15) that seed lands far from the true root and the
// 15-iteration budget exhausts without converging; pre-fix, the loop had no
// convergence check and silently returned the far-from-root H as if
// converged. The fix seeds with the Gooding-class asinh(M/e) guess, tests
// convergence on the Newton STEP (|dH| < 1e-12, matching the |dX| <= Xtol
// convention of kepler_lcd_iterate -- scale-invariant near the root, unlike
// a raw-residual test whose FP noise floor grows as O(eps*M) and would
// falsely reject valid states for |M| beyond a few thousand), and
// NaN-poisons the whole output (via kepler_nan_value, matching the LCD/IFT
// Kepler paths) when convergence is not reached.
// ---------------------------------------------------------------------------

TEST(KeplerEdgeCases, HyperbolicModerateMConverges) {
    // e=1.5, M=50: pre-fix (H=M seed) the Newton loop decreases H by roughly
    // 1 per iteration from the far-from-root seed (H=50) and never reaches
    // the true root (H* ~ 4.28) within MAXITERS=15. Post-fix, the
    // asinh(M/e) seed lands within ~0.1 of H* and the loop converges to
    // sub-tolerance step size in ~5 iterations (verified via standalone
    // probe: converged=1, iters=5, |dH| ~2.7e-16, residual ~1e-14).
    const double e = 1.5;
    const double M = 50.0;
    Vector6<double> oe;
    oe << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, M;

    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);

    // Never a finite-but-wrong value: either the recovered hyperbolic
    // anomaly satisfies the Kepler-equation residual to tight tolerance, or
    // the whole state is NaN-poisoned. For this moderate M the fixed solver
    // converges, so the NaN branch is not expected to trigger here, but the
    // disjunction is asserted defensively (mirrors the brief's failing test).
    bool residual_ok = false;
    if (rv.allFinite()) {
        double H = RecoverHyperbolicAnomaly(rv, MU_EARTH, e);
        double residual = e * std::sinh(H) - H - M;
        residual_ok = std::fabs(residual) < 1e-8;
        EXPECT_NEAR(residual, 0.0, 1e-8)
            << "hyperbolic Kepler-equation residual not near zero for moderate |M| -- "
               "Newton solve converged to the wrong root (or effectively didn't converge)";
    }
    EXPECT_TRUE(residual_ok || !rv.allFinite())
        << "classic_to_cartesian returned a finite-but-wrong state for moderate |M|";
}

TEST(KeplerEdgeCases, HyperbolicModerateMConvergesClassicToModified) {
    // Same case as above, exercised through the classic_to_modified sibling
    // (the second Newton-solve site fixed in OC §1.14).
    const double e = 1.5;
    const double M = 50.0;
    Vector6<double> oe;
    oe << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, M;

    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    ASSERT_TRUE(mee.allFinite()) << "moderate |M| hyperbolic case must not be NaN-poisoned";

    // modified_to_classic recomputes M via the closed-form (Newton-free)
    // arctanh relation, so comparing against the input M is an independent
    // residual check on the Newton-solved true longitude L.
    auto oe2 = modified_to_classic<double>(mee, MU_EARTH);
    EXPECT_NEAR(oe2[5], M, 1e-8)
        << "Recovered hyperbolic mean anomaly does not match input M -- classic_to_modified's "
           "Newton solve converged to the wrong root for moderate |M|";
}

TEST(KeplerEdgeCases, HyperbolicLargeMConvergesUnderStepSizeCheck) {
    // Under the step-size convergence test, large |M| converges to machine-
    // relative precision -- an absolute-residual test would falsely NaN-poison
    // these states because the residual's FP noise floor is O(eps*M) (~1e-13
    // at M=4e3, ~1e-10 at M=1e6), far above a fixed 1e-12 threshold, even
    // though H itself is accurate to ~1e-16 relative. M=4e3 is realistically
    // reachable (~73 days past periapsis on this a=-10000 km Earth orbit);
    // probe: M=4e3 converges in 4 iterations, M=1e6 in 3.
    const double e = 1.5;
    for (double M : {4.0e3, 1.0e6}) {
        Vector6<double> oe;
        oe << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
            30.0 * std::numbers::pi / 180.0, M;

        auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
        ASSERT_TRUE(rv.allFinite()) << "classic_to_cartesian NaN-poisoned a convergent large-M "
                                       "hyperbolic state (M="
                                    << M << ")";
        // Round-trip M recovery (Newton-free closed form in
        // cartesian_to_classic) must match the input M to tight RELATIVE
        // tolerance; probe: 1.9e-13 at M=4e3, 1.9e-10 at M=1e6 (the residual
        // amplifies H-recovery conditioning by e*cosh(H) ~ M near the
        // asymptote, so machine-accurate H yields ~1e-10 relative here).
        auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
        EXPECT_NEAR(oe2[5] / M, 1.0, 1e-8)
            << "Recovered M does not match input at M=" << M
            << " -- large-M hyperbolic Newton solve converged to the wrong root";

        auto mee = classic_to_modified<double>(oe, MU_EARTH);
        ASSERT_TRUE(mee.allFinite()) << "classic_to_modified NaN-poisoned a convergent large-M "
                                        "hyperbolic state (M="
                                     << M << ")";
        auto oe3 = modified_to_classic<double>(mee, MU_EARTH);
        EXPECT_NEAR(oe3[5] / M, 1.0, 1e-8)
            << "Recovered M via MEE round trip does not match input at M=" << M;
    }
}

TEST(KeplerEdgeCases, HyperbolicNearParabolicDivergenceNaNPoisoned) {
    // Genuine Newton divergence under the step-size check: for e barely above
    // 1 with small |M|, the Newton derivative e*cosh(H) - 1 is ~(e-1) tiny
    // near the asinh(M/e) seed, so the first step overshoots wildly and the
    // iteration either oscillates (|dH| ~ 1 forever; M=0.01) or drives
    // sinh/cosh to overflow and the step to NaN (M=0.001), never reaching
    // |dH| < 1e-12 within MAXITERS=15 (both verified via standalone probe).
    // This must surface as an explicit all-NaN poison, never a finite value
    // that merely looks plausible. (Control: same e with M=1 converges.)
    const double e = 1.0 + 1.0e-10;
    for (double M : {0.01, 0.001}) {
        Vector6<double> oe;
        oe << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
            30.0 * std::numbers::pi / 180.0, M;

        auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
        for (int i = 0; i < 6; ++i)
            EXPECT_TRUE(std::isnan(rv[i]))
                << "classic_to_cartesian component " << i
                << " should be NaN-poisoned on non-convergence (M=" << M << ")";

        auto mee = classic_to_modified<double>(oe, MU_EARTH);
        for (int i = 0; i < 6; ++i)
            EXPECT_TRUE(std::isnan(mee[i]))
                << "classic_to_modified component " << i
                << " should be NaN-poisoned on non-convergence (M=" << M << ")";
    }

    // Control: nearby convergent case must NOT be poisoned (probe: converges
    // in 8 iterations at M=1).
    Vector6<double> oe_ok;
    oe_ok << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, 1.0;
    EXPECT_TRUE(classic_to_cartesian<double>(oe_ok, MU_EARTH).allFinite());
    EXPECT_TRUE(classic_to_modified<double>(oe_ok, MU_EARTH).allFinite());
}

TEST(KeplerEdgeCases, EllipticPathUnaffectedByHyperbolicFix) {
    // Sanity check that the elliptic Newton-solve branches (untouched by
    // OC §1.14) still converge cleanly -- guards against an accidental
    // cross-branch regression from the hyperbolic-branch edit.
    Vector6<double> oe;
    oe << 7000.0, 0.1, 0.2, 0.3, 0.4, 2.5;
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    ASSERT_TRUE(rv.allFinite());
    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    ASSERT_TRUE(mee.allFinite());

    auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(oe[i], oe2[i], 1e-9) << "Element " << i << " mismatch in elliptic round trip";
}

// ---------------------------------------------------------------------------
// CODEBASE §1.1b / OC §1.14 completion: elliptic-anomaly Newton solve now
// signals non-convergence.
//
// The elliptic branches of classic_to_cartesian and classic_to_modified
// solve M = E - e*sin(E) via Newton iteration from the E = M seed. They now
// test convergence on the Newton STEP (|dE| < 1e-12, matching
// kepler_lcd_iterate's |dX| <= Xtol convention and the hyperbolic branch),
// track a `converged` flag, and NaN-poison the whole output (via
// kepler_nan_value) when MAXITERS = 15 is exhausted -- instead of the former
// residual-break loop that silently returned a finite-but-wrong state.  The
// former break tested |E - e*sinE - M| < TOL *before* stepping, which accepted
// an under-converged E near periapsis of near-parabolic orbits: a tiny residual
// need not imply a tiny step there because the 1 - e*cosE denominator shrinks.
// ---------------------------------------------------------------------------

TEST(KeplerEdgeCases, EllipticNonConvergencePoisonsOutput) {
    // Probed divergent input (scan over (e, M) near e -> 1-, standalone probe):
    // e = 1 - 1e-9, M = 1e-8 is genuinely non-convergent from the E = M seed --
    // the Newton step never falls below 1e-12 within MAXITERS = 15 (the
    // 1 - e*cosE denominator is ~1e-9 near E ~ 0, so the step oscillates /
    // overshoots).  Pre-fix this returned a finite-but-wrong state; post-fix
    // the whole output is NaN-poisoned.
    Vector6<double> oe;
    oe << 1.0e5, 1.0 - 1.0e-9, 0.1, 0.1, 0.1, /*M=*/1.0e-8;

    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_TRUE(std::isnan(rv[i]))
            << "classic_to_cartesian component " << i
            << " should be NaN-poisoned on elliptic non-convergence";

    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_TRUE(std::isnan(mee[i]))
            << "classic_to_modified component " << i
            << " should be NaN-poisoned on elliptic non-convergence";

    // Control: same near-parabolic e sampled away from periapsis (M = 1.0)
    // converges and must NOT be poisoned (probe: ~6 iters).
    Vector6<double> oe_ok;
    oe_ok << 1.0e5, 1.0 - 1.0e-9, 0.1, 0.1, 0.1, 1.0;
    EXPECT_TRUE(classic_to_cartesian<double>(oe_ok, MU_EARTH).allFinite());
    EXPECT_TRUE(classic_to_modified<double>(oe_ok, MU_EARTH).allFinite());
}

TEST(KeplerEdgeCases, NominalEllipticRoundTripUnchanged) {
    // The step-first Newton loop takes one extra step vs the former
    // residual-break loop, but for well-conditioned orbits that step is a
    // no-op at 1e-12 (standalone probe: max |E_old - E_new| = 2.1e-13 over this
    // grid).  Verify nominal elliptic orbits still produce finite states whose
    // classic->cartesian->classic round trip recovers the input elements:
    // a to ~1e-6, e to ~1e-9, and (for e > 0, where the anomaly is well-defined)
    // the mean anomaly M mod 2*pi to ~1e-8.  M recovery is the independent
    // check that the Newton-solved E is correct (cartesian_to_classic recovers
    // M closed-form, Newton-free).  e = 0 is excluded from the e/M checks: the
    // eccentricity vector is ill-defined there (catastrophic cancellation), so
    // only a-recovery and finiteness are asserted -- same reasoning as
    // NearCircularEquatorial above.
    const double PI = std::numbers::pi;
    for (double e : {0.0, 0.1, 0.7, 0.95}) {
        for (double M : {0.2, 0.9, 1.7, 2.6, 3.4, 4.3, 5.2}) {
            Vector6<double> oe;
            oe << 7000.0, e, 0.3, 0.4, 0.5, M;
            auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
            ASSERT_TRUE(rv.allFinite())
                << "nominal elliptic orbit NaN-poisoned (e=" << e << ", M=" << M << ")";
            auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
            EXPECT_NEAR(oe2[0], 7000.0, 1e-6)
                << "semi-major axis not recovered (e=" << e << ", M=" << M << ")";
            if (e > 0.0) {
                EXPECT_NEAR(oe2[1], e, 1e-9)
                    << "eccentricity not recovered (e=" << e << ", M=" << M << ")";
                double dM = std::remainder(oe2[5] - M, 2.0 * PI);
                EXPECT_NEAR(dM, 0.0, 1e-8)
                    << "mean anomaly not recovered (e=" << e << ", M=" << M
                    << ") -- Newton-solved E is wrong";
            }
        }
    }
}

TEST(KeplerEdgeCases, PropagateCartesianDtZeroValidates) {
    // CODEBASE §1.1b: the dt == 0 early return in propagate_cartesian
    // previously bypassed every input check (mu > 0, dt finite, V0 finite,
    // r0 > 0).  Those checks are now hoisted ahead of the early return, so a
    // dt == 0 call with invalid inputs raises std::invalid_argument.
    Vector6<double> rv;
    rv << 7000.0, 0.0, 0.0, 0.0, 7.5, 0.0;
    EXPECT_THROW(propagate_cartesian<double>(rv, 0.0, -1.0), std::invalid_argument);

    Vector6<double> zero = Vector6<double>::Zero();
    EXPECT_THROW(propagate_cartesian<double>(zero, 0.0, MU_EARTH), std::invalid_argument);

    // Valid dt == 0 call is still an identity (no exception, returns input).
    auto same = propagate_cartesian<double>(rv, 0.0, MU_EARTH);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(same[i], rv[i]) << "dt == 0 identity broken at component " << i;
}
