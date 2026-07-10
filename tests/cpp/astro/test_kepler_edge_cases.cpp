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

// ---------------------------------------------------------------------------
// OC §1.14: hyperbolic-anomaly Newton solve seed + non-convergence signalling
//
// Both classic_to_cartesian and classic_to_modified solve the hyperbolic
// Kepler equation M = e*sinh(H) - H via Newton iteration seeded at H = M.
// For moderate |M| (>~15) that seed lands far from the true root and the
// 15-iteration budget exhausts without converging; pre-fix, the loop had no
// residual check and silently returned the far-from-root H as if converged.
// The fix seeds with the Gooding-class asinh(M/e) guess, checks the residual
// after the loop, and NaN-poisons the whole output (via kepler_nan_value,
// matching the LCD/IFT Kepler paths) when convergence is not reached.
// ---------------------------------------------------------------------------

TEST(KeplerEdgeCases, HyperbolicModerateMConverges) {
    // e=1.5, M=50: pre-fix (H=M seed) the Newton loop decreases H by roughly
    // 1 per iteration from the far-from-root seed (H=50) and never reaches
    // the true root (H* ~ 4.28) within MAXITERS=15. Post-fix, the
    // asinh(M/e) seed lands within ~0.1 of H* and the loop converges to
    // full residual tolerance in ~5 iterations (verified via standalone
    // probe: converged=1, iters=5, final |residual| ~1.4e-14).
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

TEST(KeplerEdgeCases, HyperbolicLargeMNaNPoisonedNotFiniteWrong) {
    // M this large exceeds what the fixed asinh-seeded Newton loop can drive
    // below the absolute residual tolerance (residual scales with M, so the
    // achievable floor from double rounding, ~M*eps, exceeds TOL=1e-12) --
    // verified via standalone probe: converged=0 after 15 iterations even
    // though the final H (~14.1) is accurate to ~1e-16 relative precision.
    // This must surface as an explicit NaN-poisoned signal, never a finite
    // value that merely looks plausible.
    const double e = 1.5;
    const double M = 1.0e6;
    Vector6<double> oe;
    oe << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, M;

    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    EXPECT_FALSE(rv.allFinite());
    for (int i = 0; i < 6; ++i)
        EXPECT_TRUE(std::isnan(rv[i])) << "classic_to_cartesian component " << i
                                       << " should be NaN-poisoned on non-convergence";

    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    EXPECT_FALSE(mee.allFinite());
    for (int i = 0; i < 6; ++i)
        EXPECT_TRUE(std::isnan(mee[i])) << "classic_to_modified component " << i
                                        << " should be NaN-poisoned on non-convergence";
}

TEST(KeplerEdgeCases, HyperbolicOverflowMNaNPoisoned) {
    // M large enough that the Newton seed itself (asinh(M/e)) pushes sinh/cosh
    // into overflow (H_seed ~ ln(2M/e) > ~709.78 trips exp overflow); the
    // resulting inf/NaN residual must never leak through as a finite state.
    const double e = 1.5;
    const double M = 1.0e300;
    Vector6<double> oe;
    oe << -10000.0, e, 10.0 * std::numbers::pi / 180.0, 20.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, M;

    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    EXPECT_FALSE(rv.allFinite());

    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    EXPECT_FALSE(mee.allFinite());
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
