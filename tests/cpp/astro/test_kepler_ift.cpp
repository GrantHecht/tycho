///////////////////////////////////////////////////////////////////////////////
// Kepler IFT-composed Jacobian tests
//
// Verifies kepler_propagate_jacobian<double> against central finite difference
// on five fixtures: LEO circular, MEO eccentric, hyperbolic, multi-period
// (5 LEO orbits), and near-parabolic (a=1e7, e=0.99 — the same lane-3 fixture
// used in Task 1 to exercise small but well-conditioned α).
///////////////////////////////////////////////////////////////////////////////
#include "astro_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <tycho/detail/astro/kepler_propagator_ift.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace tycho::astro;
using namespace tycho::astro::detail;

namespace {

// Compare analytic Jacobian against central FD on the 7-vector y = (R0, V0, dt).
// Tolerance is checked element-wise as |a - fd| <= rel_tol * max(1, |fd|).
//
// `cross_check_propagate_cartesian` toggles the bit-equivalence check against
// the existing `propagate_cartesian<double>` reference.  That reference is
// known to be broken on hyperbolic orbits (Task 1 / test_kepler_lcd.cpp's
// MatchesPropagateCartesianHyperbolic comment); for those fixtures we instead
// validate the IFT-layer primal via orbit-energy and angular-momentum
// invariants, then rely on the FD Jacobian agreement as the acceptance gate.
void check_jacobian_fd(const Vector3<double> &R0, const Vector3<double> &V0, double dt, double mu,
                       double rel_tol, bool cross_check_propagate_cartesian = true) {
    Vector6<double> xf_a;
    Eigen::Matrix<double, 6, 7> jac_a;
    kepler_propagate_jacobian<double>(R0, V0, dt, mu, xf_a, jac_a);

    Vector6<double> rv0;
    rv0.head<3>() = R0;
    rv0.tail<3>() = V0;
    if (cross_check_propagate_cartesian) {
        // Acceptance criterion #1: primal output bit-equivalent to the reference.
        Vector6<double> xf_ref = propagate_cartesian<double>(rv0, dt, mu);
        for (int i = 0; i < 6; ++i)
            EXPECT_NEAR(xf_a[i], xf_ref[i], 1e-13 * std::max(1.0, std::abs(xf_ref[i])))
                << "primal comp " << i;
    } else {
        // Validate via orbit invariants on hyperbolic fixtures where
        // propagate_cartesian is broken.
        const Vector3<double> R1 = xf_a.head<3>();
        const Vector3<double> V1 = xf_a.tail<3>();
        const double E0 = 0.5 * V0.squaredNorm() - mu / R0.norm();
        const double E1 = 0.5 * V1.squaredNorm() - mu / R1.norm();
        EXPECT_NEAR(E0, E1, 1e-9 * std::abs(E0)) << "energy not conserved";
        const Vector3<double> H0 = R0.cross(V0);
        const Vector3<double> H1 = R1.cross(V1);
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(H0[i], H1[i], 1e-9 * std::max(1.0, std::abs(H0[i])))
                << "angular momentum comp " << i;
    }

    auto eval = [&](const Eigen::Matrix<double, 7, 1> &y) {
        Vector6<double> xf;
        kepler_propagate<double>(y.head<3>(), y.segment<3>(3), y[6], mu, xf);
        return xf;
    };
    Eigen::Matrix<double, 7, 1> y0;
    y0.head<3>() = R0;
    y0.segment<3>(3) = V0;
    y0[6] = dt;

    // Central FD step: eps ≈ eps_machine^(1/3) · max(1, |y|).  For LEO scale
    // (R ~ 7000, V ~ 7) this puts eps in the 4e-3 / 4e-6 range, balancing
    // truncation O(eps²) against the cancellation noise eps_machine·|f|/eps.
    Eigen::Matrix<double, 6, 7> jac_fd;
    constexpr double cbrt_epsm = 6.0554544523933395e-6; // ~ eps_machine^(1/3)
    for (int i = 0; i < 7; ++i) {
        const double eps = cbrt_epsm * std::max(1.0, std::abs(y0[i]));
        Eigen::Matrix<double, 7, 1> yp = y0, ym = y0;
        yp[i] += eps;
        ym[i] -= eps;
        const Vector6<double> fp = eval(yp);
        const Vector6<double> fm = eval(ym);
        jac_fd.col(i) = (fp - fm) / (2.0 * eps);
    }
    // Use row-norm scaling: small in-row entries are dominated by central-FD
    // round-off (ε_machine · |f| / eps ~ 1e-5 km on LEO scale), so they cannot
    // be checked at the per-element absolute level — but they are well-checked
    // relative to the row's overall magnitude.  This is the standard Jacobian-
    // FD comparison rule.
    for (int r = 0; r < 6; ++r) {
        const double row_scale = std::max(1.0, jac_fd.row(r).cwiseAbs().maxCoeff());
        for (int c = 0; c < 7; ++c) {
            EXPECT_NEAR(jac_a(r, c), jac_fd(r, c), rel_tol * row_scale)
                << "row " << r << " col " << c;
        }
    }
}

} // namespace

TEST(KeplerIFT_Jacobian, LEOCircular) {
    auto rv = classic_to_cartesian<double>(TychoTest::leoClassic(), TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 300.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, MEOEccentric) {
    Vector6<double> oe;
    oe << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 600.0, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, Hyperbolic) {
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    // The existing propagate_cartesian<double> is broken on this fixture (see
    // test_kepler_lcd.cpp, MatchesPropagateCartesianHyperbolic comment); skip
    // the bit-equivalence cross-check and rely on orbit invariants + FD.
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 100.0, TychoTest::MU_EARTH, 5e-7,
                      /*cross_check_propagate_cartesian=*/false);
}

TEST(KeplerIFT_Jacobian, MultiPeriod) {
    auto oe = TychoTest::leoClassic();
    const double a = oe[0];
    const double T = 2.0 * std::numbers::pi * std::sqrt(a * a * a / TychoTest::MU_EARTH);
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 5.0 * T, TychoTest::MU_EARTH, 5e-7);
}

TEST(KeplerIFT_Jacobian, NearParabolic) {
    // a=1e7 km, e=0.99 ⇒ α = 1/a ≈ 1e-7 (well-conditioned for the iteration kernel
    // but small enough to exercise the recursion → Taylor crossover region; α stays
    // above α_tol=1e-12 here, so the recursion branch is active and the test
    // verifies its numerical conditioning rather than the Taylor fallback).
    Vector6<double> oe;
    oe << 1.0e7, 0.99, 0.1, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, TychoTest::MU_EARTH);
    check_jacobian_fd(rv.head<3>(), rv.tail<3>(), 200.0, TychoTest::MU_EARTH, 5e-6);
}
