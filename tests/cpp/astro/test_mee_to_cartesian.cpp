///////////////////////////////////////////////////////////////////////////////
// MEEToCartesian unit tests
//
// MEEToCartesian was added in PR #42 alongside the MEEDynamics rewrite. It
// has 667 lines of generated code and was previously only exercised through
// the betts_low_thrust example. These tests pin it directly:
//
//   - Round-trip: classic -> MEE -> MEEToCartesian matches classic_to_cartesian
//     on several non-degenerate orbits (LEO, GEO, Molniya).
//   - Analytic Jacobian matches a central-difference Jacobian on a handful
//     of deterministic states.
///////////////////////////////////////////////////////////////////////////////

#include "astro_test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

namespace {

// GEO classical elements: a=42164 km, near-circular, equatorial.
tycho::Vector6<double> geoClassic() {
    tycho::Vector6<double> oe;
    oe << 42164.0, 1e-4, 1.0 * std::numbers::pi / 180.0, 0.0, 0.0,
        0.0;
    return oe;
}

// Molniya classical elements: a=26600 km, e=0.74, i=63.4, RAAN=90, w=270, M=10.
tycho::Vector6<double> molniyaClassic() {
    tycho::Vector6<double> oe;
    oe << 26600.0, 0.74, 63.4 * std::numbers::pi / 180.0,
        90.0 * std::numbers::pi / 180.0, 270.0 * std::numbers::pi / 180.0,
        10.0 * std::numbers::pi / 180.0;
    return oe;
}

void check_round_trip(const tycho::Vector6<double> &oe, double mu, double tol,
                      const char *label) {
    auto cart_ref = astro::classic_to_cartesian<double>(oe, mu);
    auto mee = astro::classic_to_modified<double>(oe, mu);

    astro::MEEToCartesian m2c(mu);
    Eigen::Matrix<double, 6, 1> cart_via_op;
    cart_via_op.setZero();
    m2c.compute_impl(mee, cart_via_op);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(cart_via_op[i], cart_ref[i], tol)
            << label << ": cartesian component " << i << " mismatch";
    }
}

} // namespace

TEST(MEEToCartesianTest, RoundTripLEO) {
    check_round_trip(leoClassic(), MU_EARTH, 1e-7, "LEO");
}

TEST(MEEToCartesianTest, RoundTripGEO) {
    check_round_trip(geoClassic(), MU_EARTH, 1e-6, "GEO");
}

TEST(MEEToCartesianTest, RoundTripMolniya) {
    // Molniya pushes eccentricity to 0.74; loosen tolerance modestly to
    // absorb the larger magnitudes (positions ~50 000 km).
    check_round_trip(molniyaClassic(), MU_EARTH, 1e-6, "Molniya");
}

TEST(MEEToCartesianTest, JacobianMatchesFiniteDifferenceLEO) {
    auto mee = astro::classic_to_modified<double>(leoClassic(), MU_EARTH);
    Eigen::VectorXd x = mee;

    astro::MEEToCartesian m2c(MU_EARTH);
    vf::GenericFunction<-1, -1> gf(m2c);
    // FD step is dimensional (km, rad); analytic Jacobian magnitudes range
    // from ~1 (angles) to ~1e4 (semi-latus rectum), so 1e-4 tol is safe.
    verify_jacobian_fd(gf, x, 1e-4);
}

TEST(MEEToCartesianTest, JacobianMatchesFiniteDifferenceGEO) {
    auto mee = astro::classic_to_modified<double>(geoClassic(), MU_EARTH);
    Eigen::VectorXd x = mee;

    astro::MEEToCartesian m2c(MU_EARTH);
    vf::GenericFunction<-1, -1> gf(m2c);
    verify_jacobian_fd(gf, x, 1e-3);
}

TEST(MEEToCartesianTest, JacobianMatchesFiniteDifferenceMolniya) {
    auto mee = astro::classic_to_modified<double>(molniyaClassic(), MU_EARTH);
    Eigen::VectorXd x = mee;

    astro::MEEToCartesian m2c(MU_EARTH);
    vf::GenericFunction<-1, -1> gf(m2c);
    verify_jacobian_fd(gf, x, 1e-3);
}
