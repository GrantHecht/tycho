///////////////////////////////////////////////////////////////////////////////
// CartesianToModified unit tests
//
// Direct posigrade Cartesian -> Modified Equinoctial Elements conversion
// (no detour through classical orbital elements / Kepler equation). Covers:
//
//   - Round-trip Cart -> MEE -> Cart matches the input.
//   - Direct path agrees with the classical-routed reference
//     (classic_to_modified . cartesian_to_classic) on non-degenerate orbits.
//   - Analytic Jacobian (codegen CartesianToModified VF) matches a
//     central-difference Jacobian on a spread of states.
///////////////////////////////////////////////////////////////////////////////

#include "astro_test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

namespace {

tycho::Vector6<double> geoClassic() {
    tycho::Vector6<double> oe;
    oe << 42164.0, 1e-4, 1.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    return oe;
}

tycho::Vector6<double> molniyaClassic() {
    tycho::Vector6<double> oe;
    oe << 26600.0, 0.74, 63.4 * std::numbers::pi / 180.0,
        90.0 * std::numbers::pi / 180.0, 270.0 * std::numbers::pi / 180.0,
        10.0 * std::numbers::pi / 180.0;
    return oe;
}

void check_round_trip(const tycho::Vector6<double> &oe, double mu, double tol,
                      const char *label) {
    auto rv = astro::classic_to_cartesian<double>(oe, mu);
    auto mee = astro::cartesian_to_modified<double>(rv, mu);
    auto rv2 = astro::modified_to_cartesian<double>(mee, mu);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(rv2[i], rv[i], tol)
            << label << ": cart component " << i
            << " mismatch in cart->MEE->cart round trip";
    }
}

void check_matches_classical_route(const tycho::Vector6<double> &oe, double mu,
                                   double tol, const char *label) {
    auto rv = astro::classic_to_cartesian<double>(oe, mu);
    auto mee_direct = astro::cartesian_to_modified<double>(rv, mu);
    auto oe2 = astro::cartesian_to_classic<double>(rv, mu);
    auto mee_routed = astro::classic_to_modified<double>(oe2, mu);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(mee_direct[i], mee_routed[i], tol)
            << label << ": MEE element " << i
            << " disagrees between direct and classical-routed paths";
    }
    // L (true longitude, element 5) is only defined mod 2pi; the two
    // formulations may disagree by an integer multiple of 2pi at the
    // atan2 branch cut. Compare on the unit circle instead.
    EXPECT_NEAR(std::cos(mee_direct[5]), std::cos(mee_routed[5]), tol)
        << label << ": cos(L) disagrees between direct and classical-routed paths";
    EXPECT_NEAR(std::sin(mee_direct[5]), std::sin(mee_routed[5]), tol)
        << label << ": sin(L) disagrees between direct and classical-routed paths";
}

} // namespace

TEST(CartesianToModified, RoundTripLEO) {
    check_round_trip(leoClassic(), MU_EARTH, 1e-7, "LEO");
}

TEST(CartesianToModified, RoundTripGEO) {
    check_round_trip(geoClassic(), MU_EARTH, 1e-6, "GEO");
}

TEST(CartesianToModified, RoundTripMolniya) {
    check_round_trip(molniyaClassic(), MU_EARTH, 1e-6, "Molniya");
}

TEST(CartesianToModified, AgreesWithClassicalRouteLEO) {
    check_matches_classical_route(leoClassic(), MU_EARTH, 1e-9, "LEO");
}

TEST(CartesianToModified, AgreesWithClassicalRouteGEO) {
    check_matches_classical_route(geoClassic(), MU_EARTH, 1e-9, "GEO");
}

TEST(CartesianToModified, AgreesWithClassicalRouteMolniya) {
    check_matches_classical_route(molniyaClassic(), MU_EARTH, 1e-9, "Molniya");
}

TEST(CartesianToModified, JacobianMatchesFiniteDifferenceLEO) {
    auto rv = astro::classic_to_cartesian<double>(leoClassic(), MU_EARTH);
    Eigen::VectorXd x = rv;

    astro::CartesianToModified c2m(MU_EARTH);
    vf::GenericFunction<-1, -1> gf(c2m);
    verify_jacobian_fd(gf, x, 1e-4);
}

TEST(CartesianToModified, JacobianMatchesFiniteDifferenceGEO) {
    auto rv = astro::classic_to_cartesian<double>(geoClassic(), MU_EARTH);
    Eigen::VectorXd x = rv;

    astro::CartesianToModified c2m(MU_EARTH);
    vf::GenericFunction<-1, -1> gf(c2m);
    verify_jacobian_fd(gf, x, 1e-3);
}

TEST(CartesianToModified, JacobianMatchesFiniteDifferenceMolniya) {
    auto rv = astro::classic_to_cartesian<double>(molniyaClassic(), MU_EARTH);
    Eigen::VectorXd x = rv;

    astro::CartesianToModified c2m(MU_EARTH);
    vf::GenericFunction<-1, -1> gf(c2m);
    verify_jacobian_fd(gf, x, 1e-3);
}
