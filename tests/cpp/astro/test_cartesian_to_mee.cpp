///////////////////////////////////////////////////////////////////////////////
// CartesianToMEE unit tests
//
// Direct posigrade Cartesian -> Modified Equinoctial Elements conversion
// (no detour through classical orbital elements / Kepler equation). Covers:
//
//   - Round-trip Cart -> MEE -> Cart matches the input.
//   - Direct path agrees with the classical-routed reference
//     (classic_to_modified . cartesian_to_classic) on non-degenerate orbits.
//   - Analytic Jacobian (codegen CartesianToMEE VF) matches a
//     central-difference Jacobian on a spread of states.
//   - Analytic Hessian symmetry and adjoint-gradient consistency on the
//     same states (catches sign/transpose bugs PSIOPT would otherwise
//     consume silently).
//
// Orbit fixtures cover bound prograde (LEO/GEO/Molniya), high-inclination
// retrograde-adjacent (i ~ 170 deg, near the posigrade-formula singularity
// at hhat[2] -> -1), and hyperbolic (e > 1, a < 0).
///////////////////////////////////////////////////////////////////////////////

#include "astro_test_utils.h"
#include "test_utils.h"
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

// High inclination near the posigrade formulation's singularity at
// 1 + hhat[2] = 0 (i.e. hhat[2] -> -1, retrograde i -> pi).  i = 170 deg
// stresses the regime most likely to surface a sign/branch bug without
// landing on the singularity itself.
tycho::Vector6<double> highInclinationClassic() {
    tycho::Vector6<double> oe;
    oe << 8000.0, 0.05, 170.0 * std::numbers::pi / 180.0,
        0.5, 1.0, 0.1;
    return oe;
}

// Hyperbolic flyby: e > 1, a < 0.  Tycho's classic_to_cartesian +
// classic_to_modified handle the e > 1 branch (via H instead of E in the
// Kepler equation); the direct cartesian_to_modified has no e < 1
// precondition.
tycho::Vector6<double> hyperbolicClassic() {
    tycho::Vector6<double> oe;
    oe << -10000.0, 1.5, 30.0 * std::numbers::pi / 180.0,
        0.0, 0.0, 0.5;
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

void check_jacobian_fd(const tycho::Vector6<double> &oe, double mu, double tol) {
    auto rv = astro::classic_to_cartesian<double>(oe, mu);
    Eigen::VectorXd x = rv;

    astro::CartesianToMEE c2m(mu);
    vf::GenericFunction<-1, -1> gf(c2m);
    verify_jacobian_fd(gf, x, tol);
}

void check_hessian_consistency(const tycho::Vector6<double> &oe, double mu,
                               double tol) {
    auto rv = astro::classic_to_cartesian<double>(oe, mu);
    Eigen::VectorXd x = rv;
    Eigen::VectorXd lm = Eigen::VectorXd::Ones(6);

    astro::CartesianToMEE c2m(mu);
    verify_hessian_consistency(c2m, x, lm, tol);
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

TEST(CartesianToMEE, RoundTripLEO) {
    check_round_trip(leoClassic(), MU_EARTH, 1e-7, "LEO");
}

TEST(CartesianToMEE, RoundTripGEO) {
    check_round_trip(geoClassic(), MU_EARTH, 1e-6, "GEO");
}

TEST(CartesianToMEE, RoundTripMolniya) {
    check_round_trip(molniyaClassic(), MU_EARTH, 1e-6, "Molniya");
}

TEST(CartesianToMEE, AgreesWithClassicalRouteLEO) {
    check_matches_classical_route(leoClassic(), MU_EARTH, 1e-9, "LEO");
}

TEST(CartesianToMEE, AgreesWithClassicalRouteGEO) {
    check_matches_classical_route(geoClassic(), MU_EARTH, 1e-9, "GEO");
}

TEST(CartesianToMEE, AgreesWithClassicalRouteMolniya) {
    check_matches_classical_route(molniyaClassic(), MU_EARTH, 1e-9, "Molniya");
}

TEST(CartesianToMEE, RoundTripHighInclination) {
    check_round_trip(highInclinationClassic(), MU_EARTH, 1e-7,
                     "HighInclination");
}

TEST(CartesianToMEE, RoundTripHyperbolic) {
    check_round_trip(hyperbolicClassic(), MU_EARTH, 1e-6, "Hyperbolic");
}

TEST(CartesianToMEE, AgreesWithClassicalRouteHighInclination) {
    check_matches_classical_route(highInclinationClassic(), MU_EARTH, 1e-9,
                                  "HighInclination");
}

TEST(CartesianToMEE, AgreesWithClassicalRouteHyperbolic) {
    check_matches_classical_route(hyperbolicClassic(), MU_EARTH, 1e-9,
                                  "Hyperbolic");
}

TEST(CartesianToMEE, JacobianMatchesFiniteDifferenceLEO) {
    check_jacobian_fd(leoClassic(), MU_EARTH, 1e-4);
}

TEST(CartesianToMEE, JacobianMatchesFiniteDifferenceGEO) {
    check_jacobian_fd(geoClassic(), MU_EARTH, 1e-3);
}

TEST(CartesianToMEE, JacobianMatchesFiniteDifferenceMolniya) {
    check_jacobian_fd(molniyaClassic(), MU_EARTH, 1e-3);
}

TEST(CartesianToMEE, JacobianMatchesFiniteDifferenceHighInclination) {
    check_jacobian_fd(highInclinationClassic(), MU_EARTH, 1e-3);
}

TEST(CartesianToMEE, JacobianMatchesFiniteDifferenceHyperbolic) {
    check_jacobian_fd(hyperbolicClassic(), MU_EARTH, 1e-3);
}

// Hessian consistency: symmetry (hx == hx^T) and adjoint-gradient
// consistency (gx == jx^T * lm).  Tolerance is loose enough to absorb
// double-precision accumulation across the codegen-emitted body; tighter
// tolerances tend to fire on the larger-magnitude orbits even with a
// correct implementation.

TEST(CartesianToMEE, HessianConsistencyLEO) {
    check_hessian_consistency(leoClassic(), MU_EARTH, 1e-9);
}

TEST(CartesianToMEE, HessianConsistencyGEO) {
    check_hessian_consistency(geoClassic(), MU_EARTH, 1e-9);
}

TEST(CartesianToMEE, HessianConsistencyMolniya) {
    check_hessian_consistency(molniyaClassic(), MU_EARTH, 1e-9);
}

TEST(CartesianToMEE, HessianConsistencyHighInclination) {
    check_hessian_consistency(highInclinationClassic(), MU_EARTH, 1e-9);
}

TEST(CartesianToMEE, HessianConsistencyHyperbolic) {
    check_hessian_consistency(hyperbolicClassic(), MU_EARTH, 1e-9);
}
