///////////////////////////////////////////////////////////////////////////////
// MEEDynamics unit tests
//
// The MEE dynamics generator was rewritten in PR #42 and was previously only
// covered at integration time via the dionysus low-thrust example. These
// tests pin the zero-thrust circular-orbit invariants and a scale/finite
// check at a non-trivial MEE state so a codegen regression shows up at the
// unit level.
///////////////////////////////////////////////////////////////////////////////

#include "astro_test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

namespace {

// MEE input layout: [p, f, g, h, k, L, ur, ut, un]
Eigen::Matrix<double, 9, 1>
build_mee_input(const tycho::Vector6<double> &mee, double ur, double ut, double un) {
    Eigen::Matrix<double, 9, 1> x;
    x.head<6>() = mee;
    x[6] = ur;
    x[7] = ut;
    x[8] = un;
    return x;
}

} // namespace

TEST(MEEDynamicsTest, CircularEquatorialZeroThrustOnlyLDotIsNonzero) {
    // Pure circular equatorial orbit at mu=1, a=1 so the mean motion is 1.
    // All five non-angle element rates must vanish; only L̇ should remain.
    const double mu = 1.0;
    const double a = 1.0;
    tycho::Vector6<double> mee;
    mee << a, 0.0, 0.0, 0.0, 0.0, 0.0; // p, f, g, h, k, L

    astro::MEEDynamics dyn(mu);
    auto x = build_mee_input(mee, 0.0, 0.0, 0.0);
    Eigen::Matrix<double, 6, 1> fx;
    fx.setZero();
    dyn.compute_impl(x, fx);

    const double expected_L_dot = std::sqrt(mu / (a * a * a));
    EXPECT_NEAR(fx[0], 0.0, 1e-12) << "ṗ";
    EXPECT_NEAR(fx[1], 0.0, 1e-12) << "ḟ";
    EXPECT_NEAR(fx[2], 0.0, 1e-12) << "ġ";
    EXPECT_NEAR(fx[3], 0.0, 1e-12) << "ḣ";
    EXPECT_NEAR(fx[4], 0.0, 1e-12) << "k̇";
    EXPECT_NEAR(fx[5], expected_L_dot, 1e-12) << "L̇";
}

TEST(MEEDynamicsTest, CircularEquatorialZeroThrustLDotSweepsInL) {
    // Circular-equatorial invariant: L̇ = sqrt(mu/a^3) is independent of L.
    const double mu = MU_EARTH;
    const double a = 7000.0;
    const double expected_L_dot = std::sqrt(mu / (a * a * a));

    astro::MEEDynamics dyn(mu);
    for (double L : {0.0, 0.5, 1.2, std::numbers::pi, 2.0 * std::numbers::pi - 0.1}) {
        tycho::Vector6<double> mee;
        mee << a, 0.0, 0.0, 0.0, 0.0, L;
        auto x = build_mee_input(mee, 0.0, 0.0, 0.0);
        Eigen::Matrix<double, 6, 1> fx;
        fx.setZero();
        dyn.compute_impl(x, fx);
        // sqrt(mu/a^3) at the Earth scale is ~1e-3 rad/s; analytic closed-form
        // permits a tight tolerance — anything looser would mask sign/column
        // codegen errors.
        EXPECT_NEAR(fx[5], expected_L_dot, 1e-12) << "L̇ drifted at L=" << L;
        EXPECT_NEAR(fx[0], 0.0, 1e-12) << "ṗ drifted at L=" << L;
    }
}

TEST(MEEDynamicsTest, NonTrivialMEEStateAllRatesFinite) {
    // Non-degenerate elliptical inclined orbit: all rates must be finite and
    // ṗ must be nonzero when tangential thrust is present.
    const double mu = MU_EARTH;
    auto oe = leoClassic();
    auto mee = classic_to_modified<double>(oe, mu);

    astro::MEEDynamics dyn(mu);
    auto x = build_mee_input(mee, 0.0, 1e-3, 0.0);
    Eigen::Matrix<double, 6, 1> fx;
    fx.setZero();
    dyn.compute_impl(x, fx);

    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(std::isfinite(fx[i])) << "non-finite rate at index " << i;
    }
    EXPECT_GT(std::abs(fx[0]), 0.0) << "tangential thrust should drive ṗ ≠ 0";
}

TEST(MEEDynamicsTest, JacobianMatchesFiniteDifference) {
    // Pin the analytic Jacobian against a central-difference Jacobian on
    // several deterministic states with non-zero thrust. Catches sign flips,
    // swapped f/g or h/k columns, and missing terms in the codegen output.
    const double mu = MU_EARTH;
    astro::MEEDynamics dyn(mu);
    vf::GenericFunction<-1, -1> gf(dyn);

    auto mee_leo = classic_to_modified<double>(leoClassic(), mu);

    struct Case {
        Eigen::Matrix<double, 9, 1> x;
        double tol;
        const char *label;
    };
    std::vector<Case> cases;
    cases.push_back({build_mee_input(mee_leo, 0.0, 1e-3, 0.0), 1e-4, "LEO tangential"});
    cases.push_back({build_mee_input(mee_leo, 1e-3, 0.0, 5e-4), 1e-4, "LEO radial+normal"});
    cases.push_back({build_mee_input(mee_leo, -2e-4, 3e-4, -1e-4), 1e-4, "LEO mixed"});

    // A second deterministic state at a different L to break any L-symmetry
    // that could mask a Jacobian column error.
    auto mee_shifted = mee_leo;
    mee_shifted[5] = 1.7; // L
    cases.push_back({build_mee_input(mee_shifted, 0.0, 1e-3, 0.0), 1e-4, "LEO L=1.7"});
    cases.push_back({build_mee_input(mee_shifted, 5e-4, -2e-4, 1e-3), 1e-4, "LEO L=1.7 mixed"});

    for (const auto &c : cases) {
        SCOPED_TRACE(c.label);
        Eigen::VectorXd x = c.x;
        verify_jacobian_fd(gf, x, c.tol);
    }
}
