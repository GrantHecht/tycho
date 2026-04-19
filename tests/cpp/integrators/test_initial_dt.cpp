///////////////////////////////////////////////////////////////////////////////
// Hairer-Wanner initial-dt unit tests
//
// Validates the two-stage algorithm:
//   d₀ = norm(u0 / sk); d₁ = norm(f₀ / sk)
//   dt₀ = 0.01 · (d₀/d₁) if both > 1e-5, else smalldt
//   u₁ = u0 + dt₀·f₀
//   f₁ = f(u₁, t + dt₀)
//   d₂ = norm((f₁ - f₀)/sk) / dt₀
//   dt₁ = if max(d₁,d₂) ≤ 1e-15: max(smalldt, dt₀·1e-3)
//         else: (0.01/max(d₁,d₂))^(1/(order+1))
//   dt = tdir · min(100·dt₀, dt₁)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>

#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/initial_dt.h"
#include <Eigen/Core>
#include <cmath>
#include <gtest/gtest.h>

using tycho::integrators::ErrorNormType;
using tycho::integrators::estimate_initial_dt;

TEST(InitialDtTest, KeplerLEOYieldsReasonableFirstStep) {
    tycho::astro::Kepler kep(398600.4418);
    Eigen::Matrix<double, 7, 1> x0;
    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    x0 << r0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(6, 1e-9);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(6, 1e-10);
    double tf = 1000.0; // seconds
    double dt0 = estimate_initial_dt(kep, x0, tf, atol, rtol, /*order=*/5,
                                     ErrorNormType::RMS);
    EXPECT_GT(dt0, 1e-6);
    EXPECT_LT(dt0, 1.0);
    EXPECT_GT(dt0, 0.0);
}

TEST(InitialDtTest, BackwardIntegrationReturnsNegativeDt) {
    tycho::astro::Kepler kep(398600.4418);
    Eigen::Matrix<double, 7, 1> x0;
    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    x0 << r0, 0.0, 0.0, 0.0, v0, 0.0, 1000.0; // t0 = 1000
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(6, 1e-9);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(6, 1e-10);
    double tf = 0.0; // backward
    double dt0 = estimate_initial_dt(kep, x0, tf, atol, rtol, 5,
                                     ErrorNormType::RMS);
    EXPECT_LT(dt0, 0.0);
}

TEST(InitialDtTest, SmallProblemReturnsSmallDt) {
    tycho::astro::Kepler kep(1.0);
    Eigen::Matrix<double, 7, 1> x0;
    x0 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0; // near-circular normalized
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(6, 1e-12);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(6, 1e-13);
    double tf = 2.0;
    double dt0 = estimate_initial_dt(kep, x0, tf, atol, rtol, 7,
                                     ErrorNormType::RMS);
    EXPECT_GT(dt0, 0.0);
    EXPECT_LT(dt0, 2.0);
}

///////////////////////////////////////////////////////////////////////////////
// Step-size guard regressions — pin the <= 0.0 tightening that closed the
// fixed-step divide-by-zero hole (def_step_size_ == 0 with HW-initdt
// disabled would otherwise crash deep in integrate_impl).
///////////////////////////////////////////////////////////////////////////////
TEST(InitialDtTest, ConstructorRejectsZeroStep) {
    tycho::astro::Kepler kep(1.0);
    EXPECT_THROW(tycho::integrators::Integrator<tycho::astro::Kepler>(kep, tycho::integrators::IVPAlg::DOPRI87, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(tycho::integrators::Integrator<tycho::astro::Kepler>(kep, tycho::integrators::IVPAlg::DOPRI87, -1.0),
                 std::invalid_argument);
}

TEST(InitialDtTest, SetInitialStepSizeRejectsZero) {
    tycho::astro::Kepler kep(1.0);
    tycho::integrators::Integrator<tycho::astro::Kepler> integ(kep, tycho::integrators::IVPAlg::DOPRI87, 0.01);
    EXPECT_THROW(integ.set_initial_step_size(0.0), std::invalid_argument);
    EXPECT_THROW(integ.set_initial_step_size(-0.5), std::invalid_argument);

    // A positive value succeeds and disables the HW auto-initdt.
    integ.set_initial_step_size(0.25);
    EXPECT_FALSE(integ.get_auto_initial_dt());
}
