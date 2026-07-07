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

#include "integrator_test_utils.h"
#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/initial_dt.h"
#include <Eigen/Core>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using tycho::integrators::ErrorNormType;
using tycho::integrators::estimate_initial_dt;

///////////////////////////////////////////////////////////////////////////////
// Constant-derivative early-out (OrdinaryDiffEq parity): when the trial step
// reproduces f₀ exactly, OrdinaryDiffEq short-circuits with
// `f₀ == f₁ && return tdir·max(dtmin, 100·dt₀)`. Trigger via the SHO ODE at the
// fixed point x=0, v=0 where f ≡ 0: the d₀/d₁ < 1e-5 branch sets dt₀ = smalldt =
// 1e-6, the trial step leaves the state unchanged, f₁ == f₀, and the early-out
// returns 100·dt₀ = 1e-4 (NOT the pre-parity 1e-6, and NOT clamped to dtmax).
///////////////////////////////////////////////////////////////////////////////
TEST(InitialDtTest, ConstDerivativeStartReturnsHundredDt0) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 0.0, 0.0, 0.0; // at fixed point — f(x0) = (v=0, -x=0) = 0
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(2, 1e-12);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(2, 1e-13);
    double tf = 1.0;
    double dt0 = estimate_initial_dt(sho, x0, tf, atol, rtol, /*order=*/5, ErrorNormType::RMS);
    EXPECT_NEAR(dt0, 1e-4, 1e-7) << "f₀==f₁ early-out returns 100·dt₀ = 100·smalldt = 1e-4";
    EXPECT_GT(dt0, 0.0) << "dt must remain finite and positive for forward integration";
}

// When per-component scaling sk[i] = atol + |x0|*rtol collapses to zero,
// the residual scaling produces NaN and the HW initdt path must throw
// rather than silently return non-finite (or zero) dt. Pins the family
// of finite/zero guards in initial_dt.h (the f0 check, the f1 check,
// and the final signed_dt check) — any of them firing satisfies the
// user-visible contract. The substring match keeps the test resilient
// to which guard catches a given configuration.
TEST(InitialDtTest, ZeroTolsOnZeroStateThrowsFromHWInitialDt) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 0.0, 0.0, 0.0;
    Eigen::VectorXd atol = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd rtol = Eigen::VectorXd::Zero(2);
    try {
        (void)estimate_initial_dt(sho, x0, /*tf=*/1.0, atol, rtol, /*order=*/5, ErrorNormType::RMS);
        FAIL() << "estimate_initial_dt should have thrown on degenerate scaling";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Hairer-Wanner initial-dt"), std::string::npos)
            << "throw must originate in the HW initdt path: " << msg;
    }
}

// Degenerate per-component scaling throws intrinsically via the non-finite-dt0
// guard, CONSISTENTLY under both RMS and MAX norms. Component 1 (v) carries a
// zero scaling: atol[1]=0 and x0[1]=0 make sk[1]=atol+|v0|*rtol=0, which still
// passes the driver's abs+rel>0 precheck (rel[1]>0). The 0/0 scaled residual is
// NaN; under both RMS (squaredNorm propagates NaN) and MAX (the NaN-propagating
// maxCoeff guard — error_norm.h) the estimate d0 goes NaN, so dt0 goes NaN, and
// the intrinsic `!isfinite(dt0)` guard aborts BEFORE the trial step (rather than
// relying on the NaN to propagate through f(x1) — which only throws if the
// dynamics don't launder it). This is a static tolerance misconfiguration
// (atol=0 on a zero component), not a recoverable transient — throwing with a
// clear diagnostic is correct, and matches OrdinaryDiffEq, which returns dt=NaN
// -> ReturnCode.DtNaN (an abort) here. Recovery from a *transient* NaN during
// stepping is the driver's job (reject-and-shrink), not this one-shot estimate's.
TEST(InitialDtTest, DegenerateScalingThrowsViaNonFiniteDt0) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0; // x=1 (finite scaling), v=0 (zero scaling on comp 1)
    Eigen::VectorXd atol(2);
    atol << 1e-12, 0.0; // comp 1 abs_tol == 0
    Eigen::VectorXd rtol(2);
    rtol << 1e-13, 1e-6; // comp 1 rel_tol > 0 → passes the driver abs+rel>0 gate,
                         // but sk[1] = 0 + |v0|*rtol[1] = 0 since v0 == 0.
    // Both norms must behave identically now that MAX propagates NaN.
    for (ErrorNormType norm : {ErrorNormType::RMS, ErrorNormType::MAX}) {
        try {
            (void)estimate_initial_dt(sho, x0, /*tf=*/1.0, atol, rtol, /*order=*/5, norm);
            FAIL() << "degenerate scaling must throw via the non-finite-dt0 guard";
        } catch (const std::runtime_error &e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("degenerate error scaling"), std::string::npos)
                << "should throw from the intrinsic non-finite-dt0 guard; got: " << msg;
            EXPECT_NE(msg.find("Hairer-Wanner initial-dt"), std::string::npos)
                << "should identify the HW initial-dt site; got: " << msg;
        }
    }
}

// A non-finite integration endpoint (t0 or tf is NaN/Inf) must throw
// intrinsically — the estimator cannot rely on its callers to pre-validate, and
// a NaN tf would otherwise slip past the `dtmax == 0` short-circuit and be
// silently laundered by the std::min({...}) raw-estimate fold (std::min drops a
// NaN in any but its first argument), returning a finite, wrong-signed step.
TEST(InitialDtTest, NonFiniteEndpointThrows) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(2, 1e-9);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(2, 1e-9);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double bad_tf : {nan, inf}) {
        try {
            (void)estimate_initial_dt(sho, x0, bad_tf, atol, rtol, /*order=*/5, ErrorNormType::RMS);
            FAIL() << "non-finite tf must throw";
        } catch (const std::runtime_error &e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("non-finite integration endpoint"), std::string::npos)
                << "should throw from the intrinsic endpoint guard; got: " << msg;
        }
    }
}

TEST(InitialDtTest, KeplerLEOYieldsReasonableFirstStep) {
    tycho::astro::Kepler kep(398600.4418);
    Eigen::Matrix<double, 7, 1> x0;
    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    x0 << r0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(6, 1e-9);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(6, 1e-10);
    double tf = 1000.0; // seconds
    double dt0 = estimate_initial_dt(kep, x0, tf, atol, rtol, /*order=*/5, ErrorNormType::RMS);
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
    double dt0 = estimate_initial_dt(kep, x0, tf, atol, rtol, 5, ErrorNormType::RMS);
    EXPECT_LT(dt0, 0.0);
}

TEST(InitialDtTest, SmallProblemReturnsSmallDt) {
    tycho::astro::Kepler kep(1.0);
    Eigen::Matrix<double, 7, 1> x0;
    x0 << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0; // near-circular normalized
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(6, 1e-12);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(6, 1e-13);
    double tf = 2.0;
    double dt0 = estimate_initial_dt(kep, x0, tf, atol, rtol, 7, ErrorNormType::RMS);
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
    EXPECT_THROW(tycho::integrators::Integrator<tycho::astro::Kepler>(
                     kep, tycho::integrators::IVPAlg::DOPRI87, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(tycho::integrators::Integrator<tycho::astro::Kepler>(
                     kep, tycho::integrators::IVPAlg::DOPRI87, -1.0),
                 std::invalid_argument);
}

TEST(InitialDtTest, SetInitialStepSizeRejectsZero) {
    tycho::astro::Kepler kep(1.0);
    tycho::integrators::Integrator<tycho::astro::Kepler> integ(
        kep, tycho::integrators::IVPAlg::DOPRI87, 0.01);
    EXPECT_THROW(integ.set_initial_step_size(0.0), std::invalid_argument);
    EXPECT_THROW(integ.set_initial_step_size(-0.5), std::invalid_argument);

    // A positive value succeeds and disables the HW auto-initdt.
    integ.set_initial_step_size(0.25);
    EXPECT_FALSE(integ.get_auto_initial_dt());
}

// -----------------------------------------------------------------------------
// HW auto-initdt OFF: fixed-step integration with def_step_size completes
// and reaches the requested tf. Previously this branch was only exercised as
// a side-effect in NaN-propagation tests, never with a finite-state problem
// where success is the contract.
// -----------------------------------------------------------------------------
TEST(InitialDtTest, HairerWannerOff_FixedStepReachesTf) {
    tycho::astro::Kepler kep(398600.4418);
    tycho::integrators::Integrator<tycho::astro::Kepler> integ(
        kep, tycho::integrators::IVPAlg::DOPRI87, /*def_step=*/0.1);

    // Force HW off and fixed-step mode so def_step_size IS the step size.
    integ.set_initial_step_size(0.1);
    ASSERT_FALSE(integ.get_auto_initial_dt());
    integ.adaptive_ = false;

    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    tycho::integrators::Integrator<tycho::astro::Kepler>::IntegRet x0;
    x0[0] = r0;
    x0[1] = 0.0;
    x0[2] = 0.0;
    x0[3] = 0.0;
    x0[4] = v0;
    x0[5] = 0.0;
    x0[6] = 0.0;

    const double tf = 5.0;
    auto xf = integ.integrate(x0, tf);

    // Time axis must equal tf (fixed-step path rescales h to land exactly).
    EXPECT_NEAR(xf[6], tf, 1e-12) << "fixed-step HW-off should land exactly on tf";
    // State must have evolved from x0 — position/velocity along the orbit.
    const double r_final = std::sqrt(xf[0] * xf[0] + xf[1] * xf[1] + xf[2] * xf[2]);
    EXPECT_NEAR(r_final, r0, 1.0) << "circular LEO radius should be preserved over tf=5s";
    EXPECT_NE(xf[0], x0[0]) << "state must have evolved";
}

///////////////////////////////////////////////////////////////////////////////
// dtmax clamp (OrdinaryDiffEq parity): the returned first step must never
// exceed the integration span |tf - t0|. Over a tiny span the well-conditioned
// SHO estimate (~5e-3) would otherwise overshoot the whole interval; it must be
// clamped to the span.
///////////////////////////////////////////////////////////////////////////////
TEST(InitialDtClamp, DtmaxClampsToSpan) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(2, 1e-6);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(2, 1e-6);
    const double tf = 1.0e-4; // span far smaller than the unclamped estimate
    const double dt = estimate_initial_dt(sho, x0, tf, atol, rtol, /*order=*/4, ErrorNormType::RMS);
    EXPECT_GT(dt, 0.0);
    EXPECT_LE(dt, tf) << "first step must be clamped to the span dtmax = |tf - t0|";
}

// §1.4/dtmax: zero-duration span (tf == t0) is a no-op — estimate_initial_dt
// returns a zero step (no integration), it does not throw. (The drivers
// short-circuit H == 0 and return the initial state upstream anyway.)
TEST(InitialDtClamp, ZeroSpanReturnsZeroStep) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 5.0;
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(2, 1e-6);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(2, 1e-6);
    const double dt =
        estimate_initial_dt(sho, x0, /*tf=*/5.0, atol, rtol, /*order=*/4, ErrorNormType::RMS);
    EXPECT_EQ(dt, 0.0);
}

// dtmin floor: at very large |t0| the ULP spacing of the time axis exceeds any
// well-conditioned step estimate, so the returned magnitude must be floored to
// dtmin = nextafter(max(eps(t0), eps(tf))) rather than a sub-representable value.
// Construct t0 = 1e16 (where one ULP ≈ 2) with tf one ULP above: the raw estimate
// (min(100·dt0, dt1, dtmax)) is ≤ dtmax = one ULP < dtmin, so the final
// max(dtmin, dt_raw) returns exactly dtmin. Pins the floor path (commit 7d98c49a)
// that the t0≈0 clamp tests never exercise.
TEST(InitialDtClamp, DtminFloorDominatesAtLargeTime) {
    TychoTest::SHO sho(0.0);
    const double t0 = 1.0e16;
    const double tf = std::nextafter(t0, std::numeric_limits<double>::infinity());
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, t0;
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(2, 1e-6);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(2, 1e-6);

    const double dt = estimate_initial_dt(sho, x0, tf, atol, rtol, /*order=*/4, ErrorNormType::RMS);

    const auto ulp = [](double t) {
        const double a = std::abs(t);
        return std::nextafter(a, std::numeric_limits<double>::infinity()) - a;
    };
    const double dtmin =
        std::nextafter(std::max(ulp(t0), ulp(tf)), std::numeric_limits<double>::infinity());
    EXPECT_GT(dt, 0.0) << "forward step must stay positive";
    EXPECT_DOUBLE_EQ(dt, dtmin) << "returned magnitude must be floored to dtmin at large |t|";
}

// Backward, tiny-span: the dtmax clamp must hold in magnitude space and the
// returned step must carry the correct (negative) sign via tdir.
TEST(InitialDtClamp, BackwardTinySpanClampsWithCorrectSign) {
    TychoTest::SHO sho(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 1.0; // t0 = 1.0
    Eigen::VectorXd atol = Eigen::VectorXd::Constant(2, 1e-6);
    Eigen::VectorXd rtol = Eigen::VectorXd::Constant(2, 1e-6);
    const double tf = 1.0 - 1.0e-4; // backward, span 1e-4 far below the estimate
    const double dt = estimate_initial_dt(sho, x0, tf, atol, rtol, /*order=*/4, ErrorNormType::RMS);
    EXPECT_LT(dt, 0.0) << "backward integration must return a negative step";
    EXPECT_LE(std::abs(dt), 1.0e-4) << "magnitude must be clamped to the span dtmax";
}
