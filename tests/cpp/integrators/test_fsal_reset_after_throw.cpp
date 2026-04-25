///////////////////////////////////////////////////////////////////////////////
// FSAL-cache-reset-after-throw regression test.
//
// Pins the invariant enforced by the reset_fsal call at the top of
// AdaptiveDriver::integrate(): when integrate() throws mid-step (e.g., a NaN
// guard fires), stale f(x_prev) in k_fsal_ must not be reused as stage 0 of
// the NEXT integrate() call on the same driver.
//
// Strategy: integrate Kepler from a valid state to prime the FSAL cache with
// real f(xf). Then integrate from origin_state() which triggers the NaN
// finite-state guard → std::runtime_error. Catch, then integrate a valid
// problem on the same integrator and compare to a fresh-integrator run —
// endpoints must match bit-exactly.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <Eigen/Core>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {
constexpr double kMu = 398600.4418;

inline Eigen::VectorXd leo_state() {
    constexpr double r0 = 7000.0;
    Eigen::VectorXd x(7);
    x << r0, 0.0, 0.0, 0.0, std::sqrt(kMu / r0), 0.0, 0.0;
    return x;
}

inline Eigen::VectorXd origin_state() { return Eigen::VectorXd::Zero(7); }

Eigen::VectorXd integrate_leo(Integrator<astro::Kepler> &integ, double tf) {
    return integ.integrate(leo_state(), tf);
}
} // namespace

class FsalResetAfterThrowTest : public VectorFunctionFixture {};

// Parameterize across all FSAL-sensitive methods so a reintroduction of
// the leak on any one method surfaces via the matching variant.
class FsalResetAfterThrowParam : public VectorFunctionFixture,
                                 public ::testing::WithParamInterface<IVPAlg> {};

TEST_P(FsalResetAfterThrowParam, PoisonedFsalCacheClearedAtNextIntegrate) {
    const IVPAlg alg = GetParam();
    const double tf = 300.0;

    astro::Kepler kep(kMu);

    Integrator<astro::Kepler> integ_reused(kep, alg, 1.0);
    integ_reused.set_abs_tol(1e-10);
    integ_reused.set_rel_tol(1e-11);

    Integrator<astro::Kepler> integ_fresh(kep, alg, 1.0);
    integ_fresh.set_abs_tol(1e-10);
    integ_fresh.set_rel_tol(1e-11);

    // Prime the reused integrator's FSAL cache with a real integration.
    // Post-call, k_fsal_ holds f(xf_leo) with fsal_valid_=true.
    (void)integrate_leo(integ_reused, tf);

    // Force a throw mid-step by integrating from the origin — Kepler's
    // acc = -mu*r/|r|^3 is 0/0=NaN and the finite-state guard fires.
    EXPECT_THROW(integ_reused.integrate(origin_state(), tf), std::runtime_error);

    // The post-throw FSAL state is indeterminate (may or may not have been
    // updated before NaN detection). Without the reset_fsal at
    // AdaptiveDriver::integrate() entry, the next call would reuse this
    // stale state as stage 0, diverging bit-for-bit from a fresh run.
    auto xf_reused = integrate_leo(integ_reused, tf);
    auto xf_fresh = integrate_leo(integ_fresh, tf);

    ASSERT_EQ(xf_reused.size(), xf_fresh.size());
    for (Eigen::Index i = 0; i < xf_reused.size(); ++i) {
        EXPECT_EQ(xf_reused[i], xf_fresh[i])
            << "Component " << i << " differs after post-throw reuse: "
            << xf_reused[i] << " vs " << xf_fresh[i]
            << " — FSAL cache must be cleared at integrate() entry.";
    }
}

INSTANTIATE_TEST_SUITE_P(AllFsalMethods, FsalResetAfterThrowParam,
                         ::testing::Values(IVPAlg::DOPRI54, IVPAlg::DOPRI87, IVPAlg::Tsit5,
                                           IVPAlg::BS3, IVPAlg::BS5),
                         [](const auto &info) {
                             switch (info.param) {
                             case IVPAlg::DOPRI54: return "DOPRI54";
                             case IVPAlg::DOPRI87: return "DOPRI87";
                             case IVPAlg::Tsit5: return "Tsit5";
                             case IVPAlg::BS3: return "BS3";
                             case IVPAlg::BS5: return "BS5";
                             default: return "Unknown";
                             }
                         });

// Sanity: a clean sequence of two integrations on the same driver must
// already match a fresh driver (no FSAL leak on the happy path either).
TEST_F(FsalResetAfterThrowTest, CleanReuseAlsoMatchesFresh) {
    astro::Kepler kep(kMu);
    const double tf = 300.0;

    Integrator<astro::Kepler> integ_reused(kep, IVPAlg::DOPRI54, 1.0);
    integ_reused.set_abs_tol(1e-10);
    integ_reused.set_rel_tol(1e-11);
    (void)integrate_leo(integ_reused, tf);
    auto xf_reused = integrate_leo(integ_reused, tf);

    Integrator<astro::Kepler> integ_fresh(kep, IVPAlg::DOPRI54, 1.0);
    integ_fresh.set_abs_tol(1e-10);
    integ_fresh.set_rel_tol(1e-11);
    auto xf_fresh = integrate_leo(integ_fresh, tf);

    for (Eigen::Index i = 0; i < xf_reused.size(); ++i) {
        EXPECT_EQ(xf_reused[i], xf_fresh[i]);
    }
}
