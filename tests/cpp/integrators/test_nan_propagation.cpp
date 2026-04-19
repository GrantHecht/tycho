///////////////////////////////////////////////////////////////////////////////
// NaN propagation guard tests
//
// Pins the unconditional finite-state checks added to integrate_impl
// (scalar + SuperScalar batch) and estimate_initial_dt. Without these
// guards, a NaN-producing ODE silently drives the adaptive loop into
// progressive step shrinkage until max_steps trips, surfacing only as a
// generic "may be stuck" message. With the guards, the user sees an
// immediate diagnostic naming the failing site, time, step size, and
// (for batch) the trajectory index.
//
// Strategy: drive the Kepler dynamics at the origin (r = 0). Kepler's
// acceleration is -mu * r / |r|^3, which evaluates to 0/0 = NaN when
// r is the zero vector — a real, reachable ill-defined state for this
// dynamics, not a synthetic injection.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <vector>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {
constexpr double kMu = 398600.4418;

// Origin state — Kepler's acc = -mu * r / |r|^3 is 0/0 = NaN here.
inline Eigen::VectorXd origin_state() { return Eigen::VectorXd::Zero(7); }

// Nominal LEO state — used as a "good lane" companion in batch tests.
inline Eigen::VectorXd leo_state() {
    constexpr double r0 = 7000.0;
    Eigen::VectorXd x(7);
    x << r0, 0.0, 0.0, 0.0, std::sqrt(kMu / r0), 0.0, 0.0;
    return x;
}
} // namespace

class NanPropagationTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Scalar stepper path: with HW initial-dt disabled and a fixed step, the
// stepper's first call sees NaN-derivatives at the origin and produces a
// NaN xnext. The post-stepper finite check fires.
///////////////////////////////////////////////////////////////////////////////
TEST_F(NanPropagationTest, ScalarStepperThrowsOnOriginKepler) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_auto_initial_dt(false); // skip HW so we hit the stepper path
    integ.adaptive_ = false;

    auto x0 = origin_state();

    try {
        integ.integrate(x0, 100.0);
        FAIL() << "Expected runtime_error from finite-state check, got success.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("Non-finite state"), std::string::npos)
            << "Diagnostic should mention 'Non-finite state'; got: " << msg;
        EXPECT_NE(msg.find("scalar"), std::string::npos)
            << "Should identify the scalar stepper site; got: " << msg;
        EXPECT_NE(msg.find("first non-finite component"), std::string::npos)
            << "Should report the first bad component index; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Hairer-Wanner initial-dt path: when the initial state itself produces
// NaN derivatives, estimate_initial_dt should throw before the loop ever
// starts, with a message identifying the HW site.
///////////////////////////////////////////////////////////////////////////////
TEST_F(NanPropagationTest, HairerWannerThrowsOnOriginKepler) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);
    // HW is on by default after construction.

    auto x0 = origin_state();

    try {
        integ.integrate(x0, 100.0);
        FAIL() << "Expected runtime_error from HW-initdt finite check, got success.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("Non-finite state"), std::string::npos)
            << "Diagnostic should mention 'Non-finite state'; got: " << msg;
        EXPECT_NE(msg.find("Hairer-Wanner"), std::string::npos)
            << "Should identify the HW initial-dt site; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD batch path: a divergent batch with one bad lane should throw with
// the trajectory index of the offending lane embedded in the message —
// proving the per-lane diagnostic is wired correctly even when other
// lanes are well-conditioned.
///////////////////////////////////////////////////////////////////////////////
TEST_F(NanPropagationTest, BatchPathReportsOffendingTrajectoryIndex) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_auto_initial_dt(false);
    integ.adaptive_ = false;
    integ.vectorize_batch_calls_ = true;

    using K = Integrator<astro::Kepler>::IntegRet;
    K x_good, x_bad;
    auto good = leo_state();
    auto bad = origin_state();
    for (int i = 0; i < 7; ++i) {
        x_good[i] = good[i];
        x_bad[i] = bad[i];
    }

    // Lane 1 is the bad one — assert the diagnostic names it.
    std::vector<K> x0s = {x_good, x_bad, x_good};
    Eigen::VectorXd tfs(3);
    tfs << 100.0, 100.0, 100.0;

    try {
        integ.integrate(x0s, tfs);
        FAIL() << "Expected runtime_error from batch finite check, got success.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("Non-finite state"), std::string::npos)
            << "Diagnostic should mention 'Non-finite state'; got: " << msg;
        EXPECT_NE(msg.find("SuperScalar"), std::string::npos)
            << "Should identify the SuperScalar batch site; got: " << msg;
        EXPECT_NE(msg.find("trajectory=1"), std::string::npos)
            << "Should name the offending trajectory index; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}
