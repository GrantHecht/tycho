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

// Parametrized variant — exercises the same NaN sites under each of the three
// controllers. Proves controller switching does not regress the NaN guards
// (P0.3: inline std::isfinite(err_norm) guard) and that the diagnostic still
// fires even when PI/PID history-state machinery is active.
class NanPropagationControllerTest : public VectorFunctionFixture,
                                     public ::testing::WithParamInterface<IVPController> {};

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
        EXPECT_NE(msg.find("AdaptiveDriver::stepper.step"), std::string::npos)
            << "Should identify the AdaptiveDriver stepper site; got: " << msg;
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

// Parametrized HW path — swap the controller before integrate and confirm the
// HW-initdt guard still fires first. If my err_norm guard were to misorder
// against HW or against the per-stepper finite-state check, this would catch
// it (the error message would shift) under at least one controller.
TEST_P(NanPropagationControllerTest, HairerWannerThrowsOnOriginKepler) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_controller(GetParam());
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    auto x0 = origin_state();
    try {
        integ.integrate(x0, 100.0);
        FAIL() << "Expected runtime_error from HW-initdt finite check.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("Hairer-Wanner"), std::string::npos) << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error.";
    }
}

// Parametrized scalar-adaptive path: HW off + adaptive on + pathological
// dynamics. Exercises the err_norm / xnext finite checks under each controller
// and proves PI/PID history state (errold_, err_[], qold_) does not cause the
// guard to fail silently.
TEST_P(NanPropagationControllerTest, ScalarAdaptiveThrowsOnOriginKepler) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_controller(GetParam());
    integ.set_auto_initial_dt(false);
    integ.set_abs_tol(1e-6);
    integ.set_rel_tol(1e-9);

    auto x0 = origin_state();
    try {
        integ.integrate(x0, 100.0);
        FAIL() << "Expected runtime_error under controller " << static_cast<int>(GetParam());
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        // Either xnext-finite check fires first, or err_norm guard. Both are
        // acceptable exit points — neither should leak past to a max_steps bail.
        const bool hit_xnext = msg.find("Non-finite state") != std::string::npos;
        const bool hit_errnorm = msg.find("Non-finite error norm") != std::string::npos;
        EXPECT_TRUE(hit_xnext || hit_errnorm) << "Expected xnext or err_norm guard; got: " << msg;
        EXPECT_EQ(msg.find("max_steps"), std::string::npos)
            << "Must not fall through to max_steps diagnostic: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error.";
    }
}

INSTANTIATE_TEST_SUITE_P(AllControllers, NanPropagationControllerTest,
                         ::testing::Values(IVPController::I, IVPController::PI,
                                           IVPController::PID));

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
        EXPECT_NE(msg.find("ParallelDriver::stepper.step"), std::string::npos)
            << "Should identify the ParallelDriver stepper site; got: " << msg;
        EXPECT_NE(msg.find("trajectory=1"), std::string::npos)
            << "Should name the offending trajectory index; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}

// -----------------------------------------------------------------------------
// Event VF NaN guard (event_handler.h:80-87).
// An event function that returns a non-finite value on a finite state
// must surface immediately with t + event-index context. Without the
// guard, `vprev * vnext < 0` silently evaluates to false under IEEE
// 754 when either operand is NaN, dropping the crossing with no signal.
// -----------------------------------------------------------------------------
TEST_F(NanPropagationTest, EventVFReturningNaNThrowsWithContext) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1e-10);
    integ.set_rel_tol(1e-10);

    // Build a constant event VF that always returns NaN. Passes through the
    // VectorFunction call site, produces non-finite output on a finite state,
    // must trip the guard on the first post-step crossing check.
    Eigen::VectorXd nan_out(1);
    nan_out[0] = std::numeric_limits<double>::quiet_NaN();
    GenericFunction<-1, 1> nan_event = Constant<-1, 1>(7, nan_out);
    std::vector<Integrator<astro::Kepler>::EventPack> events;
    events.push_back({nan_event, 0, 0});

    // Use a nominal LEO state — dynamics stay finite, so only the event VF
    // can trip the finite check.
    auto x0_vec = leo_state();
    Integrator<astro::Kepler>::IntegRet x0;
    for (int i = 0; i < 7; ++i)
        x0[i] = x0_vec[i];
    double tf = 100.0;

    try {
        integ.integrate(x0, tf, events);
        FAIL() << "Expected runtime_error from event VF NaN guard.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("non-finite"), std::string::npos)
            << "Diagnostic should mention 'non-finite'; got: " << msg;
        EXPECT_NE(msg.find("Event function"), std::string::npos)
            << "Should name the event site; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}

// -----------------------------------------------------------------------------
// STM API NaN contract. `integrate_stm` wraps `integrate_impl` + STM chain
// (stm_driver.h). On an ill-defined state (Kepler origin), the upstream
// finite-state check fires before the STM chain runs; this pins the
// end-to-end contract that the STM API does not silently return a
// NaN-laced Jacobian. If a future change removes the upstream check, the
// `check_stm_finite_or_throw` guard in STMDriver is the last line of
// defense — and these tests still expect a throw.
// -----------------------------------------------------------------------------
TEST_F(NanPropagationTest, IntegrateStmThrowsAtOriginSingularity) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    auto x0 = origin_state();
    EXPECT_THROW({ auto r = integ.integrate_stm(x0, 100.0); }, std::runtime_error);
}

// Batch STM + Hessian API. Same contract as above for the batch path: the
// single-element batch must throw rather than silently return a tuple
// containing NaN.
TEST_F(NanPropagationTest, IntegrateStm2ThrowsAtOriginSingularity) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    using K = Integrator<astro::Kepler>::IntegRet;
    K x0;
    auto x0_vec = origin_state();
    for (int i = 0; i < 7; ++i)
        x0[i] = x0_vec[i];
    K lf;
    for (int i = 0; i < 7; ++i)
        lf[i] = 1.0;

    std::vector<K> x0s = {x0};
    Eigen::VectorXd tfs(1);
    tfs[0] = 100.0;
    std::vector<K> lfs = {lf};

    EXPECT_THROW({ auto r = integ.integrate_stm2(x0s, tfs, lfs); }, std::runtime_error);
}
