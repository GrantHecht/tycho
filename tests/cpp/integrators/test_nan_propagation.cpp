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
// Adaptive midpoint guards: verify the new check_state_finite_or_throw calls
// inside the AdaptiveDriver/ParallelDriver adaptive branches catch the two
// midpoint-NaN paths the err_norm chokepoint cannot reach:
//
//   A.1 — extra-stage compute (Vern7/8/9, BS5) returns NaN, which propagates
//         into xnext_mid via the Bmid weighted sum while xnext / xnext_est
//         (computed from the standard stages only) stay finite. The new guard
//         fires inside the adaptive branch with substring "(midpoint)".
//
//   A.2 — xnext_mid itself is finite, but the post-step
//         ode.compute(xnext_mid, xdot_mid) call hits a singular RHS (e.g. 1/r
//         dynamics whose midpoint reconstruction lands on the singularity),
//         producing NaN xdot_mid. The new guard fires before push_back into
//         the user's deriv buffer, with substring "(midpoint deriv)".
//
// Strategy: NaN-injection ODE keyed on the input state's time component.
//   • BS5 (ExtraC[0] = 1/2): NaN at t = h/2 hits the extra-stage compute,
//     corrupts xnext_mid, A.1 guard fires before A.2's compute call.
//   • DOPRI87 (no extra stages, no main stage at c=1/2): NaN at t = h/2 hits
//     only the line-392 ode.compute(xnext_mid, xdot_mid) call, A.2 fires.
// -----------------------------------------------------------------------------

namespace {

// SHO variant that returns NaN derivatives when its input state's time
// component matches `nan_at_t_` to within `tol_`. The trigger is keyed on the
// state-vector time slot (x[2] for the 2-state SHO) so it activates uniformly
// across stage / extra-stage / midpoint-deriv compute calls regardless of
// FSAL bookkeeping or step counter.
struct TimeKeyedNaNSHO
    : oc::StaticODE<TimeKeyedNaNSHO, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                    vf::DenseDerivativeMode::FDiffFwd> {
    using Base = oc::StaticODE<TimeKeyedNaNSHO, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                               vf::DenseDerivativeMode::FDiffFwd>;

    // Default to a NaN trigger value so the equality check against any finite
    // input time always evaluates false — this makes the default-constructed
    // ODE behave like plain SHO. Required because Integrator's delegating
    // constructor default-constructs ode_ before copying the user-supplied one.
    double nan_at_t_ = std::numeric_limits<double>::quiet_NaN();
    double tol_ = 1e-12;

    TimeKeyedNaNSHO() { this->set_ode_size(2, 0, 0); }
    TimeKeyedNaNSHO(double nan_at_t, double tol = 1e-12) : nan_at_t_(nan_at_t), tol_(tol) {
        this->set_ode_size(2, 0, 0);
    }

    template <class InType, class OutType>
    inline void compute_impl(vf::CVecRef<InType> x_, vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        // NOTE: auto& (not auto) so writes propagate to fx_; const_cast_derived
        // returns Derived&, and `auto` would deduce Derived (a copy).
        auto &fx = fx_.const_cast_derived();
        const double t = static_cast<double>(x_[2]);
        if (std::abs(t - nan_at_t_) < tol_) {
            fx[0] = Scalar(std::numeric_limits<double>::quiet_NaN());
            fx[1] = Scalar(std::numeric_limits<double>::quiet_NaN());
            return;
        }
        fx[0] = Scalar(x_[1]);
        fx[1] = Scalar(-x_[0]);
    }
};

} // namespace

// A.1 — BS5 with NaN injected at t = h/2 corrupts xnext_mid via the extra-stage
// compute (BS5 ExtraC[0] = 1/2). Standard stages of BS5 evaluate at c values
// {0, 1/6, 2/9, 3/7, 2/3, 3/4, 1, 1, 1} — none equals 1/2 — so xnext / xnext_est
// stay finite, err_norm guard does NOT fire, and the new midpoint guard inside
// the adaptive branch catches the NaN-laced xnext_mid.
TEST_F(NanPropagationTest, AdaptiveMidpointStateGuardFiresOnExtraStageNaN) {
    // Initial step under HW-disabled adaptive: with def_step_size = 0.1 and
    // tf = 0.1, AdaptiveDriver computes h = 0.9 * H/numsteps = 0.9 * 0.1/2 =
    // 0.045 (numsteps = abs(0.1/0.1)+1 = 2). First step's midpoint is at
    // t = 0 + 0.045/2 = 0.0225. BS5's extra stage 0 fires at the same time.
    //
    // Loose tolerances ensure step 1 is accepted on first try (no controller
    // rejection that would shrink h and shift the midpoint off the trigger).
    constexpr double kFirstStepMidpoint = 0.0225;
    TimeKeyedNaNSHO ode(kFirstStepMidpoint);
    Integrator<TimeKeyedNaNSHO> integ(ode, IVPAlg::BS5, 0.1);
    integ.set_auto_initial_dt(false); // make initial dt deterministic
    integ.set_abs_tol(1.0);
    integ.set_rel_tol(1.0);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    std::vector<Integrator<TimeKeyedNaNSHO>::EventPack> events;

    try {
        (void)integ.integrate_dense(x0, 0.1, events, /*alloutput=*/true);
        FAIL() << "Expected runtime_error from adaptive midpoint guard, got success.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("Non-finite state"), std::string::npos)
            << "Diagnostic should mention 'Non-finite state'; got: " << msg;
        EXPECT_NE(msg.find("AdaptiveDriver::stepper.step (midpoint)"), std::string::npos)
            << "Should identify the new adaptive-branch midpoint guard; got: " << msg;
        EXPECT_EQ(msg.find("Non-finite error norm"), std::string::npos)
            << "err_norm guard must NOT fire here — extra-stage NaN does not flow into "
               "xnext or xnext_est; got: "
            << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}

// A.2 — DOPRI87 with NaN injected at t = h/2 hits ONLY the post-step
// ode.compute(xnext_mid, xdot_mid) call. DOPRI87 has InterpStages = 0 (no
// extra stages) and no main stage at c = 1/2, so all stage evaluations stay
// finite and xnext_mid is finite. The new xdot_mid guard catches the NaN
// derivative before push_back into the user's deriv buffer.
//
// Loose tolerances ensure step 1 is accepted on first try (no controller
// rejection that would shift the midpoint t off the trigger).
TEST_F(NanPropagationTest, AdaptiveMidpointDerivGuardFiresOnSingularRhs) {
    constexpr double kFirstStepMidpoint = 0.0225;
    TimeKeyedNaNSHO ode(kFirstStepMidpoint);
    Integrator<TimeKeyedNaNSHO> integ(ode, IVPAlg::DOPRI87, 0.1);
    integ.set_auto_initial_dt(false);
    integ.set_abs_tol(1.0);
    integ.set_rel_tol(1.0);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    std::vector<Integrator<TimeKeyedNaNSHO>::EventPack> events;

    try {
        (void)integ.integrate_dense(x0, 0.1, events, /*alloutput=*/true);
        FAIL() << "Expected runtime_error from midpoint-deriv guard, got success.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("Non-finite state"), std::string::npos)
            << "Diagnostic should mention 'Non-finite state'; got: " << msg;
        EXPECT_NE(msg.find("AdaptiveDriver::stepper.step (midpoint deriv)"), std::string::npos)
            << "Should identify the new midpoint-deriv guard; got: " << msg;
        EXPECT_EQ(msg.find("Non-finite error norm"), std::string::npos)
            << "err_norm guard must NOT fire — singular RHS only manifests at "
               "ode.compute(xnext_mid); got: "
            << msg;
        EXPECT_EQ(msg.find("(midpoint)\""), std::string::npos)
            << "The state-only midpoint guard must NOT fire — xnext_mid itself is "
               "finite for DOPRI87 (no extra stages); got: "
            << msg;
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

// -----------------------------------------------------------------------------
// Non-finite adjvars surfaces as runtime_error — either via STMDriver's
// Hessian guard (adjhess = ∂²(lf^T · x_final)/∂x₀² absorbs NaN lf and trips
// first), or via the localized adjgrad guard in
// compute_jacobian_adjointgradient_adjointhessian_impl (adjgrad = jx^T ·
// adjvars), whichever fires in the current wiring. The critical contract
// is "non-finite adjoint input → throw", not which guard catches it; both
// paths reject silent propagation of a NaN gradient into the solver.
// -----------------------------------------------------------------------------
TEST_F(NanPropagationTest, NonFiniteAdjvarsThrowsFromAdjointGradientPath) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 1.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    const int n_in = integ.input_rows();
    const int n_out = integ.output_rows();

    Eigen::VectorXd x(n_in);
    auto leo = leo_state();
    for (int i = 0; i < 7; ++i)
        x[i] = leo[i];
    x[n_in - 1] = 10.0;

    Eigen::VectorXd adjvars(n_out);
    adjvars.setConstant(1.0);
    adjvars[0] = std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd fx(n_out);
    Eigen::MatrixXd jx(n_out, n_in);
    Eigen::VectorXd adjgrad(n_in);
    Eigen::MatrixXd adjhess(n_in, n_in);
    fx.setZero();
    jx.setZero();
    adjgrad.setZero();
    adjhess.setZero();

    try {
        integ.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, adjvars);
        FAIL() << "Expected runtime_error from non-finite adjvars; got silent success.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        const bool from_adjgrad_guard = msg.find("Non-finite adjoint gradient") != std::string::npos;
        const bool from_stm_hessian_guard =
            msg.find("Non-finite STM output") != std::string::npos &&
            msg.find("hessian") != std::string::npos;
        EXPECT_TRUE(from_adjgrad_guard || from_stm_hessian_guard)
            << "Diagnostic should come from either the adjgrad guard or the STM hessian "
               "guard; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error; got a different exception type.";
    }
}
