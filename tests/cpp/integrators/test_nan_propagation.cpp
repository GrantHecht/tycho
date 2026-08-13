///////////////////////////////////////////////////////////////////////////////
// NaN propagation / recovery tests
//
// Covers how the integrator responds to non-finite state along two axes:
//
//   * Unrecoverable NaN DYNAMICS at the initial point (estimate_initial_dt) and
//     in fixed-step mode still THROW with a site-naming diagnostic — a NaN at x0
//     cannot be resolved by step control, and OrdinaryDiffEq errors here too.
//
//   * A non-finite error norm / midpoint in the ADAPTIVE loop is treated as a
//     rejected step: the driver shrinks h and retries (OrdinaryDiffEq parity —
//     EEst == NaN fails EEst <= 1). A *transient* singularity resolves at smaller
//     h and the integration completes (InteriorPointSolver / multiple-shooting recovery); a
//     *persistent* singularity (state never moves) reject-shrinks h to underflow
//     and the zero-progress stall guard throws a bounded, specific diagnostic.
//
// Strategy: drive the Kepler dynamics at the origin (r = 0). Kepler's
// acceleration is -mu * r / |r|^3, which evaluates to 0/0 = NaN when
// r is the zero vector — a real, reachable ill-defined state for this
// dynamics, not a synthetic injection. A time-keyed NaN SHO injects a
// transient (single-time) NaN for the recovery cases.
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

// Parametrized scalar-adaptive path: HW off + adaptive on + a PERSISTENT
// singularity (Kepler at the origin, where acc = -mu*r/|r|^3 = 0/0 = NaN for ANY
// step size). The non-finite reject-and-shrink path — which lets a *transient*
// NaN recover at smaller h — cannot resolve a persistent singularity: xi never
// moves (rejected steps don't advance the state), so every retry re-produces the
// NaN, h shrinks until it underflows the representable spacing of t, and the
// zero-progress stall guard fires. This confirms the failure is bounded and
// diagnosed (not an infinite loop or silent hang) under every controller. The
// non-finite reject bypasses the controller (it can't derive a factor from a
// NaN EEst), so all three controllers behave identically here — the test proves
// PI/PID history state does not change the terminal diagnostic.
TEST_P(NanPropagationControllerTest, ScalarAdaptivePersistentSingularityStalls) {
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
        // A persistent singularity reject-shrinks h to underflow, then the
        // zero-progress stall guard throws the "underflowed" diagnostic (which
        // also names a singularity as a possible cause).
        EXPECT_NE(msg.find("underflowed"), std::string::npos)
            << "persistent singularity should reject-shrink to a stall diagnostic; got: " << msg;
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

// A.1 — a TRANSIENT midpoint NaN must RECOVER via reject-and-shrink, not abort.
// BS5's extra stage (ExtraC[0] = 1/2) evaluates at the step midpoint; a NaN
// injected at the single time t = h/2 corrupts xnext_mid while xnext / xnext_est
// (standard stages, c values {0, 1/6, 2/9, 3/7, 2/3, 3/4, 1, 1, 1} — none 1/2)
// stay finite, so the err_norm guard does not fire and the adaptive midpoint
// finiteness check sees the NaN. Under the reject-and-shrink policy this rejects
// the step and shrinks h, moving every evaluation point off the single trigger
// time; the integration then completes past t = 0.0225 and never revisits it.
// This is the transient-singularity recovery the InteriorPointSolver / multiple-shooting use
// case needs — the integrator must not throw.
TEST_F(NanPropagationTest, AdaptiveTransientMidpointNaNRecoversByShrinking) {
    // First step (HW off, def_step 0.1, tf 0.1): h = 0.9 * H/numsteps = 0.045,
    // midpoint at t = 0.0225 = BS5 extra-stage time — the injected NaN. Loose
    // tolerances so the only rejection is the NaN one.
    constexpr double kFirstStepMidpoint = 0.0225;
    TimeKeyedNaNSHO ode(kFirstStepMidpoint);
    Integrator<TimeKeyedNaNSHO> integ(ode, IVPAlg::BS5, 0.1);
    integ.set_auto_initial_dt(false); // make initial dt deterministic
    integ.set_abs_tol(1.0);
    integ.set_rel_tol(1.0);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    std::vector<Integrator<TimeKeyedNaNSHO>::EventPack> events;

    Integrator<TimeKeyedNaNSHO>::DenseEventRet traj;
    EXPECT_NO_THROW({ traj = integ.integrate_dense(x0, 0.1, events, /*alloutput=*/true); })
        << "a transient midpoint NaN must recover by rejecting and shrinking h, not throw";
    // Recovery must actually reach tf with a finite state — not merely "not throw"
    // (a recovery that restored FSAL wrong or landed on NaN would still pass a bare
    // EXPECT_NO_THROW if it happened to complete).
    const auto &grid = std::get<0>(traj);
    ASSERT_FALSE(grid.empty());
    const auto &xf = grid.back();
    EXPECT_NEAR(xf[2], 0.1, 1e-9) << "recovered trajectory must reach tf";
    EXPECT_TRUE(std::isfinite(xf[0]) && std::isfinite(xf[1]))
        << "recovered final state must be finite";
}

// A TRANSIENT full-step-node NaN must RECOVER via the PRIMARY err_norm reject
// branch (not the midpoint guard). DOPRI54's last stage (c = 1) evaluates at the
// step-end node t0 + h; a NaN injected at the single time t = 0.045 (= the first
// step's end node with HW off, def_step 0.1, tf 0.1) flows into xnext AND
// xnext_est, so err_norm goes non-finite and the driver rejects + shrinks h. The
// shrunk retry's nodes move off 0.045 and the integration completes; adaptive
// re-alignment never lands another evaluation exactly on 0.045 (a measure-zero
// coincidence at tol 1e-12). Recovery must produce the SAME final state as the
// clean SHO (NaN trigger disabled) — this pins correct-state recovery through the
// err_norm branch, which the persistent-singularity stall test only drives to
// termination, never to recovery. DOPRI54 is FSAL, so the scalar restore_fsal on
// the err_norm reject path is exercised here.
TEST_F(NanPropagationTest, AdaptiveTransientErrNormNaNRecoversByShrinking) {
    constexpr double kFirstStepNode = 0.045;
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto run = [&](double nan_at_t) {
        TimeKeyedNaNSHO ode(nan_at_t);
        Integrator<TimeKeyedNaNSHO> integ(ode, IVPAlg::DOPRI54, 0.1);
        integ.set_auto_initial_dt(false);
        integ.set_abs_tol(1e-8);
        integ.set_rel_tol(1e-8);
        return integ.integrate(x0, 0.1);
    };

    // Clean reference: identical integration with the trigger disabled.
    auto ref = run(std::numeric_limits<double>::quiet_NaN());
    Integrator<TimeKeyedNaNSHO>::IntegRet got;
    EXPECT_NO_THROW({ got = run(kFirstStepNode); })
        << "a transient full-step-node NaN must recover via err_norm reject-and-shrink";
    for (int i = 0; i < 2; ++i)
        EXPECT_NEAR(got[i], ref[i], 1e-7)
            << "recovered state must match the clean reference; component " << i;
    EXPECT_NEAR(got[2], 0.1, 1e-9) << "recovered trajectory must reach tf";
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
// Batch (SIMD) reject-and-shrink recovery. The scalar path is covered above; the
// per-lane batch path has its own reject-and-shrink block (parallel_driver.h),
// and for FSAL methods a distinct FSAL-consistency concern on a rejected lane.
// Five trajectories force a SECOND SIMD pack (W = 4), so trajectory index 4 is
// packed — the exact configuration in which a lane-vs-trajectory index confusion
// in the FSAL reject path would corrupt SuperScalar-buffer memory. DOPRI54 is
// FSAL. A single-time NaN at every lane's first step-end node (t = 0.045) makes
// each lane's err_norm non-finite once; every lane must reject, shrink, and
// recover to the SAME state the scalar driver produces.
// -----------------------------------------------------------------------------
TEST_F(NanPropagationTest, BatchTransientNaNRecoversAndMatchesScalar) {
    constexpr double kFirstStepNode = 0.045;
    TimeKeyedNaNSHO ode(kFirstStepNode);
    using I = Integrator<TimeKeyedNaNSHO>;
    using R = I::IntegRet;

    std::vector<R> x0s(5);
    for (int j = 0; j < 5; ++j) {
        x0s[j][0] = 1.0 + 0.1 * j; // distinct amplitude per lane
        x0s[j][1] = 0.0;
        x0s[j][2] = 0.0; // t0
    }
    Eigen::VectorXd tfs(5);
    tfs.setConstant(0.1);

    auto run = [&](bool vectorize) {
        I integ(ode, IVPAlg::DOPRI54, 0.1);
        integ.set_auto_initial_dt(false);
        integ.set_abs_tol(1e-8);
        integ.set_rel_tol(1e-8);
        integ.vectorize_batch_calls_ = vectorize;
        return integ.integrate(x0s, tfs);
    };

    std::vector<R> scalar, batch;
    ASSERT_NO_THROW({ scalar = run(false); });
    EXPECT_NO_THROW({ batch = run(true); })
        << "vectorized batch transient NaN must recover per-lane, not throw";
    ASSERT_EQ(scalar.size(), 5u);
    ASSERT_EQ(batch.size(), 5u);
    for (int j = 0; j < 5; ++j)
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(batch[j][i], scalar[j][i], 1e-9)
                << "vectorized batch must match scalar after recovery; traj=" << j << " comp=" << i;
}

// Batch persistent singularity: a lane that never resolves under step reduction
// (Kepler at the origin, acc = 0/0 = NaN for any h) must reject-shrink to an
// underflow stall with a per-lane diagnostic — the batch analog of
// ScalarAdaptivePersistentSingularityStalls. The singular lane is index 4 (second
// SIMD pack) on the FSAL DOPRI54 method, so the reject path runs with a
// trajectory index >= the SIMD width.
TEST_F(NanPropagationTest, BatchPersistentSingularityStalls) {
    astro::Kepler kep(kMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 1.0);
    integ.set_auto_initial_dt(false);
    integ.set_abs_tol(1e-6);
    integ.set_rel_tol(1e-9);
    integ.vectorize_batch_calls_ = true; // adaptive on (default)

    using K = Integrator<astro::Kepler>::IntegRet;
    K good, bad;
    auto g = leo_state();
    auto b = origin_state();
    for (int i = 0; i < 7; ++i) {
        good[i] = g[i];
        bad[i] = b[i];
    }
    std::vector<K> x0s = {good, good, good, good, bad}; // singular lane = index 4
    Eigen::VectorXd tfs(5);
    tfs.setConstant(100.0);

    try {
        integ.integrate(x0s, tfs);
        FAIL() << "batch persistent singularity must stall-throw";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("underflowed"), std::string::npos)
            << "persistent singular lane should reject-shrink to a stall; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error.";
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
