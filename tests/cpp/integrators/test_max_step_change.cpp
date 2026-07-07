// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// max_step_change clamp + setter-validation tests.
//
// The per-step growth clamp lives in AdaptiveDriver and ParallelDriver; the
// prior review noted that neither site had any test pinning it, so a
// regression that dropped the clamp, inverted the comparison, or divided
// by zero would have gone silent.
//
// These tests force the controller onto a huge growth proposal (tiny
// err_norm on first accepted step) and assert the driver-side clamp
// capped h at h * max_step_change.

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

// -----------------------------------------------------------------------------
// Setter validation: reject values that degenerate the clamp semantics
// (<= 1 produces either a divide-or-shrink that collapses dt to zero, or
// a NaN/Inf that propagates through the step loop).
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, MaxStepChangeSetterRejectsValuesAtOrBelowOne) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);

    EXPECT_THROW(integ.set_max_step_change(1.0), std::invalid_argument);
    EXPECT_THROW(integ.set_max_step_change(0.5), std::invalid_argument);
    EXPECT_THROW(integ.set_max_step_change(-2.0), std::invalid_argument);
    EXPECT_THROW(integ.set_max_step_change(0.0), std::invalid_argument);
    EXPECT_THROW(integ.set_max_step_change(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(integ.set_max_step_change(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);

    // Values > 1 accepted; round-trip via the getter.
    integ.set_max_step_change(2.5);
    EXPECT_DOUBLE_EQ(integ.get_max_step_change(), 2.5);
    integ.set_max_step_change(10.0);
    EXPECT_DOUBLE_EQ(integ.get_max_step_change(), 10.0);
}

namespace {
// Constant-rate ODE dx/dt = 1. The embedded error is identically zero, so the
// controller proposes its MAXIMUM growth on every step — making max_step_change
// the binding constraint on every steady-state step. SHO is unsuitable here: its
// ideal step size stabilises once cruising and the controller stops wanting to
// grow, so after the first-step exemption (OrdinaryDiffEq qmax_first_step parity)
// jumps straight to cruising, the clamp never binds again. ConstRate keeps the
// clamp binding on every step after the first.
struct ConstRate : oc::StaticODE<ConstRate, 1, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                                  vf::DenseDerivativeMode::FDiffFwd> {
    ConstRate() { this->set_ode_size(1, 0, 0); }
    template <class InType, class OutType>
    inline void compute_impl(vf::CVecRef<InType> x_, vf::CVecRef<OutType> fx_) const {
        auto &fx = fx_.const_cast_derived();
        fx[0] = typename InType::Scalar(1.0);
    }
};
} // namespace

// -----------------------------------------------------------------------------
// Steady-state clamp: after the first (exempt) step, no accepted-step growth
// ratio may exceed max_step_change. On ConstRate the controller wants maximum
// growth every step, so the clamp binds at exactly its value on each steady step.
// Note the first-step exemption: step 1 is the initial h and step 2 carries the
// unclamped first-step growth, so the steady-state invariant is checked from the
// third accepted step onward.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, MaxStepChangeClamp_BoundsSteadyStateGrowth) {
    ConstRate ode;
    Eigen::Vector2d x0;
    x0 << 0.0, 0.0; // x = 0, t = 0
    Integrator<ConstRate> integ(ode, IVPAlg::DOPRI87, 1.0);
    integ.set_initial_step_size(1.0); // HW off; adaptive stays on
    integ.set_abs_tol(1e-4);
    integ.set_rel_tol(1e-4);
    const double clamp = 1.5;
    integ.set_max_step_change(clamp);

    auto traj = integ.integrate_dense(x0, 1.0e6);
    ASSERT_GE(traj.size(), 6u) << "need several steady-state steps to probe the clamp";
    constexpr int kTimeIdx = 1; // 1-state ODE → time in slot 1

    int checked = 0;
    for (size_t k = 3; k + 1 < traj.size(); ++k) {
        const double h_prev = traj[k - 1][kTimeIdx] - traj[k - 2][kTimeIdx];
        const double h_cur = traj[k][kTimeIdx] - traj[k - 1][kTimeIdx];
        if (h_prev <= 0.0)
            continue;
        EXPECT_LE(h_cur / h_prev, clamp * (1.0 + 1e-9))
            << "steady-state growth exceeded max_step_change at accepted step " << k;
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no steady-state steps were exercised";
}

// -----------------------------------------------------------------------------
// Regression guard: a tighter clamp forces more iterations to cover the same
// interval (the post-first-step ramp is geometric in the clamp factor). ConstRate
// keeps the controller perpetually wanting to grow, so the clamp — not the
// dynamics — governs the step count after the first step. If a future edit
// deletes or inverts the clamp, steps_1p5 drops to or below steps_10 and this
// fails.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, MaxStepChangeClamp_TighterBoundTakesMoreSteps) {
    ConstRate ode;
    Eigen::Vector2d x0;
    x0 << 0.0, 0.0;
    const double tf = 1.0e6;

    auto run_with_clamp = [&](double clamp) {
        Integrator<ConstRate> integ(ode, IVPAlg::DOPRI87, 1.0);
        integ.set_initial_step_size(1.0);
        integ.set_abs_tol(1e-4);
        integ.set_rel_tol(1e-4);
        integ.set_max_step_change(clamp);
        (void)integ.integrate(x0, tf);
        return integ.get_naccept() + integ.get_nreject();
    };

    const int64_t steps_1p5 = run_with_clamp(1.5);
    const int64_t steps_10 = run_with_clamp(10.0);

    EXPECT_GT(steps_1p5, steps_10)
        << "max_step_change=1.5 took " << steps_1p5 << " steps vs " << steps_10
        << " for max_step_change=10.0 — clamp appears to have regressed";
    EXPECT_GT(steps_10, 0);
}
