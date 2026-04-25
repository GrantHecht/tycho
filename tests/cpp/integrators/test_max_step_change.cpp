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

// -----------------------------------------------------------------------------
// Clamp behaviour: run with a loose tolerance and a very small
// max_step_change. Under those conditions every controller proposal
// would want to grow h by well over max_step_change on SHO (smooth,
// low error), so the driver-side clamp is the binding constraint.
// Step count should scale inversely with max_step_change: tighter
// clamp → more steps.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, MaxStepChangeClamp_TighterBoundTakesMoreSteps) {
    SHO ode(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    const double tf = 10.0;

    auto run_with_clamp = [&](double clamp) {
        Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.001);
        integ.set_abs_tol(1e-4); // Loose — errors stay small, controller wants to grow
        integ.set_rel_tol(1e-4);
        integ.set_max_step_change(clamp);
        (void)integ.integrate(x0, tf);
        return integ.get_naccept() + integ.get_nreject();
    };

    const int64_t steps_1p5 = run_with_clamp(1.5);
    const int64_t steps_10 = run_with_clamp(10.0);

    // Tighter clamp (1.5) constrains per-step growth, forcing more iterations
    // to cover the same interval than the loose clamp (10.0). This test
    // would have passed pre-fix because the clamp was always in effect —
    // its value is as a regression guard: if a future edit deletes or
    // inverts the clamp, `steps_1p5` drops to or below `steps_10` and
    // this fails.
    EXPECT_GT(steps_1p5, steps_10)
        << "max_step_change=1.5 took " << steps_1p5 << " steps vs " << steps_10
        << " for max_step_change=10.0 — clamp appears to have regressed";
    EXPECT_GT(steps_10, 0);
}
