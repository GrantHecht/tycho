///////////////////////////////////////////////////////////////////////////////
// Step controller unit tests — Julia-form IController (SP3).
//
// Validates IController::update() against Julia's OrdinaryDiffEqCore form:
//   q   = EEst^(1/k) / γ,   clipped to [1/qmax, 1/qmin]
//   dt' = dt / q
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/integrators/step_controller.h"
#include <cmath>
#include <gtest/gtest.h>

using tycho::integrators::IController;

TEST(StepControllerTest, ResetClearsState) {
    IController ctrl;
    ctrl.update(0.1, 1.5, 5, 1); // reject, stores qold
    ctrl.reset();
    EXPECT_NEAR(ctrl.qold_, 1.0, 1e-15);
}

TEST(StepControllerTest, JuliaForm_EEstHalfShrinksByRoot) {
    IController ctrl; // default: gamma=0.9, qmin=1/5, qmax=10
    // EEst = 0.5 (accept), order = 5 (DOPRI54 error order + 1 = 5 denominator)
    auto out = ctrl.update(/*h=*/0.1, /*err_norm=*/0.5, /*order=*/5, /*naccept=*/1);
    EXPECT_TRUE(out.accepted);
    // q = EEst^(1/(order+1))/gamma = 0.5^(1/6) / 0.9
    // dt_new = h / q = 0.1 * 0.9 / 0.5^(1/6)
    EXPECT_NEAR(out.dt_new, 0.1 * 0.9 / std::pow(0.5, 1.0 / 6.0), 1e-12);
}

TEST(StepControllerTest, JuliaForm_EEstLargeRejects) {
    IController ctrl;
    auto out = ctrl.update(0.1, 2.0, 5, 1);
    EXPECT_FALSE(out.accepted);
    // On reject: dt_new = h/q (same formula, clipped to [1/qmax, 1/qmin])
    EXPECT_LT(out.dt_new, 0.1);
}

TEST(StepControllerTest, JuliaForm_EEstZeroGivesMaxGrowth) {
    IController ctrl;
    auto out = ctrl.update(0.1, 0.0, 5, 1);
    EXPECT_TRUE(out.accepted);
    // q = 1/qmax = 0.1, dt_new = h/q = 10*h
    EXPECT_NEAR(out.dt_new, 0.1 * ctrl.qmax, 1e-12);
}

TEST(StepControllerTest, JuliaForm_QsteadyDeadband) {
    IController ctrl;
    ctrl.qsteady_min = 0.5;
    ctrl.qsteady_max = 2.0;
    // EEst=0.9 → q ≈ 0.9^(1/6)/0.9 ≈ 0.9824/0.9 ≈ 1.092 → in [0.5, 2], snaps to 1
    auto out = ctrl.update(0.1, 0.9, 5, 1);
    EXPECT_TRUE(out.accepted);
    EXPECT_NEAR(out.dt_new, 0.1, 1e-12);
}

TEST(StepControllerTest, JuliaForm_FirstStepQmaxOverride) {
    IController ctrl;
    ctrl.qmax_first_step = 10000.0;
    // naccept=0 triggers first-step qmax override (allows very large dt growth).
    auto out = ctrl.update(0.1, 0.0, 5, 0);
    EXPECT_TRUE(out.accepted);
    // dt_new = h * qmax_first_step = 0.1 * 10000 = 1000
    EXPECT_NEAR(out.dt_new, 0.1 * 10000.0, 1e-6);
}

TEST(StepControllerTest, JuliaForm_ResetClearsState) {
    IController ctrl;
    ctrl.update(0.1, 1.5, 5, 1); // reject, stores qold
    ctrl.reset();
    // After reset, a zero-error step gets full qmax (first-step override via naccept=0)
    auto out = ctrl.update(0.1, 0.0, 5, 0);
    EXPECT_TRUE(out.accepted);
    EXPECT_NEAR(out.dt_new, 0.1 * ctrl.qmax_first_step, 1e-6);
}
