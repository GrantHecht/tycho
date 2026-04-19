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

///////////////////////////////////////////////////////////////////////////////
// validate() rejects misconfiguration before it can produce silently-bad
// step cadence. Each branch covers one invariant per controller kind.
///////////////////////////////////////////////////////////////////////////////
TEST(StepControllerTest, IControllerValidateRejectsBadConfig) {
    using tycho::integrators::IController;
    {
        IController c;
        c.gamma = 0.0; // gamma must be > 0
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c;
        c.gamma = 1.5; // gamma must be <= 1
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c;
        c.qmin = 0.0; // qmin must be > 0
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c;
        c.qmin = 1.5; // qmin must be < 1
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c;
        c.qmax = 0.5; // qmax must be > 1
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c;
        c.qmax = 100.0;
        c.qmax_first_step = 50.0; // must be >= qmax
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c;
        c.qsteady_min = 2.0;
        c.qsteady_max = 1.0; // must be >= qsteady_min
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        IController c; // defaults must be valid
        EXPECT_NO_THROW(c.validate());
    }
}

TEST(StepControllerTest, PIControllerValidateRejectsBadConfig) {
    using tycho::integrators::PIController;
    {
        PIController c;
        c.beta1 = -1.0;
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIController c;
        c.beta2 = -0.5;
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIController c;
        c.qoldinit = 0.0; // used as denominator, must be > 0
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIController c; // defaults must be valid
        EXPECT_NO_THROW(c.validate());
    }
}

TEST(StepControllerTest, PIDControllerValidateRejectsBadConfig) {
    using tycho::integrators::PIDController;
    {
        PIDController c;
        c.beta3 = -0.1;
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIDController c;
        c.accept_safety = 0.0;
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIDController c;
        c.accept_safety = 1.5;
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIDController c;
        c.qsteady_min = 2.0;
        c.qsteady_max = 1.0;
        EXPECT_THROW(c.validate(), std::invalid_argument);
    }
    {
        PIDController c; // defaults must be valid
        EXPECT_NO_THROW(c.validate());
    }
}
