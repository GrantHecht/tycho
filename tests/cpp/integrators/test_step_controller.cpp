///////////////////////////////////////////////////////////////////////////////
// Step controller unit tests
//
// Validates IController::propose_step against the current integrator formula:
//   step_frac_ * h * pow(accerr / err, 1.0 / (error_order_ + err_pow_fac_))
// which is equivalent to:
//   safety * h * pow(1.0 / err_norm, 1.0 / (order + exponent_bias))
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/integrators/step_controller.h"
#include <cmath>
#include <gtest/gtest.h>

using tycho::integrators::IController;

TEST(StepControllerTest, AcceptedStepIncreasesH) {
    IController ctrl;
    double h = 0.1;
    double err_norm = 0.5; // < 1 means accepted
    double h_new = ctrl.propose_step(h, err_norm, 5);
    EXPECT_GT(h_new, h) << "Accepted step (err_norm < 1) should increase h";
}

TEST(StepControllerTest, RejectedStepDecreasesH) {
    IController ctrl;
    double h = 0.1;
    double err_norm = 2.0; // > 1 means rejected
    double h_new = ctrl.propose_step(h, err_norm, 5);
    EXPECT_LT(h_new, h) << "Rejected step (err_norm > 1) should decrease h";
}

TEST(StepControllerTest, BoundaryErrNormOne) {
    IController ctrl;
    double h = 0.1;
    double h_new = ctrl.propose_step(h, 1.0, 5);
    // pow(1/1, 1/6) = 1, so h_new = safety * h
    EXPECT_NEAR(h_new, ctrl.safety * h, 1e-15);
}

TEST(StepControllerTest, MatchesCurrentFormulaOrder5) {
    // Current: step_frac_ * h * pow(accerr / err, 1.0 / (error_order_ + err_pow_fac_))
    // With step_frac_=0.9, err_pow_fac_=1, error_order_=4 (DOPRI54 error order)
    double step_frac = 0.9;
    double err_pow_fac = 1.0;
    double error_order = 4.0;
    double h = 0.05;
    double accerr = 1e-12; // abs_tol (the acceptance threshold)
    double err = 5e-13;    // actual error
    double err_norm = err / accerr; // = 0.5

    double h_current = step_frac * h * std::pow(accerr / err, 1.0 / (error_order + err_pow_fac));

    IController ctrl;
    ctrl.safety = step_frac;
    ctrl.exponent_bias = err_pow_fac;
    double h_new = ctrl.propose_step(h, err_norm, static_cast<int>(error_order));

    EXPECT_NEAR(h_new, h_current, 1e-15);
}

TEST(StepControllerTest, MatchesCurrentFormulaOrder8) {
    double step_frac = 0.9;
    double err_pow_fac = 1.0;
    double error_order = 7.0;
    double h = 10.0;
    double accerr = 1e-12;
    double err = 3e-12; // rejected: err > accerr
    double err_norm = err / accerr; // = 3.0

    double h_current = step_frac * h * std::pow(accerr / err, 1.0 / (error_order + err_pow_fac));

    IController ctrl;
    ctrl.safety = step_frac;
    ctrl.exponent_bias = err_pow_fac;
    double h_new = ctrl.propose_step(h, err_norm, static_cast<int>(error_order));

    EXPECT_NEAR(h_new, h_current, 1e-12);
}

TEST(StepControllerTest, DefaultParametersMatchIntegrator) {
    IController ctrl;
    EXPECT_EQ(ctrl.safety, 0.9);
    EXPECT_EQ(ctrl.exponent_bias, 1.0);
    EXPECT_EQ(ctrl.prev_err_norm_, 0.0);
}

TEST(StepControllerTest, AcceptUpdatesHistory) {
    IController ctrl;
    EXPECT_EQ(ctrl.prev_err_norm_, 0.0);
    ctrl.accept(0.5);
    EXPECT_EQ(ctrl.prev_err_norm_, 0.5);
}

TEST(StepControllerTest, ResetClearsHistory) {
    IController ctrl;
    ctrl.accept(0.7);
    ctrl.reset();
    EXPECT_EQ(ctrl.prev_err_norm_, 0.0);
}
