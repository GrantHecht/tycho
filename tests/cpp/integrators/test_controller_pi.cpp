///////////////////////////////////////////////////////////////////////////////
// PIController unit tests — bit-exact against Julia's PIController for
// DOPRI54 (β₁=17/100, β₂=4/100) defaults.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/integrators/step_controller.h"
#include <cmath>
#include <gtest/gtest.h>

using tycho::integrators::PIController;

namespace {
constexpr double GAMMA = 0.9;
constexpr double QMIN = 1.0 / 5.0;
constexpr double QMAX = 10.0;
constexpr double QOLDINIT = 1.0e-4;

PIController dopri54_defaults() {
    PIController c;
    c.beta1 = 17.0 / 100.0; // DOPRI54 override
    c.beta2 = 4.0 / 100.0;
    c.gamma = GAMMA;
    c.qmin = QMIN;
    c.qmax = QMAX;
    c.qoldinit = QOLDINIT;
    c.errold_ = QOLDINIT;
    return c;
}

// Julia formula, for reference computation in tests.
double julia_pi_q(double err_norm, double errold, const PIController &c, double qmax_eff) {
    if (err_norm == 0.0) return 1.0 / qmax_eff;
    double q11 = std::pow(err_norm, c.beta1);
    double qtmp = q11 / std::pow(errold, c.beta2);
    double qclip = std::max(1.0 / qmax_eff, std::min(1.0 / c.qmin, qtmp / c.gamma));
    return qclip;
}
} // namespace

TEST(PIControllerTest, FirstStepMatchesJulia) {
    PIController c = dopri54_defaults();
    double h = 0.1;
    double err_norm = 0.6;
    auto out = c.update(h, err_norm, /*order=*/4, /*naccept=*/0);
    double q_expected = julia_pi_q(err_norm, QOLDINIT, c, c.qmax_first_step);
    EXPECT_TRUE(out.accepted);
    EXPECT_NEAR(out.q, q_expected, 1e-14);
    EXPECT_NEAR(out.dt_new, h / q_expected, 1e-14);
}

TEST(PIControllerTest, SubsequentStepUsesErrold) {
    PIController c = dopri54_defaults();
    // Step 1: accept EEst=0.5
    c.update(0.1, 0.5, 4, 0);
    EXPECT_NEAR(c.errold_, std::max(0.5, QOLDINIT), 1e-14);
    // Step 2: another accept
    double h2 = 0.15;
    double err_norm2 = 0.3;
    auto out = c.update(h2, err_norm2, 4, 1);
    double q_expected = julia_pi_q(err_norm2, 0.5, c, c.qmax);
    EXPECT_TRUE(out.accepted);
    EXPECT_NEAR(out.q, q_expected, 1e-14);
}

TEST(PIControllerTest, RejectUsesQ11NotQ) {
    PIController c = dopri54_defaults();
    double h = 0.1;
    double err_norm = 2.5; // rejects
    auto out = c.update(h, err_norm, 4, 1);
    EXPECT_FALSE(out.accepted);
    // dt shrink uses q_reject = min(1/qmin, q11/gamma)
    double q11 = std::pow(err_norm, c.beta1);
    double q_reject = std::min(1.0 / c.qmin, q11 / c.gamma);
    EXPECT_NEAR(out.dt_new, h / q_reject, 1e-14);
    // errold_ unchanged after reject
    EXPECT_NEAR(c.errold_, QOLDINIT, 1e-14);
}

TEST(PIControllerTest, ZeroErrGivesQmax) {
    PIController c = dopri54_defaults();
    auto out = c.update(0.1, 0.0, 4, 1);
    EXPECT_TRUE(out.accepted);
    EXPECT_NEAR(out.dt_new, 0.1 * c.qmax, 1e-12);
}

TEST(PIControllerTest, ResetClearsState) {
    PIController c = dopri54_defaults();
    c.update(0.1, 0.5, 4, 1);
    c.reset();
    EXPECT_NEAR(c.errold_, c.qoldinit, 1e-14);
    EXPECT_NEAR(c.q11_, 1.0, 1e-14);
}

TEST(PIControllerTest, BackwardIntegrationPreservesSign) {
    PIController c = dopri54_defaults();
    // Backward: h < 0. Controller returns h/q where q > 0, so sign preserved.
    auto out = c.update(/*h=*/-0.1, /*err_norm=*/0.3, /*order=*/4, /*naccept=*/1);
    EXPECT_TRUE(out.accepted);
    EXPECT_LT(out.dt_new, 0.0) << "Backward step must stay negative";
    // And reject path:
    auto out2 = c.update(/*h=*/-0.1, /*err_norm=*/3.0, 4, 2);
    EXPECT_FALSE(out2.accepted);
    EXPECT_LT(out2.dt_new, 0.0);
}

TEST(PIControllerTest, QsteadyDeadbandSnapsToOne) {
    PIController c = dopri54_defaults();
    c.qsteady_min = 0.5;
    c.qsteady_max = 2.0;
    c.errold_ = 1.0;
    auto out = c.update(0.1, 0.9, 4, 1);
    EXPECT_TRUE(out.accepted);
    // q = 0.9^0.17 / 1.0^0.04 / 0.9 ≈ 0.982 / 0.9 ≈ 1.091 → in [0.5, 2.0] → snaps to 1
    EXPECT_NEAR(out.dt_new, 0.1, 1e-12);
}

///////////////////////////////////////////////////////////////////////////////
// Clamp paths — exercise the qmax growth ceiling and the qmin shrink floor
// on the accept branch. The common path does not hit either clamp; previous
// coverage only exercised them via err_norm == 0 (degenerate). These tests
// pin both clamps with realistic err_norm values.
///////////////////////////////////////////////////////////////////////////////
TEST(PIControllerTest, QmaxGrowthClampHitsOnTinyError) {
    PIController c = dopri54_defaults();
    c.errold_ = 1.0; // neutralize the proportional history term
    // Tiny err_norm → q_raw very small → clamped to 1/qmax → dt_new = h * qmax
    auto out = c.update(/*h=*/0.1, /*err_norm=*/1.0e-12, /*order=*/4, /*naccept=*/5);
    EXPECT_TRUE(out.accepted);
    EXPECT_NEAR(out.dt_new, 0.1 * c.qmax, 1e-12)
        << "Expected dt_new clamped to h * qmax via the lower 1/qmax clip on q.";
    EXPECT_NEAR(out.q, 1.0 / c.qmax, 1e-12) << "Output q should reflect the clamp value.";
}

TEST(PIControllerTest, QminShrinkClampHitsOnAccept) {
    PIController c = dopri54_defaults();
    // Set errold microscopic so q11/errold^beta2 blows past 1/qmin even after
    // gamma scaling. With beta2=0.04 the dependence is weak, so we need an
    // extreme errold to clear the clamp threshold.
    //   q_pre = (err_norm^beta1 / errold^beta2) / gamma must exceed 1/qmin = 5.
    c.errold_ = 1.0e-50;
    auto out = c.update(/*h=*/0.1, /*err_norm=*/0.95, /*order=*/4, /*naccept=*/5);
    ASSERT_TRUE(out.accepted) << "err_norm < 1 should accept";
    // q is clamped to 1/qmin → dt_new = h * qmin (the maximum permitted shrink).
    EXPECT_NEAR(out.dt_new, 0.1 * c.qmin, 1e-12)
        << "Expected dt_new floored at h * qmin via the upper 1/qmin clip on q.";
    EXPECT_NEAR(out.q, 1.0 / c.qmin, 1e-12);
}
