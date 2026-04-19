///////////////////////////////////////////////////////////////////////////////
// PIDController unit tests — bit-exact against Julia's PIDController with
// Söderlind H211PI defaults (β₁=1/6, β₂=1/6, β₃=0).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/integrators/step_controller.h"
#include <cmath>
#include <gtest/gtest.h>

using tycho::integrators::PIDController;

namespace {
constexpr double ACCEPT_SAFETY = 0.81;

PIDController h211pi() {
    PIDController c;
    c.beta1 = 1.0 / 6.0;
    c.beta2 = 1.0 / 6.0;
    c.beta3 = 0.0;
    c.accept_safety = ACCEPT_SAFETY;
    return c;
}
} // namespace

TEST(PIDControllerTest, LimiterShape) {
    EXPECT_NEAR(PIDController::default_limiter(1.0), 1.0, 1e-15);
    // limiter(0) = 1 + atan(-1) ≈ 0.2146
    EXPECT_NEAR(PIDController::default_limiter(0.0), 1.0 + std::atan(-1.0), 1e-15);
    EXPECT_NEAR(PIDController::default_limiter(1e9), 1.0 + std::atan(1e9 - 1.0), 1e-12);
}

TEST(PIDControllerTest, FirstStepMatchesJuliaFormula) {
    PIDController c = h211pi();
    double h = 0.1;
    double err_norm = 0.5;
    int order = 5;
    auto out = c.update(h, err_norm, order, 0);
    double k = order + 1.0;
    double eps_min = 2.220446049250313e-16;
    double eest = std::max(err_norm, eps_min);
    double err1 = 1.0 / eest;
    double factor = std::pow(err1, c.beta1 / k) * std::pow(1.0, c.beta2 / k) *
                    std::pow(1.0, c.beta3 / k);
    factor = PIDController::default_limiter(factor);
    EXPECT_NEAR(out.dt_new, h * factor, 1e-14);
    EXPECT_TRUE(out.accepted); // factor > accept_safety
}

TEST(PIDControllerTest, RejectsWhenFactorBelowSafety) {
    PIDController c = h211pi();
    double h = 0.1;
    // limiter(x) = 1 + atan(x-1); for accept_safety=0.81, need x < 1 + tan(-0.19)
    // ≈ 0.808. With k=6, β₁=β₂=1/6, β₃=0 and err_[1]=err_[2]=1 (initial):
    //   factor_raw = (1/EEst)^(1/36); factor < 0.808 → EEst > 0.808^(-36) ≈ 2150.
    // Use err_norm = 1e6 to comfortably clear the threshold.
    auto out = c.update(h, 1e6, 5, 1);
    EXPECT_FALSE(out.accepted);
    EXPECT_LT(out.dt_new, h);
}

TEST(PIDControllerTest, HistoryShiftsOnAccept) {
    // Julia PIDController semantics (controllers.jl step_accept_controller!):
    //   controller.err[3] = controller.err[2]   // shift "two back"
    //   controller.err[2] = controller.err[1]   // shift "one back" <- current
    // In zero-indexed C++: err_[2] := err_[1]; err_[1] := err_[0]
    // So after step 2 accept, err_[2] holds step 1's 1/EEst value.
    PIDController c = h211pi();
    c.update(0.1, 0.5, 5, 0);       // step 1 accept; err_[0]=2
    double err1_before = c.err_[0]; // = 2 (= 1/0.5)
    c.update(0.1, 0.3, 5, 1);       // step 2 accept; err_[0]=3.333
    EXPECT_NEAR(c.err_[2], err1_before, 1e-15);
}

TEST(PIDControllerTest, HistoryUnchangedOnReject) {
    PIDController c = h211pi();
    c.update(0.1, 0.5, 5, 0);
    auto err_snap = c.err_;
    c.update(0.1, 1e6, 5, 1); // reject (see RejectsWhenFactorBelowSafety)
    // err_[0] IS updated even on reject (Julia does this — controller.err[1] = inv(EEst))
    // but err_[1] and err_[2] are only shifted on accept.
    EXPECT_NEAR(c.err_[1], err_snap[1], 1e-15);
    EXPECT_NEAR(c.err_[2], err_snap[2], 1e-15);
}

TEST(PIDControllerTest, ResetRestoresInitialState) {
    PIDController c = h211pi();
    c.update(0.1, 0.5, 5, 0);
    c.update(0.1, 0.3, 5, 1);
    c.reset();
    EXPECT_NEAR(c.err_[0], 1.0, 1e-15);
    EXPECT_NEAR(c.err_[1], 1.0, 1e-15);
    EXPECT_NEAR(c.err_[2], 1.0, 1e-15);
    EXPECT_NEAR(c.qold_, 1.0, 1e-15);
}

///////////////////////////////////////////////////////////////////////////////
// β₃ ≠ 0 path — H312PID variant. Covers the third-derivative term that
// H211PI defaults zero out. Pre-coverage relied on β₃=0 making err_[2]^0=1
// implicitly, so a sign error in the β₃ exponent or in err_[2]'s update
// would have gone unnoticed.
//
// Scenario: H312PID-like (β₁=1/18, β₂=1/9, β₃=1/18). Drive three accepts
// with controlled err_norm history, then validate that step 3's factor
// matches the closed-form ε₁^(β₁/k) · ε₂^(β₂/k) · ε₃^(β₃/k) (post-limiter).
///////////////////////////////////////////////////////////////////////////////
TEST(PIDControllerTest, Beta3NonZero_H312PID_FactorMatchesClosedForm) {
    PIDController c;
    c.beta1 = 1.0 / 18.0;
    c.beta2 = 1.0 / 9.0;
    c.beta3 = 1.0 / 18.0; // exercises the third-derivative term
    c.accept_safety = 0.81;

    int order = 5;
    double k = order + 1.0;

    // Step 1: err_norm = 0.5 → err_[0] := 2.0; history shifted (err_[2]:=1, err_[1]:=2)
    c.update(0.1, 0.5, order, 0);
    // Step 2: err_norm = 0.4 → err_[0] := 2.5; history shifted (err_[2]:=2, err_[1]:=2.5)
    c.update(0.1, 0.4, order, 1);

    // Capture the history we expect to feed step 3.
    double err1_pre = c.err_[1]; // 2.5  (from step 2's err_[0])
    double err2_pre = c.err_[2]; // 2.0  (from step 1's err_[0])

    // Step 3: err_norm = 0.6 → err_[0] := 1/0.6
    double h = 0.05;
    double en3 = 0.6;
    auto out = c.update(h, en3, order, 2);
    double err0_post = 1.0 / en3;

    double factor = std::pow(err0_post, c.beta1 / k) * std::pow(err1_pre, c.beta2 / k) *
                    std::pow(err2_pre, c.beta3 / k);
    factor = PIDController::default_limiter(factor);

    EXPECT_TRUE(out.accepted) << "factor " << factor << " should clear accept_safety";
    EXPECT_NEAR(out.dt_new, h * factor, 1e-13)
        << "Step 3 factor must include the β₃ contribution from err_[2].";
}
