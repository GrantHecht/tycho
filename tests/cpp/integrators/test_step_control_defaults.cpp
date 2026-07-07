///////////////////////////////////////////////////////////////////////////////
// Step-control default tests
//
// Two OrdinaryDiffEq-parity behaviors that had no direct coverage:
//
//   * Per-method controller gain defaults (method_order_for /
//     default_lund_betas_for / controller_defaults_for). These are the numeric
//     Lund-gain formulas the drivers rely on; a transposed beta1/beta2 or a
//     wrong method order would silently degrade step-size control. Pinned here
//     against the exact OrdinaryDiffEq beta1_default/beta2_default values.
//
//   * First-accepted-step exemption from the max_step_change clamp. OrdinaryDiffEq
//     lets the first step grow far past the steady-state cap (qmax_first_step) so a
//     conservative initial dt recovers quickly; Tycho's max_step_change extension
//     must NOT clamp that first step. Pinned via a constant-rate ODE (zero embedded
//     error → every proposed step is accepted, so the growth is directly visible in
//     the dense time grid).
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <variant>

using namespace tycho;
using namespace tycho::integrators;

// Test probe: the controller-default helpers are `protected` static members of
// Integrator. A thin subclass re-exposes them so the numeric defaults can be
// pinned directly, without having to round-trip through a full integrate().
template <class ODE> struct ControllerDefaultsProbe : Integrator<ODE> {
    using Integrator<ODE>::controller_defaults_for;
    using Integrator<ODE>::default_lund_betas_for;
    using Integrator<ODE>::method_order_for;
};

using Probe = ControllerDefaultsProbe<TychoTest::SHO>;

///////////////////////////////////////////////////////////////////////////////
// method_order_for — the algorithm order that drives every derived Lund gain.
///////////////////////////////////////////////////////////////////////////////
TEST(ControllerDefaults, MethodOrders) {
    EXPECT_EQ(Probe::method_order_for(IVPAlg::DOPRI54), 5);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::DOPRI87), 8);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::Tsit5), 5);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::BS3), 3);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::BS5), 5);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::Vern7), 7);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::Vern8), 8);
    EXPECT_EQ(Probe::method_order_for(IVPAlg::Vern9), 9);
}

///////////////////////////////////////////////////////////////////////////////
// default_lund_betas_for — generic β2 = 2/(5·order), β1 = 1/order − 3β2/4
// (= 7/(10·order)); DOPRI54 overrides to (17/100, 4/100).
///////////////////////////////////////////////////////////////////////////////
TEST(ControllerDefaults, LundBetasGenericAndDopri54Override) {
    double b1 = 0.0, b2 = 0.0;

    // DOPRI54 special-case (matches OrdinaryDiffEq's DP5 beta1/beta2_default).
    Probe::default_lund_betas_for(IVPAlg::DOPRI54, b1, b2);
    EXPECT_DOUBLE_EQ(b1, 17.0 / 100.0);
    EXPECT_DOUBLE_EQ(b2, 4.0 / 100.0);

    // DOPRI87 generic, order 8.
    Probe::default_lund_betas_for(IVPAlg::DOPRI87, b1, b2);
    EXPECT_DOUBLE_EQ(b2, 2.0 / (5.0 * 8.0));
    EXPECT_DOUBLE_EQ(b1, 1.0 / 8.0 - 3.0 * b2 / 4.0);
    EXPECT_DOUBLE_EQ(b1, 7.0 / 80.0); // closed form of the generic law

    // Vern7 generic, order 7.
    Probe::default_lund_betas_for(IVPAlg::Vern7, b1, b2);
    EXPECT_DOUBLE_EQ(b2, 2.0 / 35.0);
    EXPECT_DOUBLE_EQ(b1, 1.0 / 10.0);
}

///////////////////////////////////////////////////////////////////////////////
// controller_defaults_for — off-preference and PID selection must use the
// method-order Lund gains, NOT the struct-default order-5 / (1,0,0) values.
///////////////////////////////////////////////////////////////////////////////
TEST(ControllerDefaults, OffPreferencePIUsesMethodOrderGains) {
    // DOPRI87's preferred controller is I; requesting PI must still yield the
    // order-8 Lund gains, not the struct-default order-5 betas.
    auto cv = Probe::controller_defaults_for(IVPAlg::DOPRI87, IVPController::PI);
    ASSERT_TRUE(std::holds_alternative<PIController>(cv));
    const auto &c = std::get<PIController>(cv);
    EXPECT_DOUBLE_EQ(c.beta1_, 7.0 / 80.0);
    EXPECT_DOUBLE_EQ(c.beta2_, 1.0 / 20.0);
}

TEST(ControllerDefaults, Dopri54PIKeepsSpecialGains) {
    auto cv = Probe::controller_defaults_for(IVPAlg::DOPRI54, IVPController::PI);
    ASSERT_TRUE(std::holds_alternative<PIController>(cv));
    const auto &c = std::get<PIController>(cv);
    EXPECT_DOUBLE_EQ(c.beta1_, 17.0 / 100.0);
    EXPECT_DOUBLE_EQ(c.beta2_, 4.0 / 100.0);
}

TEST(ControllerDefaults, PIDUsesOrderScaledLundWithBeta3Zero) {
    auto cv = Probe::controller_defaults_for(IVPAlg::DOPRI87, IVPController::PID);
    ASSERT_TRUE(std::holds_alternative<PIDController>(cv));
    const auto &c = std::get<PIDController>(cv);
    // Order-8 Lund pair with β3 = 0 — NOT the pure-integral (1, 0, 0) struct
    // default that a regression removing the PID branch would restore.
    EXPECT_DOUBLE_EQ(c.beta1_, 7.0 / 80.0);
    EXPECT_DOUBLE_EQ(c.beta2_, 1.0 / 20.0);
    EXPECT_DOUBLE_EQ(c.beta3_, 0.0);
}

TEST(ControllerDefaults, PreferredControllerReturnedForMatchingKind) {
    // DOPRI87's default kind is I — requesting I returns the tuned I controller.
    auto cv = Probe::controller_defaults_for(IVPAlg::DOPRI87, IVPController::I);
    EXPECT_TRUE(std::holds_alternative<IController>(cv));
    // DOPRI54's default kind is PI — requesting PI returns the tuned PI controller.
    auto cv2 = Probe::controller_defaults_for(IVPAlg::DOPRI54, IVPController::PI);
    EXPECT_TRUE(std::holds_alternative<PIController>(cv2));
}

///////////////////////////////////////////////////////////////////////////////
// First-step exemption from max_step_change (OrdinaryDiffEq qmax_first_step
// parity).
//
// A constant-rate ODE dx/dt = 1 has identically-zero embedded error, so the
// controller proposes its maximum growth (qmax_first_step) after step 1 and that
// step is ACCEPTED unconditionally — making the first-step growth directly
// observable in the dense time grid. With the exemption, h2/h1 tracks the
// controller's large factor; before the fix it was pinned to exactly
// max_step_change = 10.
///////////////////////////////////////////////////////////////////////////////
namespace {
struct ConstRate : oc::StaticODE<ConstRate, 1, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                                 vf::DenseDerivativeMode::FDiffFwd> {
    ConstRate() { this->set_ode_size(1, 0, 0); }
    template <class InType, class OutType>
    inline void compute_impl(vf::CVecRef<InType> x_, vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        auto &fx = fx_.const_cast_derived();
        fx[0] = Scalar(1.0); // dx/dt = 1, no dependence on x or t
    }
};
} // namespace

TEST(FirstStepGrowth, FirstAcceptedStepExemptFromMaxStepChange) {
    ConstRate ode;
    Integrator<ConstRate> integ(ode, IVPAlg::DOPRI54, /*def_step=*/0.01);
    // HW off so h1 is deterministically derived from def_step; adaptive stays on.
    integ.set_initial_step_size(0.01);
    ASSERT_FALSE(integ.get_auto_initial_dt());
    integ.set_abs_tol(1e-3);
    integ.set_rel_tol(1e-3);

    Eigen::Vector2d x0;
    x0 << 0.0, 0.0; // x = 0, t = 0
    auto traj = integ.integrate_dense(x0, 1000.0);

    ASSERT_GE(traj.size(), 3u) << "need at least two accepted steps to measure growth";
    constexpr int kTimeIdx = 1; // 1-state ODE → time occupies slot 1
    const double h1 = traj[1][kTimeIdx] - traj[0][kTimeIdx];
    const double h2 = traj[2][kTimeIdx] - traj[1][kTimeIdx];
    ASSERT_GT(h1, 0.0);
    EXPECT_GT(h2 / h1, 10.5)
        << "the first accepted step must be exempt from the max_step_change = 10 clamp "
           "(OrdinaryDiffEq qmax_first_step parity); before the fix this ratio was pinned "
           "at 10. h1="
        << h1 << " h2=" << h2;
}
