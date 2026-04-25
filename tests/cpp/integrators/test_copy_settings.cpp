///////////////////////////////////////////////////////////////////////////////
// Coverage for Integrator::copy_settings_from — verifies every user-settable
// configuration field round-trips across a differently-parameterized
// Integrator instance. Guards against silent field drops when new settable
// members are added to Integrator<> without being threaded into
// copy_settings_from, and against regressions in the cross-DODE code path
// exercised by the shooter/reintegrator construction sites in ode_phase.h.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <variant>

namespace {

using namespace tycho;
using namespace tycho::integrators;
using TychoTest::SHO;

// A second 2-state ODE distinct from SHO at the type level, so
// copy_settings_from exercises its cross-DODE template path. dx/dt = v,
// dv/dt = -x - 0.1*v (damped harmonic oscillator). Shape matches SHO so
// the same abs/rel tolerance vectors are valid on both targets.
struct DampedSHO_Impl : ODESize<2, 0, 0> {
    static auto Definition(double /*unused*/) {
        auto args = Arguments<3>(); // [x, v, t]
        auto x = args.coeff<0>();
        auto v = args.coeff<1>();
        return StackedOutputs{v, (-1.0) * x + (-0.1) * v};
    }
};
BUILD_ODE_FROM_EXPRESSION(DampedSHO, DampedSHO_Impl, double);

} // namespace

TEST(CopySettingsFromTest, AllFieldsRoundTripAcrossDifferentDODE) {
    // Source: SHO integrator with every settable field pushed off its
    // construction defaults to a distinctive value.
    SHO sho_ode(0.0);
    Integrator<SHO> src(sho_ode, IVPAlg::DOPRI54, 0.01);

    src.adaptive_ = false;
    // abs/rel tolerance vectors are sized to x_vars (=2 for SHO).
    src.abs_tols_ = Eigen::VectorXd::Constant(2, 7.5e-9);
    src.rel_tols_ = Eigen::VectorXd::Constant(2, 3.25e-10);
    src.set_max_steps(4242);
    src.enable_vectorization_ = false;
    src.vectorize_batch_calls_ = false;
    src.error_norm_type_ = ErrorNormType::MAX;
    src.max_step_change_ = 3.75;
    src.event_tol_ = 1.234e-7;
    src.max_event_iters_ = 37;
    src.use_hairer_wanner_initdt_ = false;
    src.def_step_size_ = 0.123456;

    // Overwrite the controller variant with a tuned PIDController; the
    // default for DOPRI54 is a PIController, so PID guarantees a different
    // variant alternative is copied.
    PIDController pid;
    pid.beta1_ = 0.13;
    pid.beta2_ = 0.07;
    pid.beta3_ = 0.04;
    pid.accept_safety_ = 0.92;
    pid.qsteady_min_ = 0.88;
    pid.qsteady_max_ = 1.12;
    src.controller_variant_ = pid;

    // Source was constructed with DOPRI54; target will default to DOPRI87
    // so rk_method_ on the two sides is guaranteed to differ before copy.

    // Target: DampedSHO integrator with a completely different set of
    // defaults so the post-copy equality is a real change, not a tautology.
    DampedSHO damped_ode(0.0);
    Integrator<DampedSHO> tgt(damped_ode, IVPAlg::DOPRI87, 0.05);
    tgt.adaptive_ = true;
    tgt.set_max_steps(100);
    tgt.max_step_change_ = 10.0;
    tgt.event_tol_ = 1.0e-12;
    tgt.max_event_iters_ = 5;
    tgt.use_hairer_wanner_initdt_ = true;
    tgt.def_step_size_ = 0.05;

    // Snapshot observable fields that copy_settings_from MUST NOT touch,
    // so we can verify they're unchanged after the copy.
    const int64_t pre_naccept = tgt.get_naccept();
    const int64_t pre_nreject = tgt.get_nreject();
    const int64_t pre_failed_event = tgt.get_failed_event_count();

    tgt.copy_settings_from(src);

    // Every user-settable field must match the source after the copy.
    EXPECT_EQ(tgt.adaptive_, src.adaptive_);
    ASSERT_EQ(tgt.abs_tols_.size(), src.abs_tols_.size());
    EXPECT_TRUE(tgt.abs_tols_.isApprox(src.abs_tols_));
    ASSERT_EQ(tgt.rel_tols_.size(), src.rel_tols_.size());
    EXPECT_TRUE(tgt.rel_tols_.isApprox(src.rel_tols_));
    EXPECT_EQ(tgt.get_max_steps(), src.get_max_steps());
    EXPECT_EQ(tgt.enable_vectorization_, src.enable_vectorization_);
    EXPECT_EQ(tgt.vectorize_batch_calls_, src.vectorize_batch_calls_);
    EXPECT_EQ(tgt.get_method(), src.get_method());
    EXPECT_EQ(tgt.error_norm_type_, src.error_norm_type_);
    EXPECT_DOUBLE_EQ(tgt.max_step_change_, src.max_step_change_);
    EXPECT_DOUBLE_EQ(tgt.event_tol_, src.event_tol_);
    EXPECT_EQ(tgt.max_event_iters_, src.max_event_iters_);
    EXPECT_EQ(tgt.use_hairer_wanner_initdt_, src.use_hairer_wanner_initdt_);
    EXPECT_DOUBLE_EQ(tgt.def_step_size_, src.def_step_size_);

    // Controller variant: same alternative index and the PID parameter
    // fields round-trip. We don't compare every internal state field —
    // the transported subset is what copy_settings_from is contracted to
    // deliver.
    ASSERT_EQ(tgt.controller_variant_.index(), src.controller_variant_.index())
        << "Controller variant alternative did not copy";
    ASSERT_TRUE(std::holds_alternative<PIDController>(tgt.controller_variant_));
    const auto &tgt_pid = std::get<PIDController>(tgt.controller_variant_);
    const auto &src_pid = std::get<PIDController>(src.controller_variant_);
    EXPECT_DOUBLE_EQ(tgt_pid.beta1_, src_pid.beta1_);
    EXPECT_DOUBLE_EQ(tgt_pid.beta2_, src_pid.beta2_);
    EXPECT_DOUBLE_EQ(tgt_pid.beta3_, src_pid.beta3_);
    EXPECT_DOUBLE_EQ(tgt_pid.accept_safety_, src_pid.accept_safety_);
    EXPECT_DOUBLE_EQ(tgt_pid.qsteady_min_, src_pid.qsteady_min_);
    EXPECT_DOUBLE_EQ(tgt_pid.qsteady_max_, src_pid.qsteady_max_);

    // Per-call output counters must NOT be touched by copy_settings_from.
    EXPECT_EQ(tgt.get_naccept(), pre_naccept);
    EXPECT_EQ(tgt.get_nreject(), pre_nreject);
    EXPECT_EQ(tgt.get_failed_event_count(), pre_failed_event);
}

TEST(CopySettingsFromTest, CopyingMethodRebuildsSTMStepper) {
    DampedSHO ode(0.0);
    Integrator<DampedSHO> src(ode, IVPAlg::DOPRI54, 0.2);
    src.adaptive_ = false;
    src.use_hairer_wanner_initdt_ = false;

    Integrator<DampedSHO> tgt(ode, IVPAlg::DOPRI87, 0.05);
    tgt.copy_settings_from(src);

    Eigen::Vector3d x0;
    x0 << 1.0, -0.25, 0.0;
    constexpr double tf = 1.3;

    auto [ref_xf, ref_stm] = src.integrate_stm(x0, tf);
    auto [copied_xf, copied_stm] = tgt.integrate_stm(x0, tf);

    EXPECT_TRUE(copied_xf.isApprox(ref_xf, 1e-13));
    ASSERT_EQ(copied_stm.rows(), ref_stm.rows());
    ASSERT_EQ(copied_stm.cols(), ref_stm.cols());
    EXPECT_TRUE(copied_stm.isApprox(ref_stm, 1e-12))
        << "copy_settings_from must rebuild the method-coupled STM stepper";
}
