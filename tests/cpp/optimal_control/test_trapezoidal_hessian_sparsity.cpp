///////////////////////////////////////////////////////////////////////////////
// Trapezoidal Hessian-sparsity hook dispatch + phase wiring
//
// Regression for VF_REVIEW 1.1: the sparsity hook was misspelled
// (hessian_elem_is_non_zero) and never dispatched by the CRTP base, so
// slot counting stayed dense while add_hessian_elem skipped sparsely.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, TrapezoidalHessianSparsityHookDispatch) {
    BrachODE ode(9.81);
    TrapezoidalDefects<BrachODE> trap(ode);

    trap.enable_hessian_sparsity_ = false;
    const int dense = trap.num_kkt_elements(false, true);

    trap.enable_hessian_sparsity_ = true;
    const int sparse = trap.num_kkt_elements(false, true);

    // Pre-fix: the base hook (always-true) is dispatched either way -> equal.
    EXPECT_LT(sparse, dense);
    EXPECT_GT(sparse, 0);
}

namespace {

// Build a fresh trapezoidal Brachistochrone phase (independent instance per
// call, since ODEPhaseBase::transcribe() only runs the transcription once).
std::shared_ptr<ODEPhase<BrachODE>> make_trap_brach_phase(bool enable_sparsity) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_defects = 32;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::Trapezoidal, traj,
                                                       n_defects);
    phase->enable_hessian_sparsity_ = enable_sparsity;

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);

    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    return phase;
}

} // namespace

TEST_F(OptimalControlTest, TrapezoidalSparsityMatchesDenseSolve) {
    auto solve_with = [&](bool sparsity) {
        auto phase = make_trap_brach_phase(sparsity);
        auto status = phase->solve_optimize();
        EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
        return phase->return_traj().back()[3]; // tf
    };

    const double tf_dense = solve_with(false);
    const double tf_sparse = solve_with(true);

    EXPECT_NEAR(tf_dense, 1.8013, 0.01);
    EXPECT_NEAR(tf_sparse, tf_dense, 1e-6);
}
