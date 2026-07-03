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

// Sparse vs dense KKT have different explicit-zero structure, so Pardiso
// ordering/pivoting differs; cross-solve agreement is bounded by PSIOPT's
// convergence tolerance, not machine epsilon.
constexpr double kCrossSolveTol = 1e-5;

} // namespace

// Exercises the plain (non-BlockConstant) trapezoidal wiring site.
TEST_F(OptimalControlTest, TrapezoidalSparsityMatchesDenseSolve) {
    auto solve_with = [&](bool sparsity) {
        // Fresh phase per solve: transcription only runs once per phase.
        auto phase = make_brach_phase(100, 32, TranscriptionModes::Trapezoidal);
        phase->enable_hessian_sparsity_ = sparsity;
        auto status = phase->solve_optimize();
        EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
        return phase->return_traj().back()[3]; // tf
    };

    const double tf_dense = solve_with(false);
    const double tf_sparse = solve_with(true);

    EXPECT_NEAR(tf_dense, 1.8013, 0.01);
    EXPECT_NEAR(tf_sparse, tf_dense, kCrossSolveTol);
}

// Exercises the BlockConstant + UV>0 trapezoidal wiring site (the
// Blocked_ODE_Wrapper branch whose assignment was previously commented out).
TEST_F(OptimalControlTest, TrapezoidalSparsityMatchesDenseSolveBlockConstant) {
    auto solve_with = [&](bool sparsity) {
        auto phase = make_brach_phase(100, 32, TranscriptionModes::Trapezoidal);
        phase->set_control_mode(ControlModes::BlockConstant);
        phase->enable_hessian_sparsity_ = sparsity;
        auto status = phase->solve_optimize();
        EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
        return phase->return_traj().back()[3]; // tf
    };

    const double tf_dense = solve_with(false);
    const double tf_sparse = solve_with(true);

    // BlockConstant has coarser control resolution, allow wider tolerance.
    EXPECT_NEAR(tf_dense, 1.8013, 0.05);
    EXPECT_NEAR(tf_sparse, tf_dense, kCrossSolveTol);
}
