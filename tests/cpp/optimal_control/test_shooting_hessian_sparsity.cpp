///////////////////////////////////////////////////////////////////////////////
// CentralShooting Hessian-sparsity hook dispatch + phase wiring
//
// Regression for OC_REVIEW 1.13: CentralShootingDefect declared and assigned
// enable_hessian_sparsity_ but never built a static nonzero mask nor overrode
// the CRTP hooks (hessian_elem_is_nonzero / add_hessian_elem), so shooting-mode
// KKT Hessian blocks were counted and filled densely regardless of the flag --
// diverging from the trapezoidal-defect contract. These tests mirror
// test_trapezoidal_hessian_sparsity.cpp: the static mask must reduce the KKT
// slot count and the sparse solve must agree with the dense solve.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, ShootingHessianSparsityHookDispatch) {
    BrachODE ode(9.81);
    Integrator<BrachODE> integ(ode, 0.01);
    CentralShootingDefect defect(ode, integ);

    defect.enable_hessian_sparsity_ = false;
    const int dense = defect.num_kkt_elements(false, true);

    defect.enable_hessian_sparsity_ = true;
    const int sparse = defect.num_kkt_elements(false, true);

    // The static mask leaves only the endpoint-to-endpoint cross block zero, so
    // sparse must be strictly fewer than dense but still positive.
    EXPECT_LT(sparse, dense);
    EXPECT_GT(sparse, 0);
}

namespace {

// Sparse vs dense KKT have different explicit-zero structure, so Pardiso
// ordering/pivoting differs; cross-solve agreement is bounded by InteriorPointSolver's
// convergence tolerance, not machine epsilon.
constexpr double kCrossSolveTol = 1e-5;

} // namespace

TEST_F(OptimalControlTest, ShootingSparsityMatchesDenseSolve) {
    auto solve_with = [&](bool sparsity) {
        // Fresh phase per solve: transcription only runs once per phase.
        auto phase = make_brach_phase(100, 32, TranscriptionModes::CentralShooting);
        phase->enable_hessian_sparsity_ = sparsity;
        tycho::solvers::InteriorPointSolver ipm;
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        EXPECT_LE(status, tycho::ConvergenceFlags::ACCEPTABLE);
        return phase->return_traj().back()[3]; // tf
    };

    const double tf_dense = solve_with(false);
    const double tf_sparse = solve_with(true);

    EXPECT_NEAR(tf_dense, 1.8013, 0.05);
    EXPECT_NEAR(tf_sparse, tf_dense, kCrossSolveTol);
}
