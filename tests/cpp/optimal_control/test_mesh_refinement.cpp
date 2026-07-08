///////////////////////////////////////////////////////////////////////////////
// Mesh refinement tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, MeshRefinementConvergence) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer_->set_print_level(0);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-4); // relaxed tolerance
    phase->set_max_mesh_iters(5);
    phase->print_mesh_info_ = false;

    phase->solve_optimize();
    EXPECT_TRUE(phase->mesh_converged_) << "Mesh should converge with relaxed tolerance";
}

TEST_F(OptimalControlTest, MeshRefinementIterates) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer_->set_print_level(0);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-7); // tight tolerance forces refinement
    phase->set_max_mesh_iters(3);
    phase->print_mesh_info_ = false;

    phase->solve_optimize();
    EXPECT_GT(phase->mesh_iters_.size(), 0u) << "Should have at least one mesh iteration";
}

///////////////////////////////////////////////////////////////////////////////
// OC review §1.8 — construct-and-discard throw guard
//
// The `else` branch in the `get_space` lambdas of return_costate_traj() /
// return_traj_error() is unreachable through the shipping transcription enum
// (num_tran_card_states_ is always 2, 3, or 4), so there is no way to fire the
// new `throw` in a black-box test. This is a defensiveness smoke test
// confirming the supported modes still succeed without throwing, documenting
// that the throw only guards a future unsupported transcription mode.
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, CostateErrorEstimationSupportedModesDoNotThrow) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer_->set_print_level(0);

    phase->solve_optimize();
    EXPECT_NO_THROW((void)phase->return_costate_traj());
    EXPECT_NO_THROW((void)phase->return_traj_error());
}
