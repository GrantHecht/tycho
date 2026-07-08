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

///////////////////////////////////////////////////////////////////////////////
// OC review §1.4 — calc_switches() normalized-tolerance test degenerates for
// non-O(1) controls
//
// `unddiff` (meant to test the normalized `und` matrix against
// rel_switch_tol_) was computed from the same raw expression as `udiff`, so
// `und` was dead. For a control whose magnitude isn't O(1), that makes the
// "relative" check degenerate into another absolute-scale check: every raw
// step whose magnitude exceeds rel_switch_tol_ (0.3) gets flagged as a
// switch, not just genuine jumps.
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, ControlSwitchDetectionUsesNormalizedDiff) {
    auto phase = make_bangbang_phase(/*u_scale=*/1000.0);

    Eigen::VectorXd sw = phase->calc_switches();

    ASSERT_EQ(sw.size(), 1) << "Only the true control jump should be flagged as a switch; a raw "
                               "(non-normalized) diff test spuriously flags every "
                               "large-magnitude step as a switch.";
    EXPECT_NEAR(sw[0], kBangBangSwitchTime, 1e-2);
}
