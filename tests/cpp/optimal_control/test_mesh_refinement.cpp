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
