///////////////////////////////////////////////////////////////////////////////
// Mesh refinement tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, MeshRefinementConvergence) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer->PrintLevel = 0;
    phase->setAdaptiveMesh(true);
    phase->setMeshTol(1e-4); // relaxed tolerance
    phase->setMaxMeshIters(5);
    phase->PrintMeshInfo = false;

    phase->solve_optimize();
    EXPECT_TRUE(phase->MeshConverged) << "Mesh should converge with relaxed tolerance";
}

TEST_F(OptimalControlTest, MeshRefinementIterates) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer->PrintLevel = 0;
    phase->setAdaptiveMesh(true);
    phase->setMeshTol(1e-7); // tight tolerance forces refinement
    phase->setMaxMeshIters(3);
    phase->PrintMeshInfo = false;

    phase->solve_optimize();
    EXPECT_GT(phase->MeshIters.size(), 0u) << "Should have at least one mesh iteration";
}
