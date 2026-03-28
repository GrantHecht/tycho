///////////////////////////////////////////////////////////////////////////////
// Phase construction tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, BrachPhaseConstruct) {
    auto phase = make_brach_phase();
    // Phase should be constructable without error
    SUCCEED();
}

TEST_F(OptimalControlTest, BrachPhaseTranscribe) {
    auto phase = make_brach_phase();
    phase->transcribe(false, false);
    // After transcription, we can query the trajectory
    auto traj = phase->return_traj();
    EXPECT_GT(traj.size(), 0u);
}

TEST_F(OptimalControlTest, BrachPhaseBoundaryConstraints) {
    auto phase = make_brach_phase();
    phase->transcribe(false, false);
    auto traj = phase->return_traj();

    // Initial trajectory should have correct front boundary (from the guess)
    EXPECT_EQ(traj.front().size(), 5);
}
