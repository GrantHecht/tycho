///////////////////////////////////////////////////////////////////////////////
// FeasibilityStallDetector truth table: the windowed no-progress signal that
// lets a zero-objective feasibility stage dispatch restoration.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/feasibility_stall.h"

#include <gtest/gtest.h>

namespace ts = tycho::solvers;

TEST(FeasStallDetector, FirstObservationSetsBaselineAndNeverFires) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    EXPECT_EQ(d.best_theta_, 1.0);
    EXPECT_EQ(d.iters_without_improvement_, 0);
}

TEST(FeasStallDetector, FiresAfterWindowWithoutImprovement) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.0)) << "premature fire at " << i;
    EXPECT_TRUE(d.observe(1.0)); // the kFeasStallWindow-th stalled observation
}

TEST(FeasStallDetector, GrowingViolationCountsAsStalled) {
    ts::FeasibilityStallDetector d;
    double theta = 1.0;
    EXPECT_FALSE(d.observe(theta));
    bool fired = false;
    for (int i = 0; i < ts::kFeasStallWindow; ++i) {
        theta *= 1.05; // worsening, as in the recorded stall trace
        fired = d.observe(theta);
    }
    EXPECT_TRUE(fired);
    EXPECT_EQ(d.best_theta_, 1.0); // best-seen never moved
}

TEST(FeasStallDetector, SufficientImprovementRestartsWindow) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(1.0));
    // One >1% improvement of the best resets the counter entirely.
    EXPECT_FALSE(d.observe(0.9));
    for (int i = 1; i < ts::kFeasStallWindow; ++i)
        EXPECT_FALSE(d.observe(0.9)) << "window did not restart at " << i;
    EXPECT_TRUE(d.observe(0.9));
}

TEST(FeasStallDetector, SubThresholdImprovementDoesNotRestartWindow) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    bool fired = false;
    for (int i = 0; i < ts::kFeasStallWindow; ++i)
        fired = d.observe(0.995); // 0.5% better: below the 1% threshold
    EXPECT_TRUE(fired);
}

TEST(FeasStallDetector, ResetReArmsCompletely) {
    ts::FeasibilityStallDetector d;
    EXPECT_FALSE(d.observe(1.0));
    for (int i = 0; i < ts::kFeasStallWindow; ++i)
        d.observe(1.0);
    d.reset();
    EXPECT_EQ(d.iters_without_improvement_, 0);
    EXPECT_FALSE(d.observe(5.0)); // any value is a fresh baseline after reset
    EXPECT_EQ(d.best_theta_, 5.0);
}
