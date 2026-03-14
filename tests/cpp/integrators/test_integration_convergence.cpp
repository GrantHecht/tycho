///////////////////////////////////////////////////////////////////////////////
// Convergence order tests
//
// Verifies that DOPRI54 and DOPRI87 achieve their expected convergence
// orders via h-refinement on the Simple Harmonic Oscillator.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, DOPRI54ConvergenceOrder) {
    double h1 = 0.1, h2 = 0.05;
    double e1 = sho_error("DOPRI54", h1);
    double e2 = sho_error("DOPRI54", h2);

    // Expected order ~5 (uses DOPRI5 stepper): slope = log(e1/e2) / log(h1/h2)
    double slope = std::log(e1 / e2) / std::log(h1 / h2);
    EXPECT_GT(slope, 4.0) << "DOPRI54 convergence order too low: " << slope;
    EXPECT_LT(slope, 7.0) << "DOPRI54 convergence order unexpectedly high: " << slope;
}

TEST_F(IntegratorTest, DOPRI87ConvergenceOrder) {
    // Use larger steps to stay in the truncation-error regime (not roundoff).
    // The integrator internally recomputes step count and actual step size,
    // so the nominal h ratio is approximate; accept order >= 6.
    double h1 = 0.5, h2 = 0.25;
    double e1 = sho_error("DOPRI87", h1);
    double e2 = sho_error("DOPRI87", h2);

    double slope = std::log(e1 / e2) / std::log(h1 / h2);
    EXPECT_GT(slope, 6.0) << "DOPRI87 convergence order too low: " << slope;
    EXPECT_LT(slope, 11.0) << "DOPRI87 convergence order unexpectedly high: " << slope;
}
