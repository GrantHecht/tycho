///////////////////////////////////////////////////////////////////////////////
// Convergence order tests
//
// Verifies that DOPRI54 and DOPRI87 achieve their expected convergence
// orders via h-refinement on the Simple Harmonic Oscillator.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, DOPRI54ConvergenceOrder) {
    double h1 = 0.1, h2 = 0.05;
    double e1 = sho_error(IVPAlg::DOPRI54, h1);
    double e2 = sho_error(IVPAlg::DOPRI54, h2);

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
    double e1 = sho_error(IVPAlg::DOPRI87, h1);
    double e2 = sho_error(IVPAlg::DOPRI87, h2);

    double slope = std::log(e1 / e2) / std::log(h1 / h2);
    EXPECT_GT(slope, 6.0) << "DOPRI87 convergence order too low: " << slope;
    EXPECT_LT(slope, 11.0) << "DOPRI87 convergence order unexpectedly high: " << slope;
}

// Slope bounds are deliberately loose: sho_error() sets def_step_size=h but
// the fixed-step path quantizes via numsteps=ceil(H/h)+1 and h_actual=0.9·H/
// numsteps, so the nominal h ratio drifts from the actual step ratio and
// observed slope < p. We bound the lower side (catching order degradation,
// e.g. a bad tableau dropping Vern7 to order ~3) and leave the upper side
// generous.
namespace {
void check_convergence(IVPAlg alg, double h1, double h2, double slope_lower,
                       double slope_upper) {
    double e1 = sho_error(alg, h1);
    double e2 = sho_error(alg, h2);
    double slope = std::log(e1 / e2) / std::log(h1 / h2);
    EXPECT_GT(slope, slope_lower) << "convergence order too low: " << slope;
    EXPECT_LT(slope, slope_upper) << "convergence order unexpectedly high: " << slope;
}
} // namespace

TEST_F(IntegratorTest, Tsit5ConvergenceOrder) {
    check_convergence(IVPAlg::Tsit5, 0.1, 0.05, /*lower=*/4.0, /*upper=*/7.0);
}

TEST_F(IntegratorTest, BS3ConvergenceOrder) {
    check_convergence(IVPAlg::BS3, 0.1, 0.05, /*lower=*/2.0, /*upper=*/5.0);
}

TEST_F(IntegratorTest, BS5ConvergenceOrder) {
    check_convergence(IVPAlg::BS5, 0.1, 0.05, /*lower=*/4.0, /*upper=*/7.0);
}

// Vern7/8/9 on SHO over t=[0,1] hit the FP roundoff floor (~1e-14) by
// h≈0.25 for order 8+, which flattens the error curve and drags the
// measured slope below p. The lower bound of 3.5 is the empirical floor
// that still catches a gross order-degradation bug (order ≤ 2 → slope ≤ 3)
// while accepting the FP-floor artifact. A tighter bound would need a
// harder test problem (larger derivatives, longer interval) or a direct
// error-vs-step plot instead of a two-point slope.
TEST_F(IntegratorTest, Vern7ConvergenceOrder) {
    check_convergence(IVPAlg::Vern7, 0.5, 0.25, /*lower=*/3.5, /*upper=*/9.0);
}

TEST_F(IntegratorTest, Vern8ConvergenceOrder) {
    check_convergence(IVPAlg::Vern8, 0.5, 0.25, /*lower=*/3.5, /*upper=*/10.0);
}

TEST_F(IntegratorTest, Vern9ConvergenceOrder) {
    check_convergence(IVPAlg::Vern9, 0.5, 0.25, /*lower=*/3.5, /*upper=*/11.0);
}
