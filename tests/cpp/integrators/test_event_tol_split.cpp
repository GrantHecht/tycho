///////////////////////////////////////////////////////////////////////////////
// Split event tolerances (INTEGRATORS_REVIEW §3.2). A large-scale event
// function (huge slope) must refine successfully when the *residual* tolerance
// is scaled to the function's magnitude, while a tight *abscissa* tolerance
// still locates the crossing time accurately. Pre-split (single event_tol) this
// is impossible: loosening the one tol to admit the residual also loosens the
// abscissa and mislocates. This test uses the new split API, so it does not
// compile against pre-split code (that is the "red").
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {
// Event = 1e8 * x[0]. For SHO x = cos(t), this crosses zero at t = π/2, 3π/2
// with an enormous slope (|dg/dt| ~ 1e8).
GenericFunction<-1, 1> large_scale_component0_event() {
    auto args = Arguments<3>();
    return GenericFunction<-1, 1>(args.coeff<0>() * 1.0e8);
}
} // namespace

class EventTolSplitTest : public VectorFunctionFixture {};

TEST_F(EventTolSplitTest, LargeScaleEventRefinesWithLooseResidualTol) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);
    // Residual tol scaled to the event magnitude (|f| <= 1e2 ~ |x| <= 1e-6);
    // abscissa tol tight so the located time is still accurate.
    integ.set_event_residual_tol(1.0e2);
    integ.set_event_abscissa_tol(1.0e-9);

    std::vector<EventPack> events;
    events.push_back({large_scale_component0_event(), event_direction::Any, 0});

    auto [xf, eventlocs] =
        integ.integrate(Eigen::Vector3d(1.0, 0.0, 0.0), 2.0 * std::numbers::pi, events);
    (void)xf;

    ASSERT_EQ(eventlocs.size(), 1u);
    ASSERT_EQ(eventlocs[0].size(), 2u) << "Two crossings of cos(t) over one period.";
    ASSERT_TRUE(eventlocs[0][0].has_value()) << "Refinement must succeed, not fail.";
    EXPECT_EQ(integ.get_failed_event_count(), 0);
    // Located time still accurate despite the loose residual tol.
    EXPECT_NEAR((*eventlocs[0][0])[2], 0.5 * std::numbers::pi, 1e-6);
}

TEST_F(EventTolSplitTest, SettersRejectNonPositive) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    EXPECT_THROW(integ.set_event_residual_tol(0.0), std::invalid_argument);
    EXPECT_THROW(integ.set_event_residual_tol(-1.0), std::invalid_argument);
    EXPECT_THROW(integ.set_event_abscissa_tol(0.0), std::invalid_argument);
    EXPECT_THROW(integ.set_event_abscissa_tol(std::nan("")), std::invalid_argument);
    integ.set_event_residual_tol(1e-3);
    integ.set_event_abscissa_tol(1e-9);
    EXPECT_DOUBLE_EQ(integ.get_event_residual_tol(), 1e-3);
    EXPECT_DOUBLE_EQ(integ.get_event_abscissa_tol(), 1e-9);
}

// Exercises the abscissa-tol path with a normal-scale event and an explicitly
// tight abscissa tolerance: a normal-scale crossing (g = x, slope ~1) is
// located accurately. NOTE: this does not *isolate* the abscissa wiring as a
// red-green swap test — the refinement runs a fixed 2-iteration first bisect
// then Newton, so the abscissa tolerance only governs the conditional deep
// bisect that a smooth event does not deterministically trigger; the abscissa
// path is additionally covered by every event-location test (they assert
// crossing times). The residual half IS isolated by
// LargeScaleEventRefinesWithLooseResidualTol above.
TEST_F(EventTolSplitTest, TightAbscissaTolLocatesNormalScaleCrossing) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);
    integ.set_event_residual_tol(1e-8);
    integ.set_event_abscissa_tol(1e-10);

    std::vector<EventPack> events;
    auto args = Arguments<3>();
    events.push_back({GenericFunction<-1, 1>(args.coeff<0>()), event_direction::Any, 0});

    auto [xf, eventlocs] =
        integ.integrate(Eigen::Vector3d(1.0, 0.0, 0.0), 2.0 * std::numbers::pi, events);
    (void)xf;
    ASSERT_EQ(eventlocs.size(), 1u);
    ASSERT_EQ(eventlocs[0].size(), 2u);
    ASSERT_TRUE(eventlocs[0][0].has_value());
    EXPECT_EQ(integ.get_failed_event_count(), 0);
    EXPECT_NEAR((*eventlocs[0][0])[2], 0.5 * std::numbers::pi, 1e-7);
}
