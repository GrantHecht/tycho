///////////////////////////////////////////////////////////////////////////////
// Endpoint zero-crossing detection (INTEGRATORS_REVIEW §1.2).
//
// A crossing whose event value is exactly 0.0 at an accepted-step endpoint
// must be recorded exactly once — not dropped (pre-fix: vprod == 0 on both
// adjacent steps, so `vprod < 0` never fires), and not double-counted on the
// following step (where vprev == 0).
//
// These unit-test EventHandler::check_crossings directly with crafted event
// values, so the boundary is hit deterministically without relying on adaptive
// step placement.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <vector>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {
// Event VF = state component 0 (state = [x, v, t], t_var = 2).
GenericFunction<-1, 1> component0_event() {
    auto args = Arguments<3>();
    return GenericFunction<-1, 1>(args.coeff<0>());
}
} // namespace

class EventBoundaryTest : public VectorFunctionFixture {};

// vprev = -1 (below), vnext = 0 exactly at the endpoint: one Rising crossing.
TEST_F(EventBoundaryTest, ExactEndpointZeroRecordedOnce) {
    std::vector<EventPack> events;
    events.push_back({component0_event(), event_direction::Any, 0});

    std::vector<Vector1<double>> prev(1), next(1);
    prev[0][0] = -1.0; // below zero on the previous accepted state
    next[0].setZero();

    std::vector<std::vector<Eigen::Vector2d>> eventtimes(1);

    Eigen::Vector3d xnext;
    xnext << 0.0, 0.5, 1.0; // component 0 == 0 -> vnext == 0 exactly
    bool brk =
        EventHandler::check_crossings(events, prev, next, xnext, /*t_var=*/2, eventtimes, 0.5);
    (void)brk;
    EXPECT_EQ(eventtimes[0].size(), 1u) << "Exact-endpoint zero crossing must be recorded once.";
}

// Following step: vprev == 0 (the endpoint), vnext = +1 (above). Must NOT
// record again (no double count).
TEST_F(EventBoundaryTest, EndpointZeroNotDoubleCountedNextStep) {
    std::vector<EventPack> events;
    events.push_back({component0_event(), event_direction::Any, 0});

    std::vector<Vector1<double>> prev(1), next(1);
    prev[0][0] = 0.0; // sitting exactly on zero from the prior endpoint
    next[0].setZero();

    std::vector<std::vector<Eigen::Vector2d>> eventtimes(1);
    Eigen::Vector3d xnext;
    xnext << 1.0, 0.5, 2.0; // vnext == +1
    EventHandler::check_crossings(events, prev, next, xnext, 2, eventtimes, 1.0);
    EXPECT_EQ(eventtimes[0].size(), 0u)
        << "A crossing already recorded at the endpoint must not double-count.";
}

// Rising-only filter must accept an endpoint zero approached from below.
TEST_F(EventBoundaryTest, RisingFilterAcceptsEndpointApproachedFromBelow) {
    std::vector<EventPack> events;
    events.push_back({component0_event(), event_direction::Rising, 0});
    std::vector<Vector1<double>> prev(1), next(1);
    prev[0][0] = -1.0;
    next[0].setZero();
    std::vector<std::vector<Eigen::Vector2d>> eventtimes(1);
    Eigen::Vector3d xnext;
    xnext << 0.0, 0.5, 1.0;
    EventHandler::check_crossings(events, prev, next, xnext, 2, eventtimes, 0.5);
    EXPECT_EQ(eventtimes[0].size(), 1u) << "Endpoint zero from below is a rising crossing.";
}

// Falling-only filter must reject that same below->zero endpoint.
TEST_F(EventBoundaryTest, FallingFilterRejectsEndpointApproachedFromBelow) {
    std::vector<EventPack> events;
    events.push_back({component0_event(), event_direction::Falling, 0});
    std::vector<Vector1<double>> prev(1), next(1);
    prev[0][0] = -1.0;
    next[0].setZero();
    std::vector<std::vector<Eigen::Vector2d>> eventtimes(1);
    Eigen::Vector3d xnext;
    xnext << 0.0, 0.5, 1.0;
    EventHandler::check_crossings(events, prev, next, xnext, 2, eventtimes, 0.5);
    EXPECT_EQ(eventtimes[0].size(), 0u) << "Endpoint zero from below is NOT a falling crossing.";
}
