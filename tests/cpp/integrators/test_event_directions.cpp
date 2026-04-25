///////////////////////////////////////////////////////////////////////////////
// Event direction filter and edge-case timing coverage.
//
// Pre-PR coverage exercised `check_crossings`'s direction filter only with
// direction=0 (Any). The +1 (Rising) / -1 (Falling) branches at
// event_handler.h:64 were never tripped. These tests pin:
//   1. Rising-only detects a rising crossing and ignores a falling one.
//   2. Falling-only detects a falling crossing and ignores a rising one.
//   3. Two events fired in the same step both record.
//   4. Multiple crossings across many steps behave as expected.
//
// ODE: SHO x'' = -x, state = [x, v, t]. Over t ∈ [0, 2π] with x(0)=1, v(0)=0:
//    x(t) = cos(t)    — rising crossings of x=0 at t=3π/2 only (sin flips sign)
//    v(t) = -sin(t)   — falling crossings of v=0 at t=π,  rising at t=2π
// Rising "x" means x increasing through zero; falling means x decreasing.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <optional>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {

// Integrator over one full SHO period at tight tol.
std::tuple<Eigen::Vector3d, std::vector<std::vector<std::optional<Eigen::Vector3d>>>>
sho_integrate_with_events(const std::vector<EventPack> &events) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 2.0 * std::numbers::pi;
    return integ.integrate(x0, tf, events);
}

GenericFunction<-1, 1> make_component_zero_event(int idx) {
    auto args = Arguments<3>(); // [x, v, t]
    if (idx == 0)
        return GenericFunction<-1, 1>(args.coeff<0>());
    if (idx == 1)
        return GenericFunction<-1, 1>(args.coeff<1>());
    throw std::logic_error("unsupported idx");
}

} // namespace

class EventDirectionTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Rising-only filter on x (component 0): over [0, 2π], x=cos(t) has one
// rising crossing at t=3π/2. Assert exactly one crossing recorded.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventDirectionTest, RisingOnlyDetectsRisingCrossingOnly) {
    auto gf_x = make_component_zero_event(0);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Rising, 0});

    auto [xf, eventlocs] = sho_integrate_with_events(events);
    (void)xf;

    ASSERT_EQ(eventlocs.size(), 1u);
    EXPECT_EQ(eventlocs[0].size(), 1u) << "Expected exactly one rising crossing of x=cos(t) over 2π.";
    if (eventlocs[0].size() == 1) {
        ASSERT_TRUE(eventlocs[0][0].has_value());
        double t_event = (*eventlocs[0][0])[2];
        EXPECT_NEAR(t_event, 1.5 * std::numbers::pi, 1e-6) << "Rising crossing should be at 3π/2.";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Falling-only filter on x (component 0): one falling crossing at t=π/2.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventDirectionTest, FallingOnlyDetectsFallingCrossingOnly) {
    auto gf_x = make_component_zero_event(0);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Falling, 0});

    auto [xf, eventlocs] = sho_integrate_with_events(events);
    (void)xf;

    ASSERT_EQ(eventlocs.size(), 1u);
    EXPECT_EQ(eventlocs[0].size(), 1u) << "Expected exactly one falling crossing of x=cos(t) over 2π.";
    if (eventlocs[0].size() == 1) {
        ASSERT_TRUE(eventlocs[0][0].has_value());
        double t_event = (*eventlocs[0][0])[2];
        EXPECT_NEAR(t_event, 0.5 * std::numbers::pi, 1e-6) << "Falling crossing should be at π/2.";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Any-direction filter on x: two crossings over one period. Sanity check
// that `Any` is the default and detects both.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventDirectionTest, AnyDirectionDetectsBothCrossings) {
    auto gf_x = make_component_zero_event(0);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Any, 0});

    auto [xf, eventlocs] = sho_integrate_with_events(events);
    (void)xf;

    ASSERT_EQ(eventlocs.size(), 1u);
    EXPECT_EQ(eventlocs[0].size(), 2u) << "Expected two crossings of x=cos(t) over one period.";
}

///////////////////////////////////////////////////////////////////////////////
// Two simultaneous event functions: x=0 crosses twice over [0, 2π]; v=0
// starts at t=0 (not counted as crossing since vprev=0) and crosses once at
// t=π. Both groups must populate independently.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventDirectionTest, TwoSimultaneousEventsBothRecord) {
    auto gf_x = make_component_zero_event(0);
    auto gf_v = make_component_zero_event(1);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Any, 0});
    events.push_back({gf_v, event_direction::Any, 0});

    auto [xf, eventlocs] = sho_integrate_with_events(events);
    (void)xf;

    ASSERT_EQ(eventlocs.size(), 2u);
    EXPECT_EQ(eventlocs[0].size(), 2u) << "x=cos(t) has two crossings over [0, 2π].";
    EXPECT_EQ(eventlocs[1].size(), 1u) << "v=-sin(t) has one mid-period crossing at t=π.";
}

///////////////////////////////////////////////////////////////////////////////
// Stop-on-nth-crossing: Rising crossing of x, stop_count=1. Integration
// must halt early at t≈3π/2, well before tf=2π.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventDirectionTest, StopCountHaltsIntegrationEarly) {
    auto gf_x = make_component_zero_event(0);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Rising, 1});

    auto [xf, eventlocs] = sho_integrate_with_events(events);

    double t_final = xf[2];
    EXPECT_LT(t_final, 2.0 * std::numbers::pi - 0.1)
        << "Integration should halt at the rising crossing, well before tf=2π.";
    EXPECT_NEAR(t_final, 1.5 * std::numbers::pi, 0.1)
        << "Integration should halt just past the rising crossing at 3π/2.";
    ASSERT_EQ(eventlocs.size(), 1u);
    EXPECT_EQ(eventlocs[0].size(), 1u);
}

// Out-of-domain direction silently coerces inside the sign-product check;
// reject at construction time.
TEST_F(EventDirectionTest, InvalidDirectionRejected) {
    auto gf_x = make_component_zero_event(0);
    EXPECT_THROW((EventPack{gf_x, /*direction=*/2, /*stop_count=*/0}), std::invalid_argument);
    EXPECT_THROW((EventPack{gf_x, /*direction=*/-2, /*stop_count=*/0}), std::invalid_argument);
    EXPECT_THROW((EventPack{gf_x, /*direction=*/42, /*stop_count=*/0}), std::invalid_argument);
}

TEST_F(EventDirectionTest, NegativeStopCountRejected) {
    auto gf_x = make_component_zero_event(0);
    EXPECT_THROW((EventPack{gf_x, /*direction=*/0, /*stop_count=*/-1}), std::invalid_argument);
    EXPECT_THROW((EventPack{gf_x, /*direction=*/0, /*stop_count=*/-42}), std::invalid_argument);
}

///////////////////////////////////////////////////////////////////////////////
// Post-construction mutation can take EventPack out of its invariant
// envelope (public fields, direct assignment). The integrate-entry
// validate_events call must catch that, with the failing index in the error
// message.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventDirectionTest, PostConstructionMutationCaughtAtIntegrate) {
    auto gf_x = make_component_zero_event(0);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Any, 0});
    events.push_back({gf_x, event_direction::Any, 0});
    events[1].direction = 2; // out of {-1, 0, 1}; would silently coerce in check_crossings

    EXPECT_THROW(sho_integrate_with_events(events), std::invalid_argument);

    try {
        sho_integrate_with_events(events);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_NE(std::string(e.what()).find("events[1]"), std::string::npos)
            << "error message must identify the offending event index; got: " << e.what();
    }
}

TEST_F(EventDirectionTest, PostConstructionNegativeStopCountCaughtAtIntegrate) {
    auto gf_x = make_component_zero_event(0);
    std::vector<EventPack> events;
    events.push_back({gf_x, event_direction::Any, 0});
    events[0].stop_count = -1;

    EXPECT_THROW(sho_integrate_with_events(events), std::invalid_argument);
}
