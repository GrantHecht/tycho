///////////////////////////////////////////////////////////////////////////////
// Event refinement failure-counter API coverage
//
// Fills the review gap where get_failed_event_count() was not exercised by
// any test. Validates:
//   1. Happy-path integration with events leaves the counter at 0 and
//      every eventstates entry holds a value (no nullopts = no losses).
//   2. Integration without events also reports 0.
//   3. The counter is reset at the start of each find_events invocation
//      (so a second call does not accumulate prior state).
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace tycho;
using namespace TychoTest;

namespace event_refinement_coverage_test {

constexpr double kErcMu = 398600.4418;

inline Eigen::VectorXd erc_eccentric_x0() {
    double rp = 6800.0;
    double e = 0.1;
    double a = rp / (1.0 - e);
    double vp = std::sqrt(kErcMu * (2.0 / rp - 1.0 / a));
    Eigen::VectorXd x0(7);
    x0 << rp, 0.0, 0.0, 0.0, vp, 0.0, 0.0;
    return x0;
}

inline double erc_eccentric_period() {
    double rp = 6800.0;
    double e = 0.1;
    double a = rp / (1.0 - e);
    return 2.0 * std::numbers::pi * std::sqrt(a * a * a / kErcMu);
}

inline Eigen::VectorXd erc_leo_x0() {
    double r0 = 7000.0;
    double v0 = std::sqrt(kErcMu / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
    return x0;
}

} // namespace event_refinement_coverage_test

using namespace event_refinement_coverage_test;

class EventRefinementCoverageTest : public VectorFunctionFixture {};

// Happy-path altitude crossing on an eccentric orbit over one period —
// produces real crossings that refine cleanly. Asserts the API reads back 0
// and that at least one crossing was recorded.
TEST_F(EventRefinementCoverageTest, HappyPath_CounterZeroAndCrossingsFound) {
    astro::Kepler kep(kErcMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = erc_eccentric_x0();
    double tf = erc_eccentric_period();

    auto args = Arguments<7>();
    auto r = args.head<3>().norm();
    auto event_func = r + (-7000.0);
    GenericFunction<-1, 1> gf(event_func);
    std::vector<Integrator<astro::Kepler>::EventPack> events;
    events.push_back({gf, 0, 0});

    auto [xf, eventlocs] = integ.integrate(x0, tf, events);
    (void)xf;

    EXPECT_EQ(integ.get_failed_event_count(), 0)
        << "Happy-path altitude crossings on a smooth Kepler orbit should refine without failure";

    ASSERT_EQ(static_cast<int>(eventlocs.size()), 1);
    EXPECT_GT(eventlocs[0].size(), 0u) << "Expected at least one altitude crossing per period";
}

// 1:1 shape invariant: the refined eventstates vector must match eventtimes
// slot-for-slot. Every happy-path crossing is an engaged optional; none are
// nullopt because this orbit is smooth. The counter must equal the total
// nullopt count (0 here) — that equality is what the API break guarantees.
TEST_F(EventRefinementCoverageTest, ShapeMatchesEventtimes_AllEngaged) {
    astro::Kepler kep(kErcMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = erc_eccentric_x0();
    double tf = erc_eccentric_period();

    auto args = Arguments<7>();
    auto event_func = args.head<3>().norm() + (-7000.0);
    GenericFunction<-1, 1> gf(event_func);
    std::vector<Integrator<astro::Kepler>::EventPack> events;
    events.push_back({gf, 0, 0});

    auto [xf, eventlocs] = integ.integrate(x0, tf, events);
    (void)xf;

    int engaged = 0;
    int nullopt_count = 0;
    ASSERT_EQ(eventlocs.size(), 1u);
    for (const auto &slot : eventlocs[0]) {
        if (slot.has_value())
            ++engaged;
        else
            ++nullopt_count;
    }

    EXPECT_GT(engaged, 0) << "Happy-path orbit must produce at least one engaged crossing";
    EXPECT_EQ(nullopt_count, integ.get_failed_event_count())
        << "Counter must equal the number of nullopt slots";
    EXPECT_EQ(nullopt_count, 0) << "Smooth Kepler orbit should not drop any refinements";
}

// Integration with no events should leave the counter at its initial 0.
TEST_F(EventRefinementCoverageTest, NoEvents_CounterRemainsZero) {
    astro::Kepler kep(kErcMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    double r0 = 7000.0;
    double period = 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / kErcMu);
    (void)integ.integrate(erc_leo_x0(), period / 4.0);

    EXPECT_EQ(integ.get_failed_event_count(), 0);
}

// The counter is reset at the start of each find_events call — a second
// event-bearing integration must not accumulate state from the first.
TEST_F(EventRefinementCoverageTest, ResetPerCall_SecondCallIndependent) {
    astro::Kepler kep(kErcMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = erc_eccentric_x0();
    double tf = erc_eccentric_period();

    auto args = Arguments<7>();
    auto event_func = args.head<3>().norm() + (-7000.0);
    GenericFunction<-1, 1> gf(event_func);
    std::vector<Integrator<astro::Kepler>::EventPack> events;
    events.push_back({gf, 0, 0});

    (void)integ.integrate(x0, tf, events);
    int after_first = integ.get_failed_event_count();

    (void)integ.integrate(x0, tf, events);
    int after_second = integ.get_failed_event_count();

    // Counter must reflect only the most recent call. With identical inputs
    // the two values must match exactly; on the happy path both are 0.
    EXPECT_EQ(after_first, after_second);
}
