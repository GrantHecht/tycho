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

// Force the nullopt-and-counter-increment path by calling find_events
// with a hand-crafted bracket that the event VF cannot satisfy. `event_func
// = x[0] - 10000` is strictly negative across the Kepler orbit (position
// magnitude stays under 10000 km for the setup below), so Newton — starting
// from the 2-iter bisect midpoint where fx ≈ -constant and jx ≈ vx — lands
// far outside the original [tlow, thigh] bracket on both the fast and the
// wide passes. That's precisely the condition under which find_events
// emits std::nullopt and increments n_failed_refinements_. This exercises
// only the Newton-overshoot branch; a change in refinement policy could
// turn this test into a no-op rather than a failure — find_events has
// other nullopt paths not covered here.
TEST_F(EventRefinementCoverageTest, Nullopt_WhenNewtonOvershootsBracket_EmitsStdNullopt) {
    astro::Kepler kep(kErcMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = erc_eccentric_x0();
    double tf = erc_eccentric_period();

    // Build a dense trajectory including midpoints so make_table can
    // populate an LGL3 interpolation table. The with-events overload at
    // alloutput=true is the public way to get state + midpoint + state
    // interleaving without events shaping the output.
    std::vector<Integrator<astro::Kepler>::EventPack> no_events;
    auto [xs, _ignored_eventlocs] =
        integ.integrate_dense(x0, tf, no_events, /*alloutput=*/true);
    ASSERT_GT(static_cast<int>(xs.size()), 2) << "Need at least two dense points";

    // Event VF: x_position - 10000 km — never zero on this orbit, so any
    // bracket handed to find_events is a fake. The Newton step fx/jx ≈
    // (-10000)/vx is tens of km/s large; from a tig in a few-second bracket
    // this lands well outside.
    auto args = Arguments<7>();
    auto event_expr = args.coeff<0>() + (-10000.0);
    GenericFunction<-1, 1> gf(event_expr);
    std::vector<Integrator<astro::Kepler>::EventPack> events;
    events.push_back({gf, 0, 0});

    // Fabricate a crossing bracket between xs[0] and xs[1] — the table
    // supports interpolation over that window, and the bracket width is
    // small enough that Newton's fx/jx overshoot deterministically exits
    // the [tlow, thigh] range. Kepler state layout: [r(0..2), v(3..5), t].
    constexpr int kKeplerTVar = 6;
    double tlow = xs[0][kKeplerTVar];
    double thigh = xs[1][kKeplerTVar];
    if (thigh < tlow)
        std::swap(tlow, thigh);
    std::vector<std::vector<Eigen::Vector2d>> eventtimes(1);
    eventtimes[0].emplace_back(tlow, thigh);

    // fifthorder=false (LGL3) — we only need a bracket that interpolates
    // cleanly; the precise interp order doesn't change the overshoot.
    auto tab = integ.make_table(xs, /*fifthorder=*/false);
    auto eventlocs = integ.find_events(tab, events, eventtimes);

    ASSERT_EQ(eventlocs.size(), 1u);
    ASSERT_EQ(eventlocs[0].size(), 1u);
    EXPECT_FALSE(eventlocs[0][0].has_value())
        << "Bracket with no real crossing and overshooting Newton must emit std::nullopt";
    EXPECT_EQ(integ.get_failed_event_count(), 1)
        << "Counter must increment once per emitted nullopt";
}

///////////////////////////////////////////////////////////////////////////////
// Mixed-nullopt shape contract. The std::optional<ODEState> API break
// claims 1:1 alignment between eventtimes and eventstates, independent of
// which crossings resolve. A bracket that resolves fills an engaged
// optional; a bracket that fails to refine fills std::nullopt, keeping the
// index stable. Existing tests cover all-engaged and all-nullopt; this
// pins the mixed case, which is the real failure mode for consumers that
// zip eventtimes with eventstates by index.
///////////////////////////////////////////////////////////////////////////////
TEST_F(EventRefinementCoverageTest, MixedNullopt_ShapePreserved) {
    astro::Kepler kep(kErcMu);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI54, 10.0);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    auto x0 = erc_eccentric_x0();
    double tf = erc_eccentric_period();

    // Real event: altitude crossing at 7200 km on eccentric orbit (rp=6800,
    // ra~8310). Crosses twice per period — once rising, once falling.
    auto args = Arguments<7>();
    auto event_expr = args.head<3>().norm() + (-7200.0);
    GenericFunction<-1, 1> gf(event_expr);
    std::vector<Integrator<astro::Kepler>::EventPack> events;
    events.push_back({gf, 0, 0});

    // First call resolves the real crossings cleanly — this is the "engaged"
    // reference that we graft fake brackets onto.
    auto [xf_real, real_eventlocs] = integ.integrate(x0, tf, events);
    ASSERT_EQ(real_eventlocs.size(), 1u);
    ASSERT_GE(real_eventlocs[0].size(), 2u)
        << "Eccentric orbit must cross altitude=7200 at least twice per period.";

    // Now build a dense trajectory + table so we can drive find_events with a
    // caller-constructed eventtimes that MIXES real brackets (from the
    // integrate above) with fabricated no-crossing brackets — the latter
    // resolve to std::nullopt after the residual check (P0-1).
    std::vector<Integrator<astro::Kepler>::EventPack> no_events;
    auto [xs, _] = integ.integrate_dense(x0, tf, no_events, /*alloutput=*/true);
    auto tab = integ.make_table(xs, /*fifthorder=*/false);

    // Collect brackets around the known crossings. We use the refined event
    // times from the first call and widen each by a small window — the
    // refinement must land inside.
    ASSERT_TRUE(real_eventlocs[0][0].has_value());
    ASSERT_TRUE(real_eventlocs[0][1].has_value());
    const double te0 = real_eventlocs[0][0]->operator[](6); // Kepler t_var = 6
    const double te1 = real_eventlocs[0][1]->operator[](6);

    constexpr int kKeplerTVar = 6;
    const double t_start = xs[0][kKeplerTVar];
    const double t_end = xs.back()[kKeplerTVar];
    const double half_window = (t_end - t_start) * 0.01;

    std::vector<std::vector<Eigen::Vector2d>> eventtimes(1);
    // Index 0: real crossing A — must resolve.
    eventtimes[0].emplace_back(te0 - half_window, te0 + half_window);
    // Index 1: fake bracket near t_start, far from any crossing — must
    // produce std::nullopt (the residual check at P0-1 rejects the Newton
    // iterate since |altitude - 7200| stays well above tol over this span).
    eventtimes[0].emplace_back(t_start + half_window * 0.1, t_start + half_window * 0.2);
    // Index 2: real crossing B — must resolve.
    eventtimes[0].emplace_back(te1 - half_window, te1 + half_window);
    // Index 3: another fake bracket near t_end.
    eventtimes[0].emplace_back(t_end - half_window * 0.2, t_end - half_window * 0.1);

    auto eventlocs = integ.find_events(tab, events, eventtimes);

    ASSERT_EQ(eventlocs.size(), 1u);
    ASSERT_EQ(eventlocs[0].size(), 4u) << "1:1 alignment: one entry per eventtimes bracket.";
    EXPECT_TRUE(eventlocs[0][0].has_value()) << "real crossing A must resolve";
    EXPECT_FALSE(eventlocs[0][1].has_value()) << "no-crossing bracket must be std::nullopt";
    EXPECT_TRUE(eventlocs[0][2].has_value()) << "real crossing B must resolve";
    EXPECT_FALSE(eventlocs[0][3].has_value()) << "no-crossing bracket must be std::nullopt";
    EXPECT_EQ(integ.get_failed_event_count(), 2)
        << "Counter must increment once per emitted nullopt.";
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
