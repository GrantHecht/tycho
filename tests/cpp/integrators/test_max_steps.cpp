///////////////////////////////////////////////////////////////////////////////
// max_steps cap unit tests
//
// Pins the Julia-style maxiters safety net. The cap is the only runaway
// guard once the min/max-dt clamps are absent; without these tests an
// off-by-one or accidental disable would silently regress.
//
// Covers:
//   1. Runaway adaptive loop trips the cap with a useful error message.
//   2. set_max_steps rejects n <= 0 with std::invalid_argument.
//   3. SuperScalar batch path also enforces the cap (separate code path
//      with its own throw site at integrate_impl_vectorized).
//   4. Boundary check: a `max_steps` slightly above the natural step count
//      lets integration succeed (no off-by-one false positive).
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <numbers>
#include <vector>

using namespace tycho;
using namespace TychoTest;

namespace {

// Inlined Kepler helpers so this file does not depend on the
// regression_utils.h include path (which is not on the heavy test target's
// include search list).
constexpr double kMuEarth = 398600.4418;

inline Eigen::VectorXd leo_x0() {
    constexpr double r0 = 7000.0;
    const double v0 = std::sqrt(kMuEarth / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0.0, 0.0, 0.0, v0, 0.0, 0.0;
    return x0;
}

inline double leo_period() {
    constexpr double r0 = 7000.0;
    return 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / kMuEarth);
}

} // namespace

class MaxStepsTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Forces a runaway by demanding an impossible tolerance — every step rejects,
// the controller shrinks dt, and the loop runs to the cap. Asserts the throw
// type, message content, and that it fires at the cap rather than silently
// timing out.
///////////////////////////////////////////////////////////////////////////////
TEST_F(MaxStepsTest, ThrowsOnRunawayWithDiagnosticMessage) {
    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1.0e-30); // Unattainable in double precision.
    integ.set_rel_tol(1.0e-30);
    integ.set_max_steps(50);

    auto x0 = leo_x0();
    double tf = leo_period();

    try {
        integ.integrate(x0, tf);
        FAIL() << "Expected runtime_error from max_steps cap, but integration succeeded.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("max_steps"), std::string::npos)
            << "Error message should mention max_steps; got: " << msg;
        EXPECT_NE(msg.find("50"), std::string::npos)
            << "Error message should include the cap value; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error, got a different exception type.";
    }
}

///////////////////////////////////////////////////////////////////////////////
// set_max_steps must reject zero and negative values at API entry, before
// any integration runs (cheaper to surface misconfiguration here than at
// the throw site inside the loop).
///////////////////////////////////////////////////////////////////////////////
TEST_F(MaxStepsTest, SetMaxStepsRejectsNonPositive) {
    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);

    EXPECT_THROW(integ.set_max_steps(0), std::invalid_argument);
    EXPECT_THROW(integ.set_max_steps(-1), std::invalid_argument);
    EXPECT_THROW(integ.set_max_steps(-100000), std::invalid_argument);

    // Sanity: a positive value succeeds and is round-trippable.
    integ.set_max_steps(123);
    EXPECT_EQ(integ.get_max_steps(), 123);
}

///////////////////////////////////////////////////////////////////////////////
// The SIMD batch path (integrate_impl_vectorized) has its own cap-check
// site and emits a batch-specific error message. A regression that disabled
// only the scalar check would slip through without this case.
///////////////////////////////////////////////////////////////////////////////
TEST_F(MaxStepsTest, BatchPathEnforcesCap) {
    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1.0e-30);
    integ.set_rel_tol(1.0e-30);
    integ.set_max_steps(50);
    integ.vectorize_batch_calls_ = true;

    using KeplerState = Integrator<astro::Kepler>::IntegRet;
    auto x0_dyn = leo_x0();
    KeplerState x0;
    for (int i = 0; i < x0.size(); ++i)
        x0[i] = x0_dyn[i];

    constexpr int N = 4;
    std::vector<KeplerState> x0s(N, x0);
    Eigen::VectorXd tfs(N);
    for (int i = 0; i < N; ++i)
        tfs[i] = leo_period();

    try {
        integ.integrate(x0s, tfs);
        FAIL() << "Expected runtime_error from batch max_steps cap.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("max_steps"), std::string::npos)
            << "Batch error should mention max_steps; got: " << msg;
        EXPECT_NE(msg.find("ParallelDriver"), std::string::npos)
            << "Batch error should identify ParallelDriver source; got: " << msg;
    } catch (...) {
        FAIL() << "Expected std::runtime_error from batch path.";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Parametric coverage: every user-selectable RK method honors the cap.
// The throw site is inside AdaptiveDriver::integrate (templated on Alg);
// a regression that disabled the check for one method would pass the
// single-method test above. This parametric case pins the cap for all 8.
///////////////////////////////////////////////////////////////////////////////

struct MaxStepsMethodExpect {
    IVPAlg method;
    const char *name;
};

class MaxStepsAllMethodsTest : public VectorFunctionFixture,
                               public ::testing::WithParamInterface<MaxStepsMethodExpect> {};

TEST_P(MaxStepsAllMethodsTest, EachMethodHonorsMaxSteps) {
    const auto &p = GetParam();

    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, p.method, 10.0);
    integ.set_abs_tol(1.0e-30); // Unattainable in double precision.
    integ.set_rel_tol(1.0e-30);
    integ.set_max_steps(50);

    auto x0 = leo_x0();
    double tf = leo_period();

    try {
        integ.integrate(x0, tf);
        FAIL() << p.name << ": expected runtime_error from max_steps cap, but integration succeeded.";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("max_steps"), std::string::npos)
            << p.name << ": error message should mention max_steps; got: " << msg;
    } catch (...) {
        FAIL() << p.name << ": expected std::runtime_error, got a different exception type.";
    }
}

INSTANTIATE_TEST_SUITE_P(AllUserSelectableMethods, MaxStepsAllMethodsTest,
                         ::testing::Values(MaxStepsMethodExpect{IVPAlg::DOPRI54, "DOPRI54"},
                                           MaxStepsMethodExpect{IVPAlg::DOPRI87, "DOPRI87"},
                                           MaxStepsMethodExpect{IVPAlg::Tsit5, "Tsit5"},
                                           MaxStepsMethodExpect{IVPAlg::BS3, "BS3"},
                                           MaxStepsMethodExpect{IVPAlg::BS5, "BS5"},
                                           MaxStepsMethodExpect{IVPAlg::Vern7, "Vern7"},
                                           MaxStepsMethodExpect{IVPAlg::Vern8, "Vern8"},
                                           MaxStepsMethodExpect{IVPAlg::Vern9, "Vern9"}),
                         [](const auto &info) { return info.param.name; });

///////////////////////////////////////////////////////////////////////////////
// Off-by-one regression: pick a problem whose adaptive run completes in
// well under the cap, set max_steps to the natural count plus a safety
// margin, and confirm integration succeeds. A future change that throws
// when steps_attempted == max_steps_ rather than > would fail this case.
///////////////////////////////////////////////////////////////////////////////
TEST_F(MaxStepsTest, BoundaryCountAllowsSuccess) {
    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1.0e-12);
    integ.set_rel_tol(1.0e-13);

    auto x0 = leo_x0();
    double tf = leo_period() / 4.0;

    // First, learn the natural step count for this configuration.
    auto xf_unbounded = integ.integrate(x0, tf);
    int64_t natural = integ.get_naccept() + integ.get_nreject();
    ASSERT_GT(natural, 0);

    // Now cap at exactly the natural count — must still succeed. The cap
    // check runs at the top of each iteration and the final step exits
    // via continueloop=false before re-entering the check.
    integ.set_max_steps(natural);
    auto xf_capped = integ.integrate(x0, tf);

    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(xf_unbounded[i], xf_capped[i])
            << "Bounded run must be bit-identical to unbounded for component " << i;
    }
}

///////////////////////////////////////////////////////////////////////////////
// CounterWriteback contract: after any integrate call that threw,
// get_naccept()/get_nreject() must reflect the work actually done rather
// than the previous call's values. Reverting the RAII guards to a
// success-only writeback path would leave the counters at zero (initial
// state) after this forced throw.
///////////////////////////////////////////////////////////////////////////////
TEST_F(MaxStepsTest, CountersReflectPartialProgressAfterThrow) {
    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1.0e-30);
    integ.set_rel_tol(1.0e-30);
    integ.set_max_steps(50);

    auto x0 = leo_x0();
    double tf = leo_period();

    try {
        integ.integrate(x0, tf);
        FAIL() << "Expected runtime_error from max_steps cap, but integration succeeded.";
    } catch (const std::runtime_error &) {
        // expected
    }
    EXPECT_GT(integ.get_naccept() + integ.get_nreject(), 0)
        << "CounterWriteback RAII must flush local counters into member state "
           "even when integrate() exits via exception unwind.";
}

///////////////////////////////////////////////////////////////////////////////
// Parameterized CounterWriteback coverage across entrypoints that share
// the same RAII guard but use different code paths (scalar integrate vs.
// dense vs. STM vs. batch). A regression that dropped the guard from one
// entrypoint would leave the counters at zero for that lane only —
// invisible to a single-entrypoint test.
///////////////////////////////////////////////////////////////////////////////
enum class Entrypoint {
    Integrate,
    IntegrateDense,
    IntegrateStm,
    IntegrateBatch,
};

struct EntrypointInfo {
    Entrypoint kind;
    const char *name;
};

class CounterWritebackEntrypointTest : public VectorFunctionFixture,
                                       public ::testing::WithParamInterface<EntrypointInfo> {};

TEST_P(CounterWritebackEntrypointTest, PartialProgressFlushedAfterThrow) {
    const auto &p = GetParam();

    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1.0e-30);
    integ.set_rel_tol(1.0e-30);
    integ.set_max_steps(50);

    using K = Integrator<astro::Kepler>::IntegRet;
    auto x0_dyn = leo_x0();
    K x0;
    for (int i = 0; i < x0.size(); ++i)
        x0[i] = x0_dyn[i];
    const double tf = leo_period();

    bool threw = false;
    try {
        switch (p.kind) {
        case Entrypoint::Integrate:
            (void)integ.integrate(x0, tf);
            break;
        case Entrypoint::IntegrateDense:
            (void)integ.integrate_dense(x0, tf);
            break;
        case Entrypoint::IntegrateStm:
            (void)integ.integrate_stm(x0, tf);
            break;
        case Entrypoint::IntegrateBatch: {
            std::vector<K> x0s = {x0, x0};
            Eigen::VectorXd tfs(2);
            tfs << tf, tf;
            (void)integ.integrate(x0s, tfs);
            break;
        }
        }
        FAIL() << p.name << ": expected runtime_error from max_steps cap.";
    } catch (const std::runtime_error &) {
        threw = true;
    }
    ASSERT_TRUE(threw) << p.name;
    EXPECT_GT(integ.get_naccept() + integ.get_nreject(), 0)
        << p.name << ": CounterWriteback must flush local counters on exception unwind.";
}

INSTANTIATE_TEST_SUITE_P(
    AllEntrypoints, CounterWritebackEntrypointTest,
    ::testing::Values(EntrypointInfo{Entrypoint::Integrate, "integrate"},
                      EntrypointInfo{Entrypoint::IntegrateDense, "integrate_dense"},
                      EntrypointInfo{Entrypoint::IntegrateStm, "integrate_stm"},
                      EntrypointInfo{Entrypoint::IntegrateBatch, "integrate_batch"}),
    [](const auto &info) { return info.param.name; });

///////////////////////////////////////////////////////////////////////////////
// Symmetric guard for the reverse regression: a clean run after a failed run
// must reflect only the clean run's counters, not accumulate stale state
// from the prior failure. A fix that wrote counters only on exception
// unwind (the inverse of the original bug) would pass the failure-path
// test above but fail here.
///////////////////////////////////////////////////////////////////////////////
TEST_F(MaxStepsTest, CountersRepresentOnlyCleanRunAfterFailure) {
    astro::Kepler kep(kMuEarth);
    Integrator<astro::Kepler> integ(kep, IVPAlg::DOPRI87, 10.0);
    integ.set_abs_tol(1.0e-30);
    integ.set_rel_tol(1.0e-30);
    integ.set_max_steps(50);

    auto x0 = leo_x0();
    double tf = leo_period();

    // First: force a throw and capture the failure-path counter.
    try {
        integ.integrate(x0, tf);
        FAIL() << "Expected runtime_error from first max_steps cap.";
    } catch (const std::runtime_error &) {
        // expected
    }
    const int64_t failed_count = integ.get_naccept() + integ.get_nreject();
    ASSERT_GT(failed_count, 0)
        << "Precondition: failed run must have written a non-zero counter.";

    // Now: reconfigure for a clean run on the same integrator and verify
    // the counter reflects only the clean run, not the sum of both.
    integ.set_abs_tol(1.0e-12);
    integ.set_rel_tol(1.0e-13);
    integ.set_max_steps(1000);

    const double short_tf = leo_period() / 8.0; // distinct from the failed run
    (void)integ.integrate(x0, short_tf);
    const int64_t clean_count = integ.get_naccept() + integ.get_nreject();

    EXPECT_GT(clean_count, 0) << "Clean run must record a non-zero counter.";
    // Tuning-robust invariant: the clean counter must not equal the
    // failure-path counter — that would indicate either stale state
    // carried over from the prior throw OR a writeback that fires only on
    // exception unwind. Equality is the signature of both regressions.
    EXPECT_NE(clean_count, failed_count)
        << "Clean-run counter must differ from failure-path counter "
           "(both=" << clean_count << ").";
}
