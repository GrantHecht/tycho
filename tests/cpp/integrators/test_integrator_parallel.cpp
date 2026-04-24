///////////////////////////////////////////////////////////////////////////////
// Parallel integrate determinism pin — covers the P1 data race that was
// introduced when `integrate_impl` mutated `controller_variant_`,
// `naccept_`, and `nreject_` on the shared `Integrator` instance from
// inside concurrent `integrate_parallel_impl` workers.
//
// Pre-fix behavior: multiple workers racing on the shared mutable
// controller + counters produced nondeterministic step counts and, at
// worst, torn controller state that diverged from the serial path.
// These tests fail pre-fix by asserting bit-identical repeated runs
// and per-trajectory agreement with a serial reference.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <limits>
#include <vector>

using namespace tycho;
using namespace TychoTest;
using tycho::integrators::IVPController;

namespace {

constexpr int kBatchSize = 32;
constexpr int kNParts = 4;
constexpr double kTol = 1e-12;

// Build a batch of SHO initial conditions with varied phase so adaptive
// step control decisions differ across trajectories — exactly the regime
// where a shared controller race would surface.
std::vector<Eigen::Vector3d> make_batch() {
    std::vector<Eigen::Vector3d> x0s(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        double phase = double(i) / double(kBatchSize);
        x0s[i] << std::cos(phase), -std::sin(phase), 0.0;
    }
    return x0s;
}

Eigen::VectorXd make_tfs() {
    Eigen::VectorXd tfs(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        tfs[i] = 1.0 + 0.25 * (double(i) / double(kBatchSize));
    }
    return tfs;
}

template <class Integ> void tighten(Integ &integ) {
    integ.set_abs_tol(kTol);
    integ.set_rel_tol(kTol);
}

} // namespace

TEST_F(IntegratorTest, ParallelIntegrateMatchesSerial) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    std::vector<Eigen::Vector3d> serial(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        serial[i] = integ.integrate(x0s[i], tfs[i]);
    }

    ScopedThreadCount threads(kNParts);
    auto parallel = integ.integrate_parallel(x0s, tfs, kNParts);

    ASSERT_EQ(parallel.size(), serial.size());
    for (int i = 0; i < kBatchSize; i++) {
        EXPECT_NEAR(parallel[i][0], serial[i][0], 1e-13) << "trajectory " << i;
        EXPECT_NEAR(parallel[i][1], serial[i][1], 1e-13) << "trajectory " << i;
    }
}

TEST_F(IntegratorTest, ParallelIntegrateDeterministicAcrossRepeats) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    ScopedThreadCount threads(kNParts);

    auto run1 = integ.integrate_parallel(x0s, tfs, kNParts);
    auto run2 = integ.integrate_parallel(x0s, tfs, kNParts);
    auto run3 = integ.integrate_parallel(x0s, tfs, kNParts);

    for (int i = 0; i < kBatchSize; i++) {
        // Bit-identical determinism across runs: pre-fix this fails
        // because the shared controller's PI state would tear under
        // concurrent update().
        EXPECT_EQ(run1[i][0], run2[i][0]) << "trajectory " << i;
        EXPECT_EQ(run1[i][1], run2[i][1]) << "trajectory " << i;
        EXPECT_EQ(run2[i][0], run3[i][0]) << "trajectory " << i;
        EXPECT_EQ(run2[i][1], run3[i][1]) << "trajectory " << i;
    }
}

TEST_F(IntegratorTest, ParallelIntegrateReflectsSummedBatchCounts) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    // Baseline: accumulate the serial batch by calling integrate() on each
    // trajectory; the member counters hold the last call's counts after
    // each invocation (CounterWriteback overwrites per call). Sum by
    // re-querying after every integrate to get the batch total.
    int serial_total_naccept = 0;
    int serial_total_nreject = 0;
    for (int i = 0; i < kBatchSize; ++i) {
        (void)integ.integrate(x0s[i], tfs[i]);
        serial_total_naccept += integ.get_naccept();
        serial_total_nreject += integ.get_nreject();
    }

    ScopedThreadCount threads(kNParts);
    auto _p = integ.integrate_parallel(x0s, tfs, kNParts);
    (void)_p;

    // Contract: parallel paths write the summed per-trajectory counts
    // into the member counters on scope exit via BatchCounterWriteback.
    // The sum must match the serial-path total because both paths
    // integrate the same trajectories deterministically.
    EXPECT_EQ(integ.get_naccept(), serial_total_naccept);
    EXPECT_EQ(integ.get_nreject(), serial_total_nreject);
}

TEST_F(IntegratorTest, ParallelIntegrateCountersMatchCompletedLanesOnThrow) {
    // Contract: when a lane throws mid-batch, BatchCounterWriteback records
    // only the counts of trajectories that completed before the first
    // failure. Unstarted lanes contribute zero. With kNParts=1 the work
    // serializes within a single worker, so lanes after the poisoned lane
    // never run and their per-trajectory counts stay at 0.
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    // Poison lane K: a NaN x0 trips the non-finite guard inside integrate.
    constexpr int kPoisonLane = 4;
    x0s[kPoisonLane][0] = std::numeric_limits<double>::quiet_NaN();

    // Baseline: serial counts for lanes [0, kPoisonLane) — these are the
    // only lanes that should contribute to the parallel counter sum
    // before the exception propagates out of the single worker.
    int expected_naccept = 0;
    int expected_nreject = 0;
    for (int i = 0; i < kPoisonLane; ++i) {
        (void)integ.integrate(x0s[i], tfs[i]);
        expected_naccept += integ.get_naccept();
        expected_nreject += integ.get_nreject();
    }

    ScopedThreadCount threads(1);
    EXPECT_THROW((void)integ.integrate_parallel(x0s, tfs, /*n_parts=*/1), std::exception);

    // After the throw, member counters reflect only the completed lanes.
    // With n_parts=1, the single worker processes lanes in order and stops
    // at kPoisonLane, so the sum is exactly the serial sum over [0, K).
    EXPECT_EQ(integ.get_naccept(), expected_naccept);
    EXPECT_EQ(integ.get_nreject(), expected_nreject);
}

TEST_F(IntegratorTest, ParallelIntegrateDenseMatchesSerial) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    std::vector<std::vector<Eigen::Vector3d>> serial(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        serial[i] = integ.integrate_dense(x0s[i], tfs[i]);
    }

    ScopedThreadCount threads(kNParts);
    auto parallel = integ.integrate_dense_parallel(x0s, tfs, kNParts);

    ASSERT_EQ(parallel.size(), serial.size());
    for (int i = 0; i < kBatchSize; i++) {
        ASSERT_EQ(parallel[i].size(), serial[i].size()) << "trajectory " << i;
        const auto &p_last = parallel[i].back();
        const auto &s_last = serial[i].back();
        EXPECT_NEAR(p_last[0], s_last[0], 1e-13) << "trajectory " << i;
        EXPECT_NEAR(p_last[1], s_last[1], 1e-13) << "trajectory " << i;
    }
}

TEST_F(IntegratorTest, ParallelIntegrateSTMMatchesSerial) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    std::vector<Integrator<SHO>::STMRet> serial(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        serial[i] = integ.integrate_stm(x0s[i], tfs[i]);
    }

    ScopedThreadCount threads(kNParts);
    auto parallel = integ.integrate_stm_parallel(x0s, tfs, kNParts);

    ASSERT_EQ(parallel.size(), serial.size());
    for (int i = 0; i < kBatchSize; i++) {
        const auto &p_xf = std::get<0>(parallel[i]);
        const auto &s_xf = std::get<0>(serial[i]);
        EXPECT_NEAR(p_xf[0], s_xf[0], 1e-12) << "trajectory " << i;
        EXPECT_NEAR(p_xf[1], s_xf[1], 1e-12) << "trajectory " << i;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Event-carrying parallel overloads — fills the review gap where
// `integrate_parallel(..., events, ...)` and `integrate_stm_parallel(...,
// events, ...)` had no direct test coverage (only the no-events overloads
// were exercised). Asserts that per-trajectory event location lists agree
// with the serial reference for each of the N trajectories in the batch.
///////////////////////////////////////////////////////////////////////////////

namespace {

// Event on SHO: v = -0.5 (zero-crossing of `v + 0.5`). Over the chosen tf
// range each trajectory's phase produces either 0 or 1 crossings, giving
// variance across lanes that a broken event wiring would expose.
auto make_sho_event() {
    auto args = Arguments<3>();
    auto v = args.coeff<1>();
    auto event_func = v + 0.5;
    return GenericFunction<-1, 1>(event_func);
}

// Event function engineered to fail refinement: tanh(k·(t - 0.5)) with a
// very sharp slope saturates to ±1 away from t=0.5, so its analytic
// jacobian 1 - tanh² rounds to 0 exactly in double arithmetic at any
// bracket midpoint outside a tiny neighbourhood of the root. That trips
// Newton's singular-Jacobian guard → the refinement emits std::nullopt
// and n_failed_event_refinements_ is incremented. A real sign change at
// t=0.5 ensures check_crossings registers the bracket in the first place.
auto make_overshoot_time_event() {
    auto args = Arguments<3>();
    auto t = args.coeff<2>();
    constexpr double kSlope = 1.0e6;
    auto event_func = tanh(kSlope * (t + (-0.5)));
    return GenericFunction<-1, 1>(event_func);
}

int count_null_event_locs(const Integrator<SHO>::EventLocsType &eventlocs) {
    int n = 0;
    for (const auto &group : eventlocs) {
        for (const auto &loc : group) {
            if (!loc.has_value())
                ++n;
        }
    }
    return n;
}

} // namespace

TEST_F(IntegratorTest, ParallelIntegrateWithEventsMatchesSerial) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    std::vector<Integrator<SHO>::EventPack> events;
    events.push_back({make_sho_event(), 0, 0});

    std::vector<Integrator<SHO>::IntegEventRet> serial(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        serial[i] = integ.integrate(x0s[i], tfs[i], events);
    }

    ScopedThreadCount threads(kNParts);
    auto parallel = integ.integrate_parallel(x0s, tfs, events, kNParts);

    ASSERT_EQ(parallel.size(), serial.size());
    for (int i = 0; i < kBatchSize; i++) {
        const auto &p_xf = std::get<0>(parallel[i]);
        const auto &s_xf = std::get<0>(serial[i]);
        EXPECT_NEAR(p_xf[0], s_xf[0], 1e-13) << "trajectory " << i << " xf[0]";
        EXPECT_NEAR(p_xf[1], s_xf[1], 1e-13) << "trajectory " << i << " xf[1]";

        const auto &p_evs = std::get<1>(parallel[i]);
        const auto &s_evs = std::get<1>(serial[i]);
        ASSERT_EQ(p_evs.size(), s_evs.size()) << "trajectory " << i << " event-group count";
        for (size_t g = 0; g < s_evs.size(); g++) {
            ASSERT_EQ(p_evs[g].size(), s_evs[g].size())
                << "trajectory " << i << " event count in group " << g;
            for (size_t e = 0; e < s_evs[g].size(); e++) {
                ASSERT_TRUE(s_evs[g][e].has_value()) << "serial event refinement failed unexpectedly";
                ASSERT_TRUE(p_evs[g][e].has_value())
                    << "parallel event refinement failed unexpectedly";
                EXPECT_NEAR((*p_evs[g][e])[0], (*s_evs[g][e])[0], 1e-12)
                    << "trajectory " << i << " event g=" << g << " e=" << e;
            }
        }
    }
}

TEST_F(IntegratorTest, ParallelIntegrateSTMWithEventsMatchesSerial) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    std::vector<Integrator<SHO>::EventPack> events;
    events.push_back({make_sho_event(), 0, 0});

    std::vector<Integrator<SHO>::STMEventRet> serial(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        serial[i] = integ.integrate_stm(x0s[i], tfs[i], events);
    }

    ScopedThreadCount threads(kNParts);
    auto parallel = integ.integrate_stm_parallel(x0s, tfs, events, kNParts);

    ASSERT_EQ(parallel.size(), serial.size());
    for (int i = 0; i < kBatchSize; i++) {
        const auto &p_xf = std::get<0>(parallel[i]);
        const auto &s_xf = std::get<0>(serial[i]);
        EXPECT_NEAR(p_xf[0], s_xf[0], 1e-12) << "trajectory " << i << " xf[0]";
        EXPECT_NEAR(p_xf[1], s_xf[1], 1e-12) << "trajectory " << i << " xf[1]";

        const auto &p_evs = std::get<2>(parallel[i]);
        const auto &s_evs = std::get<2>(serial[i]);
        ASSERT_EQ(p_evs.size(), s_evs.size()) << "trajectory " << i << " event-group count";
        for (size_t g = 0; g < s_evs.size(); g++) {
            ASSERT_EQ(p_evs[g].size(), s_evs[g].size())
                << "trajectory " << i << " event count in group " << g;
        }
    }
}

TEST_F(IntegratorTest, ParallelEventFailuresAggregatePerTrajectoryCounts) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 1.0);
    integ.adaptive_ = false;
    integ.set_auto_initial_dt(false);

    auto x0s = make_batch();
    Eigen::VectorXd tfs(kBatchSize);
    tfs.setConstant(1.0);

    std::vector<Integrator<SHO>::EventPack> events;
    events.push_back({make_overshoot_time_event(), 0, 0});

    ScopedThreadCount threads(kNParts);
    auto results = integ.integrate_parallel(x0s, tfs, events, kNParts);

    int expected_failed = 0;
    for (const auto &ret : results) {
        expected_failed += count_null_event_locs(std::get<1>(ret));
    }

    EXPECT_GT(expected_failed, 0);
    EXPECT_EQ(integ.get_failed_event_count(), expected_failed);
}

TEST_F(IntegratorTest, ParallelDenseAndSTMEventCountersResetPerCall) {
    SHO ode(0.0);
    auto x0s = make_batch();
    Eigen::VectorXd tfs(kBatchSize);
    tfs.setConstant(1.0);
    ScopedThreadCount threads(kNParts);

    std::vector<Integrator<SHO>::EventPack> overshoot_events;
    overshoot_events.push_back({make_overshoot_time_event(), 0, 0});
    std::vector<Integrator<SHO>::EventPack> happy_events;
    happy_events.push_back({make_sho_event(), 0, 0});

    Integrator<SHO> dense_integ(ode, IVPAlg::DOPRI54, 1.0);
    dense_integ.adaptive_ = false;
    dense_integ.set_auto_initial_dt(false);
    (void)dense_integ.integrate_parallel(x0s, tfs, overshoot_events, kNParts);
    EXPECT_GT(dense_integ.get_failed_event_count(), 0);

    auto dense_results = dense_integ.integrate_dense_parallel(x0s, tfs, happy_events, kNParts);
    int dense_expected_failed = 0;
    for (const auto &ret : dense_results) {
        dense_expected_failed += count_null_event_locs(std::get<1>(ret));
    }
    EXPECT_EQ(dense_integ.get_failed_event_count(), dense_expected_failed);
    EXPECT_EQ(dense_expected_failed, 0);

    Integrator<SHO> stm_integ(ode, IVPAlg::DOPRI54, 1.0);
    stm_integ.adaptive_ = false;
    stm_integ.set_auto_initial_dt(false);
    (void)stm_integ.integrate_parallel(x0s, tfs, overshoot_events, kNParts);
    EXPECT_GT(stm_integ.get_failed_event_count(), 0);

    auto stm_results = stm_integ.integrate_stm_parallel(x0s, tfs, happy_events, kNParts);
    int stm_expected_failed = 0;
    for (const auto &ret : stm_results) {
        stm_expected_failed += count_null_event_locs(std::get<2>(ret));
    }
    EXPECT_EQ(stm_integ.get_failed_event_count(), stm_expected_failed);
    EXPECT_EQ(stm_expected_failed, 0);
}

///////////////////////////////////////////////////////////////////////////////
// Per-lane controller pin (Batch E refactor) — the SuperScalar batch path
// previously shared one controller across lanes, so any divergent-IC batch
// would see step-cadence bleed-through (lane N's accept/reject decisions
// influenced lane M's next-step proposal). Each lane now drives its own
// controller copy, so the batch result must match the per-trajectory
// scalar `integrate(x_i, tf_i)` bit-for-bit, not just within a numerical
// tolerance.
///////////////////////////////////////////////////////////////////////////////
TEST_F(IntegratorTest, BatchIntegrateBitIdenticalToScalar) {
    SHO ode(0.0);

    // Run scalar integrations first — fresh integrator, so controller
    // state for each call starts from default reset.
    Integrator<SHO> scalar_integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(scalar_integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    std::vector<Eigen::Vector3d> serial(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
        serial[i] = scalar_integ.integrate(x0s[i], tfs[i]);
    }

    // Now the batch path with vectorize_batch_calls_ enabled (SIMD).
    // Pre-Batch-E this produced lane-cross-talk on divergent ICs.
    Integrator<SHO> batch_integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(batch_integ);
    batch_integ.vectorize_batch_calls_ = true;

    auto batch = batch_integ.integrate(x0s, tfs);

    ASSERT_EQ(batch.size(), serial.size());
    for (int i = 0; i < kBatchSize; i++) {
        // Bit-exact equality: per-lane controller state must reproduce
        // the scalar trajectory exactly (no FP drift from shared state).
        EXPECT_EQ(batch[i][0], serial[i][0]) << "trajectory " << i << " state[0]";
        EXPECT_EQ(batch[i][1], serial[i][1]) << "trajectory " << i << " state[1]";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Parameterized determinism pin — extends the bit-identical guarantee from
// the (DOPRI54 + PI) baseline to the cartesian product of {DOPRI54,
// DOPRI87, Vern7} × {I, PI, PID}. The original race fix touched
// `controller_variant_` plumbing; per-controller internal state
// (`err_[]`, `qold_`, `errold_`) lives inside the variant alternatives,
// so a regression that re-introduced shared mutation on a different
// method+controller pairing would slip past the DOPRI54-only test.
///////////////////////////////////////////////////////////////////////////////
class ParallelDeterminismParamTest
    : public ::testing::TestWithParam<std::tuple<IVPAlg, IVPController>> {};

TEST_P(ParallelDeterminismParamTest, BitIdenticalAcrossRepeats) {
    auto [method, controller] = GetParam();
    SHO ode(0.0);
    Integrator<SHO> integ(ode, method, 0.01);
    integ.set_abs_tol(kTol);
    integ.set_rel_tol(kTol);
    integ.set_controller(controller);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    ScopedThreadCount threads(kNParts);

    auto run1 = integ.integrate_parallel(x0s, tfs, kNParts);
    auto run2 = integ.integrate_parallel(x0s, tfs, kNParts);
    auto run3 = integ.integrate_parallel(x0s, tfs, kNParts);

    for (int i = 0; i < kBatchSize; i++) {
        EXPECT_EQ(run1[i][0], run2[i][0]) << "trajectory " << i;
        EXPECT_EQ(run1[i][1], run2[i][1]) << "trajectory " << i;
        EXPECT_EQ(run2[i][0], run3[i][0]) << "trajectory " << i;
        EXPECT_EQ(run2[i][1], run3[i][1]) << "trajectory " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    MethodControllerCombos, ParallelDeterminismParamTest,
    ::testing::Combine(::testing::Values(IVPAlg::DOPRI54, IVPAlg::DOPRI87, IVPAlg::Vern7),
                       ::testing::Values(IVPController::I, IVPController::PI, IVPController::PID)),
    [](const ::testing::TestParamInfo<std::tuple<IVPAlg, IVPController>> &info) {
        const char *method_name = "?";
        switch (std::get<0>(info.param)) {
        case IVPAlg::DOPRI54: method_name = "DOPRI54"; break;
        case IVPAlg::DOPRI87: method_name = "DOPRI87"; break;
        case IVPAlg::Vern7: method_name = "Vern7"; break;
        default: break;
        }
        const char *ctrl_name = "?";
        switch (std::get<1>(info.param)) {
        case IVPController::I: ctrl_name = "I"; break;
        case IVPController::PI: ctrl_name = "PI"; break;
        case IVPController::PID: ctrl_name = "PID"; break;
        }
        return std::string(method_name) + "_" + ctrl_name;
    });

///////////////////////////////////////////////////////////////////////////////
// Segmented STM parallelism — parameterized over nParts so a sign flip or
// transpose at a specific segment-boundary chain link surfaces regardless
// of how many partitions the user picks. Pre-S7 only nParts=4 was tested.
///////////////////////////////////////////////////////////////////////////////
class SegmentedSTMParallelTest : public ::testing::TestWithParam<int> {};

TEST_P(SegmentedSTMParallelTest, MatchesSerialForVariousPartitionCounts) {
    const int nparts = GetParam();
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    integ.set_abs_tol(kTol);
    integ.set_rel_tol(kTol);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 2.0;

    auto [serial_xf, serial_stm] = integ.integrate_stm(x0, tf);

    ScopedThreadCount threads(nparts);
    auto [seg_xf, seg_stm] = integ.integrate_stm_parallel(x0, tf, nparts);

    EXPECT_NEAR(seg_xf[0], serial_xf[0], 1e-11) << "nparts=" << nparts;
    EXPECT_NEAR(seg_xf[1], serial_xf[1], 1e-11) << "nparts=" << nparts;
    ASSERT_EQ(seg_stm.rows(), serial_stm.rows());
    ASSERT_EQ(seg_stm.cols(), serial_stm.cols());
    for (int i = 0; i < seg_stm.rows(); i++) {
        for (int j = 0; j < seg_stm.cols(); j++) {
            EXPECT_NEAR(seg_stm(i, j), serial_stm(i, j), 1e-10)
                << "nparts=" << nparts << " STM(" << i << ", " << j << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(VariousPartitionCounts, SegmentedSTMParallelTest,
                         ::testing::Values(1, 2, 4, 8),
                         [](const ::testing::TestParamInfo<int> &info) {
                             return "nparts" + std::to_string(info.param);
                         });
