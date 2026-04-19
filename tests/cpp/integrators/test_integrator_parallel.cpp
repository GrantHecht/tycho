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
#include <vector>

using namespace tycho;
using namespace TychoTest;

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

TEST_F(IntegratorTest, ParallelIntegrateLeavesCountersUntouched) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    auto x0s = make_batch();
    auto tfs = make_tfs();

    // Seed the counters with a known serial-run value.
    auto _ignore = integ.integrate(x0s[0], tfs[0]);
    (void)_ignore;
    const int seed_naccept = integ.get_naccept();
    const int seed_nreject = integ.get_nreject();

    ScopedThreadCount threads(kNParts);
    auto _p = integ.integrate_parallel(x0s, tfs, kNParts);
    (void)_p;

    // Contract: parallel workers use private local counters and never
    // write to the `Integrator` members, so the values from the prior
    // serial call survive.
    EXPECT_EQ(integ.get_naccept(), seed_naccept);
    EXPECT_EQ(integ.get_nreject(), seed_nreject);
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

TEST_F(IntegratorTest, SegmentedSTMParallelMatchesSerial) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI54, 0.01);
    tighten(integ);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 2.0;

    auto [serial_xf, serial_stm] = integ.integrate_stm(x0, tf);

    ScopedThreadCount threads(kNParts);
    auto [seg_xf, seg_stm] = integ.integrate_stm_parallel(x0, tf, kNParts);

    EXPECT_NEAR(seg_xf[0], serial_xf[0], 1e-11);
    EXPECT_NEAR(seg_xf[1], serial_xf[1], 1e-11);
    ASSERT_EQ(seg_stm.rows(), serial_stm.rows());
    ASSERT_EQ(seg_stm.cols(), serial_stm.cols());
    for (int i = 0; i < seg_stm.rows(); i++) {
        for (int j = 0; j < seg_stm.cols(); j++) {
            EXPECT_NEAR(seg_stm(i, j), serial_stm(i, j), 1e-10)
                << "STM element (" << i << ", " << j << ")";
        }
    }
}
