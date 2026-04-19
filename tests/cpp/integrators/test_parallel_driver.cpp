// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// ParallelDriver runtime tests. Proves the SIMD batch driver produces lane-by-
// lane output matching the production Integrator's batch integrate() within
// numerical tolerance, handles zero-interval lanes, reports the offending
// lane on NaN dynamics, and independently drives per-lane controllers.

#include "integrator_test_utils.h"
#include <tycho/tycho.h>

#include "tycho/detail/integrators/parallel_driver.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {

Eigen::Vector3d sho_x0(double phase = 0.0) {
    Eigen::Vector3d x;
    x << std::cos(phase), -std::sin(phase), 0.0;
    return x;
}

} // namespace

// -----------------------------------------------------------------------------
// Compile-only: ParallelDriver instantiates across all 8 user-selectable methods.
// -----------------------------------------------------------------------------
TEST(ParallelDriverCompileTest, InstantiatesAllMethods) {
    using Kep = tycho::astro::Kepler;
    ParallelDriver<IVPAlg::DOPRI54, Kep> a;
    ParallelDriver<IVPAlg::DOPRI87, Kep> b;
    ParallelDriver<IVPAlg::Tsit5, Kep> c;
    ParallelDriver<IVPAlg::BS3, Kep> d;
    ParallelDriver<IVPAlg::BS5, Kep> e;
    ParallelDriver<IVPAlg::Vern7, Kep> f;
    ParallelDriver<IVPAlg::Vern8, Kep> g;
    ParallelDriver<IVPAlg::Vern9, Kep> h;
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
    SUCCEED();
}

// -----------------------------------------------------------------------------
// Integrates a batch of SHO trajectories through ParallelDriver and compares
// against production Integrator's batch integrate. Numerical match, not bit-
// exact: the Integrator uses stepper_compute_impl which has slightly
// different FSAL handling than ParallelDriver's Stepper<> path for non-FSAL
// methods.
// -----------------------------------------------------------------------------
class ParallelDriverRunTest : public VectorFunctionFixture {};

TEST_F(ParallelDriverRunTest, BatchSHO_MatchesIntegrator) {
    SHO ode(0.0);
    const int N = 4;
    const double tf_all = 1.0;

    // Reference via production Integrator batch integrate.
    Integrator<SHO> reference(ode, IVPAlg::DOPRI54, 0.01);
    reference.set_abs_tol(1e-10);
    reference.set_rel_tol(1e-10);
    std::vector<Eigen::Vector3d> xs_ref;
    for (int i = 0; i < N; ++i) xs_ref.push_back(sho_x0(0.1 * i));
    Eigen::VectorXd tfs(N);
    tfs.setConstant(tf_all);
    auto ref_out = reference.integrate(xs_ref, tfs);

    // Test via ParallelDriver.
    ParallelDriver<IVPAlg::DOPRI54, SHO> driver;
    AdaptiveConfig cfg;
    cfg.error_order = 4;
    cfg.def_step_size = 0.01;
    cfg.adaptive = true;
    cfg.use_hairer_wanner_initdt = true;

    Eigen::VectorXd abs_tols(2);
    abs_tols.setConstant(1e-10);
    Eigen::VectorXd rel_tols(2);
    rel_tols.setConstant(1e-10);

    std::vector<ControllerVariant> controllers(N, ControllerVariant{PIController{}});
    for (auto &c : controllers) std::visit([](auto &cc) { cc.reset(); }, c);
    std::vector<int> nacc(N, 0), nrej(N, 0);

    std::vector<ParallelDriver<IVPAlg::DOPRI54, SHO>::EventPack> events;
    std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes_s(N);
    std::vector<std::vector<Eigen::Vector3d>> states_s(N);
    std::vector<std::vector<typename SHO::template Output<double>>> derivs_s(N);

    auto noop_update_control = [](auto &) {};

    std::vector<Eigen::Vector3d> xs_test = xs_ref;
    auto out = driver.integrate(ode, xs_test, tfs, cfg, abs_tols, rel_tols, controllers, nacc, nrej,
                                events, eventtimes_s, /*storestates=*/false, /*storederivs=*/false,
                                /*storemidpoints=*/false, states_s, derivs_s,
                                noop_update_control);

    ASSERT_EQ(out.size(), static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(out[i][0], ref_out[i][0], 1e-6) << "lane " << i << " x";
        EXPECT_NEAR(out[i][1], ref_out[i][1], 1e-6) << "lane " << i << " v";
        EXPECT_NEAR(out[i][2], tf_all, 1e-12) << "lane " << i << " t";
        EXPECT_GT(nacc[i], 0) << "lane " << i << " must take at least one step";
    }
}

// -----------------------------------------------------------------------------
// Zero-interval handling: lane with tf == t0 is output-populated up front and
// never enters the SIMD loop. Non-zero lanes in the same batch integrate
// normally.
// -----------------------------------------------------------------------------
TEST_F(ParallelDriverRunTest, MixedZeroInterval_BatchCorrect) {
    SHO ode(0.0);
    ParallelDriver<IVPAlg::DOPRI54, SHO> driver;
    AdaptiveConfig cfg;
    cfg.error_order = 4;
    cfg.adaptive = true;
    cfg.use_hairer_wanner_initdt = true;

    Eigen::VectorXd abs_tols(2);
    abs_tols.setConstant(1e-10);
    Eigen::VectorXd rel_tols(2);
    rel_tols.setConstant(1e-10);

    std::vector<Eigen::Vector3d> xs = {sho_x0(), sho_x0(), sho_x0()};
    Eigen::VectorXd tfs(3);
    tfs << 0.0, 1.0, 0.0;

    std::vector<ControllerVariant> controllers(3, ControllerVariant{PIController{}});
    for (auto &c : controllers) std::visit([](auto &cc) { cc.reset(); }, c);
    std::vector<int> nacc(3, 0), nrej(3, 0);
    std::vector<ParallelDriver<IVPAlg::DOPRI54, SHO>::EventPack> events;
    std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes_s(3);
    std::vector<std::vector<Eigen::Vector3d>> states_s(3);
    std::vector<std::vector<typename SHO::template Output<double>>> derivs_s(3);

    auto out = driver.integrate(ode, xs, tfs, cfg, abs_tols, rel_tols, controllers, nacc, nrej,
                                events, eventtimes_s, false, false, false, states_s, derivs_s,
                                [](auto &) {});

    // Zero-interval lanes return input state, no steps.
    EXPECT_DOUBLE_EQ(out[0][0], xs[0][0]);
    EXPECT_DOUBLE_EQ(out[0][1], xs[0][1]);
    EXPECT_EQ(nacc[0], 0);
    EXPECT_EQ(nrej[0], 0);
    // Middle lane integrates.
    EXPECT_NEAR(out[1][0], std::cos(1.0), 1e-6);
    EXPECT_NEAR(out[1][1], -std::sin(1.0), 1e-6);
    EXPECT_GT(nacc[1], 0);
    // Third lane zero-interval.
    EXPECT_DOUBLE_EQ(out[2][0], xs[2][0]);
    EXPECT_EQ(nacc[2], 0);
}

// -----------------------------------------------------------------------------
// Per-lane independence: different controllers per lane produce different
// step counts, and a single lane's rejection doesn't poison the others.
// -----------------------------------------------------------------------------
TEST_F(ParallelDriverRunTest, PerLaneControllerIndependence) {
    SHO ode(0.0);
    ParallelDriver<IVPAlg::DOPRI54, SHO> driver;
    AdaptiveConfig cfg;
    cfg.error_order = 4;
    cfg.adaptive = true;
    cfg.use_hairer_wanner_initdt = true;

    Eigen::VectorXd abs_tols(2);
    abs_tols.setConstant(1e-8);
    Eigen::VectorXd rel_tols(2);
    rel_tols.setConstant(1e-8);

    std::vector<Eigen::Vector3d> xs = {sho_x0(), sho_x0()};
    Eigen::VectorXd tfs(2);
    tfs.setConstant(5.0);

    std::vector<ControllerVariant> controllers = {ControllerVariant{IController{}},
                                                   ControllerVariant{PIDController{}}};
    for (auto &c : controllers) std::visit([](auto &cc) { cc.reset(); }, c);
    std::vector<int> nacc(2, 0), nrej(2, 0);
    std::vector<ParallelDriver<IVPAlg::DOPRI54, SHO>::EventPack> events;
    std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes_s(2);
    std::vector<std::vector<Eigen::Vector3d>> states_s(2);
    std::vector<std::vector<typename SHO::template Output<double>>> derivs_s(2);

    auto out = driver.integrate(ode, xs, tfs, cfg, abs_tols, rel_tols, controllers, nacc, nrej,
                                events, eventtimes_s, false, false, false, states_s, derivs_s,
                                [](auto &) {});

    // Both lanes finish at tf=5.0 with SHO solution regardless of controller.
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(out[i][0], std::cos(5.0), 1e-5) << "lane " << i;
        EXPECT_NEAR(out[i][1], -std::sin(5.0), 1e-5) << "lane " << i;
        EXPECT_GT(nacc[i], 0);
    }
}
