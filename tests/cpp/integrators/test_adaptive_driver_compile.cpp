// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// AdaptiveDriver compile + runtime sanity tests.
//
// The compile-only tests verify the driver instantiates for every public RK
// method. The runtime test drives the full integrate() template body across
// all three controllers and compares results against the production Integrator.

#include "integrator_test_utils.h"
#include <tycho/tycho.h>

#include "tycho/detail/integrators/adaptive_driver.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

// -----------------------------------------------------------------------------
// Compile-only instantiation across all 8 RK methods.
// -----------------------------------------------------------------------------
TEST(AdaptiveDriverCompileTest, InstantiatesAllMethods) {
    using Kep = tycho::astro::Kepler;
    AdaptiveDriver<IVPAlg::DOPRI54, Kep, double> a;
    AdaptiveDriver<IVPAlg::DOPRI87, Kep, double> b;
    AdaptiveDriver<IVPAlg::Tsit5, Kep, double> c;
    AdaptiveDriver<IVPAlg::BS3, Kep, double> d;
    AdaptiveDriver<IVPAlg::BS5, Kep, double> e;
    AdaptiveDriver<IVPAlg::Vern7, Kep, double> f;
    AdaptiveDriver<IVPAlg::Vern8, Kep, double> g;
    AdaptiveDriver<IVPAlg::Vern9, Kep, double> h;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)f;
    (void)g;
    (void)h;
    SUCCEED();
}

// -----------------------------------------------------------------------------
// Runtime: AdaptiveDriver integrates SHO and matches the production Integrator
// to within numerical tolerance across all three controllers. Forces the full
// integrate() template body to instantiate so any type mismatch surfaces.
// -----------------------------------------------------------------------------
class AdaptiveDriverRunTest : public VectorFunctionFixture,
                              public ::testing::WithParamInterface<IVPController> {};

TEST_P(AdaptiveDriverRunTest, IntegrateSHOMatchesIntegrator) {
    using AD = AdaptiveDriver<IVPAlg::DOPRI54, SHO, double>;
    using State = typename AD::ODEState;

    SHO ode(0.0);

    // Reference via production Integrator.
    Integrator<SHO> reference(ode, IVPAlg::DOPRI54, 0.01);
    reference.set_abs_tol(1e-10);
    reference.set_rel_tol(1e-10);
    reference.set_controller(GetParam());

    State x0(3);
    x0 << 1.0, 0.0, 0.0;
    const double tf = 1.0;
    auto ref_xf = reference.integrate(x0, tf);

    AD driver;
    AdaptiveConfig cfg;
    cfg.error_order = 4; // DOPRI54
    cfg.def_step_size = 0.01;
    cfg.adaptive = true;
    cfg.use_hairer_wanner_initdt = true;

    Eigen::VectorXd abs_tols(2);
    abs_tols.setConstant(1e-10);
    Eigen::VectorXd rel_tols(2);
    rel_tols.setConstant(1e-10);

    tycho::integrators::ControllerVariant controller;
    switch (GetParam()) {
    case IVPController::I:
        controller = IController{};
        break;
    case IVPController::PI:
        controller = PIController{};
        break;
    case IVPController::PID:
        controller = PIDController{};
        break;
    }
    reset_controller(controller);
    int naccept = 0, nreject = 0;

    std::vector<AD::EventPack> events;
    std::vector<std::vector<Eigen::Vector2d>> eventtimes;
    std::vector<State> states;
    std::vector<typename AD::ODEDeriv> derivs;
    auto noop_update_control = [](State &) {};

    State drv_xf = driver.integrate(ode, x0, tf, cfg, abs_tols, rel_tols, controller, naccept,
                                    nreject, events, eventtimes,
                                    /*storestates=*/false, /*storederivs=*/false,
                                    /*storemidpoints=*/false, states, derivs, noop_update_control);

    EXPECT_GT(naccept, 0) << "driver must actually take steps";
    // Both sides integrate SHO at tol=1e-10 with the same controller kind
    // through the same underlying step math. 1e-9 gives a tight floor
    // consistent with tolerance accumulation over ~tens of steps, and
    // would catch any real drift introduced by future driver edits.
    // Pre-tighten: the loose 1e-6 band could hide step-sizing regressions
    // well above what tol=1e-10 would permit.
    EXPECT_NEAR(drv_xf[0], ref_xf[0], 1e-9);
    EXPECT_NEAR(drv_xf[1], ref_xf[1], 1e-9);
    EXPECT_NEAR(drv_xf[2], ref_xf[2], 1e-12);
}

INSTANTIATE_TEST_SUITE_P(AllControllers, AdaptiveDriverRunTest,
                         ::testing::Values(IVPController::I, IVPController::PI,
                                           IVPController::PID));

// -----------------------------------------------------------------------------
// Zero-interval short-circuit in AdaptiveDriver: returns input, no NaN leak.
// -----------------------------------------------------------------------------
TEST_F(VectorFunctionFixture, AdaptiveDriver_ZeroIntervalReturnsInput) {
    using AD = AdaptiveDriver<IVPAlg::DOPRI54, SHO, double>;
    using State = typename AD::ODEState;

    SHO ode(0.0);
    AD driver;
    AdaptiveConfig cfg;
    cfg.error_order = 4;

    Eigen::VectorXd abs_tols(2);
    abs_tols.setConstant(1e-10);
    Eigen::VectorXd rel_tols(2);
    rel_tols.setConstant(1e-10);

    State x0(3);
    x0 << 1.0, 0.0, 0.0;

    tycho::integrators::ControllerVariant controller = IController{};
    int naccept = 0, nreject = 0;
    std::vector<AD::EventPack> events;
    std::vector<std::vector<Eigen::Vector2d>> eventtimes;
    std::vector<State> states;
    std::vector<typename AD::ODEDeriv> derivs;
    auto noop_update_control = [](State &) {};

    State xf = driver.integrate(ode, x0, 0.0, cfg, abs_tols, rel_tols, controller, naccept, nreject,
                                events, eventtimes, /*storestates=*/true,
                                /*storederivs=*/true, /*storemidpoints=*/false, states, derivs,
                                noop_update_control);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(xf[i], x0[i]);
        EXPECT_TRUE(std::isfinite(xf[i]));
    }
    EXPECT_EQ(naccept, 0);
    EXPECT_EQ(nreject, 0);
    ASSERT_FALSE(states.empty());
    for (const auto &d : derivs) {
        EXPECT_TRUE(d.allFinite());
    }
}

// -----------------------------------------------------------------------------
// Tolerance validation: both-zero abs/rel throws with a component-indexed msg.
// -----------------------------------------------------------------------------
TEST_F(VectorFunctionFixture, AdaptiveDriver_BothZeroTolsThrows) {
    using AD = AdaptiveDriver<IVPAlg::DOPRI54, SHO, double>;
    using State = typename AD::ODEState;

    SHO ode(0.0);
    AD driver;
    AdaptiveConfig cfg;
    cfg.error_order = 4;

    Eigen::VectorXd abs_tols(2);
    abs_tols.setZero();
    Eigen::VectorXd rel_tols(2);
    rel_tols.setZero();

    State x0(3);
    x0 << 1.0, 0.0, 0.0;

    tycho::integrators::ControllerVariant controller = IController{};
    int naccept = 0, nreject = 0;
    std::vector<AD::EventPack> events;
    std::vector<std::vector<Eigen::Vector2d>> eventtimes;
    std::vector<State> states;
    std::vector<typename AD::ODEDeriv> derivs;
    auto noop_update_control = [](State &) {};

    EXPECT_THROW(driver.integrate(ode, x0, 1.0, cfg, abs_tols, rel_tols, controller, naccept,
                                  nreject, events, eventtimes, false, false, false, states, derivs,
                                  noop_update_control),
                 std::invalid_argument);
}
