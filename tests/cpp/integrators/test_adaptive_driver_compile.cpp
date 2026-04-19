///////////////////////////////////////////////////////////////////////////////
// AdaptiveDriver compile-only sanity test
//
// Verifies the scaffold in adaptive_driver.h instantiates with the SP3
// controller types. Not production-used — the production integrate() path
// lives in Integrator<DODE>. Keeping this compile test prevents bitrot.
// The `AdaptiveDriverRunTest` below additionally forces the full
// `integrate()` template body to instantiate so the `check_crossings`
// call at adaptive_driver.h:198 type-checks against the
// `std::vector<Vector1<double>>` buffer types.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <tycho/tycho.h>

#include "tycho/detail/integrators/adaptive_driver.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

TEST(AdaptiveDriverCompileTest, InstantiatesWithIController) {
    using Kep = tycho::astro::Kepler;
    AdaptiveDriver<IVPAlg::DOPRI54, IController, Kep, double> d;
    (void)d;
    SUCCEED();
}

TEST(AdaptiveDriverCompileTest, InstantiatesWithPIController) {
    using Kep = tycho::astro::Kepler;
    AdaptiveDriver<IVPAlg::DOPRI54, PIController, Kep, double> d;
    (void)d;
    SUCCEED();
}

TEST(AdaptiveDriverCompileTest, InstantiatesWithPIDController) {
    using Kep = tycho::astro::Kepler;
    AdaptiveDriver<IVPAlg::DOPRI54, PIDController, Kep, double> d;
    (void)d;
    SUCCEED();
}

// Runtime test — forces `AdaptiveDriver::integrate()` template body
// instantiation, which is the only way the `check_crossings` signature
// mismatch surfaces.
class AdaptiveDriverRunTest : public VectorFunctionFixture {};

TEST_F(AdaptiveDriverRunTest, IntegrateSHOMatchesSerialIntegrator) {
    using AD = AdaptiveDriver<IVPAlg::DOPRI54, IController, SHO, double>;
    using State = typename AD::ODEState;

    SHO ode(0.0);

    // Reference trajectory via the production Integrator.
    Integrator<SHO> reference(ode, IVPAlg::DOPRI54, 0.01);
    reference.set_abs_tol(1e-10);
    reference.set_rel_tol(1e-10);

    State x0(3);
    x0 << 1.0, 0.0, 0.0;
    const double tf = 1.0;
    auto ref_xf = reference.integrate(x0, tf);

    // Drive the AdaptiveDriver through its `integrate()` entry.
    AD driver;
    driver.abs_tols_.setConstant(ode.x_vars(), 1e-10);
    driver.rel_tols_.setConstant(ode.x_vars(), 1e-10);
    driver.def_step_size_ = 0.01;

    std::vector<AD::EventPack> events;
    std::vector<std::vector<Eigen::Vector2d>> eventtimes;
    std::vector<State> states;
    std::vector<typename AD::ODEDeriv> derivs;
    auto noop_update_control = [](State &) {};

    State drv_xf = driver.integrate(ode, x0, tf, events, eventtimes,
                                    /*storestates=*/false, /*storederivs=*/false,
                                    /*storemidpoints=*/false, states, derivs,
                                    noop_update_control);

    // Loose tolerance: the AdaptiveDriver uses a fresh controller and
    // default initial-dt heuristics that differ slightly from the
    // production Integrator. Correctness, not bit-identity.
    EXPECT_NEAR(drv_xf[0], ref_xf[0], 1e-6);
    EXPECT_NEAR(drv_xf[1], ref_xf[1], 1e-6);
}
