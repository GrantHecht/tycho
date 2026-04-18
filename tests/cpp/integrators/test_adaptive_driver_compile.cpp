///////////////////////////////////////////////////////////////////////////////
// AdaptiveDriver compile-only sanity test
//
// Verifies the scaffold in adaptive_driver.h instantiates with the SP3
// controller types. Not production-used — the production integrate() path
// lives in Integrator<DODE>. Keeping this compile test prevents bitrot.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>

#include "tycho/detail/integrators/adaptive_driver.h"
#include <gtest/gtest.h>

using namespace tycho::integrators;

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
