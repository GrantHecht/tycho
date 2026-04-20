///////////////////////////////////////////////////////////////////////////////
// STM coverage for every user-selectable RK method.
//
// Previously, integrate_stm was only exercised on DOPRI87 (via
// test_integration_stm.cpp). A tableau / augmented-state typo in any of the
// 6 new SP2 methods (Tsit5, BS3, BS5, Vern7, Vern8, Vern9) would not have
// surfaced. This suite pins the SHO rotation-block STM against the analytic
// matrix [[cos(t), sin(t)], [-sin(t), cos(t)]] for every method.
//
// Tolerances: SHO at atol/rtol=1e-12 admits near-machine-epsilon accuracy
// for higher-order methods; relaxed for BS3 (order 3).
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <cmath>
#include <string>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {

struct STMMethodExpect {
    IVPAlg method;
    const char *name;
    double tolerance; // per-component tol on the 2x2 rotation block
};

std::string param_name(const ::testing::TestParamInfo<STMMethodExpect> &info) {
    return info.param.name;
}

} // namespace

class STMAllMethodsTest : public ::testing::TestWithParam<STMMethodExpect> {};

TEST_P(STMAllMethodsTest, SHORotationBlockMatchesAnalytic) {
    const auto &p = GetParam();

    SHO ode(0.0);
    Integrator<SHO> integ(ode, p.method, 0.01);
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 1.0;

    auto [xf, stm] = integ.integrate_stm(x0, tf);

    const double ct = std::cos(tf);
    const double st = std::sin(tf);

    EXPECT_NEAR(stm(0, 0), ct, p.tolerance) << p.name << ": dx/dx0";
    EXPECT_NEAR(stm(0, 1), st, p.tolerance) << p.name << ": dx/dv0";
    EXPECT_NEAR(stm(1, 0), -st, p.tolerance) << p.name << ": dv/dx0";
    EXPECT_NEAR(stm(1, 1), ct, p.tolerance) << p.name << ": dv/dv0";
}

INSTANTIATE_TEST_SUITE_P(
    AllUserSelectableMethods, STMAllMethodsTest,
    ::testing::Values(STMMethodExpect{IVPAlg::DOPRI54, "DOPRI54", 1e-8},
                      STMMethodExpect{IVPAlg::DOPRI87, "DOPRI87", 1e-8},
                      STMMethodExpect{IVPAlg::Tsit5, "Tsit5", 1e-8},
                      STMMethodExpect{IVPAlg::BS3, "BS3", 1e-5},
                      STMMethodExpect{IVPAlg::BS5, "BS5", 1e-8},
                      STMMethodExpect{IVPAlg::Vern7, "Vern7", 1e-8},
                      STMMethodExpect{IVPAlg::Vern8, "Vern8", 1e-8},
                      STMMethodExpect{IVPAlg::Vern9, "Vern9", 1e-8}),
    param_name);
