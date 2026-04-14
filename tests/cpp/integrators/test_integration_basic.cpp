///////////////////////////////////////////////////////////////////////////////
// Basic integration tests
//
// Tests fundamental integrator correctness using the Simple Harmonic
// Oscillator and Brachistochrone ODEs.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, SHOKnownSolutionAtPi) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0; // x=1, v=0, t=0

    auto xf = integ.integrate(x0, std::numbers::pi);
    // At t=pi: x=cos(pi)=-1, v=-sin(pi)=0
    EXPECT_NEAR(xf[0], -1.0, 1e-10);
    EXPECT_NEAR(xf[1], 0.0, 1e-10);
}

TEST_F(IntegratorTest, SHOKnownSolutionAt2Pi) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf = integ.integrate(x0, 2.0 * std::numbers::pi);
    // At t=2pi: x=cos(2pi)=1, v=-sin(2pi)=0
    EXPECT_NEAR(xf[0], 1.0, 1e-9);
    EXPECT_NEAR(xf[1], 0.0, 1e-9);
}

TEST_F(IntegratorTest, ForwardBackwardReversibility) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    // Integrate forward to t=1
    auto xf = integ.integrate(x0, 1.0);
    // Integrate backward from t=1 to t=0
    auto xr = integ.integrate(xf, 0.0);

    EXPECT_NEAR(xr[0], x0[0], 1e-9);
    EXPECT_NEAR(xr[1], x0[1], 1e-9);
}

TEST_F(IntegratorTest, BrachistochroneSmokeTest) {
    BrachODE ode(9.81);
    // Need to set control for integration
    Eigen::VectorXd u(1);
    u << std::numbers::pi / 4.0; // constant control
    Integrator<BrachODE> integ(ode, 0.01, u);
    integ.set_abs_tol(1e-12);

    Eigen::VectorXd x0(5);
    x0 << 0, 10, 0.1, 0, std::numbers::pi / 4.0;

    auto xf = integ.integrate(x0, 1.0);
    EXPECT_EQ(xf.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(xf[i])) << "Component " << i << " not finite";
    }
}

// IVPAlg::DOPRI5 and IVPAlg::RK4Classic are internal template-dispatch tags
// used by the adaptive stepper and constexpr rk_steppers branches; the
// Integrator constructor's set_method call must reject them at runtime with a
// diagnostic that names the user-facing alternatives. Pins the
// expose-then-throw contract so a future template refactor cannot silently
// change the surface-level behavior.
TEST_F(IntegratorTest, SetMethodRejectsInternalDispatchTags) {
    SHO ode(0.0);

    try {
        Integrator<SHO> bad(ode, IVPAlg::DOPRI5, 0.01);
        FAIL() << "Integrator(..., DOPRI5, ...) should have thrown";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("DOPRI5"), std::string::npos) << msg;
        EXPECT_NE(msg.find("DOPRI54"), std::string::npos)
            << "diagnostic should direct the user to DOPRI54: " << msg;
    }

    try {
        Integrator<SHO> bad(ode, IVPAlg::RK4Classic, 0.01);
        FAIL() << "Integrator(..., RK4Classic, ...) should have thrown";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("RK4Classic"), std::string::npos) << msg;
        EXPECT_NE(msg.find("DOPRI54"), std::string::npos)
            << "diagnostic should direct the user to a runtime-selectable method: " << msg;
    }
}
