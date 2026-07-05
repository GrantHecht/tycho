///////////////////////////////////////////////////////////////////////////////
// Non-finite tf / x0 must be rejected immediately (INTEGRATORS_REVIEW §1.5),
// not ground through max_steps. (§1.4 numsteps-reserve clamp is implemented in
// both drivers and exercised by every integration test's entry path; it has no
// robust standalone red-green trigger because fixed-step mode is not publicly
// toggleable, so it is covered by inspection + no-crash across the suite.)
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <limits>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

class IntegrateInputValidationTest : public VectorFunctionFixture {};

TEST_F(IntegrateInputValidationTest, NonFiniteTfRejected) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    Eigen::Vector3d x0(1.0, 0.0, 0.0);
    double nan_tf = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)integ.integrate(x0, nan_tf), std::invalid_argument);
    double inf_tf = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)integ.integrate(x0, inf_tf), std::invalid_argument);
}

TEST_F(IntegrateInputValidationTest, NonFiniteX0Rejected) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    Eigen::Vector3d x0(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
    EXPECT_THROW((void)integ.integrate(x0, 1.0), std::invalid_argument);
}

TEST_F(IntegrateInputValidationTest, FiniteInputStillIntegrates) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    Eigen::Vector3d x0(1.0, 0.0, 0.0);
    EXPECT_NO_THROW((void)integ.integrate(x0, 1.0));
}
