///////////////////////////////////////////////////////////////////////////////
// A direct Stepper::step call with tf == t0 (h == 0) must throw rather than
// silently produce NaN derivatives via the 1/h FSAL scaling
// (INTEGRATORS_REVIEW §3.3). Drivers short-circuit H==0; direct callers do not.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

class ZeroStepGuardTest : public VectorFunctionFixture {};

TEST_F(ZeroStepGuardTest, DirectStepZeroDurationThrows) {
    SHO ode(0.0);
    Stepper<IVPAlg::DOPRI87, SHO, double> stepper;
    Eigen::Vector3d x0(1.0, 0.0, 0.0); // t0 == 0
    Eigen::Vector3d xf, xf_est, xf_mid;
    xf.setZero();
    xf_est.setZero();
    xf_mid.setZero();
    auto noop = [](Eigen::Vector3d &) {};
    // tf == t0 => h == 0
    EXPECT_THROW((stepper.template step<false>(ode, x0, 0.0, xf, xf_est, xf_mid, noop)),
                 std::invalid_argument);
}

TEST_F(ZeroStepGuardTest, NonzeroStepStillWorks) {
    SHO ode(0.0);
    Stepper<IVPAlg::DOPRI87, SHO, double> stepper;
    Eigen::Vector3d x0(1.0, 0.0, 0.0);
    Eigen::Vector3d xf, xf_est, xf_mid;
    xf.setZero();
    xf_est.setZero();
    xf_mid.setZero();
    auto noop = [](Eigen::Vector3d &) {};
    EXPECT_NO_THROW((stepper.template step<false>(ode, x0, 0.1, xf, xf_est, xf_mid, noop)));
}
