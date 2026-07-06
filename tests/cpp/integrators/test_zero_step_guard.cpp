///////////////////////////////////////////////////////////////////////////////
// A direct Stepper::step call with tf == t0 (h == 0) must throw rather than
// silently produce NaN derivatives via the 1/h FSAL scaling
// (INTEGRATORS_REVIEW §3.3). Drivers short-circuit H==0; direct callers do not.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <string>
#include <vector>

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

///////////////////////////////////////////////////////////////////////////////
// Zero-progress stall (§3.3 driver-level): at large |t| the controller shrinks
// h below the ULP of t, so tnext = t + h rounds back to t and the step would
// run with an effective h == 0. Both drivers must throw a specific, actionable
// diagnostic (mentioning max_steps + "underflowed") rather than grinding to the
// max_steps cap, poisoning the FSAL cache, or silently self-healing.
///////////////////////////////////////////////////////////////////////////////
TEST_F(ZeroStepGuardTest, AdaptiveDriverUnderflowStallThrows) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    // t0 = 1e16: ULP(1e16) ~ 2, and the SHO first step (~5e-2) rounds away, so
    // tnext == t0 on the first step. H = 100 != 0 (not the zero-interval case).
    Eigen::Vector3d x0(1.0, 0.0, 1.0e16);
    const double tf = 1.0e16 + 100.0;
    try {
        (void)integ.integrate(x0, tf);
        FAIL() << "expected a zero-progress stall throw at large |t|";
    } catch (const std::runtime_error &e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("underflowed"), std::string::npos) << msg;
        EXPECT_NE(msg.find("max_steps"), std::string::npos) << msg;
    }
}

TEST_F(ZeroStepGuardTest, ParallelDriverUnderflowStallThrows) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.vectorize_batch_calls_ = true; // exercise ParallelDriver (the default path)
    std::vector<Eigen::Vector3d> x0s(1, Eigen::Vector3d(1.0, 0.0, 1.0e16));
    Eigen::VectorXd tfs(1);
    tfs << 1.0e16 + 100.0;
    try {
        (void)integ.integrate(x0s, tfs);
        FAIL() << "ParallelDriver should throw the same stall diagnostic as AdaptiveDriver";
    } catch (const std::runtime_error &e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("underflowed"), std::string::npos) << msg;
    }
}
