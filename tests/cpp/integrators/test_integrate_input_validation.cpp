///////////////////////////////////////////////////////////////////////////////
// Integrator entry-point input validation:
//   §1.5 — non-finite tf / x0 rejected immediately (both the scalar
//          AdaptiveDriver path and the batch ParallelDriver path), not ground
//          through max_steps.
//   §1.4 — in fixed-step mode the numsteps reserve is clamped to max_steps, but
//          h is computed from the UNCLAMPED nominal so the requested step size
//          is honored: an oversized fixed-step request hits the max_steps guard
//          promptly (bounded reserve) instead of silently integrating coarser.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

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

// §1.5 on the batch ParallelDriver path (default vectorize_batch_calls_=true):
// a non-finite tf or x0 on any lane must be rejected per-lane.
TEST_F(IntegrateInputValidationTest, BatchNonFiniteLaneRejected) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.vectorize_batch_calls_ = true; // exercise ParallelDriver
    std::vector<Eigen::Vector3d> x0s(3, Eigen::Vector3d(1.0, 0.0, 0.0));

    Eigen::VectorXd tfs_nan(3);
    tfs_nan << 1.0, std::numeric_limits<double>::quiet_NaN(), 1.0;
    EXPECT_THROW((void)integ.integrate(x0s, tfs_nan), std::invalid_argument);

    Eigen::VectorXd tfs_ok(3);
    tfs_ok << 1.0, 1.0, 1.0;
    x0s[2] = Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0.0, 0.0);
    EXPECT_THROW((void)integ.integrate(x0s, tfs_ok), std::invalid_argument);
}

// §1.4: fixed-step mode (adaptive_ = false) with a tiny step over a unit span
// must throw max_steps promptly. If the clamp had coarsened h (the pre-fix bug),
// the run would instead complete in ~max_steps coarse steps without throwing.
TEST_F(IntegrateInputValidationTest, FixedStepNumstepsClampThrowsMaxSteps) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_initial_step_size(1.0e-6); // tiny fixed step; disables HW auto-initdt
    integ.adaptive_ = false;             // true fixed-step mode
    integ.set_max_steps(1000);
    Eigen::Vector3d x0(1.0, 0.0, 0.0);
    try {
        (void)integ.integrate(x0, 1.0); // span 1.0 / 1e-6 => ~1e6 steps >> 1000
        FAIL() << "expected a max_steps runtime_error for the oversized fixed-step run";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("max_steps"), std::string::npos)
            << "diagnostic should mention max_steps; got: " << e.what();
    }
}

// §1.4 on the batch ParallelDriver path: the fixed-step max_steps guard lives in
// BOTH drivers, but the scalar test above only exercises AdaptiveDriver. Without
// the mirror nacc[itmp]++ in parallel_driver.h, a batch fixed-step run counts no
// steps and grinds out its full nominal count instead of throwing. This pins the
// batch guard (delete the mirror increment and this test alone catches it).
TEST_F(IntegrateInputValidationTest, BatchFixedStepNumstepsClampThrowsMaxSteps) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_initial_step_size(1.0e-6);  // tiny fixed step; disables HW auto-initdt
    integ.adaptive_ = false;              // true fixed-step mode
    integ.vectorize_batch_calls_ = true;  // exercise ParallelDriver (default path)
    integ.set_max_steps(1000);
    std::vector<Eigen::Vector3d> x0s(1, Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::VectorXd tfs(1);
    tfs << 1.0; // span 1.0 / 1e-6 => ~1e6 steps >> 1000
    try {
        (void)integ.integrate(x0s, tfs);
        FAIL() << "ParallelDriver fixed-step run should throw max_steps, not grind to completion";
    } catch (const std::runtime_error &e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("max_steps"), std::string::npos) << msg;
        EXPECT_NE(msg.find("ParallelDriver"), std::string::npos) << msg;
    }
}
