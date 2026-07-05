///////////////////////////////////////////////////////////////////////////////
// Batch integrate entry points must validate size agreement between x0s and
// tfs (and lfs for integrate_stm2) — INTEGRATORS_REVIEW §3.1. Pre-fix these
// three entry points index tfs[i]/lfs[i] with no check -> OOB read in release.
///////////////////////////////////////////////////////////////////////////////
#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <vector>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

class BatchSizeValidationTest : public VectorFunctionFixture {};

TEST_F(BatchSizeValidationTest, IntegrateRejectsMismatchedTfs) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    std::vector<Eigen::Vector3d> x0s(3, Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::VectorXd tfs(2); // deliberately shorter than x0s
    tfs << 1.0, 2.0;
    EXPECT_THROW(integ.integrate(x0s, tfs), std::invalid_argument);
}

TEST_F(BatchSizeValidationTest, IntegrateStmRejectsMismatchedTfs) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    std::vector<Eigen::Vector3d> x0s(3, Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::VectorXd tfs(2);
    tfs << 1.0, 2.0;
    EXPECT_THROW(integ.integrate_stm(x0s, tfs), std::invalid_argument);
}

TEST_F(BatchSizeValidationTest, IntegrateStm2RejectsMismatchedLfs) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    std::vector<Eigen::Vector3d> x0s(2, Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::VectorXd tfs(2);
    tfs << 1.0, 2.0;
    std::vector<Eigen::Vector3d> lfs(1, Eigen::Vector3d(1.0, 0.0, 0.0)); // wrong size
    EXPECT_THROW(integ.integrate_stm2(x0s, tfs, lfs), std::invalid_argument);
}

TEST_F(BatchSizeValidationTest, IntegrateStm2RejectsMismatchedTfs) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    std::vector<Eigen::Vector3d> x0s(3, Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::VectorXd tfs(2); // shorter than x0s
    tfs << 1.0, 2.0;
    std::vector<Eigen::Vector3d> lfs(3, Eigen::Vector3d(1.0, 0.0, 0.0)); // matches x0s
    EXPECT_THROW(integ.integrate_stm2(x0s, tfs, lfs), std::invalid_argument);
}

TEST_F(BatchSizeValidationTest, MatchedSizesStillWork) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    std::vector<Eigen::Vector3d> x0s(2, Eigen::Vector3d(1.0, 0.0, 0.0));
    Eigen::VectorXd tfs(2);
    tfs << 1.0, 2.0;
    EXPECT_NO_THROW((void)integ.integrate(x0s, tfs));
}
