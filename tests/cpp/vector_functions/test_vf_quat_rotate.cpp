///////////////////////////////////////////////////////////////////////////////
// quat_rotate finite-difference derivative test
//
// Verifies that the VF operator's analytic Jacobian matches a central-
// difference finite-difference Jacobian on a deterministic random input.
// quat_rotate assumes a unit quaternion (scalar-last) — the test seeds
// with a normalized quaternion so the precondition is met.
///////////////////////////////////////////////////////////////////////////////

#include "vf_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho::vf;
using namespace TychoTest;

class QuatRotateFDTest : public VectorFunctionFixture {};

// quat_rotate overload with a constant vector on the RHS.
// The (VF, VF) overload is exercised end-to-end by several examples
// (cart_pole, optimal_docking) so we focus the unit test on the (VF, const)
// path which isolates the quaternion-input derivative handling.
TEST_F(QuatRotateFDTest, ConstantVecOverloadJacobianMatchesFD) {
    auto args = Arguments<4>();
    auto q = args.head<4>();
    Eigen::Vector3d v_const(1.0, -0.5, 2.0);
    auto rotated = quat_rotate(q, v_const);

    GenericFunction<-1, -1> gf(rotated);

    Eigen::VectorXd x(4);
    Eigen::Vector4d qvec(0.1, 0.4, -0.3, 0.8);
    qvec.normalize();
    x = qvec;

    verify_jacobian_fd(gf, x, 1e-5);
}
