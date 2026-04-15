///////////////////////////////////////////////////////////////////////////////
// cwise_product (Eigen-overload) finite-difference derivative test
//
// Verifies the VF operator's analytic Jacobian against a central-difference
// Jacobian for both argument orders (VF * VectorXd and VectorXd * VF).
///////////////////////////////////////////////////////////////////////////////

#include "vf_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho::vf;
using namespace TychoTest;

class CwiseProductFDTest : public VectorFunctionFixture {};

TEST_F(CwiseProductFDTest, VFTimesEigenVectorJacobianMatchesFD) {
    auto args = Arguments<4>();
    Eigen::VectorXd v(4);
    v << 2.0, -1.5, 0.75, 3.0;
    auto out = cwise_product(args, v);
    GenericFunction<-1, -1> gf(out);

    Eigen::VectorXd x = deterministic_random_vector(4, 402, -5.0, 5.0);
    verify_jacobian_fd(gf, x, 1e-6);
}

TEST_F(CwiseProductFDTest, EigenVectorTimesVFJacobianMatchesFD) {
    auto args = Arguments<4>();
    Eigen::VectorXd v(4);
    v << -2.0, 0.5, 1.25, -3.0;
    auto out = cwise_product(v, args);
    GenericFunction<-1, -1> gf(out);

    Eigen::VectorXd x = deterministic_random_vector(4, 403, -3.0, 3.0);
    verify_jacobian_fd(gf, x, 1e-6);
}
