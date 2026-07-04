///////////////////////////////////////////////////////////////////////////////
// FunctionVectorProduct_Impl adjoint-Hessian cross-term regression tests
//
// Covers VF_REVIEW 1.3: the Vsize==2 (complex/imaginary) product's
// adjoint-Hessian cross term used the unpatched complex-multiplication
// matrix [[l0,-l1],[l1,l0]] instead of the required [[l0,l1],[l1,-l0]] —
// column-1 signs were wrong whenever both operands depend on the inputs.
// The Vsize==3 (cross) and Vsize==4 (quaternion) products already carry
// correct sign patches; those cases are exercised here purely as
// regression guards.
//
// Both operands in each case are linear in disjoint input blocks, so each
// operand's own Hessian is exactly zero (compute_jacobian_adjointgradient_
// adjointhessian on a linear/Segment function contributes nothing to
// adjhess) and the adjoint Hessian reported by
// compute_jacobian_adjointgradient_adjointhessian is *entirely* the cross
// term under test — an FD mismatch here can only come from the cross-term
// multiplier matrix.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

class VFVectorProductsHessian : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Vsize == 2 — complex/imaginary product (the buggy case).
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFVectorProductsHessian, ImagProductAdjointHessianMatchesFD) {
    auto args = Arguments<4>();
    auto f1 = args.template head<2>();          // linear -> own Hessian is zero
    auto f2 = args.template tail<2>() * 2.0;    // linear -> own Hessian is zero
    FunctionImagProduct<decltype(f1), decltype(f2)> prod(f1, f2);
    EXPECT_EQ(prod.output_rows(), 2);

    Eigen::VectorXd x = deterministic_random_vector(4, 41, 0.5, 2.0);
    Eigen::VectorXd lm = deterministic_random_vector(2, 42, -1.0, 1.0);
    // verify_adjoint_hessian_fd differences the analytic Jacobian, so validate
    // the Jacobian independently first (sibling-suite convention).
    verify_jacobian_fd(prod, x, 1e-5);
    verify_adjoint_hessian_fd(prod, x, lm, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// Vsize == 3 — cross product (regression guard; already correct).
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFVectorProductsHessian, CrossProductAdjointHessianStillCorrect) {
    auto args6 = Arguments<6>();
    auto a = args6.template head<3>();          // linear -> own Hessian is zero
    auto b = args6.template tail<3>() * 2.0;    // linear -> own Hessian is zero
    auto cp = a.cross(b);
    EXPECT_EQ(cp.output_rows(), 3);

    Eigen::VectorXd x6 = deterministic_random_vector(6, 43, 0.5, 2.0);
    Eigen::VectorXd lm3 = deterministic_random_vector(3, 44, -1.0, 1.0);
    verify_jacobian_fd(cp, x6, 1e-5);
    verify_adjoint_hessian_fd(cp, x6, lm3, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// Vsize == 4 — quaternion product (regression guard; already correct).
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFVectorProductsHessian, QuatProductAdjointHessianStillCorrect) {
    auto args8 = Arguments<8>();
    auto q1 = args8.template head<4>();         // linear -> own Hessian is zero
    auto q2 = args8.template tail<4>() * 2.0;   // linear -> own Hessian is zero
    FunctionQuatProduct<decltype(q1), decltype(q2)> qp(q1, q2);
    EXPECT_EQ(qp.output_rows(), 4);

    Eigen::VectorXd x8 = deterministic_random_vector(8, 45, 0.5, 2.0);
    Eigen::VectorXd lm4 = deterministic_random_vector(4, 46, -1.0, 1.0);
    verify_jacobian_fd(qp, x8, 1e-5);
    verify_adjoint_hessian_fd(qp, x8, lm4, 1e-4);
}

} // namespace
