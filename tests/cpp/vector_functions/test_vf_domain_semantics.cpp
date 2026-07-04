///////////////////////////////////////////////////////////////////////////////
// Dynamic sub-domain semantics regression tests
//
// Covers VF_REVIEW 1.2 (CwiseSum dynamic sub-domain Jacobian assembly used
// .size() [== 2N] as a loop bound with transposed (i,0)/(i,1) DomainMatrix
// indexing instead of .cols() [== N] and (0,i)/(1,i)), VF_REVIEW 1.4 (dot
// product's sds==0 adjoint-Hessian cross term missing the * adjvars[0]
// scaling applied by its sibling branches), and the VF_REVIEW 3.8 JType1
// typo (aliased Func2's Jacobian type instead of Func1's).
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

class VFDomainSemantics : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// CwiseSum over a dynamic operand with a single covering sub-domain.
//
// Wrapping a static expression in a fully-erased GenericFunction<-1,-1>
// collapses its reported input_domain() to a single [0, IR) block, so the
// resulting CwiseSum's runtime sub_domains is a 2x1 DomainMatrix. The buggy
// loop used sub_domains.size() (== 2) as the bound with transposed (i,0)/
// (i,1) indexing, reading out of the matrix's single column already on the
// first iteration.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFDomainSemantics, DynamicCwiseSumSingleBlockDomain) {
    auto args = Arguments<4>();
    GenericFunction<-1, -1> inner(args * 2.0);
    auto s = inner.sum();

    Eigen::VectorXd x = deterministic_random_vector(4, 21, 1.0, 3.0);
    verify_jacobian_fd(s, x, 1e-5);

    Eigen::VectorXd lm = deterministic_random_vector(1, 22, -1.0, 1.0);
    verify_adjoint_consistency(s, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// CwiseSum over a dynamic operand with two disjoint sub-domains.
//
// Two genuinely dynamic segments (Arguments<-1> parent) covering disjoint
// index ranges [0,2) and [4,2) of a 6-wide input merge (via
// DomainHolder<-1>::set_input_domain) into a real 2-column DomainMatrix —
// the case the buggy loop mishandled beyond the first iteration.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFDomainSemantics, DynamicCwiseSumTwoBlockDomain) {
    auto args = Arguments<-1>(6);
    auto a = args.segment(0, 2);
    auto b = args.segment(4, 2);
    auto combo = 2.0 * a + 3.0 * b;
    auto s = combo.sum();

    Eigen::VectorXd x = deterministic_random_vector(6, 23, 1.0, 3.0);
    verify_jacobian_fd(s, x, 1e-5);
}

///////////////////////////////////////////////////////////////////////////////
// Dot-product adjoint Hessian with a dynamic (fully-erased) left operand.
//
// Exercises FunctionDotProduct_Impl's Func1::InputIsDynamic branch, whose
// sds==0 cross term was missing the * adjvars[0] scaling applied by both
// sibling (dynamic-loop and static-constexpr) branches, plus the JType1
// typo that aliased Func2's Jacobian type for a buffer func1.compute_jacobian
// fills.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFDomainSemantics, DynamicDotProductAdjointHessian) {
    auto args = Arguments<4>();
    GenericFunction<-1, -1> f1(args.head<2>().sin());
    GenericFunction<-1, -1> f2(args.tail<2>() * 2.0);
    auto d = f1.dot(f2);

    Eigen::VectorXd x = deterministic_random_vector(4, 31, 0.5, 1.5);
    Eigen::VectorXd lm = deterministic_random_vector(1, 32, -1.0, 1.0);
    verify_adjoint_hessian_fd(d, x, lm, 1e-4);
}

} // namespace
