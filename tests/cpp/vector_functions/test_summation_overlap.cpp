///////////////////////////////////////////////////////////////////////////////
// Summation overlap regression tests (VF_REVIEW 1.10)
//
// The sum-of-segments fast paths in summation.h (TwoFunctionSum_Impl and
// MultiFunctionSum_Impl) accumulate every operand's contribution out of one
// combined Jacobian temp. Each segment operand reads the *diagonal* of its own
// column block from that combined temp and writes it into the target's columns.
// When two operands share the same segment start (an identical column range),
// the combined temp's diagonal already holds the *sum* of both scale factors,
// yet each operand applies it in full — double-counting the shared columns
// (e.g. a*x.head(3) + b*x.head(3) yields 2(a+b) instead of (a+b)).
//
// (Partial column overlap between two equal-size segments with *different*
// starts does NOT double-count: each segment only touches the diagonal of its
// own block, so distinct starts map to distinct (row,col) entries and the
// result is correct. Only a shared range collides, so the observable bug needs
// identical/overlapping ranges.)
//
// The disjointness gate is a compile-time `segments_disjoint_` derived from each
// operand's `INPUT_DOMAIN::sub_domains`, NOT the runtime `input_domain()` (which
// for a static-size function always spans the whole [0, IR) block and cannot
// distinguish sub-ranges). The fast path is disabled only on a *provable*
// overlap of concrete compile-time ranges; runtime-start segments (start == -1)
// cannot be proven to overlap and keep the fast path, matching the pre-existing
// behavior. Each test asserts the expected `segments_disjoint_` value so it
// verifies the gating decision, not merely the numeric result.
//
// These tests compose the segment sum with a nonlinear outer op (.norm()),
// which routes the Jacobian through right_jacobian_product / accumulate_jacobian
// so the fast path is exercised.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

class SummationOverlap : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// TwoFunctionSum_Impl: identical (fully overlapping) scaled segments.
//   a*x.head(3) + b*x.head(3)  -> double-counts the shared [0,3) block.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SummationOverlap, OverlappingScaledSegmentsInComposition) {
    auto args = Arguments<5>();
    auto s = args.template head<3>() * 2.0 + args.template head<3>() * 3.0; // identical ranges
    static_assert(decltype(s)::is_sum_of_segments,
                  "test must exercise the TwoFunctionSum sum-of-segments fast path");
    static_assert(!decltype(s)::segments_disjoint_,
                  "identical ranges must be gated OFF the fast path (route to base)");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(5, 51, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);

    Eigen::VectorXd lm = deterministic_random_vector(1, 52, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// MultiFunctionSum_Impl: repeated segment among three addends.
//   Flattening rule: f2 + TwoFunctionSum(f1a,f1b) -> MultiFunctionSum<f1a,f1b,f2>.
//   Here a + (b + c) gives operands {b, c, a} with c and a sharing [0,3).
///////////////////////////////////////////////////////////////////////////////

TEST_F(SummationOverlap, MultiFunctionSumRepeatedSegment) {
    auto args = Arguments<6>();
    auto a = args.template head<3>() * 2.0;
    auto b = args.template tail<3>() * 3.0;
    auto c = args.template head<3>() * 4.0; // same range as a
    auto s = a + (b + c);                    // -> MultiFunctionSum<b, c, a>
    static_assert(decltype(s)::IsSumofSegments,
                  "test must exercise the MultiFunctionSum all-segments fast path");
    static_assert(!decltype(s)::segments_disjoint_,
                  "repeated range must be gated OFF the fast path (route to base)");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(6, 61, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);

    Eigen::VectorXd lm = deterministic_random_vector(1, 62, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// TwoFunctionSum_Impl func1_is_sumordiff branch: nested sum + repeated segment.
//   (a*x.head(3) + b*x.tail(3)) + c*x.head(3)  -> outer func2 (c) shares a's range.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SummationOverlap, NestedSumRepeatedSegment) {
    auto args = Arguments<6>();
    auto a = args.template head<3>() * 2.0;
    auto b = args.template tail<3>() * 3.0;
    auto c = args.template head<3>() * 4.0; // same range as a
    auto s = (a + b) + c;                    // TwoFunctionSum<TwoFunctionSum<a,b>, c>
    static_assert(!decltype(s)::is_sum_of_segments && decltype(s)::func1_is_sumordiff &&
                      decltype(s)::func2_is_segment,
                  "test must exercise the func1_is_sumordiff && func2_is_segment fast path");
    static_assert(std::decay_t<decltype(s.func1)>::is_sum_of_segments,
                  "inner sum must itself be a sum-of-segments to hit the sub-path");
    static_assert(!decltype(s)::segments_disjoint_,
                  "outer func2 sharing an inner segment range must gate OFF the fast path");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(6, 63, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);
    Eigen::VectorXd lm = deterministic_random_vector(1, 632, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// Disjoint two-segment sum must still take (and correctly execute) the fast path.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SummationOverlap, DisjointSegmentsStillCorrect) {
    auto args = Arguments<6>();
    auto s = args.template head<3>() * 2.0 + args.template tail<3>() * 3.0; // disjoint -> fast path
    static_assert(decltype(s)::is_sum_of_segments,
                  "test must exercise the sum-of-segments fast path");
    static_assert(decltype(s)::segments_disjoint_,
                  "compile-time disjoint ranges must KEEP the fast path (no perf regression)");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(6, 64, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);
    Eigen::VectorXd lm = deterministic_random_vector(1, 65, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// Disjoint multi-segment sum must still take the MultiFunctionSum fast path.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SummationOverlap, DisjointMultiSegmentsStillCorrect) {
    auto args = Arguments<6>();
    auto a = args.template segment<2>(0) * 2.0;
    auto b = args.template segment<2>(2) * 3.0;
    auto c = args.template segment<2>(4) * 4.0;
    auto s = a + (b + c); // MultiFunctionSum over three disjoint segments
    static_assert(decltype(s)::IsSumofSegments,
                  "test must exercise the MultiFunctionSum all-segments fast path");
    // Runtime-start segments cannot be proven to overlap at compile time, so the
    // fast path is kept (matching pre-existing behavior; correct when disjoint).
    static_assert(decltype(s)::segments_disjoint_,
                  "runtime-start segments must keep the fast path (overlap unprovable)");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(6, 66, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);
    // NOTE: only the Jacobian is checked here. The adjoint gradient/Hessian of a
    // sum of *runtime-start* segments (segment<N>(k), whose compile-time
    // INPUT_DOMAIN start is -1) are computed incorrectly by the domain-aware KKT
    // assembly: CompositeDomain::contains_elem treats the unresolved -1 start as a
    // literal offset, collapsing the covered range to ~index 0, so the Hessian
    // populates only row 0. This is a PRE-EXISTING domain-machinery bug, distinct
    // from the §1.10 compile-time overlap double-count fixed here, and reachable
    // only via runtime-start segments. The obvious global fix (treat a -1 start as
    // full coverage) regresses real optimizations (multi-spacecraft continuation
    // diverges) by widening KKT sparsity, so it needs a targeted fix + PSIOPT
    // review out of scope for this PR. Tracked as a backlog item; the disjoint
    // *compile-time*-start Hessian path is covered by DisjointSegmentsStillCorrect.
}
