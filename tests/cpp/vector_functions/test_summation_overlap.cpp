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
// overlap of concrete compile-time ranges. A runtime-start segment
// (`segment<N>(k)` with runtime `k`) has an unknown compile-time offset;
// `SingleDomain` widens its descriptor to the full `[0, IR)` range, so such a
// segment is (correctly) reported as overlapping everything and its sums route
// to the base path. Each test asserts the expected `segments_disjoint_` value so
// it verifies the gating decision, not merely the numeric result.
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
    auto s = a + (b + c); // MultiFunctionSum over three runtime-start segments
    static_assert(decltype(s)::IsSumofSegments,
                  "test must exercise the MultiFunctionSum all-segments path");
    // A runtime-start segment (segment<N>(k) with runtime k) has compile-time
    // INPUT_DOMAIN start -1; SingleDomain now widens that to the full [0,IR)
    // range (see function_domains.h), so the disjointness gate cannot prove
    // these disjoint and routes the sum to the correct base path -- rather than
    // the fast path, which double-counts and (via the old contains_elem -1 bug)
    // produced a row-0-only, and even out-of-bounds, adjoint Hessian.
    static_assert(!decltype(s)::segments_disjoint_,
                  "runtime-start segments report a full-range domain -> route to base");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(6, 66, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);
    Eigen::VectorXd lm = deterministic_random_vector(1, 662, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// Partial-coverage runtime-start segments: the untouched input columns must
// stay zero in the adjoint Hessian (regression guard for the SingleDomain
// full-range widening -- it must widen to the true [0,IR), not over-write
// columns the function never reads, and not leave the -1 leaf OOB).
///////////////////////////////////////////////////////////////////////////////

TEST_F(SummationOverlap, PartialCoverageRuntimeSegmentsHessian) {
    auto args = Arguments<8>();                 // cols 2,5 read by no segment
    auto a = args.template segment<2>(0) * 2.0; // cols 0,1
    auto b = args.template segment<2>(3) * 3.0; // cols 3,4
    auto c = args.template segment<2>(6) * 4.0; // cols 6,7
    auto s = a + (b + c);                        // equal-size (2) segments -> valid sum
    static_assert(decltype(s)::IsSumofSegments,
                  "test must exercise the MultiFunctionSum all-segments path");
    auto f = s.norm();

    Eigen::VectorXd x = deterministic_random_vector(8, 71, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-5);
    Eigen::VectorXd lm = deterministic_random_vector(1, 72, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4);
}
