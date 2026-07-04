///////////////////////////////////////////////////////////////////////////////
// Input-domain correctness regression tests (VF_REVIEW 1.11 + 1.15)
//
// 1.11: IfElseFunction's INPUT_DOMAIN unioned only the two branches' domains,
// dropping the test predicate's read-set entirely. For a *static*-size
// IfElseFunction this is not runtime-observable: DomainHolder<IR>::input_domain()
// (the static specialization) always reports the trivial full [0,IR) range
// regardless of the compile-time INPUT_DOMAIN typedef, and set_input_domain is
// a no-op. For a *dynamic*-size (IRC=-1) IfElseFunction, DomainHolder<-1> does
// store and merge the exact sub-domains passed to set_input_domain, so the
// drop is directly observable: build an IfElseFunction whose test predicate
// reads a column neither branch reads, and check whether ite.input_domain()
// covers that column.
//
// Also verifies ConditionalStatement/ConstantConditional (conditional.h) and
// GenericConditional/GenericComparative (generic_conditional.h /
// generic_comparative.h) each expose an input_domain() (required so
// IfElseFunction's runtime union can include the predicate at all), and that
// NestedCallAndAppendChain (call_and_append.h) still compiles for the only
// current (fully static) use pattern after removing its too-narrow INPUT_DOMAIN
// typedef and adding a compile-time-sizes static_assert (VF_REVIEW 1.15).
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

class VFInputDomains : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// VF_REVIEW 1.11a — predicate types expose input_domain()
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFInputDomains, ConditionalStatementDomainIsUnionOfOperands) {
    // Dynamic (IRC=-1) Arguments so each operand's coeff(i) reports a real
    // narrow sub-domain ({i,1}) rather than the trivial full [0,IR) block a
    // static Arguments would give — otherwise a "return lhs only" bug would
    // still cover both columns and the union wouldn't be discriminated.
    auto args = Arguments<-1>(3);
    auto lhs = args.coeff(2); // reads column 2 only
    auto rhs = args.coeff(0); // reads column 0 only

    ConditionalStatement cond(lhs, ConditionalFlags::GreaterThanFlag, rhs);
    DomainMatrix dmn = cond.input_domain();

    bool covers_col0 = false, covers_col2 = false, covers_col1 = false;
    for (int i = 0; i < dmn.cols(); i++) {
        if (dmn(0, i) <= 0 && 0 < dmn(0, i) + dmn(1, i))
            covers_col0 = true;
        if (dmn(0, i) <= 1 && 1 < dmn(0, i) + dmn(1, i))
            covers_col1 = true;
        if (dmn(0, i) <= 2 && 2 < dmn(0, i) + dmn(1, i))
            covers_col2 = true;
    }
    EXPECT_TRUE(covers_col0); // rhs operand — fails if union drops rhs
    EXPECT_TRUE(covers_col2); // lhs operand — fails if union drops lhs
    EXPECT_FALSE(covers_col1); // neither operand reads col 1 — union must not invent it
}

TEST_F(VFInputDomains, ConstantConditionalDomainIsEmpty) {
    ConstantConditional cc(5, true);
    DomainMatrix dmn = cc.input_domain();
    EXPECT_EQ(dmn.cols(), 0);
}

TEST_F(VFInputDomains, GenericConditionalDomainCoversInputRows) {
    auto args = Arguments<2>();
    auto lhs = args.coeff<0>();
    auto rhs = args.coeff<1>();
    ConditionalStatement cond(lhs, ConditionalFlags::GreaterThanFlag, rhs);
    GenericConditional<2> gc(cond);

    DomainMatrix dmn = gc.input_domain();
    // Erased predicate: conservative full-range domain.
    int total = 0;
    for (int i = 0; i < dmn.cols(); i++)
        total += dmn(1, i);
    EXPECT_EQ(total, gc.input_rows());
}

TEST_F(VFInputDomains, GenericComparativeDomainCoversInputRows) {
    auto args = Arguments<2>();
    auto lhs = args.coeff<0>();
    auto rhs = args.coeff<1>();
    ConditionalStatement cond(lhs, ConditionalFlags::LessThanFlag, rhs);
    GenericComparative<2> gc(cond);

    DomainMatrix dmn = gc.input_domain();
    int total = 0;
    for (int i = 0; i < dmn.cols(); i++)
        total += dmn(1, i);
    EXPECT_EQ(total, gc.input_rows());
}

///////////////////////////////////////////////////////////////////////////////
// VF_REVIEW 1.11b — IfElseFunction's runtime domain union includes the
// predicate. Only observable for a dynamic-size (IRC=-1) IfElseFunction; see
// the file-level comment above for why the static case can't discriminate.
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFInputDomains, IfElseDynamicDomainIncludesPredicate) {
    // Dynamic (IRC=-1) Arguments so that Segment(...).input_domain() reports
    // real, non-trivial sub-domains at runtime (DomainHolder<-1>), unlike the
    // static case where DomainHolder<IR>::input_domain() always trivially
    // returns the full [0,IR) range.
    auto args = Arguments<-1>(6);

    // coeff(i) returns a statically-scalar (ORC=1) Segment<-1,1,-1>, which is
    // what operator> and IfElseFunction's scalar branches require.
    auto pred_scl = args.coeff(4);    // predicate reads column 4 only
    auto true_branch = args.coeff(0); // true branch reads column 0 only
    auto false_branch = args.coeff(2); // false branch reads column 2 only

    auto cond = pred_scl > 0.0; // ConditionalStatement<Segment, Constant>, IRC=-1

    auto ite = IfElseFunction(cond, true_branch, false_branch);
    ASSERT_EQ(ite.IRC, -1) << "test requires a dynamic-size IfElseFunction to be "
                              "runtime-observable -- see file-level comment";
    EXPECT_EQ(ite.input_rows(), 6);

    DomainMatrix dmn = ite.input_domain();
    bool covers_col4 = false;
    for (int i = 0; i < dmn.cols(); i++)
        if (dmn(0, i) <= 4 && 4 < dmn(0, i) + dmn(1, i))
            covers_col4 = true;

    // VF_REVIEW 1.11: pre-fix, IfElseFunction's set_input_domain call only
    // passed {true_func_.input_domain(), false_func_.input_domain()} (columns
    // 0 and 2), never test_func_.input_domain() (column 4) -- so this would
    // fail pre-fix regardless of what ConditionalStatement::input_domain()
    // itself returns, since it was never part of the union at all.
    EXPECT_TRUE(covers_col4);

    // Functional sanity check: still computes correctly post-fix.
    Eigen::VectorXd x = deterministic_random_vector(6, 9001, -5.0, 5.0);
    x[4] = 1.0; // predicate true: x[4] > 0
    Eigen::VectorXd fx(1);
    fx.setZero();
    ite.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], x[0]);

    x[4] = -1.0; // predicate false: x[4] > 0 is false
    ite.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], x[2]);
}

///////////////////////////////////////////////////////////////////////////////
// VF_REVIEW 1.15 — NestedCallAndAppendChain compile-time-sizes static_assert.
// This is a compile-time-only regression: the only production use
// (rk_steppers.h) is fully static, so this test just re-exercises a small
// fully-static chain to confirm the static_assert doesn't reject valid
// static-size usage (would be a compile error, not a runtime failure, if
// broken).
///////////////////////////////////////////////////////////////////////////////

namespace {
struct DoubleIt : VectorFunction<DoubleIt, 2, 2> {
    using Base = VectorFunction<DoubleIt, 2, 2>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = 2.0 * x;
    }
};
struct SumOuter : VectorFunction<SumOuter, 4, 1> {
    using Base = VectorFunction<SumOuter, 4, 1>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = x.sum();
    }
};
} // namespace

TEST_F(VFInputDomains, StaticNestedCallAndAppendChainStillCompiles) {
    DoubleIt inner;
    SumOuter outer;
    NestedCallAndAppendChain chain(outer, inner);

    EXPECT_EQ(chain.input_rows(), 2);
    EXPECT_EQ(chain.output_rows(), 1);

    Eigen::Vector2d x(3.0, 4.0);
    Eigen::VectorXd fx(1);
    chain.compute(x, fx);
    // Chain vector = [x0, x1, 2*x0, 2*x1]; SumOuter sums all 4 entries.
    EXPECT_DOUBLE_EQ(fx[0], x[0] + x[1] + 2.0 * x[0] + 2.0 * x[1]);

    // Full domain default: input_domain() should cover [0, 2).
    DomainMatrix dmn = chain.input_domain();
    int total = 0;
    for (int i = 0; i < dmn.cols(); i++)
        total += dmn(1, i);
    EXPECT_EQ(total, chain.input_rows());
}

} // namespace
