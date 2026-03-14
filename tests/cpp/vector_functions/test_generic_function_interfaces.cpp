///////////////////////////////////////////////////////////////////////////////
// GenericFunction type erasure tests — Solver interfaces & Conditionals
//
// Tests ConstraintInterface/ObjectiveInterface packing,
// GenericConditional/Comparative.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Solver interfaces
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, PackIntoConstraintInterface) {
    auto args = Arguments<5>();
    auto scaled = 2.0 * args;
    GenericFunction<-1, -1> gf(scaled);
    ConstraintInterface ci(gf);

    EXPECT_EQ(ci.IRows(), 5);
    EXPECT_EQ(ci.ORows(), 5);
}

TEST_F(GenericFunctionTest, PackIntoObjectiveInterface) {
    auto args = Arguments<3>();
    auto n = args.norm();
    GenericFunction<-1, 1> gf(n);
    ObjectiveInterface oi(gf);

    EXPECT_EQ(oi.IRows(), 3);
    EXPECT_EQ(oi.ORows(), 1);
}

TEST_F(GenericFunctionTest, CopyConstraintInterface) {
    auto args = Arguments<4>();
    auto sq = args.squared_norm();
    GenericFunction<-1, 1> gf(sq);
    ConstraintInterface ci1(gf);
    ConstraintInterface ci2(ci1);

    EXPECT_EQ(ci2.IRows(), 4);
    EXPECT_EQ(ci2.ORows(), 1);
}

///////////////////////////////////////////////////////////////////////////////
// GenericConditional / Comparative
///////////////////////////////////////////////////////////////////////////////

TEST_F(GenericFunctionTest, ConditionalGreaterThan) {
    auto args = Arguments<2>();
    auto lhs = args.coeff<0>(); // scalar: x[0]
    auto rhs = args.coeff<1>(); // scalar: x[1]

    ConditionalStatement cond(lhs, ConditionalFlags::GreaterThanFlag, rhs);
    GenericConditional<2> gc(cond);

    Eigen::Vector2d x1(5.0, 3.0); // 5 > 3 → true
    EXPECT_TRUE(gc.compute(x1));

    Eigen::Vector2d x2(2.0, 7.0); // 2 > 7 → false
    EXPECT_FALSE(gc.compute(x2));

    Eigen::Vector2d x3(4.0, 4.0); // 4 > 4 → false (strict)
    EXPECT_FALSE(gc.compute(x3));
}

TEST_F(GenericFunctionTest, ConditionalCompoundAND) {
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();

    // Create a zero constant for comparison
    Eigen::VectorXd zero_val(1);
    zero_val << 0.0;
    auto zero = Constant<2, 1>(2, zero_val);

    // c1: x[0] > 0,  c2: x[1] > 0
    ConditionalStatement c1(x0, ConditionalFlags::GreaterThanFlag, zero);
    ConditionalStatement c2(x1, ConditionalFlags::GreaterThanFlag, zero);

    // AND: both must be true
    ConditionalStatement both(c1, ConditionalFlags::ANDFlag, c2);
    GenericConditional<2> gc(both);

    Eigen::Vector2d tt(1.0, 1.0); // T, T → T
    EXPECT_TRUE(gc.compute(tt));

    Eigen::Vector2d tf(1.0, -1.0); // T, F → F
    EXPECT_FALSE(gc.compute(tf));

    Eigen::Vector2d ft(-1.0, 1.0); // F, T → F
    EXPECT_FALSE(gc.compute(ft));
}

TEST_F(GenericFunctionTest, ComparativeBasic) {
    // GenericComparative has the same interface, just a distinct type
    auto args = Arguments<2>();
    auto lhs = args.coeff<0>();
    auto rhs = args.coeff<1>();

    ConditionalStatement cond(lhs, ConditionalFlags::LessThanFlag, rhs);
    GenericComparative<2> gc(cond);

    Eigen::Vector2d x1(3.0, 5.0); // 3 < 5 → true
    EXPECT_TRUE(gc.compute(x1));

    Eigen::Vector2d x2(7.0, 2.0); // 7 < 2 → false
    EXPECT_FALSE(gc.compute(x2));
}
