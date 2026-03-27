///////////////////////////////////////////////////////////////////////////////
// SizingHelpers unit tests
//
// Tests compile-time arithmetic helpers (SZ_SUM, SZ_PROD, SZ_MAX, SZ_MIN,
// SZ_DIFF, SZ_DIVOP) from src/Utils/SizingHelpers.h.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/utils/sizing_helpers.h"
#include <gtest/gtest.h>

using namespace Tycho;

TEST(SizingHelpersTest, SumPositive) {
    static_assert(SZ_SUM<3, 4>::value == 7);
    static_assert(SZ_SUM<1, 2, 3>::value == 6);
    SUCCEED();
}

TEST(SizingHelpersTest, SumWithNeg1) {
    // Any -1 argument propagates to -1
    static_assert(SZ_SUM<3, -1>::value == -1);
    static_assert(SZ_SUM<-1, 5>::value == -1);
    SUCCEED();
}

TEST(SizingHelpersTest, ProdPositive) {
    static_assert(SZ_PROD<3, 4>::value == 12);
    static_assert(SZ_PROD<2, 3, 4>::value == 24);
    SUCCEED();
}

TEST(SizingHelpersTest, ProdWithNeg1) {
    static_assert(SZ_PROD<3, -1>::value == -1);
    SUCCEED();
}

TEST(SizingHelpersTest, MaxPositive) {
    static_assert(SZ_MAX<3, 7>::value == 7);
    static_assert(SZ_MAX<10, 2>::value == 10);
    SUCCEED();
}

TEST(SizingHelpersTest, MaxWithNeg1) {
    static_assert(SZ_MAX<3, -1>::value == -1);
    SUCCEED();
}

TEST(SizingHelpersTest, MinPositive) {
    static_assert(SZ_MIN<3, 7>::value == 3);
    static_assert(SZ_MIN<10, 2>::value == 2);
    SUCCEED();
}

TEST(SizingHelpersTest, MinWithNeg1) {
    static_assert(SZ_MIN<3, -1>::value == -1);
    SUCCEED();
}

TEST(SizingHelpersTest, DiffPositive) {
    static_assert(SZ_DIFF<7, 3>::value == 4);
    SUCCEED();
}

TEST(SizingHelpersTest, DiffWithNeg1) {
    static_assert(SZ_DIFF<7, -1>::value == -1);
    static_assert(SZ_DIFF<-1, 3>::value == -1);
    SUCCEED();
}

TEST(SizingHelpersTest, DivPositive) {
    // SZ_DIVOP is a pairwise operator; use it directly to avoid the
    // SZ_BINOP fold which evaluates SZ_DIVOP<0,0>::Identity (division by zero).
    static_assert(SZ_DIVOP<12, 3>::value == 4);
    SUCCEED();
}

TEST(SizingHelpersTest, DivWithNeg1) {
    static_assert(SZ_DIVOP<12, -1>::value == -1);
    static_assert(SZ_DIVOP<-1, 3>::value == -1);
    SUCCEED();
}
