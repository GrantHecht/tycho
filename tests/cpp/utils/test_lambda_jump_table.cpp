///////////////////////////////////////////////////////////////////////////////
// LambdaJumpTable unit tests
//
// Tests compile-time dispatch table from src/Utils/LambdaJumpTable.h.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/utils/lambda_jump_table.h"
#include <gtest/gtest.h>

using namespace Tycho;

TEST(LambdaJumpTableTest, BelowFirstThreshold) {
    int dispatched = -99;
    auto f = [&](auto ic) { dispatched = decltype(ic)::value; };
    LambdaJumpTable<4, 8, 16>::run(f, 3);
    EXPECT_EQ(dispatched, 4);
}

TEST(LambdaJumpTableTest, BetweenThresholds) {
    int dispatched = -99;
    auto f = [&](auto ic) { dispatched = decltype(ic)::value; };
    LambdaJumpTable<4, 8, 16>::run(f, 6);
    EXPECT_EQ(dispatched, 8);
}

TEST(LambdaJumpTableTest, AtThreshold) {
    int dispatched = -99;
    auto f = [&](auto ic) { dispatched = decltype(ic)::value; };
    LambdaJumpTable<4, 8, 16>::run(f, 16);
    EXPECT_EQ(dispatched, 16);
}

TEST(LambdaJumpTableTest, AboveAllThresholds) {
    int dispatched = -99;
    auto f = [&](auto ic) { dispatched = decltype(ic)::value; };
    LambdaJumpTable<4, 8, 16>::run(f, 20);
    EXPECT_EQ(dispatched, -1);
}
