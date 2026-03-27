///////////////////////////////////////////////////////////////////////////////
// MathFunctions unit tests
//
// Tests factorial and factorial_div from src/Utils/MathFunctions.h.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/utils/math_functions.h"
#include <gtest/gtest.h>

using namespace tycho::utils;

TEST(MathFunctionsTest, FactorialZero) { EXPECT_EQ(factorial(0), 1); }

TEST(MathFunctionsTest, FactorialOne) { EXPECT_EQ(factorial(1), 1); }

TEST(MathFunctionsTest, FactorialFive) { EXPECT_EQ(factorial(5), 120); }

TEST(MathFunctionsTest, FactorialTen) { EXPECT_EQ(factorial(10), 3628800); }

TEST(MathFunctionsTest, FactorialDivEqualArgs) {
    // factorial_div(n, n) = 1 for any n
    EXPECT_EQ(factorial_div(5, 5), 1);
}

TEST(MathFunctionsTest, FactorialDivIGreaterJ) {
    // factorial_div(5, 3) = 5 * 4 = 20
    EXPECT_EQ(factorial_div(5, 3), 20);
}

TEST(MathFunctionsTest, FactorialDivILessJ) {
    // factorial_div(3, 5) = 5 * 4 = 20 (symmetric in product sense)
    EXPECT_EQ(factorial_div(3, 5), 20);
}

TEST(MathFunctionsTest, FactorialDivAdjacent) {
    // factorial_div(n, n-1) = n
    EXPECT_EQ(factorial_div(7, 6), 7);
    EXPECT_EQ(factorial_div(6, 7), 7);
}
