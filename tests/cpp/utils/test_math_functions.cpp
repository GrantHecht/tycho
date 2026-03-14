///////////////////////////////////////////////////////////////////////////////
// MathFunctions unit tests
//
// Tests factorial and factorialDiv from src/Utils/MathFunctions.h.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/MathFunctions.h"
#include <gtest/gtest.h>

using namespace Tycho;

TEST(MathFunctionsTest, FactorialZero) { EXPECT_EQ(factorial(0), 1); }

TEST(MathFunctionsTest, FactorialOne) { EXPECT_EQ(factorial(1), 1); }

TEST(MathFunctionsTest, FactorialFive) { EXPECT_EQ(factorial(5), 120); }

TEST(MathFunctionsTest, FactorialTen) { EXPECT_EQ(factorial(10), 3628800); }

TEST(MathFunctionsTest, FactorialDivEqualArgs) {
    // factorialDiv(n, n) = 1 for any n
    EXPECT_EQ(factorialDiv(5, 5), 1);
}

TEST(MathFunctionsTest, FactorialDivIGreaterJ) {
    // factorialDiv(5, 3) = 5 * 4 = 20
    EXPECT_EQ(factorialDiv(5, 3), 20);
}

TEST(MathFunctionsTest, FactorialDivILessJ) {
    // factorialDiv(3, 5) = 5 * 4 = 20 (symmetric in product sense)
    EXPECT_EQ(factorialDiv(3, 5), 20);
}

TEST(MathFunctionsTest, FactorialDivAdjacent) {
    // factorialDiv(n, n-1) = n
    EXPECT_EQ(factorialDiv(7, 6), 7);
    EXPECT_EQ(factorialDiv(6, 7), 7);
}
