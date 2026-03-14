///////////////////////////////////////////////////////////////////////////////
// Utils unit tests
//
// Tests small header-only utilities with no MemoryManager dependency:
// Timer, MathFunctions, SizingHelpers, EigenSTL, LambdaJumpTable,
// TypeName, GetCoreCount, STDExtensions, FunctionReturnType.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/EigenSTL.h"
#include "Utils/FunctionReturnType.h"
#include "Utils/GetCoreCount.h"
#include "Utils/LambdaJumpTable.h"
#include "Utils/MathFunctions.h"
#include "Utils/SizingHelpers.h"
#include "Utils/STDExtensions.h"
#include "Utils/Timer.h"
#include "Utils/TypeName.h"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// TimerTest
///////////////////////////////////////////////////////////////////////////////

TEST(TimerTest, DefaultNotStarted) {
    Utils::Timer t;
    EXPECT_EQ(t.count<std::chrono::microseconds>(), 0);
}

TEST(TimerTest, ConstructWithStart) {
    Utils::Timer t(true);
    // Timer is running — count should be non-negative (may be 0 on fast machines)
    EXPECT_GE(t.count<std::chrono::microseconds>(), 0);
}

TEST(TimerTest, StartStopAccumulates) {
    Utils::Timer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    t.stop();
    auto elapsed = t.count<std::chrono::milliseconds>();
    EXPECT_GT(elapsed, 0) << "Timer should have accumulated > 0 ms";
}

TEST(TimerTest, ResetClearsAccumulated) {
    Utils::Timer t(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.stop();
    EXPECT_GT(t.count<std::chrono::milliseconds>(), 0);
    t.reset();
    EXPECT_EQ(t.count<std::chrono::milliseconds>(), 0);
}

TEST(TimerTest, StopWithoutStartIsNoOp) {
    Utils::Timer t;
    t.stop(); // Should not crash
    EXPECT_EQ(t.count<std::chrono::microseconds>(), 0);
}

TEST(TimerTest, CountWithDurationTypes) {
    Utils::Timer t(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    t.stop();
    // Microseconds should be larger than milliseconds value
    auto us = t.count<std::chrono::microseconds>();
    auto ms = t.count<std::chrono::milliseconds>();
    EXPECT_GT(us, 0);
    EXPECT_GE(us, ms * 1000);
}

///////////////////////////////////////////////////////////////////////////////
// MathFunctionsTest
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// SizingHelpersTest — compile-time arithmetic
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// EigenSTLTest
///////////////////////////////////////////////////////////////////////////////

TEST(EigenSTLTest, EigenToStdDynamic) {
    Eigen::VectorXd ev(3);
    ev << 1.0, 2.0, 3.0;
    auto sv = eigenvector_to_stdvector(ev);
    ASSERT_EQ(sv.size(), 3u);
    EXPECT_DOUBLE_EQ(sv[0], 1.0);
    EXPECT_DOUBLE_EQ(sv[1], 2.0);
    EXPECT_DOUBLE_EQ(sv[2], 3.0);
}

TEST(EigenSTLTest, StdToEigenDynamic) {
    std::vector<double> sv = {4.0, 5.0, 6.0};
    auto ev = stdvector_to_eigenvector(sv);
    ASSERT_EQ(ev.size(), 3);
    EXPECT_DOUBLE_EQ(ev[0], 4.0);
    EXPECT_DOUBLE_EQ(ev[1], 5.0);
    EXPECT_DOUBLE_EQ(ev[2], 6.0);
}

TEST(EigenSTLTest, EigenToStdFixed) {
    Eigen::Vector3d ev(1.0, 2.0, 3.0);
    auto sv = eigenvector_to_stdvector(ev);
    ASSERT_EQ(sv.size(), 3u);
    EXPECT_DOUBLE_EQ(sv[0], 1.0);
}

TEST(EigenSTLTest, EmptyStdToEigen) {
    std::vector<double> sv;
    auto ev = stdvector_to_eigenvector(sv);
    EXPECT_EQ(ev.size(), 0);
}

///////////////////////////////////////////////////////////////////////////////
// LambdaJumpTableTest
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// TypeNameTest
///////////////////////////////////////////////////////////////////////////////

TEST(TypeNameTest, IntName) {
    auto name = type_name<int>();
    EXPECT_NE(name.find("int"), std::string::npos);
}

TEST(TypeNameTest, DoubleName) {
    auto name = type_name<double>();
    EXPECT_NE(name.find("double"), std::string::npos);
}

TEST(TypeNameTest, ConstRef) {
    auto name = type_name<const int &>();
    // Should contain "int" and either "const" or "&"
    EXPECT_NE(name.find("int"), std::string::npos);
}

///////////////////////////////////////////////////////////////////////////////
// GetCoreCountTest
///////////////////////////////////////////////////////////////////////////////

TEST(GetCoreCountTest, AtLeastOne) { EXPECT_GE(get_core_count(), 1); }

///////////////////////////////////////////////////////////////////////////////
// STDExtensionsTest
///////////////////////////////////////////////////////////////////////////////

TEST(STDExtensionsTest, RemoveConstRef) {
    static_assert(std::is_same_v<std::remove_const_reference<const int &>::type, int>);
    SUCCEED();
}

TEST(STDExtensionsTest, PassThroughDouble) {
    static_assert(std::is_same_v<std::remove_const_reference<double>::type, double>);
    SUCCEED();
}

///////////////////////////////////////////////////////////////////////////////
// FunctionReturnTypeTest
///////////////////////////////////////////////////////////////////////////////

static double free_func(int, float) { return 0.0; }

struct TestClass {
    int member_func(double) { return 0; }
    float const_member(int) const { return 0.0f; }
};

TEST(FunctionReturnTypeTest, FreeFunction) {
    static_assert(std::is_same_v<return_type_t<decltype(&free_func)>, double>);
    SUCCEED();
}

TEST(FunctionReturnTypeTest, MemberFunction) {
    static_assert(std::is_same_v<return_type_t<decltype(&TestClass::member_func)>, int>);
    SUCCEED();
}

TEST(FunctionReturnTypeTest, ConstMemberFunction) {
    static_assert(std::is_same_v<return_type_t<decltype(&TestClass::const_member)>, float>);
    SUCCEED();
}
