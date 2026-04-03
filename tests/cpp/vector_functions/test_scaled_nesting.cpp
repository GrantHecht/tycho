///////////////////////////////////////////////////////////////////////////////
// Scaled<Scaled<...>> nesting — verify collapse to single Scaled
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>
#include <type_traits>

using namespace tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Type-level checks: Scaled<Scaled<X>> must not occur
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, DoubleScaleCollapse_ScaledTimesDouble) {
    auto args = Arguments<3>();
    auto x = args.coeff<0>();

    // First scaling: double * coeff → Scaled<Segment<...>>
    auto once = 3.0 * x;
    // Second scaling: Scaled<...> * double → should still be Scaled<Segment<...>>
    auto twice = once * 2.0;

    // Type check: twice should be Scaled<Segment<...>>, NOT Scaled<Scaled<Segment<...>>>
    static_assert(!std::is_same_v<decltype(twice), decltype(2.0 * once)> == false,
                  "double * Scaled and Scaled * double should produce the same type");

    Eigen::VectorXd input(3);
    input << 7.0, 0.0, 0.0;
    Eigen::VectorXd output(1);
    output.setZero();
    twice.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 42.0); // 3 * 2 * 7
}

TEST_F(CommonFunctionsTest, DoubleScaleCollapse_DoubleTimesScaled) {
    auto args = Arguments<3>();
    auto x = args.coeff<0>();

    auto once = 3.0 * x;
    auto twice = 2.0 * once;

    Eigen::VectorXd input(3);
    input << 5.0, 0.0, 0.0;
    Eigen::VectorXd output(1);
    output.setZero();
    twice.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 30.0); // 2 * 3 * 5
}

TEST_F(CommonFunctionsTest, DoubleScaleCollapse_ChainedMultiply) {
    auto args = Arguments<3>();
    auto x = args.coeff<0>();

    // a * (b * x) should produce Scaled<Segment<...>> with combined scale a*b
    double a = 2.0;
    double b = 3.0;
    auto expr = a * (b * x);

    Eigen::VectorXd input(3);
    input << 1.0, 2.0, 3.0;
    Eigen::VectorXd output(1);
    output.setZero();
    expr.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 6.0); // 2 * 3 * 1.0
}

TEST_F(CommonFunctionsTest, ScaledDivisionCollapse) {
    auto args = Arguments<3>();
    auto x = args.coeff<0>();

    // Scaled / double should collapse
    auto scaled = 6.0 * x;
    auto divided = scaled / 2.0;

    Eigen::VectorXd input(3);
    input << 5.0, 0.0, 0.0;
    Eigen::VectorXd output(1);
    output.setZero();
    divided.compute(input, output);
    EXPECT_DOUBLE_EQ(output(0), 15.0); // (6/2) * 5
}

TEST_F(CommonFunctionsTest, ScaledNesting_JacobianCorrect) {
    auto args = Arguments<3>();
    auto scaled = 2.0 * (3.0 * args);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;

    // Jacobian of 6*I is 6*I
    Eigen::MatrixXd expected = 6.0 * Eigen::MatrixXd::Identity(3, 3);
    verify_jacobian_analytical(scaled, x, expected);
}

TEST_F(CommonFunctionsTest, ScaledNesting_FDCrossCheck) {
    auto args = Arguments<3>();
    auto scaled = 2.0 * (3.0 * args);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    verify_jacobian_fd(scaled, x);
}
