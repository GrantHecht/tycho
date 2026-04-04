///////////////////////////////////////////////////////////////////////////////
// Scaled and RowScaled nesting — verify collapse to avoid double-wrapping
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

// Trait to detect double-wrapping: Scaled<Scaled<T>>
template <class T> struct is_double_scaled : std::false_type {};
template <class T>
struct is_double_scaled<tycho::vf::Scaled<tycho::vf::Scaled<T>>> : std::true_type {};

// Trait to detect double-wrapping: RowScaled<RowScaled<T>>
template <class T> struct is_double_row_scaled : std::false_type {};
template <class T>
struct is_double_row_scaled<tycho::vf::RowScaled<tycho::vf::RowScaled<T>>> : std::true_type {};

TEST_F(CommonFunctionsTest, DoubleScaleCollapse_ScaledTimesDouble) {
    auto args = Arguments<3>();
    auto x = args.coeff<0>();

    // First scaling: double * coeff → Scaled<Segment<...>>
    auto once = 3.0 * x;
    // Second scaling: Scaled<...> * double → should still be Scaled<Segment<...>>
    auto twice = once * 2.0;

    // Type check: twice should be Scaled<Segment<...>>, NOT Scaled<Scaled<Segment<...>>>
    static_assert(!is_double_scaled<decltype(twice)>::value,
                  "Scaled nesting should be collapsed, not Scaled<Scaled<...>>");
    static_assert(std::is_same_v<decltype(twice), decltype(2.0 * once)>,
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

    static_assert(!is_double_scaled<decltype(twice)>::value,
                  "Scaled nesting should be collapsed, not Scaled<Scaled<...>>");

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

///////////////////////////////////////////////////////////////////////////////
// RowScaled nesting — verify collapse when chaining vector scales
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, RowScaledTimesDouble) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv(2.0, 3.0, 4.0);
    auto row_scaled = args * sv; // RowScaled<Arguments<3>>
    auto chained = row_scaled * 5.0; // should collapse to RowScaled with combined scale

    static_assert(!is_double_row_scaled<decltype(chained)>::value,
                  "RowScaled nesting should be collapsed, not RowScaled<RowScaled<...>>");

    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    chained.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 10.0); // 2 * 5
    EXPECT_DOUBLE_EQ(fx[1], 15.0); // 3 * 5
    EXPECT_DOUBLE_EQ(fx[2], 20.0); // 4 * 5
}

TEST_F(CommonFunctionsTest, DoubleTimesRowScaled) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv(2.0, 3.0, 4.0);
    auto row_scaled = args * sv;
    auto chained = 5.0 * row_scaled;

    static_assert(!is_double_row_scaled<decltype(chained)>::value,
                  "RowScaled nesting should be collapsed, not RowScaled<RowScaled<...>>");

    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    chained.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 10.0);
    EXPECT_DOUBLE_EQ(fx[1], 15.0);
    EXPECT_DOUBLE_EQ(fx[2], 20.0);
}

TEST_F(CommonFunctionsTest, RowScaledTimesVector) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv1(2.0, 3.0, 4.0);
    Eigen::Vector3d sv2(10.0, 20.0, 30.0);
    auto row_scaled = args * sv1;
    auto chained = row_scaled * sv2; // cwiseProduct of scales

    static_assert(!is_double_row_scaled<decltype(chained)>::value,
                  "RowScaled nesting should be collapsed, not RowScaled<RowScaled<...>>");

    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    chained.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 20.0);  // 2 * 10
    EXPECT_DOUBLE_EQ(fx[1], 60.0);  // 3 * 20
    EXPECT_DOUBLE_EQ(fx[2], 120.0); // 4 * 30
}

TEST_F(CommonFunctionsTest, VectorTimesRowScaled) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv1(2.0, 3.0, 4.0);
    Eigen::Vector3d sv2(10.0, 20.0, 30.0);
    auto row_scaled = args * sv1;
    auto chained = sv2 * row_scaled;

    static_assert(!is_double_row_scaled<decltype(chained)>::value,
                  "RowScaled nesting should be collapsed, not RowScaled<RowScaled<...>>");

    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    chained.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 20.0);
    EXPECT_DOUBLE_EQ(fx[1], 60.0);
    EXPECT_DOUBLE_EQ(fx[2], 120.0);
}

TEST_F(CommonFunctionsTest, RowScaledDivisionCollapse) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv(2.0, 3.0, 4.0);
    auto row_scaled = args * sv;
    auto divided = row_scaled / 2.0;

    static_assert(!is_double_row_scaled<decltype(divided)>::value,
                  "RowScaled / double should collapse, not double-wrap");

    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    divided.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 1.0);  // 2 / 2
    EXPECT_DOUBLE_EQ(fx[1], 1.5);  // 3 / 2
    EXPECT_DOUBLE_EQ(fx[2], 2.0);  // 4 / 2
}

TEST_F(CommonFunctionsTest, RowScaledNesting_JacobianFD_RowScaledDivDouble) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv(2.0, 3.0, 4.0);
    auto chained = (args * sv) / 2.0;

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    verify_jacobian_fd(chained, x);
}

TEST_F(CommonFunctionsTest, RowScaledNesting_JacobianFD_DoubleTimesRowScaled) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv(2.0, 3.0, 4.0);
    auto chained = 5.0 * (args * sv);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    verify_jacobian_fd(chained, x);
}

TEST_F(CommonFunctionsTest, RowScaledNesting_JacobianFD_RowScaledTimesDouble) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv(2.0, 3.0, 4.0);
    auto chained = (args * sv) * 5.0;

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    verify_jacobian_fd(chained, x);
}

TEST_F(CommonFunctionsTest, RowScaledNesting_JacobianFD_RowScaledTimesVector) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv1(2.0, 3.0, 4.0);
    Eigen::Vector3d sv2(10.0, 20.0, 30.0);
    auto chained = (args * sv1) * sv2;

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    verify_jacobian_fd(chained, x);
}

TEST_F(CommonFunctionsTest, RowScaledNesting_JacobianFD_VectorTimesRowScaled) {
    auto args = Arguments<3>();
    Eigen::Vector3d sv1(2.0, 3.0, 4.0);
    Eigen::Vector3d sv2(10.0, 20.0, 30.0);
    auto chained = sv2 * (args * sv1);

    Eigen::VectorXd x(3);
    x << 1.0, 2.0, 3.0;
    verify_jacobian_fd(chained, x);
}
