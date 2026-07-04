///////////////////////////////////////////////////////////////////////////////
// CommonFunctions unit tests — Arguments / Segment
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(CommonFunctionsTest, ArgumentsDimensions) {
    auto args = Arguments<5>();
    EXPECT_EQ(args.input_rows(), 5);
    EXPECT_EQ(args.output_rows(), 5);
}

TEST_F(CommonFunctionsTest, ArgumentsIdentityJacobian) {
    auto args = Arguments<4>();
    Eigen::VectorXd x(4);
    x << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd expected = Eigen::MatrixXd::Identity(4, 4);
    verify_jacobian_analytical(args, x, expected);
}

TEST_F(CommonFunctionsTest, SegmentDimensions) {
    auto args = Arguments<6>();
    auto seg = args.template segment<3, 2>(); // 3 elements starting at index 2
    EXPECT_EQ(seg.input_rows(), 6);
    EXPECT_EQ(seg.output_rows(), 3);
}

TEST_F(CommonFunctionsTest, SegmentValues) {
    auto args = Arguments<6>();
    auto seg = args.template segment<3, 2>();
    Eigen::VectorXd x(6);
    x << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    seg.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 30.0);
    EXPECT_DOUBLE_EQ(fx[1], 40.0);
    EXPECT_DOUBLE_EQ(fx[2], 50.0);
}

TEST_F(CommonFunctionsTest, SegmentJacobian) {
    auto args = Arguments<5>();
    auto seg = args.template segment<2, 1>(); // 2 elements starting at index 1
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;
    // Jacobian should be [0 1 0 0 0; 0 0 1 0 0]
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(2, 5);
    expected(0, 1) = 1.0;
    expected(1, 2) = 1.0;
    verify_jacobian_analytical(seg, x, expected);
}

// VF_REVIEW 3.2: Segment_Impl::accumulate_jacobian/accumulate_gradient used to
// silently drop ScaledDirectAssignment/ScaledPlusEqualsAssignment (empty
// `else {}`). These tags aren't produced anywhere by the expression DSL
// today, so exercise them directly against the low-level accumulate API to
// pin down the now-implemented scaled-accumulate math.
TEST_F(CommonFunctionsTest, SegmentAccumulateJacobianScaled) {
    auto args = Arguments<6>();
    auto seg = args.template segment<3, 2>(); // OR=3, ST=2, IR=6

    Eigen::MatrixXd right = Eigen::MatrixXd::Zero(3, 6);
    right(0, 2) = 1.0;
    right(1, 3) = 2.0;
    right(2, 4) = 3.0;

    const double scale = 5.0;
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(3, 6);
    seg.accumulate_jacobian(target, right, ScaledDirectAssignment<double>(scale));

    Eigen::MatrixXd expected = scale * right;
    EXPECT_TRUE(target.isApprox(expected));

    // ScaledPlusEqualsAssignment should add on top of the existing contents.
    seg.accumulate_jacobian(target, right, ScaledPlusEqualsAssignment<double>(scale));
    EXPECT_TRUE(target.isApprox(2.0 * expected));
}

TEST_F(CommonFunctionsTest, SegmentAccumulateGradientScaled) {
    auto args = Arguments<6>();
    auto seg = args.template segment<3, 2>(); // OR=3, ST=2, IR=6

    Eigen::VectorXd right(6);
    right << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0;

    const double scale = -2.0;
    Eigen::VectorXd target = Eigen::VectorXd::Zero(6);
    seg.accumulate_gradient(target, right, ScaledDirectAssignment<double>(scale));

    Eigen::VectorXd expected = Eigen::VectorXd::Zero(6);
    expected.segment(2, 3) = scale * right.segment(2, 3);
    EXPECT_TRUE(target.isApprox(expected));

    // ScaledPlusEqualsAssignment should add on top of the existing contents.
    seg.accumulate_gradient(target, right, ScaledPlusEqualsAssignment<double>(scale));
    EXPECT_TRUE(target.isApprox(2.0 * expected));
}
