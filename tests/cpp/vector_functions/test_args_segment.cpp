///////////////////////////////////////////////////////////////////////////////
// CommonFunctions unit tests — Arguments / Segment
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

class CommonFunctionsTest : public VectorFunctionFixture {};

TEST_F(CommonFunctionsTest, ArgumentsDimensions) {
    auto args = Arguments<5>();
    EXPECT_EQ(args.IRows(), 5);
    EXPECT_EQ(args.ORows(), 5);
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
    EXPECT_EQ(seg.IRows(), 6);
    EXPECT_EQ(seg.ORows(), 3);
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
