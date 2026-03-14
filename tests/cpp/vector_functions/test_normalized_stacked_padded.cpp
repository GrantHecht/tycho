///////////////////////////////////////////////////////////////////////////////
// Normalized / Stacked / Padded unit tests
//
// Extracted from test_common_functions.cpp — Normalized, Stacked, Padded
// sections.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Normalized
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, NormalizedUnitVector) {
    auto args = Arguments<3>();
    auto nrm = args.normalized();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(3);
    fx.setZero();
    nrm.compute(x, fx);
    EXPECT_NEAR(fx[0], 0.6, 1e-14);
    EXPECT_NEAR(fx[1], 0.8, 1e-14);
    EXPECT_NEAR(fx[2], 0.0, 1e-14);
    // Output norm should be 1
    EXPECT_NEAR(fx.norm(), 1.0, 1e-14);
}

TEST_F(CommonFunctionsTest, NormalizedAnalyticalJacobian) {
    auto args = Arguments<3>();
    auto nrm = args.normalized();
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    double n = x.norm();
    // J = (I - x_hat * x_hat^T) / ||x||
    Eigen::VectorXd x_hat = x / n;
    Eigen::MatrixXd expected = (Eigen::MatrixXd::Identity(3, 3) - x_hat * x_hat.transpose()) / n;
    verify_jacobian_analytical(nrm, x, expected, 1e-11);
}

TEST_F(CommonFunctionsTest, NormalizedPowerAdjointConsistency) {
    auto args = Arguments<3>();
    auto np = args.normalized_power<2>();
    Eigen::VectorXd x = deterministic_random_vector(3, 52, 1.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 53, -1.0, 1.0);
    verify_adjoint_consistency(np, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Stacked
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, StackedDimensions) {
    auto args = Arguments<5>();
    auto a = args.template head<2>();
    auto b = args.template segment<3, 2>();
    auto stacked = StackedOutputs{a, b};
    EXPECT_EQ(stacked.IRows(), 5);
    EXPECT_EQ(stacked.ORows(), 5); // 2 + 3
}

TEST_F(CommonFunctionsTest, StackedValues) {
    auto args = Arguments<5>();
    auto a = args.template head<2>();
    auto b = args.template tail<3>();
    auto stacked = StackedOutputs{a, b};
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;
    Eigen::VectorXd fx(5);
    fx.setZero();
    stacked.compute(x, fx);
    EXPECT_DOUBLE_EQ(fx[0], 1.0);
    EXPECT_DOUBLE_EQ(fx[1], 2.0);
    EXPECT_DOUBLE_EQ(fx[2], 3.0);
    EXPECT_DOUBLE_EQ(fx[3], 4.0);
    EXPECT_DOUBLE_EQ(fx[4], 5.0);
}

TEST_F(CommonFunctionsTest, StackedAdjointConsistency) {
    auto args = Arguments<6>();
    auto a = args.template head<2>();
    auto b = args.template segment<2, 2>();
    auto c = args.template tail<2>();
    auto stacked = StackedOutputs{a, b, c};
    EXPECT_EQ(stacked.ORows(), 6);

    Eigen::VectorXd x = deterministic_random_vector(6, 60, 1.0, 10.0);
    Eigen::VectorXd lm = deterministic_random_vector(6, 61, -1.0, 1.0);
    verify_adjoint_consistency(stacked, x, lm);
}

///////////////////////////////////////////////////////////////////////////////
// Padded
///////////////////////////////////////////////////////////////////////////////

TEST_F(CommonFunctionsTest, PaddedUpperDimensions) {
    auto args = Arguments<3>();
    auto padded = args.padded_upper<2>();
    EXPECT_EQ(padded.IRows(), 3);
    EXPECT_EQ(padded.ORows(), 5); // 3 + 2 zeros on top
}

TEST_F(CommonFunctionsTest, PaddedLowerDimensions) {
    auto args = Arguments<3>();
    auto padded = args.padded_lower<2>();
    EXPECT_EQ(padded.IRows(), 3);
    EXPECT_EQ(padded.ORows(), 5);
}

TEST_F(CommonFunctionsTest, PaddedAdjointConsistency) {
    auto args = Arguments<3>();
    auto padded = args.template padded<2, 3>();
    EXPECT_EQ(padded.ORows(), 8); // 2 + 3 + 3
    Eigen::VectorXd x = deterministic_random_vector(3, 62, 1.0, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(8, 63, -1.0, 1.0);
    verify_adjoint_consistency(padded, x, lm);
}
