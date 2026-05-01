///////////////////////////////////////////////////////////////////////////////
// tycho::math unit tests
//
// Direct numerical verification that the SuperScalar overloads of
// tycho::math::cos / tycho::math::sin agree lane-by-lane with
// W independent std::cos / std::sin calls.  Catches per-lane swap bugs in
// the overload bodies — the existing Enzyme tests validate via whole-body
// matching (BrachVectorizable body matches scalar Brach body, both using
// std::cos), so a lane swap inside the SuperScalar overload would still
// pass there.  Asymmetric per-lane inputs here pin each lane independently.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/math.h"
#include <gtest/gtest.h>
#include <cmath>

namespace {

TEST(TychoMath, ScalarCosMatchesStdCos) {
    EXPECT_EQ(tycho::math::cos(0.1), std::cos(0.1));
    EXPECT_EQ(tycho::math::cos(1.7), std::cos(1.7));
    EXPECT_EQ(tycho::math::cos(-0.3), std::cos(-0.3));
    EXPECT_EQ(tycho::math::cos(2.4), std::cos(2.4));
}

TEST(TychoMath, ScalarSinMatchesStdSin) {
    EXPECT_EQ(tycho::math::sin(0.1), std::sin(0.1));
    EXPECT_EQ(tycho::math::sin(1.7), std::sin(1.7));
    EXPECT_EQ(tycho::math::sin(-0.3), std::sin(-0.3));
    EXPECT_EQ(tycho::math::sin(2.4), std::sin(2.4));
}

TEST(TychoMath, SuperScalarLaneByLaneCos_W2) {
    Eigen::Array<double, 2, 1> x;
    x << 0.1, 1.7;
    auto y = tycho::math::cos(x);
    EXPECT_EQ(y[0], std::cos(0.1));
    EXPECT_EQ(y[1], std::cos(1.7));
}

TEST(TychoMath, SuperScalarLaneByLaneSin_W2) {
    Eigen::Array<double, 2, 1> x;
    x << 0.1, 1.7;
    auto y = tycho::math::sin(x);
    EXPECT_EQ(y[0], std::sin(0.1));
    EXPECT_EQ(y[1], std::sin(1.7));
}

TEST(TychoMath, SuperScalarLaneByLaneCos_W4) {
    Eigen::Array<double, 4, 1> x;
    // Asymmetric values prevent a lane-swap bug from being masked by symmetry.
    x << 0.1, 1.7, -0.3, 2.4;
    auto y = tycho::math::cos(x);
    EXPECT_EQ(y[0], std::cos(0.1));
    EXPECT_EQ(y[1], std::cos(1.7));
    EXPECT_EQ(y[2], std::cos(-0.3));
    EXPECT_EQ(y[3], std::cos(2.4));
}

TEST(TychoMath, SuperScalarLaneByLaneSin_W4) {
    Eigen::Array<double, 4, 1> x;
    x << 0.1, 1.7, -0.3, 2.4;
    auto y = tycho::math::sin(x);
    EXPECT_EQ(y[0], std::sin(0.1));
    EXPECT_EQ(y[1], std::sin(1.7));
    EXPECT_EQ(y[2], std::sin(-0.3));
    EXPECT_EQ(y[3], std::sin(2.4));
}

TEST(TychoMath, SuperScalarLaneByLaneCos_W8) {
    Eigen::Array<double, 8, 1> x;
    x << 0.1, 1.7, -0.3, 2.4, -1.1, 0.8, 3.0, -2.7;
    auto y = tycho::math::cos(x);
    for (int k = 0; k < 8; ++k) EXPECT_EQ(y[k], std::cos(x[k]));
}

TEST(TychoMath, SuperScalarLaneByLaneSin_W8) {
    Eigen::Array<double, 8, 1> x;
    x << 0.1, 1.7, -0.3, 2.4, -1.1, 0.8, 3.0, -2.7;
    auto y = tycho::math::sin(x);
    for (int k = 0; k < 8; ++k) EXPECT_EQ(y[k], std::sin(x[k]));
}

} // namespace
