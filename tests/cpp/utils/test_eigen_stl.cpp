///////////////////////////////////////////////////////////////////////////////
// EigenSTL unit tests
//
// Tests Eigen<->std::vector conversion utilities from src/Utils/EigenSTL.h.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/eigen_stl.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace Tycho;

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
