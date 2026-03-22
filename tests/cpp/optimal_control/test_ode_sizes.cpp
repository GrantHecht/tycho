///////////////////////////////////////////////////////////////////////////////
// ODESize tests
//
// Tests ODESize idx methods and the Pidxs() bug fix.
///////////////////////////////////////////////////////////////////////////////

#include "OptimalControl/ODESizes.h"
#include <gtest/gtest.h>

using namespace Tycho;

TEST(ODESizeTest, AddIdxSingleInt) {
    // Tests the bug fix: add_idx(string, int) previously dropped the name
    ODESize<3, 1, 0> ode;
    ode.add_idx("my_var", 2);

    Eigen::VectorXi expected(1);
    expected[0] = 2;
    EXPECT_EQ(ode.idx("my_var"), expected);
}

TEST(ODESizeTest, AddIdxDuplicateThrows) {
    ODESize<3, 1, 0> ode;
    Eigen::VectorXi v(1);
    v[0] = 0;
    ode.add_idx("x", v);
    EXPECT_THROW(ode.add_idx("x", v), std::invalid_argument);
}

TEST(ODESizeTest, PidxsUsesCorrectSize) {
    // Regression test: Pidxs() previously used UVars() instead of PVars()
    ODESize<3, 2, 4> ode;
    auto pidxs = ode.Pidxs();
    EXPECT_EQ(pidxs.size(), 4);  // PVars=4, not UVars=2
}
