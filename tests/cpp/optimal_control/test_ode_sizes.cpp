///////////////////////////////////////////////////////////////////////////////
// ODESize tests
//
// Tests ODESize idx methods and the Pidxs() bug fix.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/optimal_control/core/ode_sizes.h"
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

    // Verify actual index values: iota starting at XtUVars()=6
    EXPECT_EQ(pidxs[0], 6);
    EXPECT_EQ(pidxs[1], 7);
    EXPECT_EQ(pidxs[2], 8);
    EXPECT_EQ(pidxs[3], 9);
}

TEST(ODESizeTest, AddIdxEmptyVectorThrows) {
    ODESize<3, 1, 0> ode;
    Eigen::VectorXi empty(0);
    EXPECT_THROW(ode.add_idx("bad", empty), std::invalid_argument);
}

TEST(ODESizeTest, IdxMissingKeyThrows) {
    ODESize<3, 1, 0> ode;
    EXPECT_THROW(ode.idx("nonexistent"), std::invalid_argument);
}

TEST(ODESizeTest, SetIdxsEmptyVectorThrows) {
    ODESize<3, 1, 0> ode;
    FlatMap<std::string, Eigen::VectorXi> idxs;
    idxs.insert("good", Eigen::VectorXi::LinSpaced(2, 0, 1));
    idxs.insert("bad", Eigen::VectorXi(0));
    EXPECT_THROW(ode.set_idxs(idxs), std::invalid_argument);
}

TEST(ODESizeTest, PidxsWithSubindexing) {
    ODESize<3, 2, 4> ode;
    Eigen::VectorXi sub(2);
    sub << 0, 3;
    auto result = ode.Pidxs(sub);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 6);   // XtUVars() + 0
    EXPECT_EQ(result[1], 9);   // XtUVars() + 3
}

TEST(ODESizeTest, SetGetIdxsRoundtrip) {
    ODESize<3, 2, 0> ode;

    FlatMap<std::string, Eigen::VectorXi> idxs;
    Eigen::VectorXi v1(2);
    v1 << 0, 1;
    Eigen::VectorXi v2(1);
    v2 << 2;
    idxs.insert("pos", v1);
    idxs.insert("vel", v2);

    ode.set_idxs(idxs);
    auto result = ode.get_idxs();

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("pos"), v1);
    EXPECT_EQ(result.at("vel"), v2);
}
