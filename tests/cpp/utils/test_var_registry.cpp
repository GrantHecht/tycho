///////////////////////////////////////////////////////////////////////////////
// VarRegistry unit tests
///////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <tycho/detail/var_registry.h>

using namespace Tycho;

TEST(VarRegistry, SingleNameLookup) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);
    reg.add_name("t", 3);
    reg.add_name("theta", 4);

    auto idx = reg.resolve("x");
    EXPECT_EQ(idx.size(), 1);
    EXPECT_EQ(idx[0], 0);

    idx = reg.resolve("theta");
    EXPECT_EQ(idx.size(), 1);
    EXPECT_EQ(idx[0], 4);
}

TEST(VarRegistry, GroupContiguousRange) {
    VarRegistry reg(7, 3, 1);
    reg.add_group("MEEs", 0, 6); // indices 0-5
    reg.add_name("w", 6);
    reg.add_name("t", 7); // time index

    auto idx = reg.resolve("MEEs");
    EXPECT_EQ(idx.size(), 6);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(idx[i], i);
}

TEST(VarRegistry, GroupFromMembers) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);
    reg.add_group("state", {"x", "y", "v"});

    auto idx = reg.resolve("state");
    EXPECT_EQ(idx.size(), 3);
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 1);
    EXPECT_EQ(idx[2], 2);
}

TEST(VarRegistry, ResolveMultipleNames) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);
    reg.add_name("t", 3);

    auto idx = reg.resolve({"x", "y", "v", "t"});
    EXPECT_EQ(idx.size(), 4);
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 1);
    EXPECT_EQ(idx[2], 2);
    EXPECT_EQ(idx[3], 3);
}

TEST(VarRegistry, MakeInput) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);
    reg.add_name("t", 3);
    reg.add_name("theta", 4);

    auto vec = reg.make_input({{"x", 1.0}, {"y", 2.0}, {"v", 0.5}, {"theta", 0.7}});
    EXPECT_EQ(vec.size(), 5); // XtUP = 3 + 1 + 1 + 0
    EXPECT_DOUBLE_EQ(vec[0], 1.0);
    EXPECT_DOUBLE_EQ(vec[1], 2.0);
    EXPECT_DOUBLE_EQ(vec[2], 0.5);
    EXPECT_DOUBLE_EQ(vec[3], 0.0); // t not set, defaults to 0
    EXPECT_DOUBLE_EQ(vec[4], 0.7);
}

TEST(VarRegistry, MakeUnits) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);
    reg.add_name("v", 2);

    auto units = reg.make_units({{"x", 1000.0}, {"y", 1000.0}, {"v", 10.0}});
    EXPECT_EQ(units.size(), 5);
    EXPECT_DOUBLE_EQ(units[0], 1000.0);
    EXPECT_DOUBLE_EQ(units[1], 1000.0);
    EXPECT_DOUBLE_EQ(units[2], 10.0);
    EXPECT_DOUBLE_EQ(units[3], 1.0); // t unset, defaults to 1
    EXPECT_DOUBLE_EQ(units[4], 1.0); // theta unset, defaults to 1
}

TEST(VarRegistry, Accessors) {
    VarRegistry reg(7, 3, 2);
    EXPECT_EQ(reg.xvars(), 7);
    EXPECT_EQ(reg.uvars(), 3);
    EXPECT_EQ(reg.pvars(), 2);
    EXPECT_EQ(reg.xtup_size(), 7 + 1 + 3 + 2);
}

TEST(VarRegistry, DuplicateNameThrows) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    EXPECT_THROW(reg.add_name("x", 1), std::invalid_argument);
}

TEST(VarRegistry, UnknownNameThrows) {
    VarRegistry reg(3, 1, 0);
    EXPECT_THROW(reg.resolve("nonexistent"), std::invalid_argument);
}

TEST(VarRegistry, OutOfRangeIndexThrows) {
    VarRegistry reg(3, 1, 0);
    EXPECT_THROW(reg.add_name("bad", 5), std::invalid_argument); // xtup_size = 5, max idx = 4
    EXPECT_THROW(reg.add_name("bad", -1), std::invalid_argument);
}

TEST(VarRegistry, MakeInputMultiIndexThrows) {
    VarRegistry reg(3, 1, 0);
    reg.add_group("pos", 0, 2);
    EXPECT_THROW(reg.make_input({{"pos", 1.0}}), std::invalid_argument);
}

TEST(VarRegistry, EmptyAndContains) {
    VarRegistry reg(3, 1, 0);
    EXPECT_TRUE(reg.empty());
    EXPECT_FALSE(reg.contains("x"));

    reg.add_name("x", 0);
    EXPECT_FALSE(reg.empty());
    EXPECT_TRUE(reg.contains("x"));
}

TEST(VarRegistry, ResolveVector) {
    VarRegistry reg(3, 1, 0);
    reg.add_name("x", 0);
    reg.add_name("y", 1);

    std::vector<std::string> names = {"x", "y"};
    auto idx = reg.resolve(names);
    EXPECT_EQ(idx.size(), 2);
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 1);
}
