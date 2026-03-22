///////////////////////////////////////////////////////////////////////////////
// FlatMap tests
//
// Tests FlatMap<Key, Value> from src/Utils/FlatMap.h — insert, lookup,
// duplicate rejection, copy/move semantics, iteration.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/FlatMap.h"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <string>

using namespace Tycho;

TEST(FlatMapTest, DefaultConstructEmpty) {
    FlatMap<std::string, int> fm;
    EXPECT_TRUE(fm.empty());
    EXPECT_EQ(fm.size(), 0u);
}

TEST(FlatMapTest, InsertAndAt) {
    FlatMap<std::string, int> fm;
    fm.insert("a", 1);
    fm.insert("b", 2);

    EXPECT_EQ(fm.size(), 2u);
    EXPECT_FALSE(fm.empty());
    EXPECT_EQ(fm.at("a"), 1);
    EXPECT_EQ(fm.at("b"), 2);
}

TEST(FlatMapTest, InsertDuplicateThrows) {
    FlatMap<std::string, int> fm;
    fm.insert("x", 10);
    EXPECT_THROW(fm.insert("x", 20), std::invalid_argument);
}

TEST(FlatMapTest, AtMissingKeyThrows) {
    FlatMap<std::string, int> fm;
    EXPECT_THROW(fm.at("missing"), std::out_of_range);

    fm.insert("present", 42);
    EXPECT_THROW(fm.at("absent"), std::out_of_range);
}

TEST(FlatMapTest, AtMutableAccess) {
    FlatMap<std::string, int> fm;
    fm.insert("k", 5);
    fm.at("k") = 99;
    EXPECT_EQ(fm.at("k"), 99);
}

TEST(FlatMapTest, Contains) {
    FlatMap<std::string, int> fm;
    EXPECT_FALSE(fm.contains("a"));
    fm.insert("a", 1);
    EXPECT_TRUE(fm.contains("a"));
    EXPECT_FALSE(fm.contains("b"));
}

TEST(FlatMapTest, SubscriptInsertAndAccess) {
    FlatMap<std::string, int> fm;
    fm["x"] = 10;
    EXPECT_TRUE(fm.contains("x"));
    EXPECT_EQ(fm["x"], 10);

    // Overwrite via subscript
    fm["x"] = 20;
    EXPECT_EQ(fm["x"], 20);
    EXPECT_EQ(fm.size(), 1u);

    // Access non-existent key inserts default
    int val = fm["new_key"];
    EXPECT_EQ(val, 0);
    EXPECT_TRUE(fm.contains("new_key"));
    EXPECT_EQ(fm.size(), 2u);
}

TEST(FlatMapTest, Clear) {
    FlatMap<std::string, int> fm;
    fm.insert("a", 1);
    fm.insert("b", 2);
    fm.clear();
    EXPECT_TRUE(fm.empty());
    EXPECT_EQ(fm.size(), 0u);
    EXPECT_FALSE(fm.contains("a"));
}

TEST(FlatMapTest, Iteration) {
    FlatMap<std::string, int> fm;
    fm.insert("x", 10);
    fm.insert("y", 20);
    fm.insert("z", 30);

    // Verify insertion-order preserved
    std::vector<std::string> keys;
    std::vector<int> values;
    for (const auto &[k, v] : fm) {
        keys.push_back(k);
        values.push_back(v);
    }
    ASSERT_EQ(keys.size(), 3u);
    EXPECT_EQ(keys[0], "x");
    EXPECT_EQ(keys[1], "y");
    EXPECT_EQ(keys[2], "z");
    EXPECT_EQ(values[0], 10);
    EXPECT_EQ(values[1], 20);
    EXPECT_EQ(values[2], 30);
}

TEST(FlatMapTest, CopyConstruct) {
    FlatMap<std::string, int> a;
    a.insert("k", 42);

    FlatMap<std::string, int> b(a);
    EXPECT_EQ(b.at("k"), 42);

    // Independence: modifying copy doesn't affect original
    b.at("k") = 99;
    EXPECT_EQ(a.at("k"), 42);
    EXPECT_EQ(b.at("k"), 99);
}

TEST(FlatMapTest, MoveConstruct) {
    FlatMap<std::string, int> a;
    a.insert("k", 42);

    FlatMap<std::string, int> b(std::move(a));
    EXPECT_EQ(b.at("k"), 42);
    EXPECT_TRUE(a.empty()); // NOLINT — testing post-move state
}

TEST(FlatMapTest, CopyAssign) {
    FlatMap<std::string, int> a;
    a.insert("a", 1);

    FlatMap<std::string, int> b;
    b.insert("b", 2);

    b = a;
    EXPECT_EQ(b.at("a"), 1);
    EXPECT_FALSE(b.contains("b"));

    // Independence
    a.insert("c", 3);
    EXPECT_FALSE(b.contains("c"));
}

TEST(FlatMapTest, MoveAssign) {
    FlatMap<std::string, int> a;
    a.insert("k", 42);

    FlatMap<std::string, int> b;
    b = std::move(a);
    EXPECT_EQ(b.at("k"), 42);
    EXPECT_TRUE(a.empty()); // NOLINT — testing post-move state
}

TEST(FlatMapTest, EigenVectorXiValues) {
    FlatMap<std::string, Eigen::VectorXi> fm;

    Eigen::VectorXi v1(3);
    v1 << 0, 1, 2;
    Eigen::VectorXi v2(2);
    v2 << 10, 20;

    fm.insert("x", v1);
    fm.insert("u", v2);

    EXPECT_EQ(fm.size(), 2u);
    EXPECT_EQ(fm.at("x"), v1);
    EXPECT_EQ(fm.at("u"), v2);

    // Copy preserves Eigen vectors
    auto copy = fm;
    EXPECT_EQ(copy.at("x"), v1);

    // Independence: modifying copy doesn't affect original
    copy.at("x")[0] = 99;
    EXPECT_EQ(fm.at("x")[0], 0);

    // Move
    auto moved = std::move(copy);
    EXPECT_EQ(moved.at("x")[0], 99);
    EXPECT_EQ(moved.at("u"), v2);
}
