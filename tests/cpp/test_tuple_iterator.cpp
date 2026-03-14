///////////////////////////////////////////////////////////////////////////////
// TupleIterator unit tests
//
// Tests compile-time tuple iteration utilities from
// src/Utils/TupleIterator.h.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/TupleIterator.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace Tycho;

TEST(TupleIteratorTest, ForEachMixedTypes) {
    auto t = std::make_tuple(1, 2.0, std::string("three"));
    std::vector<std::string> visited;
    tuple_for_each(t, [&](const auto &elem) {
        std::ostringstream oss;
        oss << elem;
        visited.push_back(oss.str());
    });
    ASSERT_EQ(visited.size(), 3u);
    EXPECT_EQ(visited[0], "1");
    EXPECT_EQ(visited[1], "2");
    EXPECT_EQ(visited[2], "three");
}

TEST(TupleIteratorTest, ForEachEmpty) {
    auto t = std::make_tuple();
    int count = 0;
    tuple_for_each(t, [&](const auto &) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST(TupleIteratorTest, ReverseForEach) {
    auto t = std::make_tuple(1, 2, 3);
    std::vector<int> visited;
    reverse_tuple_for_each(t, [&](const auto &elem) { visited.push_back(elem); });
    ASSERT_EQ(visited.size(), 3u);
    EXPECT_EQ(visited[0], 3);
    EXPECT_EQ(visited[1], 2);
    EXPECT_EQ(visited[2], 1);
}

TEST(TupleIteratorTest, ForLoopWithIndex) {
    auto t = std::make_tuple(10, 20, 30);
    std::vector<std::pair<int, size_t>> visited;
    tuple_for_loop(t, [&](const auto &elem, auto idx) {
        visited.push_back({elem, decltype(idx)::value});
    });
    ASSERT_EQ(visited.size(), 3u);
    EXPECT_EQ(visited[0].first, 10);
    EXPECT_EQ(visited[0].second, 0u);
    EXPECT_EQ(visited[2].first, 30);
    EXPECT_EQ(visited[2].second, 2u);
}

TEST(TupleIteratorTest, ReverseForLoop) {
    auto t = std::make_tuple(10, 20, 30);
    std::vector<size_t> indices;
    reverse_tuple_for_loop(t, [&](const auto &, auto idx) { indices.push_back(decltype(idx)::value); });
    ASSERT_EQ(indices.size(), 3u);
    EXPECT_EQ(indices[0], 2u);
    EXPECT_EQ(indices[1], 1u);
    EXPECT_EQ(indices[2], 0u);
}

TEST(TupleIteratorTest, ConstexprForLoop) {
    int sum = 0;
    constexpr_for_loop(std::integral_constant<int, 0>(), std::integral_constant<int, 5>(),
                       [&](auto ic) { sum += decltype(ic)::value; });
    // 0 + 1 + 2 + 3 + 4 = 10
    EXPECT_EQ(sum, 10);
}

TEST(TupleIteratorTest, ConstexprForwardingLoop) {
    // Accumulate sum: 0 + 0 + 1 + 2 + 3 = 6
    auto result =
        constexpr_forwarding_loop(std::integral_constant<int, 0>(), std::integral_constant<int, 4>(),
                                  [](auto ic, auto input) { return input + decltype(ic)::value; }, 0);
    EXPECT_EQ(result, 6);
}

TEST(TupleIteratorTest, MakeArray) {
    auto arr = make_array<5>([](std::size_t i) { return static_cast<int>(i * i); });
    ASSERT_EQ(arr.size(), 5u);
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 1);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 9);
    EXPECT_EQ(arr[4], 16);
}

TEST(TupleIteratorTest, MakeArrayZero) {
    auto arr = make_array<0>([](std::size_t i) { return static_cast<int>(i); });
    EXPECT_EQ(arr.size(), 0u);
}

TEST(TupleIteratorTest, SingleElementTuple) {
    auto t = std::make_tuple(42);
    int value = 0;
    tuple_for_each(t, [&](const auto &elem) { value = elem; });
    EXPECT_EQ(value, 42);
}
