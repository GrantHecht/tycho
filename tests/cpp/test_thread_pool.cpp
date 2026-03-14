///////////////////////////////////////////////////////////////////////////////
// ThreadPool unit tests
//
// Tests ctpl::ThreadPool from src/Utils/ThreadPool.h — construction,
// task submission, resize, idle counts, and shutdown behavior.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/ThreadPool.h"
#include <atomic>
#include <gtest/gtest.h>
#include <vector>

TEST(ThreadPoolTest, DefaultConstruct) {
    ctpl::ThreadPool pool;
    EXPECT_EQ(pool.size(), 0);
}

TEST(ThreadPoolTest, ConstructWithN) {
    ctpl::ThreadPool pool(4);
    EXPECT_EQ(pool.size(), 4);
}

TEST(ThreadPoolTest, PushAndGetResult) {
    ctpl::ThreadPool pool(2);
    auto fut = pool.push([](int) { return 42; });
    EXPECT_EQ(fut.get(), 42);
}

TEST(ThreadPoolTest, PushMultipleTasks) {
    ctpl::ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futs;
    futs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        futs.push_back(pool.push([&counter](int) { counter.fetch_add(1); }));
    }
    for (auto &f : futs)
        f.get();
    EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPoolTest, ResizeUp) {
    ctpl::ThreadPool pool(2);
    pool.resize(6);
    EXPECT_EQ(pool.size(), 6);
}

TEST(ThreadPoolTest, ResizeDown) {
    ctpl::ThreadPool pool(6);
    pool.resize(2);
    EXPECT_EQ(pool.size(), 2);
}

TEST(ThreadPoolTest, IdleCount) {
    ctpl::ThreadPool pool(4);
    // After construction with no tasks, all threads should become idle
    // Give threads time to start and enter idle state
    // 50ms is generous; may need increase under heavy CI load
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(pool.n_idle(), 4);
}

TEST(ThreadPoolTest, StopWithWait) {
    ctpl::ThreadPool pool(2);
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.push([&counter](int) {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
    }
    pool.stop(true); // Wait for all tasks
    EXPECT_EQ(counter.load(), 50);
}
