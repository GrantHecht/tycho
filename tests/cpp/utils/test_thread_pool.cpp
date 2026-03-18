///////////////////////////////////////////////////////////////////////////////
// ThreadPool unit tests
//
// Tests BS::thread_pool from dep/thread-pool — construction,
// task submission, reset, thread counts, and shutdown behavior.
///////////////////////////////////////////////////////////////////////////////

#include <BS_thread_pool.hpp>
#include <atomic>
#include <gtest/gtest.h>
#include <vector>

TEST(ThreadPoolTest, ConstructWithZero) {
    // BS::thread_pool v5 treats 0 as "use hardware concurrency"
    BS::thread_pool<> pool(0);
    EXPECT_GT(pool.get_thread_count(), 0u);
}

TEST(ThreadPoolTest, ConstructWithN) {
    BS::thread_pool<> pool(4);
    EXPECT_EQ(pool.get_thread_count(), 4u);
}

TEST(ThreadPoolTest, SubmitAndGetResult) {
    BS::thread_pool<> pool(2);
    auto fut = pool.submit_task([] { return 42; });
    EXPECT_EQ(fut.get(), 42);
}

TEST(ThreadPoolTest, SubmitMultipleTasks) {
    BS::thread_pool<> pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futs;
    futs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        futs.push_back(pool.submit_task([&counter] { counter.fetch_add(1); }));
    }
    for (auto &f : futs)
        f.get();
    EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPoolTest, Reset) {
    BS::thread_pool<> pool(2);
    pool.reset(6);
    EXPECT_EQ(pool.get_thread_count(), 6u);
    pool.reset(2);
    EXPECT_EQ(pool.get_thread_count(), 2u);
}

TEST(ThreadPoolTest, Wait) {
    BS::thread_pool<> pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.detach_task([&counter] {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
    }
    pool.wait();
    EXPECT_EQ(counter.load(), 50);
}

TEST(ThreadPoolTest, TasksTotal) {
    BS::thread_pool<> pool(2);
    // After construction with no tasks, total should be 0
    pool.wait();
    EXPECT_EQ(pool.get_tasks_total(), 0u);
}

// ---- Global thread pool singleton tests ----

#include "Utils/ThreadPool.h"

TEST(GlobalThreadPool, DefaultThreadCount) {
    // Default should be hardware_concurrency
    int n = Tycho::get_num_threads();
    EXPECT_EQ(n, static_cast<int>(std::thread::hardware_concurrency()));
}

TEST(GlobalThreadPool, SetAndGet) {
    Tycho::set_num_threads(4);
    EXPECT_EQ(Tycho::get_num_threads(), 4);
    EXPECT_TRUE(Tycho::use_thread_pool());

    Tycho::set_num_threads(1);
    EXPECT_EQ(Tycho::get_num_threads(), 1);
    EXPECT_FALSE(Tycho::use_thread_pool());

    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}

TEST(GlobalThreadPool, SetZeroMeansSingleThreaded) {
    Tycho::set_num_threads(0);
    EXPECT_EQ(Tycho::get_num_threads(), 1);
    EXPECT_FALSE(Tycho::use_thread_pool());

    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}

TEST(GlobalThreadPool, PoolDispatch) {
    Tycho::set_num_threads(4);
    auto &pool = Tycho::thread_pool();
    EXPECT_EQ(pool.get_thread_count(), 4u);

    std::atomic<int> counter{0};
    pool.detach_sequence(0, 100, [&counter](int) { counter.fetch_add(1); });
    pool.wait();
    EXPECT_EQ(counter.load(), 100);

    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}

TEST(GlobalThreadPool, SingleThreadedBypass) {
    Tycho::set_num_threads(1);

    // When single-threaded, user code should NOT call thread_pool()
    // — instead check use_thread_pool() and run inline
    EXPECT_FALSE(Tycho::use_thread_pool());

    int result = 0;
    if (Tycho::use_thread_pool()) {
        Tycho::thread_pool().detach_task([&result] { result = 42; });
        Tycho::thread_pool().wait();
    } else {
        result = 42; // inline
    }
    EXPECT_EQ(result, 42);

    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}
