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
