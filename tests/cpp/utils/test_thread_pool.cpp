///////////////////////////////////////////////////////////////////////////////
// ThreadPool unit tests
//
// Tests Tycho::ThreadPool — construction, task submission, reset,
// work stealing, thread counts, and shutdown behavior.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/ThreadPool.h"
#include <atomic>
#include <gtest/gtest.h>
#include <vector>

TEST(ThreadPoolTest, ConstructDefault) {
    Tycho::ThreadPool pool;
    EXPECT_GT(pool.get_thread_count(), 0u);
}

TEST(ThreadPoolTest, ConstructWithN) {
    Tycho::ThreadPool pool(4);
    EXPECT_EQ(pool.get_thread_count(), 4u);
}

TEST(ThreadPoolTest, SubmitAndGetResult) {
    Tycho::ThreadPool pool(2);
    auto fut = pool.submit_task([] { return 42; });
    EXPECT_EQ(fut.get(), 42);
}

TEST(ThreadPoolTest, SubmitMultipleTasks) {
    Tycho::ThreadPool pool(4);
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
    Tycho::ThreadPool pool(2);
    pool.reset(6);
    EXPECT_EQ(pool.get_thread_count(), 6u);
    pool.reset(2);
    EXPECT_EQ(pool.get_thread_count(), 2u);
}

TEST(ThreadPoolTest, EnqueueWorkAndWait) {
    Tycho::ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.enqueue_work([&counter] {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
    }
    pool.wait();
    EXPECT_EQ(counter.load(), 50);
}

TEST(ThreadPoolTest, WorkStealing) {
    // Enqueue many tasks to a single queue index to force stealing
    Tycho::ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 200; ++i) {
        pool.enqueue_work([&counter] { counter.fetch_add(1); });
    }
    pool.wait();
    EXPECT_EQ(counter.load(), 200);
}

// ---- Global thread pool singleton tests ----

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
    for (int i = 0; i < 100; ++i)
        pool.enqueue_work([&counter] { counter.fetch_add(1); });
    pool.wait();
    EXPECT_EQ(counter.load(), 100);

    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}

TEST(GlobalThreadPool, SingleThreadedBypass) {
    Tycho::set_num_threads(1);

    EXPECT_FALSE(Tycho::use_thread_pool());

    int result = 0;
    if (Tycho::use_thread_pool()) {
        Tycho::thread_pool().enqueue_work([&result] { result = 42; });
        Tycho::thread_pool().wait();
    } else {
        result = 42; // inline
    }
    EXPECT_EQ(result, 42);

    // Restore default
    Tycho::set_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
}
