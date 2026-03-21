///////////////////////////////////////////////////////////////////////////////
// ThreadPool unit tests
//
// Tests Tycho::ThreadPool — construction, task submission, reset,
// work stealing, thread counts, and shutdown behavior.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/ThreadPool.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

// RAII guard to restore global thread count on scope exit.
// Ensures test failures don't leave stale thread counts for subsequent tests.
struct ScopedThreadCount {
    int prev;
    explicit ScopedThreadCount(int n) : prev(Tycho::get_num_threads()) {
        Tycho::set_num_threads(n);
    }
    ~ScopedThreadCount() { Tycho::set_num_threads(prev); }
    ScopedThreadCount(const ScopedThreadCount &) = delete;
    ScopedThreadCount &operator=(const ScopedThreadCount &) = delete;
};

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
    ScopedThreadCount guard(4);
    EXPECT_EQ(Tycho::get_num_threads(), 4);
    EXPECT_TRUE(Tycho::use_thread_pool());

    Tycho::set_num_threads(1);
    EXPECT_EQ(Tycho::get_num_threads(), 1);
    EXPECT_FALSE(Tycho::use_thread_pool());
}

TEST(GlobalThreadPool, SetZeroMeansSingleThreaded) {
    ScopedThreadCount guard(0);
    EXPECT_EQ(Tycho::get_num_threads(), 1);
    EXPECT_FALSE(Tycho::use_thread_pool());
}

TEST(GlobalThreadPool, PoolDispatch) {
    ScopedThreadCount guard(4);
    auto &pool = Tycho::thread_pool();
    EXPECT_EQ(pool.get_thread_count(), 4u);

    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i)
        pool.enqueue_work([&counter] { counter.fetch_add(1); });
    pool.wait();
    EXPECT_EQ(counter.load(), 100);
}

TEST(GlobalThreadPool, SingleThreadedBypass) {
    ScopedThreadCount guard(1);

    EXPECT_FALSE(Tycho::use_thread_pool());

    int result = 0;
    if (Tycho::use_thread_pool()) {
        Tycho::thread_pool().enqueue_work([&result] { result = 42; });
        Tycho::thread_pool().wait();
    } else {
        result = 42; // inline
    }
    EXPECT_EQ(result, 42);
}

// ---- Exception propagation tests ----

TEST(DispatchHelpers, ExceptionPropagation_ParallelBlocks) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(
        Tycho::parallel_blocks(
            4,
            [](int start, int /*end*/) {
                if (start == 0) // first pool task
                    throw std::runtime_error("pool exception");
            },
            4),
        std::runtime_error);
}

TEST(DispatchHelpers, ExceptionPropagation_ParallelSequence) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(
        Tycho::parallel_sequence(4,
                                 [](int i) {
                                     if (i == 1)
                                         throw std::runtime_error("sequence exception");
                                 }),
        std::runtime_error);
}

TEST(DispatchHelpers, ExceptionPropagation_ParallelTask) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(
        Tycho::parallel_task(
            2, [] { throw std::runtime_error("task exception"); }, [] {}),
        std::runtime_error);
}

TEST(DispatchHelpers, InlineException_ParallelBlocks) {
    ScopedThreadCount guard(4);
    // With count=4, nparts=4: blocks are [0,1) [1,2) [2,3) [3,4).
    // The inline block is [3,4) — throw from it.
    EXPECT_THROW(
        Tycho::parallel_blocks(
            4,
            [](int start, int /*end*/) {
                if (start == 3) // inline (last) block
                    throw std::runtime_error("inline exception");
            },
            4),
        std::runtime_error);
}

TEST(DispatchHelpers, WorkerDetection) {
    ScopedThreadCount guard(4);
    EXPECT_FALSE(Tycho::is_pool_worker());

    std::atomic<bool> worker_flag{false};
    Tycho::thread_pool().enqueue_work(
        [&worker_flag] { worker_flag.store(Tycho::is_pool_worker()); });
    Tycho::thread_pool().wait();
    EXPECT_TRUE(worker_flag.load());
}

TEST(ThreadPoolTest, ResetDrainsPending) {
    Tycho::ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) {
        pool.enqueue_work([&counter] {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            counter.fetch_add(1);
        });
    }
    pool.reset(2); // wait() + shutdown + restart
    EXPECT_EQ(counter.load(), 100);
    EXPECT_EQ(pool.get_thread_count(), 2u);
}

TEST(DispatchHelpers, StressTest_ParallelSequence) {
    ScopedThreadCount guard(4);
    for (int round = 0; round < 100; ++round) {
        std::atomic<int> counter{0};
        Tycho::parallel_sequence(4, [&counter](int) { counter.fetch_add(1); });
        EXPECT_EQ(counter.load(), 4);
    }
}
