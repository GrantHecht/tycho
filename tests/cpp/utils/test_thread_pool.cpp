///////////////////////////////////////////////////////////////////////////////
// ThreadPool unit tests
//
// Tests Tycho::ThreadPool — construction, task submission, reset,
// work stealing, thread counts, and shutdown behavior.
///////////////////////////////////////////////////////////////////////////////

#include "Utils/ThreadPool.h"
#include "test_utils.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using TychoTest::ScopedThreadCount;

/// Test-only accessor for ThreadPool::reset() (now private).
namespace Tycho {
struct ThreadPoolTestAccess {
    static void reset(ThreadPool &pool, unsigned n) { pool.reset(n); }
};
} // namespace Tycho
using Tycho::ThreadPoolTestAccess;

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
    ThreadPoolTestAccess::reset(pool, 6);
    EXPECT_EQ(pool.get_thread_count(), 6u);
    ThreadPoolTestAccess::reset(pool, 2);
    EXPECT_EQ(pool.get_thread_count(), 2u);
}

TEST(ThreadPoolTest, SubmitWorkAndWait) {
    Tycho::ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futs;
    futs.reserve(50);
    for (int i = 0; i < 50; ++i) {
        futs.push_back(pool.submit_task([&counter] {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }));
    }
    for (auto &f : futs)
        f.get();
    EXPECT_EQ(counter.load(), 50);
}

TEST(ThreadPoolTest, WorkStealing) {
    // Enqueue many more tasks than threads; round-robin distribution plus
    // varying worker speeds exercises the work-stealing path.
    Tycho::ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futs;
    futs.reserve(200);
    for (int i = 0; i < 200; ++i) {
        futs.push_back(pool.submit_task([&counter] { counter.fetch_add(1); }));
    }
    for (auto &f : futs)
        f.get();
    EXPECT_EQ(counter.load(), 200);
}

// ---- Global thread pool singleton tests ----

TEST(GlobalThreadPool, DefaultThreadCount) {
    // Default should be hardware_concurrency, clamped to at least 1
    int n = Tycho::get_num_threads();
    EXPECT_EQ(n, static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
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

TEST(GlobalThreadPool, RoundTripNumThreads) {
    // Verify the 4 -> 1 -> 4 transition: when n<=1 the pool stays alive but
    // is bypassed; going back to n>1 resets the pool to the new thread count.
    ScopedThreadCount guard(4);
    EXPECT_EQ(Tycho::get_num_threads(), 4);
    EXPECT_TRUE(Tycho::use_thread_pool());

    Tycho::set_num_threads(1);
    EXPECT_EQ(Tycho::get_num_threads(), 1);
    EXPECT_FALSE(Tycho::use_thread_pool());

    Tycho::set_num_threads(4);
    EXPECT_EQ(Tycho::get_num_threads(), 4);
    EXPECT_TRUE(Tycho::use_thread_pool());

    // Verify pool is functional after round-trip
    std::atomic<int> counter{0};
    Tycho::parallel_sequence(20, [&counter](int) { counter.fetch_add(1); });
    EXPECT_EQ(counter.load(), 20);
}

TEST(GlobalThreadPool, PoolDispatch) {
    ScopedThreadCount guard(4);
    auto &pool = Tycho::thread_pool();
    EXPECT_EQ(pool.get_thread_count(), 4u);

    std::atomic<int> counter{0};
    std::vector<std::future<void>> futs;
    futs.reserve(100);
    for (int i = 0; i < 100; ++i)
        futs.push_back(pool.submit_task([&counter] { counter.fetch_add(1); }));
    for (auto &f : futs)
        f.get();
    EXPECT_EQ(counter.load(), 100);
}

TEST(GlobalThreadPool, SingleThreadedBypass) {
    ScopedThreadCount guard(1);

    EXPECT_FALSE(Tycho::use_thread_pool());

    // With num_threads=1, dispatch helpers run inline without touching the pool.
    std::atomic<int> counter{0};
    Tycho::parallel_sequence(4, [&counter](int) { counter.fetch_add(1); });
    EXPECT_EQ(counter.load(), 4);
}

// ---- Exception propagation tests ----

TEST(DispatchHelpers, ExceptionPropagation_ParallelBlocks) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(Tycho::parallel_blocks(
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
    EXPECT_THROW(Tycho::parallel_sequence(4,
                                          [](int i) {
                                              if (i == 1)
                                                  throw std::runtime_error("sequence exception");
                                          }),
                 std::runtime_error);
}

TEST(DispatchHelpers, ExceptionPropagation_ParallelTask) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(Tycho::parallel_task(
                     2, [] { throw std::runtime_error("task exception"); }, [] {}),
                 std::runtime_error);
}

TEST(DispatchHelpers, InlineException_ParallelBlocks) {
    ScopedThreadCount guard(4);
    // With count=4, nparts=4: blocks are [0,1) [1,2) [2,3) [3,4).
    // The inline block is [3,4) — throw from it.
    EXPECT_THROW(Tycho::parallel_blocks(
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
    auto fut = Tycho::thread_pool().submit_task(
        [&worker_flag] { worker_flag.store(Tycho::is_pool_worker()); });
    fut.get();
    EXPECT_TRUE(worker_flag.load());
}

TEST(ThreadPoolTest, ResetDrainsPending) {
    Tycho::ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) {
        pool.submit_task([&counter] {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            counter.fetch_add(1);
        });
    }
    ThreadPoolTestAccess::reset(pool, 2); // wait() + shutdown + restart
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

// ---- Nested dispatch tests ----

TEST(DispatchHelpers, NestedDispatch_MainThread_Succeeds) {
    // Mirrors the NLP evalKKT pattern: parallel_task's inline arm calls
    // parallel_blocks. Safe because the inline arm runs on the main thread.
    ScopedThreadCount guard(4);
    std::atomic<int> pool_counter{0};
    std::atomic<int> inner_sum{0};

    Tycho::parallel_task(
        4, [&pool_counter] { pool_counter.fetch_add(1); },
        [&inner_sum] {
            // Inline arm — runs on main thread, nested dispatch is allowed.
            // parallel_blocks splits [0,8) into 4 blocks; each block sums its range.
            Tycho::parallel_blocks(
                8, [&inner_sum](int start, int end) { inner_sum.fetch_add(end - start); }, 4);
        });

    EXPECT_EQ(pool_counter.load(), 1);
    EXPECT_EQ(inner_sum.load(), 8); // sum of all block sizes = count
}

TEST(DispatchHelpers, NestedDispatch_PoolWorker_Throws) {
    // Dispatch from a pool worker must throw logic_error.
    // parallel_sequence runs i=0..2 on pool workers and i=3 inline.
    // The pool workers attempting nested dispatch should throw.
    ScopedThreadCount guard(4);
    EXPECT_THROW(
        Tycho::parallel_sequence(4, [](int) { Tycho::parallel_blocks(4, [](int, int) {}, 2); }),
        std::logic_error);
}

TEST(DispatchHelpers, DualException_ParallelTask) {
    // Both pool and inline arms throw. Pool exception should win.
    ScopedThreadCount guard(4);
    try {
        Tycho::parallel_task(
            2, [] { throw std::runtime_error("pool arm"); },
            [] { throw std::runtime_error("inline arm"); });
        FAIL() << "Expected exception";
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "pool arm");
    }
}

TEST(DispatchHelpers, StressTest_ManyTasks_SmallPool) {
    // 100 tasks on a 2-thread pool — forces heavy work stealing.
    ScopedThreadCount guard(2);
    for (int round = 0; round < 10; ++round) {
        std::atomic<int> counter{0};
        Tycho::parallel_sequence(100, [&counter](int) { counter.fetch_add(1); });
        EXPECT_EQ(counter.load(), 100);
    }
}

TEST(ThreadPoolTest, SubmitTask_ExceptionPropagation) {
    Tycho::ThreadPool pool(2);
    auto fut = pool.submit_task([]() -> int { throw std::runtime_error("submit_task exception"); });
    EXPECT_THROW(fut.get(), std::runtime_error);
}

// ---- Edge-case tests for dispatch helpers ----

TEST(DispatchHelpers, ParallelTask_SequentialFallback) {
    // When nparts <= 1, both callables run inline sequentially.
    // This is the path that prevents Jet/NLP deadlock with NumPartitions=1.
    ScopedThreadCount guard(4); // pool is active, but nparts=1 bypasses it
    int order = 0;
    int pool_order = -1, inline_order = -1;

    Tycho::parallel_task(1, [&] { pool_order = order++; }, [&] { inline_order = order++; });

    EXPECT_EQ(pool_order, 0);   // pool_work runs first in sequential mode
    EXPECT_EQ(inline_order, 1); // inline_work runs second
}

TEST(DispatchHelpers, ParallelBlocks_CountLessThanNparts) {
    // count < nparts triggers the clamp: nparts = std::min(nparts, count).
    // Without it, block_size would be 0 producing incorrect partitions.
    ScopedThreadCount guard(4);
    std::atomic<int> sum{0};

    Tycho::parallel_blocks(2, [&sum](int start, int end) { sum.fetch_add(end - start); }, 10);

    EXPECT_EQ(sum.load(), 2);
}

TEST(DispatchHelpers, ParallelBlocks_ZeroCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    Tycho::parallel_blocks(0, [&](int, int) { called = true; }, 4);
    EXPECT_FALSE(called);
}

TEST(DispatchHelpers, ParallelSequence_ZeroCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    Tycho::parallel_sequence(0, [&](int) { called = true; });
    EXPECT_FALSE(called);
}

TEST(DispatchHelpers, ParallelBlocks_NegativeCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    Tycho::parallel_blocks(-1, [&](int, int) { called = true; }, 4);
    EXPECT_FALSE(called);
}

TEST(DispatchHelpers, ParallelSequence_NegativeCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    Tycho::parallel_sequence(-5, [&](int) { called = true; });
    EXPECT_FALSE(called);
}

TEST(ThreadPoolTest, ConstructZeroThreadsThrows) {
    EXPECT_THROW(Tycho::ThreadPool pool(0), std::invalid_argument);
}

TEST(ThreadPoolTest, ResetZeroThreadsThrows) {
    Tycho::ThreadPool pool(2);
    EXPECT_THROW(ThreadPoolTestAccess::reset(pool, 0), std::invalid_argument);
    // Pool should still be functional after rejected reset.
    auto fut = pool.submit_task([] { return 7; });
    EXPECT_EQ(fut.get(), 7);
}

// =============================================================================
// task SBO wrapper
// =============================================================================

TEST(TaskSBOTest, DefaultConstructedIsEmpty) {
    Tycho::task t;
    EXPECT_FALSE(static_cast<bool>(t));
}

TEST(TaskSBOTest, ConstructFromLambdaIsNonEmpty) {
    bool invoked = false;
    Tycho::task t([&invoked] { invoked = true; });
    EXPECT_TRUE(static_cast<bool>(t));
    t();
    EXPECT_TRUE(invoked);
}

TEST(TaskSBOTest, MoveConstructTransfersOwnership) {
    int value = 0;
    Tycho::task t1([&value] { value = 42; });
    EXPECT_TRUE(static_cast<bool>(t1));

    Tycho::task t2(std::move(t1));
    EXPECT_FALSE(static_cast<bool>(t1));
    EXPECT_TRUE(static_cast<bool>(t2));
    t2();
    EXPECT_EQ(value, 42);
}

TEST(TaskSBOTest, MoveAssignOntoNonEmpty) {
    int a = 0, b = 0;
    Tycho::task t1([&a] { a = 1; });
    Tycho::task t2([&b] { b = 2; });

    // Move-assign t2 onto t1; old t1 callable should be destroyed.
    t1 = std::move(t2);
    EXPECT_TRUE(static_cast<bool>(t1));
    EXPECT_FALSE(static_cast<bool>(t2));
    t1();
    EXPECT_EQ(a, 0); // old callable was not invoked
    EXPECT_EQ(b, 2);
}

TEST(TaskSBOTest, MoveAssignOntoEmpty) {
    int value = 0;
    Tycho::task t1;
    Tycho::task t2([&value] { value = 99; });

    t1 = std::move(t2);
    EXPECT_TRUE(static_cast<bool>(t1));
    EXPECT_FALSE(static_cast<bool>(t2));
    t1();
    EXPECT_EQ(value, 99);
}

// =============================================================================
// Dispatch edge cases: nparts=1 with pool active
// =============================================================================

TEST(DispatchHelpers, ParallelBlocks_NpartsOne_PoolActive) {
    ScopedThreadCount guard(4);
    std::atomic<int> sum{0};
    constexpr int N = 100;
    Tycho::parallel_blocks(
        N, [&](int start, int end) { sum.fetch_add(end - start, std::memory_order_relaxed); }, 1);
    EXPECT_EQ(sum.load(), N);
}

TEST(DispatchHelpers, ParallelSequence_NOne_PoolActive) {
    ScopedThreadCount guard(4);
    std::atomic<int> count{0};
    Tycho::parallel_sequence(1, [&](int) { count.fetch_add(1, std::memory_order_relaxed); });
    EXPECT_EQ(count.load(), 1);
}
