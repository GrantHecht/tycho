///////////////////////////////////////////////////////////////////////////////
// ThreadPool unit tests
//
// Tests tycho::utils::ThreadPool — construction, task submission, reset,
// work stealing, thread counts, and shutdown behavior.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/utils/thread_pool.h"
#include "test_utils.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using TychoTest::ScopedThreadCount;

/// Test-only accessor for ThreadPool::reset() (now private).
namespace tycho::utils {
struct ThreadPoolTestAccess {
    static void reset(ThreadPool &pool, unsigned n) { pool.reset(n); }
};
} // namespace tycho::utils
using tycho::utils::ThreadPoolTestAccess;

TEST(ThreadPoolTest, ConstructDefault) {
    tycho::utils::ThreadPool pool;
    EXPECT_GT(pool.get_thread_count(), 0u);
}

TEST(ThreadPoolTest, ConstructWithN) {
    tycho::utils::ThreadPool pool(4);
    EXPECT_EQ(pool.get_thread_count(), 4u);
}

TEST(ThreadPoolTest, SubmitAndGetResult) {
    tycho::utils::ThreadPool pool(2);
    auto fut = pool.submit_task([] { return 42; });
    EXPECT_EQ(fut.get(), 42);
}

TEST(ThreadPoolTest, SubmitMultipleTasks) {
    tycho::utils::ThreadPool pool(4);
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
    tycho::utils::ThreadPool pool(2);
    ThreadPoolTestAccess::reset(pool, 6);
    EXPECT_EQ(pool.get_thread_count(), 6u);
    ThreadPoolTestAccess::reset(pool, 2);
    EXPECT_EQ(pool.get_thread_count(), 2u);
}

TEST(ThreadPoolTest, SubmitWorkAndWait) {
    tycho::utils::ThreadPool pool(4);
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
    tycho::utils::ThreadPool pool(4);
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
    int n = tycho::utils::get_num_threads();
    EXPECT_EQ(n, static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
}

TEST(GlobalThreadPool, SetAndGet) {
    ScopedThreadCount guard(4);
    EXPECT_EQ(tycho::utils::get_num_threads(), 4);
    EXPECT_TRUE(tycho::utils::use_thread_pool());

    tycho::utils::set_num_threads(1);
    EXPECT_EQ(tycho::utils::get_num_threads(), 1);
    EXPECT_FALSE(tycho::utils::use_thread_pool());
}

TEST(GlobalThreadPool, SetZeroMeansSingleThreaded) {
    ScopedThreadCount guard(0);
    EXPECT_EQ(tycho::utils::get_num_threads(), 1);
    EXPECT_FALSE(tycho::utils::use_thread_pool());
}

TEST(GlobalThreadPool, RoundTripNumThreads) {
    // Verify the 4 -> 1 -> 4 transition: when n<=1 the pool stays alive but
    // is bypassed; going back to n>1 resets the pool to the new thread count.
    ScopedThreadCount guard(4);
    EXPECT_EQ(tycho::utils::get_num_threads(), 4);
    EXPECT_TRUE(tycho::utils::use_thread_pool());

    tycho::utils::set_num_threads(1);
    EXPECT_EQ(tycho::utils::get_num_threads(), 1);
    EXPECT_FALSE(tycho::utils::use_thread_pool());

    tycho::utils::set_num_threads(4);
    EXPECT_EQ(tycho::utils::get_num_threads(), 4);
    EXPECT_TRUE(tycho::utils::use_thread_pool());

    // Verify pool is functional after round-trip
    std::atomic<int> counter{0};
    tycho::utils::parallel_sequence(20, [&counter](int) { counter.fetch_add(1); });
    EXPECT_EQ(counter.load(), 20);
}

TEST(GlobalThreadPool, PoolDispatch) {
    ScopedThreadCount guard(4);
    auto &pool = tycho::utils::thread_pool();
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

    EXPECT_FALSE(tycho::utils::use_thread_pool());

    // With num_threads=1, dispatch helpers run inline without touching the pool.
    std::atomic<int> counter{0};
    tycho::utils::parallel_sequence(4, [&counter](int) { counter.fetch_add(1); });
    EXPECT_EQ(counter.load(), 4);
}

// ---- Exception propagation tests ----

TEST(DispatchHelpers, ExceptionPropagation_ParallelBlocks) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(tycho::utils::parallel_blocks(
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
    EXPECT_THROW(tycho::utils::parallel_sequence(4,
                                          [](int i) {
                                              if (i == 1)
                                                  throw std::runtime_error("sequence exception");
                                          }),
                 std::runtime_error);
}

TEST(DispatchHelpers, ExceptionPropagation_ParallelTask) {
    ScopedThreadCount guard(4);
    EXPECT_THROW(tycho::utils::parallel_task(
                     2, [] { throw std::runtime_error("task exception"); }, [] {}),
                 std::runtime_error);
}

TEST(DispatchHelpers, InlineException_ParallelBlocks) {
    ScopedThreadCount guard(4);
    // With count=4, nparts=4: blocks are [0,1) [1,2) [2,3) [3,4).
    // The inline block is [3,4) — throw from it.
    EXPECT_THROW(tycho::utils::parallel_blocks(
                     4,
                     [](int start, int /*end*/) {
                         if (start == 3) // inline (last) block
                             throw std::runtime_error("inline exception");
                     },
                     4),
                 std::runtime_error);
}

TEST(DispatchHelpers, InlineException_ParallelSequence) {
    ScopedThreadCount guard(4);
    // With n=4, indices 0-2 go to pool, index 3 runs inline.
    EXPECT_THROW(tycho::utils::parallel_sequence(4,
                                          [](int i) {
                                              if (i == 3) // inline (last) index
                                                  throw std::runtime_error("inline exception");
                                          }),
                 std::runtime_error);
}

TEST(DispatchHelpers, WorkerDetection) {
    ScopedThreadCount guard(4);
    EXPECT_FALSE(tycho::utils::is_pool_worker());

    std::atomic<bool> worker_flag{false};
    auto fut = tycho::utils::thread_pool().submit_task(
        [&worker_flag] { worker_flag.store(tycho::utils::is_pool_worker()); });
    fut.get();
    EXPECT_TRUE(worker_flag.load());
}

TEST(ThreadPoolTest, ResetDrainsPending) {
    tycho::utils::ThreadPool pool(4);
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
        tycho::utils::parallel_sequence(4, [&counter](int) { counter.fetch_add(1); });
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

    tycho::utils::parallel_task(
        4, [&pool_counter] { pool_counter.fetch_add(1); },
        [&inner_sum] {
            // Inline arm — runs on main thread, nested dispatch is allowed.
            // parallel_blocks splits [0,8) into 4 blocks; each block sums its range.
            tycho::utils::parallel_blocks(
                8, [&inner_sum](int start, int end) { inner_sum.fetch_add(end - start); }, 4);
        });

    EXPECT_EQ(pool_counter.load(), 1);
    EXPECT_EQ(inner_sum.load(), 8); // sum of all block sizes = count
}

TEST(DispatchHelpers, NestedDispatch_PoolWorker_Throws) {
    // Dispatch from a pool worker must be rejected. parallel_sequence runs
    // i=0..2 on pool workers and i=3 inline; each attempts nested dispatch
    // and throws. DispatchContext's aggregation wraps the 3 concurrent
    // worker exceptions into a runtime_error whose message includes every
    // "nested dispatch from pool worker" string — callers can no longer
    // lose N-1 of the failures to stderr.
    ScopedThreadCount guard(4);
    try {
        tycho::utils::parallel_sequence(
            4, [](int) { tycho::utils::parallel_blocks(4, [](int, int) {}, 2); });
        FAIL() << "Expected exception";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("nested dispatch from pool worker"), std::string::npos)
            << "Aggregated message must include the underlying logic_error: " << msg;
    }
}

TEST(DispatchHelpers, DispatchContext_AggregatesUpToCapAndCountsSuppressed) {
    // DispatchContext caps captured exceptions at kMaxCaptured=16; throws
    // beyond that increment the suppressed counter and are reported in the
    // aggregated message tail. parallel_sequence(N) dispatches N-1 to the
    // pool and runs index N-1 inline; only pool throws flow through
    // DispatchContext, so to fire 17 captured-via-pool throws we need
    // n = 18 (indices 0..16 → pool, throwing; index 17 inline, also throwing
    // but rethrown only if the pool path is silent).
    ScopedThreadCount guard(4);
    constexpr int kPoolThrows = 17;
    constexpr int kSequenceN = kPoolThrows + 1;
    try {
        tycho::utils::parallel_sequence(kSequenceN, [](int i) {
            throw std::runtime_error("lane " + std::to_string(i));
        });
        FAIL() << "Expected aggregated runtime_error from " << kPoolThrows << " throwing lanes";
    } catch (const std::runtime_error &e) {
        std::string msg(e.what());
        // Header reports the total pool-captured count (stored + suppressed).
        EXPECT_NE(msg.find(std::to_string(kPoolThrows) + " concurrent worker exception(s)"),
                  std::string::npos)
            << "Header must report total throw count: " << msg;
        // Suppressed tail reports the count beyond the cap.
        EXPECT_NE(msg.find("and 1 more suppressed"), std::string::npos)
            << "Tail must report suppressed count: " << msg;
        // Cap was hit — slot index 15 must appear (last distinct stored slot).
        EXPECT_NE(msg.find("--- [15] ---"), std::string::npos)
            << "Capped slot index 15 must be present in body: " << msg;
        // Slot 16 must NOT appear — that one was suppressed, not stored.
        EXPECT_EQ(msg.find("--- [16] ---"), std::string::npos)
            << "Slot 16 must NOT be in body (suppressed by cap): " << msg;
    }
}

TEST(DispatchHelpers, DispatchContext_PreservesSingleExceptionType) {
    // The n==1 fast path rethrows the original exception unchanged so callers
    // see the typed exception (e.g., invalid_argument) rather than the
    // composed runtime_error. Verifies the typed-preservation contract that
    // decorate_trajectory's typed-rethrow ladder relies on.
    ScopedThreadCount guard(4);
    EXPECT_THROW(tycho::utils::parallel_sequence(4,
                                                 [](int i) {
                                                     if (i == 1) {
                                                         throw std::invalid_argument("typed");
                                                     }
                                                 }),
                 std::invalid_argument);
}

TEST(DispatchHelpers, DualException_ParallelTask) {
    // Both pool and inline arms throw. Pool exception should win.
    ScopedThreadCount guard(4);
    try {
        tycho::utils::parallel_task(
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
        tycho::utils::parallel_sequence(100, [&counter](int) { counter.fetch_add(1); });
        EXPECT_EQ(counter.load(), 100);
    }
}

TEST(ThreadPoolTest, SubmitTask_ExceptionPropagation) {
    tycho::utils::ThreadPool pool(2);
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

    tycho::utils::parallel_task(1, [&] { pool_order = order++; }, [&] { inline_order = order++; });

    EXPECT_EQ(pool_order, 0);   // pool_work runs first in sequential mode
    EXPECT_EQ(inline_order, 1); // inline_work runs second
}

TEST(DispatchHelpers, ParallelBlocks_CountLessThanNparts) {
    // count < nparts triggers the clamp: nparts = std::min(nparts, count).
    // Without it, block_size would be 0 producing incorrect partitions.
    ScopedThreadCount guard(4);
    std::atomic<int> sum{0};

    tycho::utils::parallel_blocks(2, [&sum](int start, int end) { sum.fetch_add(end - start); }, 10);

    EXPECT_EQ(sum.load(), 2);
}

TEST(DispatchHelpers, ParallelBlocks_ZeroCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    tycho::utils::parallel_blocks(0, [&](int, int) { called = true; }, 4);
    EXPECT_FALSE(called);
}

TEST(DispatchHelpers, ParallelSequence_ZeroCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    tycho::utils::parallel_sequence(0, [&](int) { called = true; });
    EXPECT_FALSE(called);
}

TEST(DispatchHelpers, ParallelBlocks_NegativeCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    tycho::utils::parallel_blocks(-1, [&](int, int) { called = true; }, 4);
    EXPECT_FALSE(called);
}

TEST(DispatchHelpers, ParallelSequence_NegativeCount) {
    ScopedThreadCount guard(4);
    bool called = false;
    tycho::utils::parallel_sequence(-5, [&](int) { called = true; });
    EXPECT_FALSE(called);
}

TEST(ThreadPoolTest, ConstructZeroThreadsThrows) {
    EXPECT_THROW(tycho::utils::ThreadPool pool(0), std::invalid_argument);
}

TEST(ThreadPoolTest, ResetZeroThreadsThrows) {
    tycho::utils::ThreadPool pool(2);
    EXPECT_THROW(ThreadPoolTestAccess::reset(pool, 0), std::invalid_argument);
    // Pool should still be functional after rejected reset.
    auto fut = pool.submit_task([] { return 7; });
    EXPECT_EQ(fut.get(), 7);
}

// =============================================================================
// task SBO wrapper
// =============================================================================

TEST(TaskSBOTest, DefaultConstructedIsEmpty) {
    tycho::utils::task t;
    EXPECT_FALSE(static_cast<bool>(t));
}

TEST(TaskSBOTest, ConstructFromLambdaIsNonEmpty) {
    bool invoked = false;
    tycho::utils::task t([&invoked] { invoked = true; });
    EXPECT_TRUE(static_cast<bool>(t));
    t();
    EXPECT_TRUE(invoked);
}

TEST(TaskSBOTest, MoveConstructTransfersOwnership) {
    int value = 0;
    tycho::utils::task t1([&value] { value = 42; });
    EXPECT_TRUE(static_cast<bool>(t1));

    tycho::utils::task t2(std::move(t1));
    EXPECT_FALSE(static_cast<bool>(t1));
    EXPECT_TRUE(static_cast<bool>(t2));
    t2();
    EXPECT_EQ(value, 42);
}

TEST(TaskSBOTest, MoveAssignOntoNonEmpty) {
    int a = 0, b = 0;
    tycho::utils::task t1([&a] { a = 1; });
    tycho::utils::task t2([&b] { b = 2; });

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
    tycho::utils::task t1;
    tycho::utils::task t2([&value] { value = 99; });

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
    tycho::utils::parallel_blocks(
        N, [&](int start, int end) { sum.fetch_add(end - start, std::memory_order_relaxed); }, 1);
    EXPECT_EQ(sum.load(), N);
}

TEST(DispatchHelpers, ParallelSequence_NOne_PoolActive) {
    ScopedThreadCount guard(4);
    std::atomic<int> count{0};
    tycho::utils::parallel_sequence(1, [&](int) { count.fetch_add(1, std::memory_order_relaxed); });
    EXPECT_EQ(count.load(), 1);
}
