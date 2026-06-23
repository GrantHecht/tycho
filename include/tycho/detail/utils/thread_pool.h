#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <future>
#include <latch>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace tycho::utils {

// =============================================================================
// Worker thread identification + per-dispatch synchronization
// =============================================================================

namespace detail {

/// Set to true on each pool worker thread.
inline thread_local bool g_is_pool_worker{false};

/// @internal
/// @brief Per-dispatch synchronization context used by the parallel dispatch helpers.
///
/// Lives on the caller's stack. Workers hold `&ctx` — the caller MUST call
/// `done.wait()` before `ctx` goes out of scope.
///
/// Exception aggregation: captures up to `kMaxCaptured` distinct worker exceptions.
/// If a single worker throws, `rethrow_if_exception()` rethrows that exception
/// unchanged (preserves type). If multiple workers throw concurrently, it composes
/// a `std::runtime_error` whose message concatenates every captured `.what()`.
struct DispatchContext {
    // 16 lets a 256-trajectory STM that diverges in many lanes preserve
    // enough distinct messages to debug without forcing every caller to
    // unwrap a composite. Storage is per-DispatchContext on the dispatch
    // caller's stack, so the size cost is bounded.
    static constexpr int kMaxCaptured = 16; ///< @internal Maximum number of captured exceptions.

    std::latch done;           ///< @internal Countdown latch; each worker calls count_down().
    std::atomic<int> captured{0};   ///< @internal Number of exceptions stored so far.
    std::atomic<int> suppressed{0}; ///< @internal Count of exceptions beyond kMaxCaptured.
    std::array<std::exception_ptr, kMaxCaptured> exceptions{}; ///< @internal Stored exceptions.

    /// @internal
    /// @brief Construct with a latch count equal to the number of pool tasks.
    explicit DispatchContext(std::ptrdiff_t count) : done(count) {}

    /// Store an exception. Only called from catch blocks (cold path).
    /// fetch_add serializes slot assignment; excess beyond kMaxCaptured
    /// increments the suppressed counter (surfaced in the aggregated
    /// message).
    void store_exception() noexcept {
        int idx = captured.fetch_add(1, std::memory_order_acq_rel);
        if (idx < kMaxCaptured) {
            exceptions[idx] = std::current_exception();
        } else {
            suppressed.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// Rethrow stored exception(s). MUST be called after done.wait()
    /// (the latch provides the happens-before for the slot writes).
    void rethrow_if_exception() {
        int total = captured.load(std::memory_order_acquire);
        if (total == 0)
            return;

        int n = total < kMaxCaptured ? total : kMaxCaptured;
        // Preserve the original exception type when only one worker failed;
        // aggregation is only useful when distinct root causes coexist.
        if (n == 1) {
            std::rethrow_exception(exceptions[0]);
        }

        // Build the aggregated message, but fall back to surfacing the
        // first captured exception unchanged if string allocation fails:
        // letting bad_alloc escape would mask every captured worker error
        // with a misleading OOM at the dispatcher boundary.
        std::string msg;
        try {
            msg = "[Tycho] dispatch: ";
            msg += std::to_string(total);
            msg += " concurrent worker exception(s):";
            for (int i = 0; i < n; ++i) {
                msg += "\n--- [";
                msg += std::to_string(i);
                msg += "] ---\n";
                try {
                    std::rethrow_exception(exceptions[i]);
                } catch (const std::exception &e) {
                    msg += e.what();
                } catch (...) {
                    msg += "<non-std::exception>";
                }
            }
            int extra = suppressed.load(std::memory_order_relaxed);
            if (extra > 0) {
                msg += "\n... and ";
                msg += std::to_string(extra);
                msg += " more suppressed";
            }
        } catch (const std::bad_alloc &) {
            std::rethrow_exception(exceptions[0]);
        }
        throw std::runtime_error(msg);
    }
};

} // namespace detail

/// Returns true if the calling thread is a pool worker.
inline bool is_pool_worker() noexcept { return detail::g_is_pool_worker; }

// =============================================================================
// task — SBO callable wrapper (move-only, 64-byte inline buffer)
//
// Replaces std::function<void()> as the task unit. Zero heap allocation for
// all Tycho dispatch closures (verified <= 64 bytes). The static_assert
// fires at compile time if a closure exceeds the buffer.
//
// Current largest closures (approximate, 64-bit, Clang):
//   parallel_sequence wrapper: ~20 bytes (&func, &ctx, int i)
//   parallel_blocks wrapper:   ~28 bytes (&func, &ctx, int start, int end)
//   parallel_task wrapper:     ~24 bytes (captured lambda, &ctx)
// Headroom: ~36 bytes. Sizes are compiler/ABI-dependent; the static_assert
// in the constructor (line ~95) is the actual enforcement mechanism.
// =============================================================================

/// @internal
/// @brief Move-only SBO callable wrapper used as the pool task unit.
///
/// Stores any `std::invocable` callable inline (up to 64 bytes) with zero heap
/// allocation. A `static_assert` fires at instantiation time if the closure
/// exceeds the buffer. Use `operator()()` to invoke; `operator bool()` to check
/// whether the task is non-empty.
class task {
    static constexpr std::size_t BUF_SIZE = 64;
    static constexpr std::size_t ALIGN = alignof(std::max_align_t);

    alignas(ALIGN) unsigned char m_buf[BUF_SIZE];
    void (*m_invoke)(void *) = nullptr;
    void (*m_destroy)(void *) = nullptr;
    void (*m_move)(void *src, void *dst) = nullptr;

  public:
    task() = default;

    /// @internal
    /// @brief Construct from any invocable `F`; stores the callable in the SBO buffer.
    template <typename F>
        requires std::invocable<F &>
    task(F &&f) noexcept {
        using Fn = std::decay_t<F>;
        static_assert(sizeof(Fn) <= BUF_SIZE, "Callable exceeds task SBO buffer (64 bytes)");
        static_assert(alignof(Fn) <= ALIGN);
        static_assert(std::is_nothrow_move_constructible_v<Fn>);
        ::new (m_buf) Fn(std::forward<F>(f));
        m_invoke = [](void *p) { (*static_cast<Fn *>(p))(); };
        m_destroy = [](void *p) { static_cast<Fn *>(p)->~Fn(); };
        m_move = [](void *src, void *dst) {
            ::new (dst) Fn(std::move(*static_cast<Fn *>(src)));
            static_cast<Fn *>(src)->~Fn();
        };
    }

    /// @internal
    /// @brief Move constructor; transfers ownership of the stored callable.
    task(task &&o) noexcept : m_invoke(o.m_invoke), m_destroy(o.m_destroy), m_move(o.m_move) {
        if (m_move) {
            m_move(o.m_buf, m_buf);
            o.m_invoke = nullptr;
            o.m_destroy = nullptr;
            o.m_move = nullptr;
        }
    }

    /// @internal
    /// @brief Move-assignment operator; transfers ownership of the stored callable.
    task &operator=(task &&o) noexcept {
        if (this != &o) {
            if (m_destroy)
                m_destroy(m_buf);
            m_invoke = o.m_invoke;
            m_destroy = o.m_destroy;
            m_move = o.m_move;
            if (m_move) {
                m_move(o.m_buf, m_buf);
                o.m_invoke = nullptr;
                o.m_destroy = nullptr;
                o.m_move = nullptr;
            }
        }
        return *this;
    }

    ~task() {
        if (m_destroy)
            m_destroy(m_buf);
    }

    /// @internal
    /// @brief Invoke the stored callable. Asserts that the task is non-empty.
    void operator()() {
        assert(m_invoke && "task::operator() called on empty task");
        m_invoke(m_buf);
    }
    /// @internal
    /// @brief Returns true if the task holds a callable (i.e., is non-empty).
    explicit operator bool() const noexcept { return m_invoke != nullptr; }

    task(const task &) = delete;
    task &operator=(const task &) = delete;
};

// =============================================================================
// WorkStealingQueue — per-worker task queue
//
// Mutex-based work-stealing queue. Not lock-free — Tycho dispatches O(NumPartitions)
// tasks per PSIOPT iteration with ms-scale durations, so lock contention is negligible
// vs task cost. A lock-free Chase-Lev deque would be appropriate if tasks
// were sub-microsecond, but would add complexity for no measurable benefit
// at Tycho's task granularity.
//
// Owner pushes/pops from front (LIFO — cache locality).
// Thieves steal from back (FIFO — older tasks, reduces contention with owner's LIFO pops).
// =============================================================================

/// @internal
/// @brief Per-worker mutex-based work-stealing task queue used by `ThreadPool`.
///
/// Owner pushes/pops from the front (LIFO). Thief threads steal from the back (FIFO).
/// Not lock-free: at Tycho's ms-scale task granularity, mutex overhead is negligible.
class WorkStealingQueue {
    std::deque<task> m_deque;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_done = false;

  public:
    /// @internal
    /// @brief Blocking push to front of queue (owner side, always succeeds).
    void push(task f) {
        {
            std::lock_guard lock(m_mutex);
            m_deque.push_front(std::move(f));
        }
        m_cv.notify_one();
    }

    /// @internal
    /// @brief Non-blocking push to front; moves `f` on success, leaves it untouched on failure.
    bool try_push(task &f) {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock())
            return false;
        m_deque.push_front(std::move(f));
        lock.unlock();
        m_cv.notify_one();
        return true;
    }

    /// @internal
    /// @brief Non-blocking LIFO pop from front (owner side).
    bool try_pop(task &f) {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock() || m_deque.empty())
            return false;
        f = std::move(m_deque.front());
        m_deque.pop_front();
        return true;
    }

    /// @internal
    /// @brief Non-blocking FIFO steal from back (thief side).
    bool try_steal(task &f) {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock() || m_deque.empty())
            return false;
        f = std::move(m_deque.back());
        m_deque.pop_back();
        return true;
    }

    /// @internal
    /// @brief Blocking pop from front; returns false when the queue is shut down and empty.
    bool pop(task &f) {
        std::unique_lock lock(m_mutex);
        m_cv.wait(lock, [this] { return !m_deque.empty() || m_done; });
        if (m_deque.empty())
            return false;
        f = std::move(m_deque.front());
        m_deque.pop_front();
        return true;
    }

    /// @internal
    /// @brief Signal shutdown, waking all threads blocked in `pop()`.
    void done() {
        {
            std::lock_guard lock(m_mutex);
            m_done = true;
        }
        m_cv.notify_all();
    }
};

// =============================================================================
// ThreadPool — work-stealing thread pool
//
// Each worker has its own queue. Tasks are distributed round-robin on enqueue.
// Workers try their own queue first, then steal from others. Cache-line
// padding on m_tasks_pending prevents false sharing.
// =============================================================================

struct ThreadPoolTestAccess; // forward-declared for friend access from tests

/// @internal
/// @brief Work-stealing thread pool backing the Tycho parallel dispatch helpers.
///
/// Each worker thread has its own `WorkStealingQueue`. Tasks are enqueued
/// round-robin; workers try their own queue first, then steal from others.
/// Use `set_num_threads()`, `get_num_threads()`, and `thread_pool()` for all
/// external access — do not instantiate `ThreadPool` directly.
class ThreadPool {
    std::vector<WorkStealingQueue> m_queues;
    std::vector<std::thread> m_threads;
    unsigned m_count;

    // Apple Silicon L1 cache line is 128 bytes, not 64.
    static constexpr std::size_t CL =
#if defined(__APPLE__) && defined(__aarch64__)
        128;
#elif defined(__cpp_lib_hardware_interference_size)
        std::hardware_destructive_interference_size;
#else
        64;
#endif

    alignas(CL) std::atomic<int> m_tasks_pending{0};

    // TLS enqueue index — eliminates atomic contention on burst dispatch.
    // Must be class-level static (not function-local) because enqueue_work
    // is a template — function-local would create a separate counter per
    // lambda type, breaking round-robin distribution.
    static inline thread_local unsigned tl_enqueue_index{0};

    // try_push retry multiplier: on enqueue, try K * m_count queues before
    // falling back to blocking push on the original queue.
    static constexpr unsigned K = 3;

    void start_workers() {
        for (unsigned i = 0; i < m_count; ++i) {
            m_threads.emplace_back([this, i] {
                detail::g_is_pool_worker = true;
#if defined(__APPLE__)
                char name[16];
                std::snprintf(name, sizeof(name), "tycho-pool-%u", i);
                pthread_setname_np(name);
#elif defined(__linux__)
                char name[16];
                std::snprintf(name, sizeof(name), "tycho-pool-%u", i);
                pthread_setname_np(pthread_self(), name);
#elif defined(_WIN32)
                wchar_t name[32];
                swprintf(name, sizeof(name) / sizeof(wchar_t), L"tycho-pool-%u", i);
                SetThreadDescription(GetCurrentThread(), name);
#endif
                while (true) {
                    task f;
                    // Phase 1: try own queue (LIFO), then steal from others (FIFO)
                    if (!m_queues[i].try_pop(f)) {
                        for (unsigned n = 1; n < m_count; ++n) {
                            if (m_queues[(i + n) % m_count].try_steal(f))
                                break;
                        }
                    }
                    // Phase 2: if nothing found, block on own queue.
                    // Starvation note: a worker blocked here misses work on other
                    // queues. In practice, round-robin enqueue from a single
                    // dispatcher (Tycho's pattern) distributes evenly, and push()
                    // wakes this worker when its queue gets work. A timed wait
                    // (try_pop_for) was tested but pthread_cond_timedwait overhead
                    // caused 13-17% PSIOPT regression vs pthread_cond_wait.
                    if (!f) {
                        if (!m_queues[i].pop(f))
                            break; // shutdown
                    }
                    // Execute and signal completion. enqueue_work is the
                    // private fire-and-forget path — all public callers go
                    // through submit_task() / DispatchContext which own
                    // exception propagation. A throw here means a wrapper
                    // was bypassed; swallowing would let the main thread
                    // report success while a worker died. Fatal-terminate
                    // instead so the failure is unmissable.
                    try {
                        f();
                    } catch (const std::exception &e) {
                        std::fprintf(
                            stderr, "[Tycho] FATAL: unhandled exception in enqueue_work task: %s\n",
                            e.what());
                        std::terminate();
                    } catch (...) {
                        std::fprintf(
                            stderr,
                            "[Tycho] FATAL: unhandled non-std::exception in enqueue_work task\n");
                        std::terminate();
                    }
                    if (m_tasks_pending.fetch_sub(1, std::memory_order_release) == 1)
                        m_tasks_pending.notify_all();
                }
            });
        }
    }

    /// Fire-and-forget enqueue — no per-task heap allocation (SBO task wrapper).
    /// Private: callers should use submit_task() or the dispatch helpers
    /// (parallel_blocks, parallel_sequence, parallel_task).
    template <std::invocable F> void enqueue_work(F &&f) {
        // release: not strictly required (push() mutex provides the fence),
        // but documents the intent that m_tasks_pending publishes "work was
        // submitted". The correctness-critical pairing is:
        //   worker fetch_sub(release) → wait() load(acquire)
        // which ensures wait() sees task completion effects.
        m_tasks_pending.fetch_add(1, std::memory_order_release);
        task work{std::forward<F>(f)};
        unsigned i = tl_enqueue_index++;
        for (unsigned n = 0; n < m_count * K; ++n)
            if (m_queues[(i + n) % m_count].try_push(work))
                return;
        try {
            m_queues[i % m_count].push(std::move(work));
        } catch (...) {
            m_tasks_pending.fetch_sub(1, std::memory_order_release);
            throw;
        }
    }

    /// Shut down all workers and restart with n threads.
    /// Called only from set_num_threads(), which sets pool_configuring() to
    /// prevent concurrent dispatch (best-effort TOCTOU guard).
    /// This is a configuration API, not a hot-path operation.
    void reset(unsigned n) {
        if (n == 0)
            throw std::invalid_argument("ThreadPool: thread count must be > 0");
        wait(); // drain pending tasks (one atomic load when idle)
        for (auto &q : m_queues)
            q.done();
        for (auto &t : m_threads)
            t.join();
        m_threads.clear();
        std::vector<WorkStealingQueue> fresh(n);
        m_queues.swap(fresh);
        m_count = n;
        start_workers();
    }

  public:
    /// @internal
    /// @brief Construct the pool with `threads` worker threads (default: hardware concurrency).
    explicit ThreadPool(unsigned threads = std::max(1u, std::thread::hardware_concurrency()))
        : m_queues(threads), m_count(threads) {
        if (threads == 0)
            throw std::invalid_argument("ThreadPool: thread count must be > 0");
        start_workers();
    }

    ~ThreadPool() noexcept {
        wait();
        for (auto &q : m_queues)
            q.done();
        for (auto &t : m_threads)
            t.join();
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    // Dispatch helpers and set_num_threads need private access.
    template <typename F> friend void parallel_blocks(int, F &&, int);
    template <typename F> friend void parallel_sequence(int, F &&);
    template <typename FTask, typename FInline>
    friend void parallel_task(int, FTask &&, FInline &&);
    friend void set_num_threads(int n);
    friend struct ThreadPoolTestAccess;

    /// @brief Submit a callable to the pool and return a future for its result.
    ///
    /// Cold path only (Jet full solves, Integrator STM). Prefer the dispatch
    /// helpers (`parallel_blocks`, `parallel_sequence`, `parallel_task`) for
    /// hot-path parallel work.
    ///
    /// @tparam F  Callable with signature `R()` for any return type `R`.
    /// @param f   Callable to execute on a pool worker thread.
    /// @return    `std::future<R>` that holds the callable's return value (or
    ///            any exception it throws).
    template <std::invocable F>
    auto submit_task(F &&f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using R = std::invoke_result_t<std::decay_t<F>>;
        auto pt = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto future = pt->get_future();
        enqueue_work([pt = std::move(pt)]() { (*pt)(); });
        return future;
    }

    /// @brief Block the calling thread until all enqueued tasks have completed.
    void wait() {
        int val = m_tasks_pending.load(std::memory_order_acquire);
        while (val != 0) {
            m_tasks_pending.wait(val, std::memory_order_acquire);
            val = m_tasks_pending.load(std::memory_order_acquire);
        }
    }

    /// @internal
    /// @brief Return the number of worker threads in the pool.
    unsigned get_thread_count() const noexcept { return m_count; }
};

// ---------------------------------------------------------------------------
// Global thread budget
//
// set_num_threads() is the process-global execution budget. It is meant to be
// called once at startup (or from a top-level Python script), NOT from deep
// inside algorithms. Algorithms should never change it.
// ---------------------------------------------------------------------------

/// @brief Return the process-global thread pool (Meyers singleton, fixed address).
///
/// Created lazily on first call, sized to `std::thread::hardware_concurrency()`.
/// The returned reference is stable for the lifetime of the process.
ThreadPool &thread_pool();

/// @brief Set the number of threads used for parallel work.
///
/// @param n Thread count. `n <= 1` (including 0/negative) selects
///          single-threaded mode: work runs inline on the calling thread;
///          the pool stays alive but is bypassed via `use_thread_pool()`.
///          `n > 1` resizes the pool to @p n worker threads.
void set_num_threads(int n);

/// @brief Return the current thread count setting.
///
/// @return The number of threads. 1 means single-threaded mode.
int get_num_threads();

/// @brief Return true if the global pool should be used (`num_threads > 1`).
inline bool use_thread_pool() { return get_num_threads() > 1; }

/// @brief Return true if `set_num_threads()` is currently mid-reset.
///
/// Best-effort safety net: dispatch helpers throw if this returns true,
/// catching the misuse of calling a parallel dispatch helper concurrently
/// while `set_num_threads()` is resizing the pool.
bool pool_configuring();

// ---------------------------------------------------------------------------
// Safe parallel dispatch helpers
//
// These wrap ThreadPool's enqueue_work API with:
//   1. Automatic inline fallback when use_thread_pool() is false or nparts <= 1
//   2. Per-dispatch synchronization via std::latch (no global wait())
//   3. Exception propagation — first exception from any task is rethrown
//   4. Calling thread runs the last partition inline (stays productive)
//
// Per-dispatch latches make concurrent dispatches from separate threads
// inherently safe — each dispatch waits only on its own latch.
// ---------------------------------------------------------------------------

/// @brief Partition `[0, count)` into @p nparts blocks and run `func(start, stop)` in parallel.
///
/// Dispatches `nparts-1` blocks to the thread pool and runs the last block
/// inline on the calling thread, then waits for all pool tasks to finish.
/// `nparts` is first clamped to `count` (`nparts = std::min(nparts, count)`),
/// so the single-call fallback also fires when `count == 1`.
/// Falls back to a single `func(0, count)` call when `nparts <= 1` or the
/// pool is inactive (`use_thread_pool()` is false).
///
/// @tparam F    Callable with signature `void(int start, int stop)`.
/// @param count  Total range size; no-op if `count <= 0`.
/// @param func   Callable invoked once per block with the half-open range.
/// @param nparts Requested number of partitions; clamped to `count`.
template <typename F> void parallel_blocks(int count, F &&func, int nparts) {
    if (count <= 0)
        return;
    nparts = std::min(nparts, count);
    if (nparts > 1 && use_thread_pool()) {
        if (detail::g_is_pool_worker)
            throw std::logic_error(
                "tycho::utils::parallel_blocks: nested dispatch from pool worker");
        if (pool_configuring())
            throw std::logic_error(
                "tycho::utils::parallel_blocks: dispatch during set_num_threads");
        int pool_tasks = nparts - 1;
        detail::DispatchContext ctx(pool_tasks);
        int block_size = count / nparts;
        int remainder = count % nparts;
        int start = 0;
        for (int p = 0; p < pool_tasks; ++p) {
            int end = start + block_size + (p < remainder ? 1 : 0);
            thread_pool().enqueue_work([&func, &ctx, start, end] {
                try {
                    func(start, end);
                } catch (...) {
                    ctx.store_exception();
                }
                ctx.done.count_down();
            });
            start = end;
        }
        // Run last block inline on calling thread
        std::exception_ptr inline_ex;
        try {
            func(start, count);
        } catch (...) {
            inline_ex = std::current_exception();
        }
        ctx.done.wait(); // MUST complete before ctx destruction
        ctx.rethrow_if_exception();
        if (inline_ex)
            std::rethrow_exception(inline_ex);
    } else {
        func(0, count);
    }
}

/// @brief Invoke `func(i)` for each `i` in `[0, n)` in parallel.
///
/// Dispatches `n-1` tasks to the thread pool and runs the last index
/// inline on the calling thread, then waits for all pool tasks to finish.
/// Falls back to a sequential loop when `n <= 1` or the pool is inactive.
///
/// @tparam F  Callable with signature `void(int i)`.
/// @param n    Number of iterations; no-op if `n <= 0`.
/// @param func Callable invoked once per index.
template <typename F> void parallel_sequence(int n, F &&func) {
    if (n <= 0)
        return;
    if (n > 1 && use_thread_pool()) {
        if (detail::g_is_pool_worker)
            throw std::logic_error(
                "tycho::utils::parallel_sequence: nested dispatch from pool worker");
        if (pool_configuring())
            throw std::logic_error(
                "tycho::utils::parallel_sequence: dispatch during set_num_threads");
        int pool_tasks = n - 1;
        detail::DispatchContext ctx(pool_tasks);
        for (int i = 0; i < pool_tasks; ++i)
            thread_pool().enqueue_work([&func, &ctx, i] {
                try {
                    func(i);
                } catch (...) {
                    ctx.store_exception();
                }
                ctx.done.count_down();
            });
        // Run last index inline on calling thread
        std::exception_ptr inline_ex;
        try {
            func(n - 1);
        } catch (...) {
            inline_ex = std::current_exception();
        }
        ctx.done.wait();
        ctx.rethrow_if_exception();
        if (inline_ex)
            std::rethrow_exception(inline_ex);
    } else {
        for (int i = 0; i < n; i++)
            func(i);
    }
}

/// @brief Run `pool_work` on the pool concurrently with `inline_work` on the calling thread.
///
/// Submits `pool_work` to the thread pool and immediately executes
/// `inline_work` on the caller, then waits for the pool task to finish.
/// Only dispatches when `nparts > 1` AND the pool is active — when
/// `nparts == 1` (e.g. during Jet with NumPartitions=1), both callables
/// run sequentially inline, avoiding deadlock from nested dispatch.
///
/// @tparam FTask    Callable with signature `void()` sent to the pool.
/// @tparam FInline  Callable with signature `void()` run on the calling thread.
/// @param nparts     Number of active partitions; both run inline when `<= 1`.
/// @param pool_work  Callable executed on a pool worker thread.
/// @param inline_work Callable executed on the calling thread concurrently.
template <typename FTask, typename FInline>
void parallel_task(int nparts, FTask &&pool_work, FInline &&inline_work) {
    if (nparts > 1 && use_thread_pool()) {
        if (detail::g_is_pool_worker)
            throw std::logic_error("tycho::utils::parallel_task: nested dispatch from pool worker");
        if (pool_configuring())
            throw std::logic_error("tycho::utils::parallel_task: dispatch during set_num_threads");
        detail::DispatchContext ctx(1);
        thread_pool().enqueue_work([f = std::forward<FTask>(pool_work), &ctx] {
            try {
                f();
            } catch (...) {
                ctx.store_exception();
            }
            ctx.done.count_down();
        });
        std::exception_ptr inline_ex;
        try {
            inline_work();
        } catch (...) {
            inline_ex = std::current_exception();
        }
        // Exception priority: pool exception is rethrown first. If both arms
        // throw (extremely unlikely — independent numerical failures), the
        // inline exception is discarded. Both would indicate the same
        // underlying issue; nested_exception adds complexity for no benefit.
        ctx.done.wait();
        ctx.rethrow_if_exception();
        if (inline_ex)
            std::rethrow_exception(inline_ex);
    } else {
        // Fallback: execute sequentially (pool_work first, then inline_work).
        // In the parallel path above, execution order is unspecified.
        pool_work();
        inline_work();
    }
}

} // namespace tycho::utils
