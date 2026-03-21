#pragma once

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
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace Tycho {

// =============================================================================
// Worker thread identification + per-dispatch synchronization
// =============================================================================

namespace detail {

/// Set to true on each pool worker thread.
inline thread_local bool g_is_pool_worker{false};

/// Per-dispatch synchronization context.
/// Lives on the caller's stack. Workers hold &ctx — caller MUST wait()
/// on the latch before ctx goes out of scope.
struct DispatchContext {
    std::latch done;
    std::atomic<bool> has_exception{false};
    std::exception_ptr first_exception;

    explicit DispatchContext(std::ptrdiff_t count) : done(count) {}

    /// Store the first exception. Only called from catch blocks (cold path).
    /// Uses seq_cst — correctness is obvious without reasoning about
    /// latch synchronization chains. Cost: ~0 (never executed on happy path).
    void store_exception() noexcept {
        if (!has_exception.exchange(true))
            first_exception = std::current_exception();
    }

    /// Rethrow stored exception. MUST be called after done.wait().
    void rethrow_if_exception() {
        if (has_exception.load())
            std::rethrow_exception(first_exception);
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
// Current largest closures (approximate, 64-bit):
//   parallel_sequence wrapper: ~20 bytes (&func, &ctx, int i)
//   parallel_blocks wrapper:   ~28 bytes (&func, &ctx, int start, int end)
//   parallel_task wrapper:     ~24 bytes (captured lambda, &ctx)
// Headroom: ~36 bytes
// =============================================================================

class task {
    static constexpr std::size_t BUF_SIZE = 64;
    static constexpr std::size_t ALIGN = alignof(std::max_align_t);

    alignas(ALIGN) unsigned char m_buf[BUF_SIZE];
    void (*m_invoke)(void *) = nullptr;
    void (*m_destroy)(void *) = nullptr;
    void (*m_move)(void *src, void *dst) = nullptr;

  public:
    task() = default;

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

    task(task &&o) noexcept : m_invoke(o.m_invoke), m_destroy(o.m_destroy), m_move(o.m_move) {
        if (m_move) {
            m_move(o.m_buf, m_buf);
            o.m_invoke = nullptr;
            o.m_destroy = nullptr;
            o.m_move = nullptr;
        }
    }

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

    void operator()() {
        assert(m_invoke && "task::operator() called on empty task");
        m_invoke(m_buf);
    }
    explicit operator bool() const noexcept { return m_invoke != nullptr; }

    task(const task &) = delete;
    task &operator=(const task &) = delete;
};

// =============================================================================
// WorkStealingQueue — per-worker task queue
//
// Owner pushes/pops from front (LIFO — cache locality).
// Thieves steal from back (FIFO — older, larger work units).
// Mutex-based: Tycho dispatches 10-100 tasks per cycle with ms-scale
// durations, so lock contention is negligible vs task cost.
// =============================================================================

class WorkStealingQueue {
    std::deque<task> m_deque;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_done = false;

  public:
    // Owner pushes to front (blocking, always succeeds)
    void push(task f) {
        {
            std::lock_guard lock(m_mutex);
            m_deque.push_front(std::move(f));
        }
        m_cv.notify_one();
    }

    // Non-blocking push. Moves from f on success, leaves f untouched on failure.
    bool try_push(task &f) {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock())
            return false;
        m_deque.push_front(std::move(f));
        lock.unlock();
        m_cv.notify_one();
        return true;
    }

    // Owner pops from front (LIFO, non-blocking)
    bool try_pop(task &f) {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock() || m_deque.empty())
            return false;
        f = std::move(m_deque.front());
        m_deque.pop_front();
        return true;
    }

    // Thief steals from back (FIFO, non-blocking)
    bool try_steal(task &f) {
        std::unique_lock lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock() || m_deque.empty())
            return false;
        f = std::move(m_deque.back());
        m_deque.pop_back();
        return true;
    }

    // Owner blocks until work arrives or done() is called.
    // Returns false on shutdown (m_done && empty).
    bool pop(task &f) {
        std::unique_lock lock(m_mutex);
        m_cv.wait(lock, [this] { return !m_deque.empty() || m_done; });
        if (m_deque.empty())
            return false;
        f = std::move(m_deque.front());
        m_deque.pop_front();
        return true;
    }

    // Signal shutdown — wakes all blocked pop() calls
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
                    // Phase 2: if nothing found, block on own queue
                    if (!f) {
                        if (!m_queues[i].pop(f))
                            break; // shutdown
                    }
                    // Execute and signal completion
                    try {
                        f();
                    } catch (...) {
                        // Safety net for raw enqueue_work users.
                        // Dispatch helpers handle their own exceptions via DispatchContext.
                    }
                    if (m_tasks_pending.fetch_sub(1, std::memory_order_release) == 1)
                        m_tasks_pending.notify_all();
                }
            });
        }
    }

  public:
    explicit ThreadPool(unsigned threads = std::thread::hardware_concurrency())
        : m_queues(threads), m_count(threads) {
        assert(threads > 0);
        start_workers();
    }

    ~ThreadPool() noexcept {
        for (auto &q : m_queues)
            q.done();
        for (auto &t : m_threads)
            t.join();
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    /// Fire-and-forget enqueue — zero heap allocation (SBO task wrapper).
    template <std::invocable F> void enqueue_work(F &&f) {
        m_tasks_pending.fetch_add(1, std::memory_order_release);
        task work{std::forward<F>(f)};
        unsigned i = tl_enqueue_index++;
        for (unsigned n = 0; n < m_count * K; ++n)
            if (m_queues[(i + n) % m_count].try_push(work))
                return;
        m_queues[i % m_count].push(std::move(work));
    }

    /// Submit a task and get a future for its result.
    /// Cold path only (Jet full solves, Integrator STM).
    template <std::invocable F>
    auto submit_task(F &&f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using R = std::invoke_result_t<std::decay_t<F>>;
        auto pt = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto future = pt->get_future();
        enqueue_work([pt = std::move(pt)]() { (*pt)(); });
        return future;
    }

    /// Block until all enqueued tasks have completed.
    void wait() {
        int val = m_tasks_pending.load(std::memory_order_acquire);
        while (val != 0) {
            m_tasks_pending.wait(val, std::memory_order_acquire);
            val = m_tasks_pending.load(std::memory_order_acquire);
        }
    }

    /// Shut down all workers and restart with n threads.
    /// Precondition: no concurrent enqueue_work() or submit_task() calls.
    /// This is a configuration API, not a hot-path operation.
    void reset(unsigned n) {
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

    unsigned get_thread_count() const noexcept { return m_count; }
};

// ---------------------------------------------------------------------------
// Global thread budget
//
// set_num_threads() is the process-global execution budget. It is meant to be
// called once at startup (or from a top-level Python script), NOT from deep
// inside algorithms. Algorithms should never change it.
// ---------------------------------------------------------------------------

/// Returns the process-global thread pool (Meyers singleton, fixed address).
/// Created lazily on first call, sized to hardware_concurrency().
ThreadPool &thread_pool();

/// Set the number of threads used for parallel work.
/// n <= 1: single-threaded mode (all work runs inline, pool is never touched).
/// n > 1: resize the pool to n threads.
void set_num_threads(int n);

/// Returns the current thread count setting. 1 = single-threaded mode.
int get_num_threads();

/// Fast check: returns true if the global pool should be used (num_threads > 1).
inline bool use_thread_pool() { return get_num_threads() > 1; }

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

/// Run func(start, stop) over [0, count) split into nparts blocks.
/// Dispatches N-1 blocks to pool, runs last block inline, then waits.
template <typename F> void parallel_blocks(int count, F &&func, int nparts) {
    if (count <= 0)
        return;
    nparts = std::min(nparts, count);
    if (nparts > 1 && use_thread_pool()) {
        if (detail::g_is_pool_worker)
            throw std::logic_error("Tycho::parallel_blocks: nested dispatch from pool worker");
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

/// Run func(i) for i in [0, n). Dispatches N-1 tasks to pool, runs last
/// partition inline on calling thread, then waits.
template <typename F> void parallel_sequence(int n, F &&func) {
    if (n <= 0)
        return;
    if (n > 1 && use_thread_pool()) {
        if (detail::g_is_pool_worker)
            throw std::logic_error("Tycho::parallel_sequence: nested dispatch from pool worker");
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

/// Run pool_work concurrently with inline_work on the calling thread.
/// Only dispatches when nparts > 1 AND the pool is active — when
/// nparts == 1 (e.g. during Jet with NumPartitions=1), both run inline,
/// avoiding deadlock from nested dispatch.
template <typename FTask, typename FInline>
void parallel_task(int nparts, FTask &&pool_work, FInline &&inline_work) {
    if (nparts > 1 && use_thread_pool()) {
        if (detail::g_is_pool_worker)
            throw std::logic_error("Tycho::parallel_task: nested dispatch from pool worker");
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

} // namespace Tycho
