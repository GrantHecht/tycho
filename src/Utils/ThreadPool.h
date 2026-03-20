#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

namespace Tycho {

// =============================================================================
// task — SBO callable wrapper (move-only, 64-byte inline buffer)
//
// Replaces std::function<void()> as the task unit. Zero heap allocation for
// all Tycho dispatch closures (verified <= 64 bytes). The static_assert
// fires at compile time if a closure exceeds the buffer.
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

    void operator()() { m_invoke(m_buf); }
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
// padding on m_index and m_tasks_pending prevents false sharing.
// =============================================================================

class ThreadPool {
    std::vector<WorkStealingQueue> m_queues;
    std::vector<std::thread> m_threads;
    unsigned m_count;

    static constexpr std::size_t CL =
#ifdef __cpp_lib_hardware_interference_size
        std::hardware_destructive_interference_size;
#else
        64;
#endif
    alignas(CL) std::atomic<unsigned> m_index{0};
    alignas(CL) std::atomic<int> m_tasks_pending{0};

    std::mutex m_wait_mutex;
    std::condition_variable m_wait_cv;
    static constexpr unsigned K = 3;

    void start_workers() {
        for (unsigned i = 0; i < m_count; ++i) {
            m_threads.emplace_back([this, i] {
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
                        // fire-and-forget: swallow exceptions
                    }
                    if (m_tasks_pending.fetch_sub(1, std::memory_order_release) == 1) {
                        std::lock_guard lock(m_wait_mutex);
                        m_wait_cv.notify_all();
                    }
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
        m_tasks_pending.fetch_add(1, std::memory_order_relaxed);
        task work{std::forward<F>(f)};
        unsigned i = m_index.fetch_add(1, std::memory_order_relaxed);
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
        std::unique_lock lock(m_wait_mutex);
        m_wait_cv.wait(lock,
                       [this] { return m_tasks_pending.load(std::memory_order_acquire) == 0; });
    }

    /// Shut down all workers and restart with n threads.
    void reset(unsigned n) {
        for (auto &q : m_queues)
            q.done();
        for (auto &t : m_threads)
            t.join();
        m_threads.clear();
        std::vector<WorkStealingQueue> fresh(n);
        m_queues.swap(fresh);
        m_count = n;
        m_index.store(0, std::memory_order_relaxed);
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
//   2. Enqueue N-1 tasks + run last partition inline + wait() (no future
//      overhead, calling thread stays productive)
//
// Exception safety: enqueue_work swallows exceptions from worker tasks. This is
// acceptable because NLP eval lambdas do not throw. Same as peak code.
//
// wait() invariant: ThreadPool::wait() waits for ALL enqueued tasks. This is
// safe because Tycho ensures only one parallel dispatch is active at a time
// per call chain. Nested dispatch (e.g. fillSolverCoeffs inside a KKT eval)
// works correctly: the inner wait() also waits for the outer enqueued task,
// acting as an implicit barrier.
// ---------------------------------------------------------------------------

/// Run func(start, stop) over [0, count) split into nparts blocks.
/// Dispatches N-1 blocks to pool, runs last block inline, then waits.
template <typename F> void parallel_blocks(int count, F &&func, int nparts) {
    if (count <= 0)
        return;
    nparts = std::min(nparts, count);
    if (nparts > 1 && use_thread_pool()) {
        int block_size = count / nparts;
        int remainder = count % nparts;
        int start = 0;
        for (int p = 0; p < nparts - 1; ++p) {
            int end = start + block_size + (p < remainder ? 1 : 0);
            thread_pool().enqueue_work([&func, start, end] { func(start, end); });
            start = end;
        }
        func(start, count); // last block inline
        thread_pool().wait();
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
        for (int i = 0; i < n - 1; ++i)
            thread_pool().enqueue_work([&func, i] { func(i); });
        func(n - 1); // last partition inline
        thread_pool().wait();
    } else {
        for (int i = 0; i < n; i++)
            func(i);
    }
}

/// Run a single task concurrently with inline_work on the calling thread.
/// Only dispatches to the pool when nparts > 1 AND the pool is active,
/// preventing deadlock when called from a pool worker (e.g. inside a Jet job).
/// Returns after both complete.
template <typename FTask, typename FInline>
void parallel_task(int nparts, FTask &&task, FInline &&inline_work) {
    if (nparts > 1 && use_thread_pool()) {
        thread_pool().enqueue_work(std::forward<FTask>(task));
        inline_work();
        thread_pool().wait();
    } else {
        task();
        inline_work();
    }
}

} // namespace Tycho
