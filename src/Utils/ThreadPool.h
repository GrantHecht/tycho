#pragma once
#include <BS_thread_pool.hpp>

namespace Tycho {

// ---------------------------------------------------------------------------
// Global thread budget
//
// set_num_threads() is the process-global execution budget. It is meant to be
// called once at startup (or from a top-level Python script), NOT from deep
// inside algorithms. Algorithms should never change it.
// ---------------------------------------------------------------------------

/// Returns the process-global thread pool.
/// Created lazily on first call, sized to hardware_concurrency().
BS::thread_pool<>& thread_pool();

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
// These wrap BS::thread_pool's submit_*/detach_* APIs with:
//   1. Automatic inline fallback when use_thread_pool() is false or nparts <= 1
//   2. Exception propagation via submit_*().get() (not detach_* + wait())
//
// Use these at all call sites instead of open-coding detach/wait pairs.
// ---------------------------------------------------------------------------

/// Run func(start, stop) over [0, count) split into nparts blocks.
/// Exceptions from any block propagate to the caller.
template <typename F>
void parallel_blocks(int count, F &&func, int nparts) {
    if (count <= 0)
        return;
    if (nparts > 1 && use_thread_pool()) {
        auto futures =
            thread_pool().submit_blocks(0, count, std::forward<F>(func), static_cast<size_t>(nparts));
        futures.get();
    } else {
        func(0, count);
    }
}

/// Run func(i) for i in [0, n). Each index is a separate task.
/// Exceptions from any task propagate to the caller.
template <typename F>
void parallel_sequence(int n, F &&func) {
    if (n <= 0)
        return;
    if (n > 1 && use_thread_pool()) {
        auto futures = thread_pool().submit_sequence(0, n, std::forward<F>(func));
        futures.get();
    } else {
        for (int i = 0; i < n; i++)
            func(i);
    }
}

/// Run a single task concurrently with inline_work on the calling thread.
/// Both exceptions propagate. Returns after both complete.
template <typename FTask, typename FInline>
void parallel_task(FTask &&task, FInline &&inline_work) {
    if (use_thread_pool()) {
        auto future = thread_pool().submit_task(std::forward<FTask>(task));
        inline_work();
        future.get();
    } else {
        task();
        inline_work();
    }
}

} // namespace Tycho
