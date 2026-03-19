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

/// Returns the process-global thread pool (Meyers singleton, fixed address).
/// Created lazily on first call, sized to hardware_concurrency().
BS::thread_pool<> &thread_pool();

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
// These wrap BS::thread_pool's detach_* APIs with:
//   1. Automatic inline fallback when use_thread_pool() is false or nparts <= 1
//   2. Detach N-1 tasks + run last partition inline + wait() (no promise/future
//      overhead, calling thread stays productive)
//
// Exception safety: detach_* swallows exceptions from worker tasks. This is
// acceptable because NLP eval lambdas do not throw. Same as peak code.
//
// wait() invariant: BS::thread_pool::wait() waits for ALL queued tasks, not
// per-batch. This is safe because Tycho ensures only one parallel dispatch is
// active at a time per call chain. Nested dispatch (e.g. fillSolverCoeffs
// inside a KKT eval) works correctly: the inner wait() also waits for the
// outer detached task, acting as an implicit barrier.
// ---------------------------------------------------------------------------

/// Run func(start, stop) over [0, count) split into nparts blocks.
/// All blocks dispatched to pool + wait (no inline block — BS::blocks computes
/// ranges internally and we cannot easily extract one).
/// WARNING: exceptions from worker tasks are swallowed (detach_* API).
/// NOTE: wait() is global — do not nest parallel_* calls.
template <typename F> void parallel_blocks(int count, F &&func, int nparts) {
    if (count <= 0)
        return;
    if (nparts > 1 && use_thread_pool()) {
        thread_pool().detach_blocks(0, count, std::forward<F>(func), static_cast<size_t>(nparts));
        thread_pool().wait();
    } else {
        func(0, count);
    }
}

/// Run func(i) for i in [0, n). Dispatches N-1 tasks to pool, runs last
/// partition inline on calling thread, then waits.
/// WARNING: exceptions from worker tasks are swallowed (detach_* API).
/// NOTE: wait() is global — do not nest parallel_* calls.
template <typename F> void parallel_sequence(int n, F &&func) {
    if (n <= 0)
        return;
    if (n > 1 && use_thread_pool()) {
        thread_pool().detach_sequence(0, n - 1, func); // N-1 tasks to pool
        func(n - 1);                                   // last partition inline
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
/// WARNING: exceptions from worker tasks are swallowed (detach_* API).
/// NOTE: wait() is global — do not nest parallel_* calls.
template <typename FTask, typename FInline>
void parallel_task(int nparts, FTask &&task, FInline &&inline_work) {
    if (nparts > 1 && use_thread_pool()) {
        thread_pool().detach_task(std::forward<FTask>(task));
        inline_work();
        thread_pool().wait();
    } else {
        task();
        inline_work();
    }
}

} // namespace Tycho
