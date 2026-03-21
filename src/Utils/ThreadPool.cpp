#include "ThreadPool.h"
#include <atomic>
#include <thread>

namespace {
std::atomic<int> g_num_threads{static_cast<int>(std::thread::hardware_concurrency())};

/// Best-effort guard: true while set_num_threads() is resizing the pool.
/// Dispatch helpers check this and throw if true, catching the common misuse
/// of calling set_num_threads() concurrently with active computation.
/// Note: there is a narrow TOCTOU window between the dispatch helper's check
/// and the actual enqueue_work() call. Closing it fully would require a mutex
/// on the hot path, which is unacceptable for Tycho's dispatch frequency.
std::atomic<bool> g_pool_configuring{false};
} // namespace

namespace Tycho {

ThreadPool &thread_pool() {
    static ThreadPool pool;
    return pool;
}

void set_num_threads(int n) {
    if (n <= 1) {
        g_num_threads.store(1, std::memory_order_relaxed);
        // Pool stays alive but parallel_* helpers bypass it via use_thread_pool().
        // Parked threads consume no CPU. set_num_threads is startup-only.
    } else {
        g_pool_configuring.store(true, std::memory_order_release);
        g_num_threads.store(n, std::memory_order_relaxed);
        thread_pool().reset(static_cast<unsigned>(n));
        g_pool_configuring.store(false, std::memory_order_release);
    }
}

int get_num_threads() { return g_num_threads.load(std::memory_order_relaxed); }

bool pool_configuring() { return g_pool_configuring.load(std::memory_order_acquire); }

} // namespace Tycho
