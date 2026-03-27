#include "tycho/detail/utils/thread_pool.h"
#include <atomic>
#include <mutex>
#include <thread>

namespace {
std::atomic<int> g_num_threads{static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))};

/// Best-effort guard: true while set_num_threads() is resizing the pool.
/// Dispatch helpers check this and throw if true, catching the common misuse
/// of calling set_num_threads() concurrently with active computation.
/// Note: there is a narrow TOCTOU window between the dispatch helper's check
/// and the actual enqueue_work() call. Closing it fully would require a mutex
/// on the hot path, which is unacceptable for Tycho's dispatch frequency.
std::atomic<bool> g_pool_configuring{false};

/// Serializes concurrent set_num_threads() calls. Without this, two concurrent
/// callers would race on reset() — double-joining threads and data-racing on
/// internal vectors. Cold path only (startup), zero cost on the dispatch path.
std::mutex g_set_threads_mutex;
} // namespace

namespace Tycho {

ThreadPool &thread_pool() {
    static ThreadPool pool;
    return pool;
}

void set_num_threads(int n) {
    std::lock_guard lock(g_set_threads_mutex);
    if (n <= 1) {
        g_num_threads.store(1, std::memory_order_relaxed);
        // Pool stays alive but parallel_* helpers bypass it via use_thread_pool().
        // Idle workers block on their empty queues, consuming no CPU.
        // set_num_threads is startup-only.
    } else {
        g_pool_configuring.store(true, std::memory_order_release);
        try {
            thread_pool().reset(static_cast<unsigned>(n));
            // Store after reset(): ensures use_thread_pool() sees the new count
            // only after the pool is fully resized.
            g_num_threads.store(n, std::memory_order_release);
        } catch (...) {
            g_pool_configuring.store(false, std::memory_order_release);
            throw;
        }
        g_pool_configuring.store(false, std::memory_order_release);
    }
}

int get_num_threads() { return g_num_threads.load(std::memory_order_relaxed); }

bool pool_configuring() { return g_pool_configuring.load(std::memory_order_acquire); }

} // namespace Tycho
