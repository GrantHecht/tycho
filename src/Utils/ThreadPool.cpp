#include "ThreadPool.h"
#include <atomic>
#include <thread>

namespace {
std::atomic<int> g_num_threads{static_cast<int>(std::thread::hardware_concurrency())};
} // namespace

namespace Tycho {

BS::thread_pool<> &thread_pool() {
    static BS::thread_pool<> pool;
    return pool;
}

void set_num_threads(int n) {
    if (n <= 1) {
        g_num_threads.store(1, std::memory_order_relaxed);
        // Pool stays alive but parallel_* helpers bypass it via use_thread_pool().
        // Parked threads consume no CPU. set_num_threads is startup-only.
    } else {
        g_num_threads.store(n, std::memory_order_relaxed);
        thread_pool().reset(static_cast<std::size_t>(n));
    }
}

int get_num_threads() { return g_num_threads.load(std::memory_order_relaxed); }

} // namespace Tycho
