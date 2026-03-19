#include "ThreadPool.h"
#include <atomic>
#include <memory>
#include <thread>

namespace {
std::atomic<int> g_num_threads{static_cast<int>(std::thread::hardware_concurrency())};

// Lazily-created pool, managed via unique_ptr so set_num_threads(1) can destroy it.
// set_num_threads() is documented as startup-only and must not be called
// concurrently with pool usage.
std::unique_ptr<BS::thread_pool<>> g_pool;
} // namespace

namespace Tycho {

BS::thread_pool<> &thread_pool() {
    if (!g_pool) {
        int n = std::max(g_num_threads.load(std::memory_order_relaxed), 2);
        g_pool = std::make_unique<BS::thread_pool<>>(static_cast<std::size_t>(n));
    }
    return *g_pool;
}

void set_num_threads(int n) {
    if (n <= 1) {
        g_num_threads.store(1, std::memory_order_relaxed);
        g_pool.reset(); // Destroy pool, reclaim worker threads.
                        // thread_pool() will lazily recreate if num_threads is later raised.
    } else {
        g_num_threads.store(n, std::memory_order_relaxed);
        if (g_pool) {
            g_pool->reset(static_cast<std::size_t>(n));
        } else {
            g_pool = std::make_unique<BS::thread_pool<>>(static_cast<std::size_t>(n));
        }
    }
}

int get_num_threads() { return g_num_threads.load(std::memory_order_relaxed); }

} // namespace Tycho
