#pragma once
#include <BS_thread_pool.hpp>

namespace Tycho {

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

} // namespace Tycho
