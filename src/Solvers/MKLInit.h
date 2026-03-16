#pragma once

namespace Tycho {

/// Ensure the math runtime is initialized (thread pool, CPU dispatch, memory
/// allocators).  Thread-safe (std::call_once).
///
/// On MKL: calls dsecnd() to warm up the runtime.
/// On Accelerate: sets VECLIB_MAXIMUM_THREADS and triggers a trivial BLAS call
/// to initialize the thread pool.
///
/// Returns the wall-clock time in milliseconds the first init took,
/// or 0.0 if already initialized.
double ensure_mkl_initialized();

} // namespace Tycho
