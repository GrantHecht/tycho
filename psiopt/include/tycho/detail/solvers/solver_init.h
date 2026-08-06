#pragma once

namespace tycho::solvers {

/// Ensure the math runtime is initialized (thread pool, CPU dispatch, memory
/// allocators).  Thread-safe (std::call_once).
///
/// On MKL: calls dsecnd() to warm up the runtime.
/// On Accelerate: sets VECLIB_MAXIMUM_THREADS, triggers a trivial BLAS call,
/// and warms up the sparse solver subsystem (including MT-METIS thread pool).
///
/// Returns the wall-clock time in milliseconds the first init took,
/// or 0.0 if already initialized.
double ensure_solver_initialized();

} // namespace tycho::solvers
