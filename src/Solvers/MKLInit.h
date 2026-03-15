#pragma once

namespace Tycho {

/// Ensure the MKL runtime is initialized (thread pool, CPU dispatch, memory
/// allocators).  Thread-safe (std::call_once).  No-op on Apple Accelerate.
///
/// Returns the wall-clock time in milliseconds the first init took,
/// or 0.0 if already initialized or on Accelerate.
double ensure_mkl_initialized();

} // namespace Tycho
