///////////////////////////////////////////////////////////////////////////////
// Benchmark runtime initialization — compiled once into bench_all.
// Decoupled from per-TU headers so lightweight benchmarks (utils, kepler)
// don't pay for solver/OC header processing.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/solvers.h>
#include <tycho/utils.h>
#include <fmt/core.h>

namespace {
const bool g_bench_init = [] {
    tycho::utils::BumpAllocator::resize(256, 256);
    double init_ms = tycho::solvers::ensure_solver_initialized();
#ifdef USE_ACCELERATE_SPARSE
    fmt::print("Accelerate init: {:.3f} ms (VECLIB_MAXIMUM_THREADS={})\n", init_ms,
               TYCHO_DEFAULT_QP_THREADS);
#else
    if (init_ms > 0.0)
        fmt::print("MKL init: {:.3f} ms\n", init_ms);
#endif
    return true;
}();
} // namespace
