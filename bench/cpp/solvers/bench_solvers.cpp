///////////////////////////////////////////////////////////////////////////////
// PSIOPT solver benchmarks — end-to-end convergence
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>

///////////////////////////////////////////////////////////////////////////////
// PSIOPT solve_optimize benchmarks
// Each iteration constructs a fresh phase (solver modifies internal state)
///////////////////////////////////////////////////////////////////////////////

static void BM_PSIOPT_Brach_16seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_phase(16, 3);
        auto status = phase->solve_optimize();
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("PSIOPT did not converge (16seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_PSIOPT_Brach_16seg)->Unit(benchmark::kMillisecond);

static void BM_PSIOPT_Brach_32seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_phase(32, 3);
        auto status = phase->solve_optimize();
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("PSIOPT did not converge (32seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_PSIOPT_Brach_32seg)->Unit(benchmark::kMillisecond);
