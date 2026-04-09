///////////////////////////////////////////////////////////////////////////////
// Optimal control benchmarks — OCP phase construction + transcription
///////////////////////////////////////////////////////////////////////////////

#include "../bench_phases.h"
#include <benchmark/benchmark.h>

///////////////////////////////////////////////////////////////////////////////
// Phase construction + transcription benchmarks
///////////////////////////////////////////////////////////////////////////////

static void BM_Phase_Construct_16seg(benchmark::State &state) {
    for (auto _ : state) {
        auto phase = make_brach_phase(16);
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Construct_16seg);

static void BM_Phase_Construct_64seg(benchmark::State &state) {
    for (auto _ : state) {
        auto phase = make_brach_phase(64);
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Construct_64seg);

///////////////////////////////////////////////////////////////////////////////
// Transcription-only benchmarks (exclude phase construction)
///////////////////////////////////////////////////////////////////////////////

static void BM_Phase_Transcribe_16seg(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto phase = make_brach_phase(16);
        state.ResumeTiming();
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Transcribe_16seg);

static void BM_Phase_Transcribe_64seg(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto phase = make_brach_phase(64);
        state.ResumeTiming();
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Transcribe_64seg);
