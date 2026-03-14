///////////////////////////////////////////////////////////////////////////////
// Kepler conversion throughput benchmarks (Google Benchmark)
///////////////////////////////////////////////////////////////////////////////

#include "Astro/KeplerUtils.h"
#include <benchmark/benchmark.h>

using namespace Tycho;

static constexpr double MU_EARTH = 398600.4418;

// LEO classical elements
static Vector6<double> leoClassic() {
    Vector6<double> oe;
    oe << 7000.0, 0.01, 28.5 * M_PI / 180.0, 45.0 * M_PI / 180.0, 30.0 * M_PI / 180.0,
        60.0 * M_PI / 180.0;
    return oe;
}

static void BM_ClassicToCartesian(benchmark::State &state) {
    auto oe = leoClassic();
    for (auto _ : state) {
        auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
        benchmark::DoNotOptimize(rv);
    }
}
BENCHMARK(BM_ClassicToCartesian);

static void BM_CartesianToClassic(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (auto _ : state) {
        auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
        benchmark::DoNotOptimize(oe2);
    }
}
BENCHMARK(BM_CartesianToClassic);

static void BM_RoundTrip(benchmark::State &state) {
    auto oe = leoClassic();
    for (auto _ : state) {
        auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
        auto oe2 = cartesian_to_classic<double>(rv, MU_EARTH);
        benchmark::DoNotOptimize(oe2);
    }
}
BENCHMARK(BM_RoundTrip);

static void BM_PropagateCartesian(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    double dt = 300.0; // 5 minutes
    for (auto _ : state) {
        auto rv2 = propagate_cartesian<double>(rv, dt, MU_EARTH);
        benchmark::DoNotOptimize(rv2);
    }
}
BENCHMARK(BM_PropagateCartesian);
