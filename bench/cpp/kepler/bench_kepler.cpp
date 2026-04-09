///////////////////////////////////////////////////////////////////////////////
// Kepler conversion and astrodynamics throughput benchmarks (Google Benchmark)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/detail/astro/kepler_utils.h>
#include <tycho/detail/astro/kepler_propagator.h>
#include <tycho/detail/astro/lambert_solvers.h>
#include <benchmark/benchmark.h>
#include <numbers>

using namespace tycho;
using namespace tycho::astro;

static constexpr double MU_EARTH = 398600.4418;

// LEO classical elements
static Vector6<double> leoClassic() {
    Vector6<double> oe;
    oe << 7000.0, 0.01, 28.5 * std::numbers::pi / 180.0, 45.0 * std::numbers::pi / 180.0,
        30.0 * std::numbers::pi / 180.0, 60.0 * std::numbers::pi / 180.0;
    return oe;
}

///////////////////////////////////////////////////////////////////////////////
// Classical <-> Cartesian conversions
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Modified equinoctial element conversions
///////////////////////////////////////////////////////////////////////////////

static void BM_ClassicToModified(benchmark::State &state) {
    auto oe = leoClassic();
    for (auto _ : state) {
        auto mee = classic_to_modified<double>(oe, MU_EARTH);
        benchmark::DoNotOptimize(mee);
    }
}
BENCHMARK(BM_ClassicToModified);

static void BM_ModifiedToCartesian(benchmark::State &state) {
    auto oe = leoClassic();
    auto mee = classic_to_modified<double>(oe, MU_EARTH);
    for (auto _ : state) {
        auto rv = modified_to_cartesian<double>(mee, MU_EARTH);
        benchmark::DoNotOptimize(rv);
    }
}
BENCHMARK(BM_ModifiedToCartesian);

static void BM_CartesianToModified(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (auto _ : state) {
        auto mee = cartesian_to_modified<double>(rv, MU_EARTH);
        benchmark::DoNotOptimize(mee);
    }
}
BENCHMARK(BM_CartesianToModified);

///////////////////////////////////////////////////////////////////////////////
// Kepler propagation — varying eccentricities
///////////////////////////////////////////////////////////////////////////////

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

static void BM_Propagate_Moderate(benchmark::State &state) {
    // Moderate eccentricity orbit: e=0.5, a=12000
    Vector6<double> oe;
    oe << 12000.0, 0.5, 28.5 * std::numbers::pi / 180.0, 0.0, 0.0, 0.0;
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (auto _ : state) {
        auto rv2 = propagate_cartesian<double>(rv, 300.0, MU_EARTH);
        benchmark::DoNotOptimize(rv2);
    }
}
BENCHMARK(BM_Propagate_Moderate);

static void BM_Propagate_Hyperbolic(benchmark::State &state) {
    // Hyperbolic orbit: e=1.5, a=-10000 (negative for hyperbolic)
    Vector6<double> oe;
    oe << -10000.0, 1.5, 10.0 * std::numbers::pi / 180.0, 0.0, 0.0, 0.1;
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    for (auto _ : state) {
        auto rv2 = propagate_cartesian<double>(rv, 100.0, MU_EARTH);
        benchmark::DoNotOptimize(rv2);
    }
}
BENCHMARK(BM_Propagate_Hyperbolic);

///////////////////////////////////////////////////////////////////////////////
// Lambert solver benchmarks
///////////////////////////////////////////////////////////////////////////////

static void BM_Lambert_ShortWay(benchmark::State &state) {
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << 0.0, 8000.0, 0.0;
    double dt = 2000.0;
    for (auto _ : state) {
        auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Lambert_ShortWay);

static void BM_Lambert_LongWay(benchmark::State &state) {
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << 0.0, 8000.0, 0.0;
    double dt = 5000.0;
    for (auto _ : state) {
        auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, true);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Lambert_LongWay);

static void BM_Lambert_MultiRev(benchmark::State &state) {
    Vector3<double> R1, R2;
    R1 << 7000.0, 0.0, 0.0;
    R2 << 0.0, 7000.0, 0.0;
    // Period of circular orbit at r=7000
    double T = 2.0 * std::numbers::pi * std::sqrt(7000.0 * 7000.0 * 7000.0 / MU_EARTH);
    double dt = 1.25 * T; // 1.25 revolutions
    for (auto _ : state) {
        auto result = lambert_izzo<double>(R1, R2, dt, MU_EARTH, false, 1, false);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Lambert_MultiRev);
