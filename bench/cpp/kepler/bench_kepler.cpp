///////////////////////////////////////////////////////////////////////////////
// Kepler conversion and astrodynamics throughput benchmarks (Google Benchmark)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/detail/astro/conversions/kepler_utils.h>
#include <tycho/detail/astro/kepler/kepler_propagation.h>
#include <tycho/detail/astro/kepler/kepler_propagator.h>
#include <tycho/detail/astro/kepler/lambert_solvers.h>
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

///////////////////////////////////////////////////////////////////////////////
// KeplerPropagator VF benchmarks (post-LCD-rewrite)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/detail/astro/kepler/kepler_propagator.h>

static void BM_KeplerPropagator_VF_Compute(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    Eigen::Matrix<double, 7, 1> in;
    in.head<6>() = rv;
    in[6] = 300.0;
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<double, 6, 1> out;
    for (auto _ : state) {
        prop.compute_impl(in, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_Compute);

static void BM_KeplerPropagator_VF_Jacobian(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    Eigen::Matrix<double, 7, 1> in;
    in.head<6>() = rv;
    in[6] = 300.0;
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<double, 6, 1> out;
    Eigen::Matrix<double, 6, 7> jac;
    for (auto _ : state) {
        prop.compute_jacobian_impl(in, out, jac);
        benchmark::DoNotOptimize(jac);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_Jacobian);

static void BM_KeplerPropagator_VF_AdjointHessian(benchmark::State &state) {
    auto oe = leoClassic();
    auto rv = classic_to_cartesian<double>(oe, MU_EARTH);
    Eigen::Matrix<double, 7, 1> in;
    in.head<6>() = rv;
    in[6] = 300.0;
    Eigen::Matrix<double, 6, 1> lm = Eigen::Matrix<double, 6, 1>::Constant(0.5);
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<double, 6, 1> out;
    Eigen::Matrix<double, 6, 7> jac;
    Eigen::Matrix<double, 7, 1> grad;
    Eigen::Matrix<double, 7, 7> hess;
    for (auto _ : state) {
        prop.compute_jacobian_adjointgradient_adjointhessian_impl(
            in, out, jac, grad, hess, lm);
        benchmark::DoNotOptimize(hess);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_AdjointHessian);

static void BM_KeplerPropagator_VF_Compute_SS4(benchmark::State &state) {
    using SS = Eigen::Array<double, 4, 1>;
    auto rv = classic_to_cartesian<double>(leoClassic(), MU_EARTH);
    Eigen::Matrix<SS, 7, 1> in;
    for (int i = 0; i < 6; ++i) in[i] = SS::Constant(rv[i]);
    in[6] = SS::Constant(300.0);
    KeplerPropagator prop(MU_EARTH);
    Eigen::Matrix<SS, 6, 1> out;
    for (auto _ : state) {
        prop.compute_impl(in, out);
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_KeplerPropagator_VF_Compute_SS4);
