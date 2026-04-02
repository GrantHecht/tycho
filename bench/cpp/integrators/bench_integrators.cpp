///////////////////////////////////////////////////////////////////////////////
// Integrator benchmarks — RK stepper pipeline throughput
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>

///////////////////////////////////////////////////////////////////////////////
// SHO integration benchmarks — one full period (2*pi)
///////////////////////////////////////////////////////////////////////////////

static void BM_Integrate_SHO_DOPRI54(benchmark::State &state) {
    SHO ode(0.0);
    double tf = 2.0 * std::numbers::pi;
    for (auto _ : state) {
        Integrator<SHO> integ(ode, "DOPRI54", 0.1);
        Eigen::Vector3d x0;
        x0 << 1.0, 0.0, 0.0;
        auto xf = integ.integrate(x0, tf);
        benchmark::DoNotOptimize(xf);
    }
}
BENCHMARK(BM_Integrate_SHO_DOPRI54);

static void BM_Integrate_SHO_DOPRI87(benchmark::State &state) {
    SHO ode(0.0);
    double tf = 2.0 * std::numbers::pi;
    for (auto _ : state) {
        Integrator<SHO> integ(ode, "DOPRI87", 0.1);
        Eigen::Vector3d x0;
        x0 << 1.0, 0.0, 0.0;
        auto xf = integ.integrate(x0, tf);
        benchmark::DoNotOptimize(xf);
    }
}
BENCHMARK(BM_Integrate_SHO_DOPRI87);

static void BM_Integrate_SHO_FixedStep(benchmark::State &state) {
    SHO ode(0.0);
    double tf = 2.0 * std::numbers::pi;
    for (auto _ : state) {
        Integrator<SHO> integ(ode, "DOPRI87", 0.01);
        integ.adaptive_ = false;
        Eigen::Vector3d x0;
        x0 << 1.0, 0.0, 0.0;
        auto xf = integ.integrate(x0, tf);
        benchmark::DoNotOptimize(xf);
    }
}
BENCHMARK(BM_Integrate_SHO_FixedStep);

///////////////////////////////////////////////////////////////////////////////
// Brachistochrone integration
///////////////////////////////////////////////////////////////////////////////

static void BM_Integrate_Brach_DOPRI87(benchmark::State &state) {
    BrachODE ode(9.81);
    for (auto _ : state) {
        Integrator<BrachODE> integ(ode, "DOPRI87", 0.05);
        Eigen::VectorXd x0(5);
        x0 << 0.0, 10.0, 0.01, 0.0, 1.0; // [x, y, v, t, theta]
        auto xf = integ.integrate(x0, 1.0);
        benchmark::DoNotOptimize(xf);
    }
}
BENCHMARK(BM_Integrate_Brach_DOPRI87);

static void BM_Integrate_Brach_DOPRI54(benchmark::State &state) {
    BrachODE ode(9.81);
    for (auto _ : state) {
        Integrator<BrachODE> integ(ode, "DOPRI54", 0.05);
        Eigen::VectorXd x0(5);
        x0 << 0.0, 10.0, 0.01, 0.0, 1.0;
        auto xf = integ.integrate(x0, 1.0);
        benchmark::DoNotOptimize(xf);
    }
}
BENCHMARK(BM_Integrate_Brach_DOPRI54);

///////////////////////////////////////////////////////////////////////////////
// PolarLT integration — higher-dimensional ODE (7-element state)
///////////////////////////////////////////////////////////////////////////////

static void BM_Integrate_PolarLT_DOPRI87(benchmark::State &state) {
    PolarLTODE ode(0.01);
    for (auto _ : state) {
        Integrator<PolarLTODE> integ(ode, "DOPRI87", 0.05);
        Eigen::VectorXd x0(7);
        x0 << 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.3; // [r, theta, vr, vt, t, u, alpha]
        auto xf = integ.integrate(x0, 2.0);
        benchmark::DoNotOptimize(xf);
    }
}
BENCHMARK(BM_Integrate_PolarLT_DOPRI87);

///////////////////////////////////////////////////////////////////////////////
// Dense output benchmark — 100 interpolated points
///////////////////////////////////////////////////////////////////////////////

static void BM_Integrate_SHO_Dense_100(benchmark::State &state) {
    SHO ode(0.0);
    double tf = 2.0 * std::numbers::pi;
    for (auto _ : state) {
        Integrator<SHO> integ(ode, "DOPRI87", 0.1);
        Eigen::Vector3d x0;
        x0 << 1.0, 0.0, 0.0;
        auto traj = integ.integrate_dense(x0, tf, 100);
        benchmark::DoNotOptimize(traj);
    }
}
BENCHMARK(BM_Integrate_SHO_Dense_100);
