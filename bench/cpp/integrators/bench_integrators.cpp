///////////////////////////////////////////////////////////////////////////////
// Integrator benchmarks — RK stepper pipeline throughput
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <benchmark/benchmark.h>
#include <cmath>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// ODE definitions
///////////////////////////////////////////////////////////////////////////////

struct SHO_Impl : ODESize<2, 0, 0> {
    static auto Definition(double /*unused*/) {
        auto args = Arguments<3>(); // [x, v, t]
        auto x = args.coeff<0>();
        auto v = args.coeff<1>();
        auto xdot = v;
        auto vdot = (-1.0) * x;
        return StackedOutputs{xdot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(SHO, SHO_Impl, double);

struct BrachODE_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>(); // [x, y, v, t, theta]
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();
        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);
        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(BrachODE, BrachODE_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Global init
///////////////////////////////////////////////////////////////////////////////

namespace {

struct GlobalInit {
    GlobalInit() { MemoryManager::resize(256, 256); }
};
GlobalInit g_init;

} // namespace

///////////////////////////////////////////////////////////////////////////////
// SHO integration benchmarks — one full period (2*pi)
///////////////////////////////////////////////////////////////////////////////

static void BM_Integrate_SHO_DOPRI54(benchmark::State &state) {
    SHO ode(0.0);
    double tf = 2.0 * M_PI;
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
    double tf = 2.0 * M_PI;
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
    double tf = 2.0 * M_PI;
    for (auto _ : state) {
        Integrator<SHO> integ(ode, "DOPRI87", 0.01);
        integ.Adaptive = false;
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
