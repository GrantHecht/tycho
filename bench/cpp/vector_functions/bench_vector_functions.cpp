///////////////////////////////////////////////////////////////////////////////
// VectorFunction DSL evaluation benchmarks
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <benchmark/benchmark.h>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// ODE definitions
///////////////////////////////////////////////////////////////////////////////

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

struct PolarLT_Impl : ODESize<4, 2, 0> {
    static auto Definition(double amax) {
        auto args = Arguments<7>(); // [r, theta, vr, vt, t, u, alpha]
        auto r = args.coeff<0>();
        auto vr = args.coeff<2>();
        auto vt = args.coeff<3>();
        auto u = args.coeff<5>();
        auto alpha = args.coeff<6>();
        auto rdot = vr;
        auto tdot = vt / r;
        auto vrdot = (vt * vt) / r + ((-1.0) / (r * r)) + amax * u * sin(alpha);
        auto vtdot = (vt * vr) / r * (-1.0) + amax * u * cos(alpha);
        return StackedOutputs{rdot, tdot, vrdot, vtdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(PolarLTODE, PolarLT_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Global init — MemoryManager must be sized before VF evaluation
///////////////////////////////////////////////////////////////////////////////

namespace {

struct GlobalInit {
    GlobalInit() { MemoryManager::resize(256, 256); }
};
GlobalInit g_init;

} // namespace

///////////////////////////////////////////////////////////////////////////////
// BrachODE benchmarks (5->3)
///////////////////////////////////////////////////////////////////////////////

static void BM_VF_Compute(benchmark::State &state) {
    BrachODE ode(9.81);
    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 0.5, 1.0;
    Eigen::VectorXd fx(3);
    for (auto _ : state) {
        fx.setZero();
        ode.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_Compute);

static void BM_VF_ComputeJacobian(benchmark::State &state) {
    BrachODE ode(9.81);
    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 0.5, 1.0;
    Eigen::VectorXd fx(3);
    Eigen::MatrixXd jx(3, 5);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        ode.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_ComputeJacobian);

static void BM_VF_ComputeAdjointGradient(benchmark::State &state) {
    BrachODE ode(9.81);
    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 0.5, 1.0;
    Eigen::VectorXd fx(3);
    Eigen::VectorXd gx(5);
    Eigen::VectorXd lm(3);
    lm.setOnes();
    for (auto _ : state) {
        fx.setZero();
        gx.setZero();
        ode.compute_adjointgradient(x, fx, gx, lm);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(gx);
    }
}
BENCHMARK(BM_VF_ComputeAdjointGradient);

static void BM_VF_ComputeFullVJP(benchmark::State &state) {
    BrachODE ode(9.81);
    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 0.5, 1.0;
    Eigen::VectorXd fx(3);
    Eigen::MatrixXd jx(3, 5);
    Eigen::VectorXd gx(5);
    Eigen::MatrixXd hx(5, 5);
    Eigen::VectorXd lm(3);
    lm.setOnes();
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        gx.setZero();
        hx.setZero();
        ode.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(hx);
    }
}
BENCHMARK(BM_VF_ComputeFullVJP);

///////////////////////////////////////////////////////////////////////////////
// Composition benchmark — stacked segments
///////////////////////////////////////////////////////////////////////////////

static void BM_VF_Composition(benchmark::State &state) {
    auto args = Arguments<7>();
    auto a = args.head<3>();
    auto b = args.segment<4, 3>();
    auto stacked = StackedOutputs{a, b};
    Eigen::VectorXd x(7);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;
    Eigen::VectorXd fx(7);
    Eigen::MatrixXd jx(7, 7);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        stacked.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_Composition);

///////////////////////////////////////////////////////////////////////////////
// ArgsSegment throughput
///////////////////////////////////////////////////////////////////////////////

static void BM_VF_ArgsSegment(benchmark::State &state) {
    auto args = Arguments<7>();
    auto seg = args.segment<3, 4>();
    Eigen::VectorXd x(7);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;
    Eigen::VectorXd fx(3);
    Eigen::MatrixXd jx(3, 7);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        seg.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_ArgsSegment);
