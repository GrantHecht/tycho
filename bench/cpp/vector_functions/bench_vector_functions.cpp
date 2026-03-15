///////////////////////////////////////////////////////////////////////////////
// VectorFunction DSL evaluation benchmarks
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>

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
// Composition benchmark — nested math operations
///////////////////////////////////////////////////////////////////////////////

static void BM_VF_Composition(benchmark::State &state) {
    auto args = Arguments<7>();
    auto a = args.head<3>();
    auto b = args.segment<3, 3>(); // <Size, Start>
    auto cross_ab = a.cross(b);
    auto norm_a = a.norm();
    auto stacked = StackedOutputs{cross_ab, norm_a};
    Eigen::VectorXd x(7);
    x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;
    Eigen::VectorXd fx(4);
    Eigen::MatrixXd jx(4, 7);
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
    auto seg = args.segment<3, 4>(); // <Size, Start>
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

///////////////////////////////////////////////////////////////////////////////
// CR3BP benchmarks
///////////////////////////////////////////////////////////////////////////////

static void BM_VF_CR3BP_Compute(benchmark::State &state) {
    CR3BP ode(0.01215); // Earth-Moon mass ratio
    Eigen::VectorXd x(7);
    x << 0.8, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0; // near L1
    Eigen::VectorXd fx(6);
    for (auto _ : state) {
        fx.setZero();
        ode.compute(x, fx);
        benchmark::DoNotOptimize(fx);
    }
}
BENCHMARK(BM_VF_CR3BP_Compute);

static void BM_VF_CR3BP_ComputeJacobian(benchmark::State &state) {
    CR3BP ode(0.01215);
    Eigen::VectorXd x(7);
    x << 0.8, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0;
    Eigen::VectorXd fx(6);
    Eigen::MatrixXd jx(6, 7);
    for (auto _ : state) {
        fx.setZero();
        jx.setZero();
        ode.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
    }
}
BENCHMARK(BM_VF_CR3BP_ComputeJacobian);
