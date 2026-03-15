///////////////////////////////////////////////////////////////////////////////
// Type erasure benchmarks — GenericFunction VJP dispatch, GFStorage clone
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>

///////////////////////////////////////////////////////////////////////////////
// VJP dispatch through GFConcept vtable
///////////////////////////////////////////////////////////////////////////////

template <typename ODE> static void BM_GF_VJP(benchmark::State &state, ODE ode) {
    using Dspec = DenseFunctionSpec<-1, -1>;

    GenericFunction<-1, -1> gf(ode);
    Eigen::VectorXd x_s(gf.IRows());
    x_s.setRandom();
    Eigen::VectorXd fx_s(gf.ORows());
    fx_s.setZero();
    Eigen::MatrixXd jx_s(gf.ORows(), gf.IRows());
    jx_s.setZero();
    Eigen::VectorXd gx_s(gf.IRows());
    gx_s.setZero();
    Eigen::MatrixXd hx_s(gf.IRows(), gf.IRows());
    hx_s.setZero();
    Eigen::VectorXd l_s(gf.ORows());
    l_s.setOnes();

    for (auto _ : state) {
        Dspec::InType xt(x_s);
        Dspec::OutType fxt(fx_s);
        Dspec::JacType jxt(jx_s);
        Dspec::AdjGradType gxt(gx_s);
        Dspec::AdjHessType hxt(hx_s);
        Dspec::AdjVarType lxt(l_s);
        gf.func.get().compute_jacobian_adjointgradient_adjointhessian(xt, fxt, jxt, gxt, hxt, lxt);
        benchmark::DoNotOptimize(fx_s);
    }
}

static void BM_GF_VJP_Zermelo(benchmark::State &state) { BM_GF_VJP(state, ZermeloODE(1.5)); }
static void BM_GF_VJP_Brach(benchmark::State &state) { BM_GF_VJP(state, BrachODE(9.81)); }
static void BM_GF_VJP_PolarLT(benchmark::State &state) { BM_GF_VJP(state, PolarLTODE(0.01)); }
BENCHMARK(BM_GF_VJP_Zermelo);
BENCHMARK(BM_GF_VJP_Brach);
BENCHMARK(BM_GF_VJP_PolarLT);

///////////////////////////////////////////////////////////////////////////////
// GFStorage clone (SBO deep-copy)
///////////////////////////////////////////////////////////////////////////////

template <typename ODE> static void BM_GF_Clone(benchmark::State &state, ODE ode) {
    GenericFunction<-1, -1> src(ode);
    for (auto _ : state) {
        GenericFunction<-1, -1> copy(src);
        benchmark::DoNotOptimize(copy);
    }
}

static void BM_GF_Clone_Zermelo(benchmark::State &state) { BM_GF_Clone(state, ZermeloODE(1.5)); }
static void BM_GF_Clone_Brach(benchmark::State &state) { BM_GF_Clone(state, BrachODE(9.81)); }
static void BM_GF_Clone_PolarLT(benchmark::State &state) { BM_GF_Clone(state, PolarLTODE(0.01)); }
BENCHMARK(BM_GF_Clone_Zermelo);
BENCHMARK(BM_GF_Clone_Brach);
BENCHMARK(BM_GF_Clone_PolarLT);
