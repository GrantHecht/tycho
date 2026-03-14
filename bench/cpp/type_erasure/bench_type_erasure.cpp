///////////////////////////////////////////////////////////////////////////////
// Type erasure benchmarks — GenericFunction VJP dispatch, GFStorage clone,
// TypeStorage emplace/clone
///////////////////////////////////////////////////////////////////////////////

#include "Utils/TypeStorage.h"
#include "Tycho.h"
#include <benchmark/benchmark.h>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// ODE definitions (matching bench/legacy/gf_eval/bench_gf_eval.cpp)
///////////////////////////////////////////////////////////////////////////////

struct Zermelo_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args = Arguments<4>(); // [x, y, t, theta]
        auto theta = args.coeff<3>();
        auto xdot = vMax * sin(theta) + 0.5;
        auto ydot = vMax * cos(theta) + (-0.3);
        return StackedOutputs{xdot, ydot};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloODE, Zermelo_Impl, double);

struct Brach_Impl : ODESize<3, 1, 0> {
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
BUILD_ODE_FROM_EXPRESSION(BrachODE, Brach_Impl, double);

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
// Global init
///////////////////////////////////////////////////////////////////////////////

namespace {

struct GlobalInit {
    GlobalInit() { MemoryManager::resize(256, 256); }
};
GlobalInit g_init;

} // namespace

///////////////////////////////////////////////////////////////////////////////
// VJP dispatch through GFConcept vtable
///////////////////////////////////////////////////////////////////////////////

template <typename ODE>
static void BM_GF_VJP(benchmark::State &state, ODE ode) {
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

template <typename ODE>
static void BM_GF_Clone(benchmark::State &state, ODE ode) {
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

///////////////////////////////////////////////////////////////////////////////
// TypeStorage benchmarks
///////////////////////////////////////////////////////////////////////////////

namespace {

struct BenchBase {
    virtual ~BenchBase() = default;
    virtual void clone_into(TypeStorage<BenchBase> &s) const = 0;
};

struct BenchModel : BenchBase {
    int v_;
    explicit BenchModel(int v) : v_(v) {}
    void clone_into(TypeStorage<BenchBase> &s) const override { s.emplace<BenchModel>(v_); }
};

} // namespace

static void BM_TypeStorage_EmplaceSBO(benchmark::State &state) {
    for (auto _ : state) {
        TypeStorage<BenchBase> s;
        s.emplace<BenchModel>(42);
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_TypeStorage_EmplaceSBO);

static void BM_TypeStorage_Clone(benchmark::State &state) {
    TypeStorage<BenchBase> src;
    src.emplace<BenchModel>(42);
    for (auto _ : state) {
        TypeStorage<BenchBase> copy(src);
        benchmark::DoNotOptimize(copy);
    }
}
BENCHMARK(BM_TypeStorage_Clone);
