///////////////////////////////////////////////////////////////////////////////
// bench_gf_eval.cpp — GenericFunction / GFStorage throughput benchmark
//
// Stresses the concept-based type erasure layer (PR 6: GFTypeErasure.h)
// by timing the two core operations performed on every OCP solve:
//
//   1. compute_jacobian_adjointgradient_adjointhessian — the hot VJP path.
//      Called through the GFConcept vtable (one virtual dispatch per call).
//      Three functions of increasing expression complexity are timed so that
//      dispatch overhead is visible relative to compute cost.
//
//   2. GFStorage clone — the new SBO deep-copy path (value semantics).
//      The old rubber_types implementation always heap-allocated; GFStorage
//      copies small functions with placement-new into a 128-byte inline
//      buffer.  This section measures clone throughput.
//
// Prints elapsed seconds (sum of all timed sections) to stdout.
// All other output is suppressed so the benchmark runner can parse the result.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace Tycho;
using Clock = std::chrono::high_resolution_clock;

///////////////////////////////////////////////////////////////////////////////
// Function definitions (VectorFunction DSL, BUILD_ODE_FROM_EXPRESSION)
///////////////////////////////////////////////////////////////////////////////

// --- F1: Zermelo ship routing (4→2, tiny expression) -----------------------
struct Zermelo_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args  = Arguments<4>();  // [x, y, t, theta]
        auto theta = args.coeff<3>();
        auto xdot  = vMax * sin(theta) + 0.5;
        auto ydot  = vMax * cos(theta) + (-0.3);
        return StackedOutputs{xdot, ydot};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloODE, Zermelo_Impl, double);

// --- F2: Brachistochrone (5→3, slightly more complex) ----------------------
struct Brach_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args  = Arguments<5>();  // [x, y, v, t, theta]
        auto v     = args.coeff<2>();
        auto theta = args.coeff<4>();
        auto xdot  = sin(theta) * v;
        auto ydot  = cos(theta) * v * (-1.0);
        auto vdot  = g * cos(theta);
        return StackedOutputs{xdot, ydot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(BrachODE, Brach_Impl, double);

// --- F3: Polar low-thrust orbit transfer (7→4, richer expression tree) ----
// Matches bench_topputto_lowthrust.py dynamics.
struct PolarLT_Impl : ODESize<4, 2, 0> {
    static auto Definition(double amax) {
        auto args  = Arguments<7>();  // [r, theta, vr, vt, t, u, alpha]
        auto r     = args.coeff<0>();
        auto vr    = args.coeff<2>();
        auto vt    = args.coeff<3>();
        auto u     = args.coeff<5>();
        auto alpha = args.coeff<6>();
        auto rdot  = vr;
        auto tdot  = vt / r;
        auto vrdot = (vt * vt) / r + ((-1.0) / (r * r)) + amax * u * sin(alpha);
        auto vtdot = (vt * vr) / r * (-1.0) + amax * u * cos(alpha);
        return StackedOutputs{rdot, tdot, vrdot, vtdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(PolarLTODE, PolarLT_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Benchmark helpers
//
// GenericFunction<-1,-1> wraps everything through dynamic vtable dispatch.
// DenseFunctionSpec<-1,-1>::InType  = Eigen::Ref<Eigen::VectorXd>
// DenseFunctionSpec<-1,-1>::JacType = Eigen::Ref<Eigen::MatrixXd>
// We construct those Refs from heap-allocated storage and pass directly to
// the GFConcept virtuals, mirroring what GenericFunction::SpeedTest does.
///////////////////////////////////////////////////////////////////////////////

// Time N full VJP calls through the GFConcept vtable.
// Uses DenseFunctionSpec<-1,-1> type aliases to match the exact Ref types
// expected by GFConcept's virtual signature, mirroring GenericFunction::SpeedTest.
//   InType      = Ref<const VectorXd>   AdjGradType = Ref<VectorXd>
//   OutType     = Ref<VectorXd>          AdjHessType = Ref<MatrixXd>
//   JacType     = Ref<MatrixXd>          AdjVarType  = Ref<const VectorXd>
double time_vjp(GenericFunction<-1, -1>& gf, int N) {
    using Dspec = DenseFunctionSpec<-1, -1>;

    Eigen::VectorXd x_s(gf.IRows());              x_s.setRandom();
    Eigen::VectorXd fx_s(gf.ORows());             fx_s.setZero();
    Eigen::MatrixXd jx_s(gf.ORows(), gf.IRows()); jx_s.setZero();
    Eigen::VectorXd gx_s(gf.IRows());             gx_s.setZero();
    Eigen::MatrixXd hx_s(gf.IRows(), gf.IRows()); hx_s.setZero();
    Eigen::VectorXd l_s(gf.ORows());              l_s.setOnes();

    BumpAllocator::resize(64, 64);

    volatile double dummy = 0.0;
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
        Dspec::InType      xt(x_s);
        Dspec::OutType     fxt(fx_s);
        Dspec::JacType     jxt(jx_s);
        Dspec::AdjGradType gxt(gx_s);
        Dspec::AdjHessType hxt(hx_s);
        Dspec::AdjVarType  lxt(l_s);
#if __cplusplus >= 202002L
        // PR6: dispatch through GFConcept flat vtable (GFStorage)
        gf.func.get().compute_jacobian_adjointgradient_adjointhessian(
            xt, fxt, jxt, gxt, hxt, lxt);
#else
        // Pre-PR6: dispatch through rubber_types virtual interface
        gf.compute_jacobian_adjointgradient_adjointhessian(
            xt, fxt, jxt, gxt, hxt, lxt);
#endif
        dummy += fx_s[0];
    }
    auto t1 = Clock::now();
    (void)dummy;
    return std::chrono::duration<double>(t1 - t0).count();
}

// Time N deep copies of a GenericFunction<-1,-1>.
// Tests the GFStorage::clone_into SBO path: types <= 128 bytes are copied
// via placement-new into the inline buffer with no heap allocation.
double time_clone(GenericFunction<-1, -1>& src, int N) {
    volatile double dummy = 0.0;
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
        GenericFunction<-1, -1> copy(src);   // deep clone via GFStorage
        dummy += static_cast<double>(copy.IRows());
    }
    auto t1 = Clock::now();
    (void)dummy;
    return std::chrono::duration<double>(t1 - t0).count();
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    // ------------------------------------------------------------------
    // Section 1: VJP throughput — three functions, N=500,000 calls each.
    // All three are erased into GenericFunction<-1,-1> to exercise dynamic
    // vtable dispatch through GFConcept.
    // ------------------------------------------------------------------
    constexpr int N_VJP = 500'000;

    ZermeloODE f1(1.5);
    BrachODE   f2(9.81);
    PolarLTODE f3(0.01);

    GenericFunction<-1, -1> gf1(f1);
    GenericFunction<-1, -1> gf2(f2);
    GenericFunction<-1, -1> gf3(f3);

    double elapsed = 0.0;
    elapsed += time_vjp(gf1, N_VJP);   // F1: Zermelo  4→2
    elapsed += time_vjp(gf2, N_VJP);   // F2: Brach    5→3
    elapsed += time_vjp(gf3, N_VJP);   // F3: PolarLT  7→4

    // ------------------------------------------------------------------
    // Section 2: Clone throughput — N=50,000 GFStorage deep-copies each.
    // SBO-eligible types (sizeof(GFModel) <= 128) copy inline; larger
    // types fall back to heap allocation.
    // ------------------------------------------------------------------
    constexpr int N_CLONE = 50'000;
    elapsed += time_clone(gf1, N_CLONE);
    elapsed += time_clone(gf2, N_CLONE);
    elapsed += time_clone(gf3, N_CLONE);

    std::cout << elapsed << std::endl;
    return 0;
}
