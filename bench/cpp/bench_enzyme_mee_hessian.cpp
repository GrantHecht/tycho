// =============================================================================
// MEE Hessian SIMD bench — separate TU.
//
// Why separate TU?  Under TYCHO_ENZYME_HESSIAN_STRATEGY=ForwardOverForward at
// -O3, compiling the MEE FoF SIMD Hessian path in the same TU as the rest
// of bench_enzyme.cpp triggers an Enzyme TypeAnalysis failure: "Cannot
// deduce single type of store" on a 16-byte SSE2 zero-store, with the
// memory typed both as Float@offset 8 and Pointer-to-SS4 by different
// inlined paths.  The same SIMD body compiles cleanly when isolated in
// its own TU (this file), or in the test TU at -O1.  Apparent root cause:
// at -O3, cross-template inlining of MEEBench's SuperScalar arithmetic
// alongside the other Enzyme-differentiated bench bodies creates IR
// shapes that conflict in Enzyme's type analysis.  Isolating to a single
// fixture per TU sidesteps the conflict.
//
// The MEE FoF SIMD path itself is correctness-validated by the test
// EnzymeVectorized.HessianFoFSIMDMatchesScalarized_MEE.
// =============================================================================
#include <benchmark/benchmark.h>

#include <tycho/tycho.h>

#include "tycho/detail/utils/simd_math.h"

namespace {

inline Eigen::Matrix<double, 9, 1> mee_input() {
    Eigen::Matrix<double, 9, 1> x;
    x << 1.2, 0.05, -0.03, 0.02, 0.01, 0.4, 0.1, 0.2, 0.3;
    return x;
}

}  // namespace

namespace tycho_enzyme_bench_mee {

// Local copy of MEEBench scoped to this TU so the SuperScalar instantiation
// is isolated from bench_enzyme.cpp's other Enzyme template instantiations.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct MEEBench
    : tycho::vf::VectorFunction<MEEBench<Jm, Hm>, 9, 6, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<MEEBench<Jm, Hm>, 9, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
    static constexpr bool is_vectorizable = true;
#if !defined(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverForward)
    static constexpr bool enzyme_simd_hessian_supported = false;
#endif
    double mu_, sqm_;
    MEEBench(double mu = 1.0) : mu_{mu}, sqm_{std::sqrt(mu)} {}
    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using tycho::math::cos;
        using tycho::math::sin;
        using std::sqrt;
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        Scalar x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4];
        Scalar x5 = x[5], x6 = x[6], x7 = x[7], x8 = x[8];
        Scalar sqx0 = sqrt(x0);
        Scalar x9 = Scalar(1.0 / sqm_);
        Scalar x10 = cos(x5);
        Scalar x11 = sin(x5);
        Scalar x12 = x1 * x10 + x11 * x2;
        Scalar x13 = x12 + 1.0;
        Scalar x14 = 1.0 / x13;
        Scalar x15 = x14 * x7;
        Scalar x16 = x10 * x4;
        Scalar x17 = x11 * x3;
        Scalar x18 = x14 * x8;
        Scalar x19 = x12 + 2.0;
        Scalar x20 = sqx0 * x9;
        Scalar x21 = x18 * (-x16 + x17);
        Scalar x22 = 0.5 * x18 * x20 * ((x3 * x3) + (x4 * x4) + 1.0);
        fx[0] = 2.0 * (x0 * sqx0) * x15 * x9;
        fx[1] = x20 * (x11 * x6 + x15 * (x1 + x10 * x19) + x18 * x2 * (x16 - x17));
        fx[2] = x20 * (x1 * x21 - x10 * x6 + x15 * (x11 * x19 + x2));
        fx[3] = x10 * x22;
        fx[4] = x11 * x22;
        fx[5] = x20 * (mu_ * x13 * x13 / (x0 * x0) + 1.0 * x21);
    }
};

}  // namespace tycho_enzyme_bench_mee

namespace {

using SS4 = Eigen::Array<double, 4, 1>;

void BM_MEE_HessianSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench_mee::MEEBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::EnzymeAD>;
    ODE f;
    Eigen::Matrix<SS4, 9, 1> x;
    for (int i = 0; i < 9; ++i) x(i).setConstant(mee_input()[i]);
    Eigen::Matrix<SS4, 6, 1> fx;
    Eigen::Matrix<SS4, 6, 9> jx;
    Eigen::Matrix<SS4, 9, 1> g;
    Eigen::Matrix<SS4, 9, 9> h;
    Eigen::Matrix<SS4, 6, 1> lam;
    fx.setZero(); jx.setZero(); g.setZero(); h.setZero();
    {
        const double seed[6] = {0.2, -0.5, 0.7, 0.1, -0.3, 0.4};
        for (int i = 0; i < 6; ++i)
            for (int lane = 0; lane < 4; ++lane)
                lam(i)(lane) = seed[i] + 0.05 * lane;
    }
    for (auto _ : state) {
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, g, h, lam);
        benchmark::DoNotOptimize(h);
        benchmark::DoNotOptimize(g);
        benchmark::ClobberMemory();
    }
}

}  // namespace

BENCHMARK(BM_MEE_HessianSIMD_Enzyme)->Name("BM_HessianVecSIMD_Enzyme/MEE");
