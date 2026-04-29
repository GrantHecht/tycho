// =============================================================================
// Enzyme microbenchmark suite.
//
// Phase 1: Jacobian comparisons __enzyme_fwddiff vs FDiffCentArray.
// Phase 2: Hessian comparisons (full Enzyme path) vs FDiffFwd, plus a
//          brachistochrone PSIOPT full-solve TTS comparison (gate criterion
//          §8.2: full-Enzyme TTS ≤ 1.5× FDiff reference TTS).
//
// The benchmark is built once per TYCHO_ENZYME_HESSIAN_STRATEGY value (FoR /
// FoF). Names stay neutral; users compare the two JSON dumps externally.
//
// Plugin flag is scoped to this target only (see bench/cpp/CMakeLists.txt);
// the rest of the benchmark suite is unaffected.
// =============================================================================
#include <benchmark/benchmark.h>

#include <tycho/tycho.h>

#include "tycho/detail/utils/simd_math.h"

#include "enzyme_test_dynamics.h"

namespace {

// Deterministic input vectors (avoid singular regions of CR3BP / MEE).
inline Eigen::Matrix<double, 5, 1> brach_input() {
    Eigen::Matrix<double, 5, 1> x;
    x << 1.0, 2.0, 3.0, 0.0, 0.5;
    return x;
}

inline Eigen::Matrix<double, 7, 1> cr3bp_input() {
    Eigen::Matrix<double, 7, 1> x;
    x << 0.5, 0.7, 0.1, 0.05, -0.03, 0.02, 0.0;
    return x;
}

inline Eigen::Matrix<double, 9, 1> mee_input() {
    Eigen::Matrix<double, 9, 1> x;
    x << 1.2, 0.05, -0.03, 0.02, 0.01, 0.4, 0.1, 0.2, 0.3;
    return x;
}

// Common templated body.  Held in an anonymous namespace so each ODE binding
// gets its own concrete BM_Jacobian<ODE> instantiation that the benchmark
// macro can take the address of.
template <class ODE>
inline void bm_jacobian_body(benchmark::State& state,
                             const Eigen::Matrix<double, ODE::IRC, 1>& x_in) {
    ODE f;
    Eigen::Matrix<double, ODE::IRC, 1> x = x_in;
    Eigen::Matrix<double, ODE::ORC, 1> fx;
    Eigen::Matrix<double, ODE::ORC, ODE::IRC> jx;
    fx.setZero();
    jx.setZero();
    for (auto _ : state) {
        f.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

// Per-ODE thunks so BENCHMARK_CAPTURE has a concrete function pointer.
void BM_Brach_FDiff(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::BrachFDiff>(state, brach_input());
}
void BM_Brach_Enzyme(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::BrachEnzymeAD>(state, brach_input());
}

void BM_CR3BP_FDiff(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::CR3BPFDiff>(state, cr3bp_input());
}
void BM_CR3BP_Enzyme(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::CR3BPEnzymeAD>(state, cr3bp_input());
}

void BM_MEE_FDiff(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::MEEFDiff>(state, mee_input());
}
void BM_MEE_Enzyme(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::MEEEnzymeAD>(state, mee_input());
}

// -----------------------------------------------------------------------------
// Phase 5b vectorized Jacobian benchmark.  Compares one batched call with W
// SIMD lanes against the scalar Phase 1/Phase 3 path on the same problem.
// Per-call cost / W is the figure of merit (each lane represents one
// independent VF evaluation).
// -----------------------------------------------------------------------------

}  // close anonymous namespace temporarily so Vectorizable BrachBench
   // gets external linkage (Enzyme's template-instantiated __enzyme_fwddiff
   // refuses to bind to types in an unnamed namespace).

namespace tycho_enzyme_bench {

// Brach variant marked Vectorizable=true so the Phase 5b SIMD dispatch fires.
// Trig calls route through tycho::math::cos/sin which scalarises lane-wise
// so Enzyme sees per-lane @llvm.cos.f64 / @llvm.sin.f64 — bithack-free,
// composes with TYCHO_ENZYME_BATCH_WIDTH > 1.  See
// include/tycho/detail/utils/simd_math.h.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct BrachBench
    : tycho::vf::VectorFunction<BrachBench<Jm, Hm>, 5, 3, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<BrachBench<Jm, Hm>, 5, 3, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
    static constexpr bool is_vectorizable = true;
    double g_;
    BrachBench(double g = 32.2) : g_{g} {}
    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using tycho::math::cos;
        using tycho::math::sin;
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        const Scalar v = x[2];
        const Scalar theta = x[4];
        fx[0] = sin(theta) * v;
        fx[1] = -cos(theta) * v;
        fx[2] = Scalar(g_) * cos(theta);
    }
};

// CR3BP variant marked Vectorizable=true.  Scalar(...) casts are needed
// because arithmetic mixing plain double with SS-typed Eigen ops requires
// matching scalar types under Vectorizable dispatch.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct CR3BPBench
    : tycho::vf::VectorFunction<CR3BPBench<Jm, Hm>, 7, 6, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<CR3BPBench<Jm, Hm>, 7, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
    static constexpr bool is_vectorizable = true;
    double mu_;
    CR3BPBench(double mu = 0.0123) : mu_{mu} {}
    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        tycho::Vector3<Scalar> X = x.template head<3>();
        tycho::Vector3<Scalar> V = x.template segment<3>(3);
        tycho::Vector3<Scalar> p1loc; p1loc[0] = Scalar(-mu_);
        tycho::Vector3<Scalar> p2loc; p2loc[0] = Scalar(1.0 - mu_);
        tycho::Vector3<Scalar> dvec = X - p1loc;
        tycho::Vector3<Scalar> rvec = X - p2loc;
        Scalar d = (dvec.array() * dvec.array()).sum();
        d = sqrt(d);
        Scalar r = (rvec.array() * rvec.array()).sum();
        r = sqrt(r);
        fx.template head<3>() = V;
        fx.template segment<3>(3) =
            (Scalar(-(1.0 - mu_)) / (d * d * d)) * dvec
          + (Scalar(-mu_) / (r * r * r)) * rvec;
        fx[3] += Scalar(2.0) * V[1] + X[0];
        fx[4] += Scalar(-2.0) * V[0] + X[1];
    }
};

// MEE variant marked Vectorizable=true.  See BrachBench notes above.
//
// The composite trig+sqrt+division body trips Enzyme's SuperScalar reverse-mode
// IR type analysis ("Cannot deduce single type of store") under the FoR Hessian
// strategy, so we opt out via enzyme_simd_hessian_supported=false and fall
// through to Phase 5a scalarize-per-lane.
template <tycho::vf::DenseDerivativeMode Jm, tycho::vf::DenseDerivativeMode Hm>
struct MEEBench
    : tycho::vf::VectorFunction<MEEBench<Jm, Hm>, 9, 6, Jm, Hm> {
    using Base = tycho::vf::VectorFunction<MEEBench<Jm, Hm>, 9, 6, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
    static constexpr bool is_vectorizable = true;
    static constexpr bool enzyme_simd_hessian_supported = false;
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

}  // namespace tycho_enzyme_bench

namespace {

using SS4 = Eigen::Array<double, 4, 1>;

// FDiff reference (Vectorizable + FDiffCentArray Jacobian / FDiffFwd Hessian)
// is not exercised here — the central-difference path scalarizes per-lane
// externally, so the Vectorizable benches measure the EnzymeAD SIMD path
// alone with the scalar Phase 1 / Phase 3 path as the implicit comparator.

// Vectorizable+EnzymeAD compute_jacobian routes through Phase 5a
// scalarize-per-lane by default (see dense_enzyme.h dispatch comment).
// These benchmarks measure that default path — the scalar path runs IR
// times per lane × W lanes, internally batched via Phase 3 enzyme_width.
//
// To benchmark the Phase 5b direct-SIMD path explicitly, see the
// BM_JacobianVecSIMD_* variants below which call
// simd_compute_jacobian_impl directly.

void BM_Brach_Vectorized_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::BrachBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::FDiffFwd>;
    ODE f;
    Eigen::Matrix<SS4, 5, 1> x;
    for (int i = 0; i < 5; ++i) x(i).setConstant(brach_input()[i]);
    Eigen::Matrix<SS4, 3, 1> fx;
    Eigen::Matrix<SS4, 3, 5> jx;
    fx.setZero(); jx.setZero();
    for (auto _ : state) {
        f.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

void BM_CR3BP_Vectorized_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::CR3BPBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::FDiffFwd>;
    ODE f;
    Eigen::Matrix<SS4, 7, 1> x;
    for (int i = 0; i < 7; ++i) x(i).setConstant(cr3bp_input()[i]);
    Eigen::Matrix<SS4, 6, 1> fx;
    Eigen::Matrix<SS4, 6, 7> jx;
    fx.setZero(); jx.setZero();
    for (auto _ : state) {
        f.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

void BM_MEE_Vectorized_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::MEEBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::FDiffFwd>;
    ODE f;
    Eigen::Matrix<SS4, 9, 1> x;
    for (int i = 0; i < 9; ++i) x(i).setConstant(mee_input()[i]);
    Eigen::Matrix<SS4, 6, 1> fx;
    Eigen::Matrix<SS4, 6, 9> jx;
    fx.setZero(); jx.setZero();
    for (auto _ : state) {
        f.compute_jacobian(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

// Phase 5b direct-SIMD variants: call simd_compute_jacobian_impl directly,
// bypassing the default scalarize-per-lane dispatch.  These measure the
// SuperScalar primal path (Eigen::Array<double, W, 1> arithmetic in the
// user's compute_impl, Enzyme differentiating through SIMD ops).

void BM_Brach_VectorizedSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::BrachBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::FDiffFwd>;
    ODE f;
    Eigen::Matrix<SS4, 5, 1> x;
    for (int i = 0; i < 5; ++i) x(i).setConstant(brach_input()[i]);
    Eigen::Matrix<SS4, 3, 1> fx;
    Eigen::Matrix<SS4, 3, 5> jx;
    fx.setZero(); jx.setZero();
    for (auto _ : state) {
        f.simd_compute_jacobian_impl(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

void BM_CR3BP_VectorizedSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::CR3BPBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::FDiffFwd>;
    ODE f;
    Eigen::Matrix<SS4, 7, 1> x;
    for (int i = 0; i < 7; ++i) x(i).setConstant(cr3bp_input()[i]);
    Eigen::Matrix<SS4, 6, 1> fx;
    Eigen::Matrix<SS4, 6, 7> jx;
    fx.setZero(); jx.setZero();
    for (auto _ : state) {
        f.simd_compute_jacobian_impl(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

void BM_MEE_VectorizedSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::MEEBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::FDiffFwd>;
    ODE f;
    Eigen::Matrix<SS4, 9, 1> x;
    for (int i = 0; i < 9; ++i) x(i).setConstant(mee_input()[i]);
    Eigen::Matrix<SS4, 6, 1> fx;
    Eigen::Matrix<SS4, 6, 9> jx;
    fx.setZero(); jx.setZero();
    for (auto _ : state) {
        f.simd_compute_jacobian_impl(x, fx, jx);
        benchmark::DoNotOptimize(fx);
        benchmark::DoNotOptimize(jx);
        benchmark::ClobberMemory();
    }
}

// -----------------------------------------------------------------------------
// Phase 6 Hessian SIMD bench.  Routes through the dispatch entry point
// (compute_jacobian_adjointgradient_adjointhessian) so the same bench runs
// the Phase 5a scalarize-per-lane fallback when TYCHO_ENZYME_SIMD_HESSIAN
// is OFF and the Phase 6 direct-SIMD FoR path when ON.  <EnzymeAD, EnzymeAD>
// pairing exercises the EnzymeAD Hessian.  The lam seed perturbs each lane
// by +0.05*lane so Enzyme cannot fold lane values together.
// -----------------------------------------------------------------------------

void BM_Brach_HessianSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::BrachBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::EnzymeAD>;
    ODE f;
    Eigen::Matrix<SS4, 5, 1> x;
    for (int i = 0; i < 5; ++i) x(i).setConstant(brach_input()[i]);
    Eigen::Matrix<SS4, 3, 1> fx;
    Eigen::Matrix<SS4, 3, 5> jx;
    Eigen::Matrix<SS4, 5, 1> g;
    Eigen::Matrix<SS4, 5, 5> h;
    Eigen::Matrix<SS4, 3, 1> lam;
    fx.setZero(); jx.setZero(); g.setZero(); h.setZero();
    {
        const double seed[3] = {0.7, -1.1, 0.3};
        for (int i = 0; i < 3; ++i)
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

void BM_CR3BP_HessianSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::CR3BPBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::EnzymeAD>;
    ODE f;
    Eigen::Matrix<SS4, 7, 1> x;
    for (int i = 0; i < 7; ++i) x(i).setConstant(cr3bp_input()[i]);
    Eigen::Matrix<SS4, 6, 1> fx;
    Eigen::Matrix<SS4, 6, 7> jx;
    Eigen::Matrix<SS4, 7, 1> g;
    Eigen::Matrix<SS4, 7, 7> h;
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

// MEE Hessian SIMD bench: MEEBench opts out of FoR SIMD via
// enzyme_simd_hessian_supported=false (composite trig+sqrt+division body
// trips Enzyme's SS reverse-mode TypeAnalysis), so the bench routes
// through the Phase 5a scalarize-per-lane fallback.
void BM_MEE_HessianSIMD_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::MEEBench<
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

// -----------------------------------------------------------------------------
// Phase 2: Hessian benchmarks.  Each iteration computes the full Jacobian +
// adjoint gradient + adjoint Hessian via the active EnzymeAD pathway (FoR or
// FoF, depending on TYCHO_ENZYME_HESSIAN_STRATEGY).  Compared head-to-head
// with the FDiffFwd nested-FD reference.
// -----------------------------------------------------------------------------
template <class ODE>
inline void bm_hessian_body(benchmark::State& state,
                            const Eigen::Matrix<double, ODE::IRC, 1>& x_in) {
    ODE f;
    Eigen::Matrix<double, ODE::IRC, 1> x = x_in;
    Eigen::Matrix<double, ODE::ORC, 1> fx;
    Eigen::Matrix<double, ODE::ORC, ODE::IRC> jx;
    Eigen::Matrix<double, ODE::IRC, 1> g;
    Eigen::Matrix<double, ODE::IRC, ODE::IRC> h;
    Eigen::Matrix<double, ODE::ORC, 1> lam;
    // Deterministic adjoint vector — mirrors the unit-test choice.
    if constexpr (ODE::ORC == 3) {
        lam << 0.7, -1.1, 0.3;
    } else if constexpr (ODE::ORC == 6) {
        lam << 0.2, -0.5, 0.7, 0.1, -0.3, 0.4;
    } else {
        lam.setOnes();
    }
    fx.setZero(); jx.setZero(); g.setZero(); h.setZero();

    for (auto _ : state) {
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, g, h, lam);
        benchmark::DoNotOptimize(h);
        benchmark::DoNotOptimize(g);
        benchmark::ClobberMemory();
    }
}

void BM_Hessian_Brach_FDiff(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::BrachFDiff>(state, brach_input());
}
void BM_Hessian_Brach_Enzyme(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::BrachEnzymeFull>(state, brach_input());
}

void BM_Hessian_CR3BP_FDiff(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::CR3BPFDiff>(state, cr3bp_input());
}
void BM_Hessian_CR3BP_Enzyme(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::CR3BPEnzymeFull>(state, cr3bp_input());
}

void BM_Hessian_MEE_FDiff(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::MEEFDiff>(state, mee_input());
}
void BM_Hessian_MEE_Enzyme(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::MEEEnzymeFull>(state, mee_input());
}

// -----------------------------------------------------------------------------
// Phase 2 gate criterion: brachistochrone PSIOPT full-solve TTS comparison.
// Same problem the e2e test runs; benchmarks the entire setup-+-solve cycle.
// -----------------------------------------------------------------------------
template <class BrachFunc>
inline void bm_brachistochrone_solve_body(benchmark::State& state) {
    using namespace tycho;
    using namespace tycho::oc;
    using namespace tycho::solvers;

    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0;
    constexpr double theta_guess = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_defects = 32;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    for (auto _ : state) {
        auto erased = tycho::vf::GenericFunction<-1, -1>(BrachFunc(g));
        ODE ode(std::move(erased), 3, 1, 0);
        auto phase = ode.phase(TranscriptionModes::LGL3, traj, n_defects);

        Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
        Eigen::VectorXd front_val(4);
        front_val << x0, y0, v0, t0;
        phase.add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val,
                                 ScaleModes::AUTO);

        Eigen::VectorXi back_idx(2);
        back_idx << 0, 1;
        Eigen::VectorXd back_val(2);
        back_val << xf, yf;
        phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val,
                                 ScaleModes::AUTO);

        phase.add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
        phase.add_delta_time_objective(1.0, ScaleModes::AUTO);

        auto status = phase.solve_optimize();
        benchmark::DoNotOptimize(status);
        benchmark::ClobberMemory();
    }
}

}  // namespace

// Brachistochrone (5 -> 3)
BENCHMARK(BM_Brach_FDiff)->Name("BM_Jacobian_FDiff/Brach");
BENCHMARK(BM_Brach_Enzyme)->Name("BM_Jacobian_Enzyme/Brach");

// CR3BP (7 -> 6)
BENCHMARK(BM_CR3BP_FDiff)->Name("BM_Jacobian_FDiff/CR3BP");
BENCHMARK(BM_CR3BP_Enzyme)->Name("BM_Jacobian_Enzyme/CR3BP");

// MEE (9 -> 6)
BENCHMARK(BM_MEE_FDiff)->Name("BM_Jacobian_FDiff/MEE");
BENCHMARK(BM_MEE_Enzyme)->Name("BM_Jacobian_Enzyme/MEE");

// Vectorizable+EnzymeAD compute_jacobian (default: scalarize-per-lane).
BENCHMARK(BM_Brach_Vectorized_Enzyme)->Name("BM_JacobianVec_Enzyme/Brach");
BENCHMARK(BM_CR3BP_Vectorized_Enzyme)->Name("BM_JacobianVec_Enzyme/CR3BP");
BENCHMARK(BM_MEE_Vectorized_Enzyme)->Name("BM_JacobianVec_Enzyme/MEE");

// Phase 5b direct-SIMD path (explicit simd_compute_jacobian_impl call).
BENCHMARK(BM_Brach_VectorizedSIMD_Enzyme)->Name("BM_JacobianVecSIMD_Enzyme/Brach");
BENCHMARK(BM_CR3BP_VectorizedSIMD_Enzyme)->Name("BM_JacobianVecSIMD_Enzyme/CR3BP");
BENCHMARK(BM_MEE_VectorizedSIMD_Enzyme)->Name("BM_JacobianVecSIMD_Enzyme/MEE");

// Phase 2: Hessians.  EnzymeFull = <EnzymeAD, EnzymeAD>; the active
// TYCHO_ENZYME_HESSIAN_STRATEGY chooses FoR vs FoF at compile time.
BENCHMARK(BM_Hessian_Brach_FDiff)->Name("BM_Hessian_FDiff/Brach");
BENCHMARK(BM_Hessian_Brach_Enzyme)->Name("BM_Hessian_Enzyme/Brach");
BENCHMARK(BM_Hessian_CR3BP_FDiff)->Name("BM_Hessian_FDiff/CR3BP");
BENCHMARK(BM_Hessian_CR3BP_Enzyme)->Name("BM_Hessian_Enzyme/CR3BP");
BENCHMARK(BM_Hessian_MEE_FDiff)->Name("BM_Hessian_FDiff/MEE");
BENCHMARK(BM_Hessian_MEE_Enzyme)->Name("BM_Hessian_Enzyme/MEE");

// Phase 6: SuperScalar Hessian via dispatch entry.  Flag OFF runs Phase 5a
// scalarize-per-lane; flag ON runs Phase 6 direct-SIMD FoR.
BENCHMARK(BM_Brach_HessianSIMD_Enzyme)->Name("BM_HessianVecSIMD_Enzyme/Brach");
BENCHMARK(BM_CR3BP_HessianSIMD_Enzyme)->Name("BM_HessianVecSIMD_Enzyme/CR3BP");
BENCHMARK(BM_MEE_HessianSIMD_Enzyme)->Name("BM_HessianVecSIMD_Enzyme/MEE");

// Phase 2 gate: full-solve TTS for the brachistochrone.  Each iteration
// builds the phase + solves PSIOPT, so the per-iteration cost includes
// problem setup as well as the solve itself.
void BM_Solve_Brach_FDiff(benchmark::State& state) {
    bm_brachistochrone_solve_body<tycho_enzyme_test::BrachFDiff>(state);
}
void BM_Solve_Brach_Enzyme(benchmark::State& state) {
    bm_brachistochrone_solve_body<tycho_enzyme_test::BrachEnzymeFull>(state);
}
BENCHMARK(BM_Solve_Brach_FDiff)
    ->Name("BM_FullSolve_FDiff/Brach")
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Solve_Brach_Enzyme)
    ->Name("BM_FullSolve_Enzyme/Brach")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
