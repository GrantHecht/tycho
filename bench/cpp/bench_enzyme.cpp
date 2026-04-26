// =============================================================================
// Enzyme microbenchmark suite.
//
// Phase 1: Jacobian comparisons __enzyme_fwddiff vs AutodiffFwd.
// Phase 2: Hessian comparisons (full Enzyme path) vs AutodiffFwd, plus a
//          brachistochrone PSIOPT full-solve TTS comparison (gate criterion
//          §8.2: full-Enzyme TTS ≤ 1.5× autodiff TTS).
//
// The benchmark is built once per TYCHO_ENZYME_HESSIAN_STRATEGY value (FoR /
// FoF). Names stay neutral; users compare the two JSON dumps externally.
//
// Plugin flag is scoped to this target only (see bench/cpp/CMakeLists.txt);
// the rest of the benchmark suite is unaffected.
// =============================================================================
#include <benchmark/benchmark.h>

#include <tycho/tycho.h>

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
void BM_Brach_Autodiff(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::BrachAutodiff>(state, brach_input());
}
void BM_Brach_Enzyme(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::BrachEnzymeAD>(state, brach_input());
}

void BM_CR3BP_Autodiff(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::CR3BPAutodiff>(state, cr3bp_input());
}
void BM_CR3BP_Enzyme(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::CR3BPEnzymeAD>(state, cr3bp_input());
}

void BM_MEE_Autodiff(benchmark::State& state) {
    bm_jacobian_body<tycho_enzyme_test::MEEAutodiff>(state, mee_input());
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
        using std::cos;
        using std::sin;
        using Scalar = typename InType::Scalar;
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        const Scalar v = x[2];
        const Scalar theta = x[4];
        fx[0] = sin(theta) * v;
        fx[1] = -cos(theta) * v;
        fx[2] = Scalar(g_) * cos(theta);
    }
};

}  // namespace tycho_enzyme_bench

namespace {

using SS4 = Eigen::Array<double, 4, 1>;

// AutodiffFwd does not natively support dual<Eigen::Array<...>>; the
// Tycho convention is to scalarize per-lane externally for autodiff +
// Vectorizable.  Reference comparison here is "scalar Enzyme × W lanes"
// (the Phase 1 / Phase 3 path running W times).

void BM_Brach_Vectorized_Enzyme(benchmark::State& state) {
    using ODE = tycho_enzyme_bench::BrachBench<
        tycho::vf::DenseDerivativeMode::EnzymeAD,
        tycho::vf::DenseDerivativeMode::AutodiffFwd>;
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

// -----------------------------------------------------------------------------
// Phase 2: Hessian benchmarks.  Each iteration computes the full Jacobian +
// adjoint gradient + adjoint Hessian via the active EnzymeAD pathway (FoR or
// FoF, depending on TYCHO_ENZYME_HESSIAN_STRATEGY).  Compared head-to-head
// with the AutodiffFwd nested-dual reference.
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

void BM_Hessian_Brach_Autodiff(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::BrachAutodiff>(state, brach_input());
}
void BM_Hessian_Brach_Enzyme(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::BrachEnzymeFull>(state, brach_input());
}

void BM_Hessian_CR3BP_Autodiff(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::CR3BPAutodiff>(state, cr3bp_input());
}
void BM_Hessian_CR3BP_Enzyme(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::CR3BPEnzymeFull>(state, cr3bp_input());
}

void BM_Hessian_MEE_Autodiff(benchmark::State& state) {
    bm_hessian_body<tycho_enzyme_test::MEEAutodiff>(state, mee_input());
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
BENCHMARK(BM_Brach_Autodiff)->Name("BM_Jacobian_Autodiff/Brach");
BENCHMARK(BM_Brach_Enzyme)->Name("BM_Jacobian_Enzyme/Brach");

// CR3BP (7 -> 6)
BENCHMARK(BM_CR3BP_Autodiff)->Name("BM_Jacobian_Autodiff/CR3BP");
BENCHMARK(BM_CR3BP_Enzyme)->Name("BM_Jacobian_Enzyme/CR3BP");

// MEE (9 -> 6)
BENCHMARK(BM_MEE_Autodiff)->Name("BM_Jacobian_Autodiff/MEE");
BENCHMARK(BM_MEE_Enzyme)->Name("BM_Jacobian_Enzyme/MEE");

// Phase 5b: vectorized Brach Jacobian (W=4 SIMD lanes per call, Enzyme only).
BENCHMARK(BM_Brach_Vectorized_Enzyme)->Name("BM_JacobianVec_Enzyme/Brach");

// Phase 2: Hessians.  EnzymeFull = <EnzymeAD, EnzymeAD>; the active
// TYCHO_ENZYME_HESSIAN_STRATEGY chooses FoR vs FoF at compile time.
BENCHMARK(BM_Hessian_Brach_Autodiff)->Name("BM_Hessian_Autodiff/Brach");
BENCHMARK(BM_Hessian_Brach_Enzyme)->Name("BM_Hessian_Enzyme/Brach");
BENCHMARK(BM_Hessian_CR3BP_Autodiff)->Name("BM_Hessian_Autodiff/CR3BP");
BENCHMARK(BM_Hessian_CR3BP_Enzyme)->Name("BM_Hessian_Enzyme/CR3BP");
BENCHMARK(BM_Hessian_MEE_Autodiff)->Name("BM_Hessian_Autodiff/MEE");
BENCHMARK(BM_Hessian_MEE_Enzyme)->Name("BM_Hessian_Enzyme/MEE");

// Phase 2 gate: full-solve TTS for the brachistochrone.  Each iteration
// builds the phase + solves PSIOPT, so the per-iteration cost includes
// problem setup as well as the solve itself.
void BM_Solve_Brach_Autodiff(benchmark::State& state) {
    bm_brachistochrone_solve_body<tycho_enzyme_test::BrachAutodiff>(state);
}
void BM_Solve_Brach_Enzyme(benchmark::State& state) {
    bm_brachistochrone_solve_body<tycho_enzyme_test::BrachEnzymeFull>(state);
}
BENCHMARK(BM_Solve_Brach_Autodiff)
    ->Name("BM_FullSolve_Autodiff/Brach")
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Solve_Brach_Enzyme)
    ->Name("BM_FullSolve_Enzyme/Brach")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
