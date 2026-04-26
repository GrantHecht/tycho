// =============================================================================
// Phase 1 Enzyme microbenchmark.
// Compares __enzyme_fwddiff Jacobians against AutodiffFwd dual-number Jacobians
// for Brachistochrone (5->3), CR3BP (7->6), and Modified Equinoctial Elements
// (9->6).
//
// Plugin flag is scoped to this target only (see bench/cpp/CMakeLists.txt);
// the rest of the benchmark suite is unaffected.
// =============================================================================
#include <benchmark/benchmark.h>

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

BENCHMARK_MAIN();
