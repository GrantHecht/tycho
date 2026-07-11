///////////////////////////////////////////////////////////////////////////////
// Gridded/LGL interpolation benchmarks -- 2026-07 review series PR 8 coverage
// foundation for OC §2.2 (LGL interp tpow allocation) and §2.4 (2-D interp
// coefficient cache).
//
// Prior to this file, no benchmark referenced LGLInterpTable, interp_block_gen,
// InterpTable2D, InterpTable3D, or InterpTable4D anywhere in bench/cpp/ -- see
// the PR 8 OC/INTEGRATORS fact dossier. These benches exist to gate the
// upcoming win-or-drop perf changes (fixed-max-size tpow/tpow2 buffers,
// per-cell coefficient cache) against a real baseline.
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>
#include <cmath>
#include <memory>
#include <tycho/optimal_control.h>
#include <vector>

using namespace tycho;
using namespace tycho::oc;

namespace {

///////////////////////////////////////////////////////////////////////////////
// Helper: single-state (no controls) LGLInterpTable built in the given
// transcription mode from a short analytic trajectory (x(t) = sin(t),
// xdot = cos(t)) loaded via load_exact_data() -- same shape as
// oc_test_utils.h's make_exact_lgl_table(), reimplemented locally so this
// bench TU doesn't pull in gtest via tests/cpp headers.
//
// LGL3/Trapezoidal dispatch interp_block()/interp_block_deriv() to the
// specialized, allocation-free interp_block_lgl3()/interp_block_deriv_lgl3().
// LGL5/LGL7 fall through to the general-order interp_block_gen()/
// interp_block_deriv_gen() kernels -- the ones OC review §2.2 flags for
// per-call VectorX<Scalar> tpow/tpow2 heap allocation. Use LGL5 so these
// benches actually exercise the flagged path.
///////////////////////////////////////////////////////////////////////////////
std::shared_ptr<LGLInterpTable> make_gen_dispatch_lgl_table() {
    constexpr int x_vars = 1;
    // 6 == (num_nodes - 1) is divisible by (block_size_ - 1) for LGL5 (2).
    constexpr int num_nodes = 7;
    constexpr double t_final = 1.0;

    auto tab = std::make_shared<LGLInterpTable>(x_vars, /*uv=*/0, TranscriptionModes::LGL5);

    std::vector<Eigen::VectorXd> xtudat;
    std::vector<Eigen::VectorXd> xdotdat;
    xtudat.reserve(num_nodes);
    xdotdat.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        double t = t_final * static_cast<double>(i) / (num_nodes - 1);
        Eigen::VectorXd node(2); // [x, t] -- axis_ == x_vars == 1
        node << std::sin(t), t;
        xtudat.push_back(node);
        Eigen::VectorXd deriv(1);
        deriv << std::cos(t);
        xdotdat.push_back(deriv);
    }
    tab->load_exact_data(xtudat, xdotdat);
    return tab;
}

constexpr int kSamples = 100;

/// 100 sample times spread across the table's domain, staying strictly
/// inside [0, t_final) so find_block()'s search never has to clamp at the
/// upper edge.
std::vector<double> make_sample_times() {
    std::vector<double> ts(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        ts[i] = 0.999 * static_cast<double>(i) / static_cast<double>(kSamples - 1);
    }
    return ts;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// LGLInterpTable::interpolate() -- drives interp_block_gen()'s per-call tpow.
///////////////////////////////////////////////////////////////////////////////

static void BM_LGLInterpTable_Interp(benchmark::State &state) {
    auto tab = make_gen_dispatch_lgl_table();
    auto ts = make_sample_times();

    for (auto _ : state) {
        double acc = 0.0;
        for (double t : ts) {
            Eigen::VectorXd fx = tab->interpolate(t);
            acc += fx[0];
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * kSamples);
}
BENCHMARK(BM_LGLInterpTable_Interp);

///////////////////////////////////////////////////////////////////////////////
// LGLInterpTable::interpolate_deriv() -- drives interp_block_deriv_gen()'s
// per-call tpow + tpow2.
///////////////////////////////////////////////////////////////////////////////

static void BM_LGLInterpTable_InterpDeriv(benchmark::State &state) {
    auto tab = make_gen_dispatch_lgl_table();
    auto ts = make_sample_times();

    for (auto _ : state) {
        double acc = 0.0;
        for (double t : ts) {
            Eigen::Matrix<double, -1, 2> fx = tab->interpolate_deriv(t);
            acc += fx(0, 0) + fx(0, 1);
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * kSamples);
}
BENCHMARK(BM_LGLInterpTable_InterpDeriv);

///////////////////////////////////////////////////////////////////////////////
// InterpTable2D::interp() -- OC review §2.4 reference bench: 2-D has no
// per-cell coefficient cache (rebuilds the L*Z*R 4x4 GEMM chain in
// get_amatrix() every call), unlike 3-D/4-D's cache_alpha_/cache_alphavecs().
// Analogous in structure to BM_ChebTable2D_Eval (bench_cheb.cpp).
///////////////////////////////////////////////////////////////////////////////

namespace {

double ref_fn_2d(double x, double y) { return std::sin(0.8 * x) * std::cos(0.6 * y) + 0.1 * x * y; }

Eigen::VectorXd linspace_2d(double a, double b, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i)
        v[i] = a + (b - a) * static_cast<double>(i) / static_cast<double>(n - 1);
    return v;
}

} // namespace

static void BM_InterpTable2D_Eval(benchmark::State &state) {
    auto xs = linspace_2d(0.0, 3.0, 20);
    auto ys = linspace_2d(-1.0, 2.0, 20);

    InterpTable2D::MatType zs(ys.size(), xs.size());
    for (int j = 0; j < ys.size(); ++j)
        for (int i = 0; i < xs.size(); ++i)
            zs(j, i) = ref_fn_2d(xs[i], ys[j]);

    InterpTable2D table(xs, ys, zs, InterpType::Cubic);

    // 100 query points spread across the interior of the domain (strictly
    // inside the grid bounds so every call takes the normal in-domain path,
    // not the out-of-bounds throw).
    std::vector<std::pair<double, double>> pts;
    pts.reserve(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        double s = static_cast<double>(i) / static_cast<double>(kSamples - 1);
        double x = 0.05 + s * (2.95 - 0.05);
        double y = -0.95 + s * (1.95 - (-0.95));
        pts.emplace_back(x, y);
    }

    for (auto _ : state) {
        double acc = 0.0;
        for (auto &p : pts) {
            acc += table.interp(p.first, p.second);
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * kSamples);
}
BENCHMARK(BM_InterpTable2D_Eval);
