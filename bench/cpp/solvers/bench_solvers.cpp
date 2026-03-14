///////////////////////////////////////////////////////////////////////////////
// PSIOPT solver benchmarks — end-to-end convergence
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <memory>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Helper: build a fresh Brachistochrone phase for solving
///////////////////////////////////////////////////////////////////////////////

static std::shared_ptr<ODEPhase<BrachODE>> make_brach_solver_phase(int n_segs) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;

    int n_pts = n_segs * 3 + 1;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * 1.0 * std::cos(1.0);
        pt[3] = t0 + 1.0 * s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, n_segs);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);
    phase->optimizer->PrintLevel = 3; // suppress all solver output

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// PSIOPT solve_optimize benchmarks
// Each iteration constructs a fresh phase (solver modifies internal state)
///////////////////////////////////////////////////////////////////////////////

static void BM_PSIOPT_Brach_16seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_solver_phase(16);
        auto status = phase->solve_optimize();
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("PSIOPT did not converge (16seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_PSIOPT_Brach_16seg)->Unit(benchmark::kMillisecond);

static void BM_PSIOPT_Brach_32seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_solver_phase(32);
        auto status = phase->solve_optimize();
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("PSIOPT did not converge (32seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_PSIOPT_Brach_32seg)->Unit(benchmark::kMillisecond);
