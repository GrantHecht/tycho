///////////////////////////////////////////////////////////////////////////////
// Optimal control benchmarks — OCP phase construction + transcription
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <memory>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Helper: build a Brachistochrone phase
///////////////////////////////////////////////////////////////////////////////

static std::shared_ptr<ODEPhase<BrachODE>> make_brach_phase(int n_segs) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;

    int n_pts = n_segs * 3 + 1; // LGL3 collocation
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
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

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Phase construction + transcription benchmarks
///////////////////////////////////////////////////////////////////////////////

static void BM_Phase_Construct_16seg(benchmark::State &state) {
    for (auto _ : state) {
        auto phase = make_brach_phase(16);
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Construct_16seg);

static void BM_Phase_Construct_64seg(benchmark::State &state) {
    for (auto _ : state) {
        auto phase = make_brach_phase(64);
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Construct_64seg);

///////////////////////////////////////////////////////////////////////////////
// Transcription-only benchmarks (exclude phase construction)
///////////////////////////////////////////////////////////////////////////////

static void BM_Phase_Transcribe_16seg(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto phase = make_brach_phase(16);
        state.ResumeTiming();
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Transcribe_16seg);

static void BM_Phase_Transcribe_64seg(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto phase = make_brach_phase(64);
        state.ResumeTiming();
        phase->transcribe(false, false);
        benchmark::DoNotOptimize(phase);
    }
}
BENCHMARK(BM_Phase_Transcribe_64seg);
