///////////////////////////////////////////////////////////////////////////////
// InteriorPointSolver solver benchmarks — end-to-end convergence
///////////////////////////////////////////////////////////////////////////////

#include "../bench_phases.h"
#include <benchmark/benchmark.h>
#include <tycho/solvers.h>

///////////////////////////////////////////////////////////////////////////////
// InteriorPointSolver solve_optimize benchmarks
// Each iteration constructs a fresh phase (solver modifies internal state)
///////////////////////////////////////////////////////////////////////////////

static void BM_InteriorPointSolver_Brach_16seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_phase(16);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (16seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_Brach_16seg)->Unit(benchmark::kMillisecond);

static void BM_InteriorPointSolver_Brach_32seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_phase(32);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (32seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_Brach_32seg)->Unit(benchmark::kMillisecond);

///////////////////////////////////////////////////////////////////////////////
// Larger-scale InteriorPointSolver benchmarks (PolarLT — 4 states, 2 controls)
// These exercise the solver at a more realistic scale where fixed overheads
// (MT-METIS init, Accelerate cold-start) are amortized.
///////////////////////////////////////////////////////////////////////////////

static void BM_InteriorPointSolver_PolarLT_128seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_polar_lt_phase(128);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (PolarLT 128seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_PolarLT_128seg)->Unit(benchmark::kMillisecond);

static void BM_InteriorPointSolver_PolarLT_256seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_polar_lt_phase(256);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (PolarLT 256seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_PolarLT_256seg)->Unit(benchmark::kMillisecond);

///////////////////////////////////////////////////////////////////////////////
// MT-METIS ordering variants — track Apple's MT-METIS improvements over time.
// These mirror the serial-METIS benchmarks above with PARMETIS ordering,
// which maps to MT-METIS on Accelerate (macOS 26+).
///////////////////////////////////////////////////////////////////////////////

static void BM_InteriorPointSolver_Brach_32seg_MTMetis(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_brach_phase(32);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        ipm.set_qp_ordering_mode(InteriorPointSolver::QPOrderingModes::PARMETIS);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (32seg MTMetis)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_Brach_32seg_MTMetis)->Unit(benchmark::kMillisecond);

static void BM_InteriorPointSolver_PolarLT_128seg_MTMetis(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_polar_lt_phase(128);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        ipm.set_qp_ordering_mode(InteriorPointSolver::QPOrderingModes::PARMETIS);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError(
                    "InteriorPointSolver did not converge (PolarLT 128seg MTMetis)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_PolarLT_128seg_MTMetis)->Unit(benchmark::kMillisecond);

///////////////////////////////////////////////////////////////////////////////
// Betts low-thrust benchmarks — large-scale MEE dynamics (7 states, 3 controls)
// Disabled: MEE ODE template instantiation uses 11+ GB per TU in shared header.
// Needs own dedicated TU to be viable as a C++ benchmark.
///////////////////////////////////////////////////////////////////////////////

#if 0
static void BM_InteriorPointSolver_BettsLT_32seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_betts_lt_phase(32, TranscriptionModes::LGL3, false);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (BettsLT 32seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_BettsLT_32seg)->Unit(benchmark::kMillisecond);

static void BM_InteriorPointSolver_BettsLT_64seg(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_betts_lt_phase(64, TranscriptionModes::LGL3, false);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (BettsLT 64seg)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_BettsLT_64seg)->Unit(benchmark::kMillisecond);

static void BM_InteriorPointSolver_BettsLT_MeshRefine(benchmark::State &state) {
    bool first = true;
    for (auto _ : state) {
        auto phase = make_betts_lt_phase(16, TranscriptionModes::LGL5, true);
        tycho::solvers::InteriorPointSolver ipm;
        ipm.set_print_level(3);
        auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        benchmark::DoNotOptimize(status);
        if (first) {
            if (status > tycho::ConvergenceFlags::ACCEPTABLE) {
                state.SkipWithError("InteriorPointSolver did not converge (BettsLT MeshRefine)");
            }
            first = false;
        }
    }
}
BENCHMARK(BM_InteriorPointSolver_BettsLT_MeshRefine)->Unit(benchmark::kMillisecond);
#endif
