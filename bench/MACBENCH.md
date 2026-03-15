## Benchmarking (macOS Apple Silicon)

### Overview

The C++ benchmark suite uses [Google Benchmark](https://github.com/google/benchmark) (v1.9.5, fetched automatically via CMake FetchContent). All 46 benchmarks compile into a single executable (`bench_all`) covering every major subsystem: Kepler/Lambert astrodynamics, VectorFunction DSL evaluation, type erasure dispatch, RK integrators, optimal control transcription, PSIOPT solver convergence, and utility primitives (TypeStorage, MemoryManager, ThreadPool).

A local tracking script (`bench/bench_track.sh`) records results keyed by git commit hash and compares runs to detect performance regressions.

### Building

```bash
# One-time: configure with benchmarks enabled
cmake --preset macos-llvm-release -DBUILD_CPP_BENCHMARKS=ON

# Build
cd build && ninja -j2 bench_all
```

### Running benchmarks directly

```bash
# Full run (default: each benchmark runs until statistically stable)
./build/bench/cpp/bench_all

# Quick smoke test
./build/bench/cpp/bench_all --benchmark_min_time=0.01s

# Run a subset by name filter
./build/bench/cpp/bench_all --benchmark_filter="BM_Kepler|BM_Lambert"

# JSON output for external tools
./build/bench/cpp/bench_all --benchmark_out=results.json --benchmark_out_format=json
```

### Tracking performance across commits

Results are stored in `bench/results/` (gitignored) as JSON files named by commit hash.

**Establish a baseline** (typically on `main` or before a refactor):
```bash
bench/bench_track.sh baseline
```

This runs benchmarks with 5 repetitions (configurable via `--reps N`), extracts the median of each benchmark, and saves the result as the baseline for future comparisons.

**Record results for the current commit:**
```bash
bench/bench_track.sh record
```

Same as `baseline` but does not update the baseline symlink. Use this after making changes to capture the new performance profile.

**Compare against baseline:**
```bash
# Compare HEAD against baseline
bench/bench_track.sh compare

# Compare two specific commits
bench/bench_track.sh compare abc12345 def67890

# Custom regression threshold (default: 10%)
bench/bench_track.sh compare --threshold 5.0
```

The comparison table shows old/new times, percent change, and flags regressions (`!! REGRESSION`) or improvements (`++ faster`) beyond the threshold. Exit code 2 indicates regressions were detected.

**List stored results:**
```bash
bench/bench_track.sh list
```

### Typical workflow

```bash
# 1. On main (or before changes), establish baseline
git checkout main
cd build && ninja -j2 bench_all && cd ..
bench/bench_track.sh baseline

# 2. Switch to feature branch, rebuild, record
git checkout feature/my-change
cd build && ninja -j2 bench_all && cd ..
bench/bench_track.sh record

# 3. Compare
bench/bench_track.sh compare
```

### Getting reliable measurements

- **Close other applications** during benchmark runs — browser tabs, Slack, and IDE indexing add noise.
- **Use 5+ repetitions** (the default). The script uses the median, which is robust to occasional outliers but still benefits from more samples.
- **Plug in your laptop** — macOS throttles CPU on battery.
- **Let the machine settle** after a rebuild — ccache and compiler processes heat up the CPU, which can temporarily throttle clock speeds.
- **Compare only results from the same machine.** Benchmark JSON files are not portable across hardware.
- If uncommitted changes exist, results are tagged `-dirty` in the filename and metadata. Prefer benchmarking on clean commits for reproducibility.

### Benchmark suite structure

| File | Benchmarks | What it measures |
|------|-----------|------------------|
| `kepler/bench_kepler.cpp` | 12 | Element conversions, Kepler propagation, Lambert solver |
| `vector_functions/bench_vector_functions.cpp` | 8 | VF compute, Jacobian, adjoint gradient, full VJP, composition |
| `type_erasure/bench_type_erasure.cpp` | 6 | GenericFunction VJP dispatch, GFStorage clone |
| `integrators/bench_integrators.cpp` | 7 | RK stepper throughput (DOPRI54/87, fixed-step, dense output) |
| `optimal_control/bench_optimal_control.cpp` | 4 | Phase construction + LGL3 transcription |
| `solvers/bench_solvers.cpp` | 2 | End-to-end PSIOPT convergence |
| `utils/bench_utils.cpp` | 7 | TypeStorage SBO, MemoryManager, ThreadPool dispatch |

Shared ODE definitions and the `make_brach_phase` helper live in `bench/cpp/bench_common.h`.
