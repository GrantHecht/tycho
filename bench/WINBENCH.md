## Benchmarking (Windows x64, clang-cl + MSVC)

### Overview

The C++ benchmark suite uses [Google Benchmark](https://github.com/google/benchmark) (v1.9.5, fetched automatically via CMake FetchContent). All 46 benchmarks compile into a single executable (`bench_all.exe`) covering every major subsystem: Kepler/Lambert astrodynamics, VectorFunction DSL evaluation, type erasure dispatch, RK integrators, optimal control transcription, PSIOPT solver convergence, and utility primitives (TypeStorage, MemoryManager, ThreadPool).

A local tracking script (`bench/bench_track.sh`) records results keyed by git commit hash and compares runs to detect performance regressions. On Windows the script runs under Git Bash (or WSL).

### Prerequisites

- LLVM/clang-cl (installed to `C:\Program Files\LLVM`)
- Visual Studio 2022 with C++ desktop workload (provides MSVC headers/libs)
- Intel OneAPI Base Toolkit (for MKL)
- Conda environment `tycho` with Python 3.13
- `jq` (install via `winget install jqlang.jq` or `scoop install jq`)

### Building

Building uses the shared environment setup script which loads the VS developer
environment and Intel OneAPI vars automatically.

```powershell
# One-time: configure with benchmarks enabled
powershell -File scripts/configure_tests.ps1

# Build (via the standard build script, which builds all targets including bench_all)
powershell -File scripts/config_and_build.ps1
```

Or manually after environment setup:

```powershell
. scripts\env_setup.ps1
& $cmake --build --preset x64-Clang-Release --target bench_all --parallel 8
```

The benchmark binary is located at:
```
out\build\x64-Clang-Release\bench\cpp\bench_all.exe
```

### Running benchmarks directly

From Git Bash or any shell where the binary is accessible:

```bash
BENCH="out/build/x64-Clang-Release/bench/cpp/bench_all.exe"

# Full run (default: each benchmark runs until statistically stable)
"$BENCH"

# Quick smoke test
"$BENCH" --benchmark_min_time=0.01s

# Run a subset by name filter
"$BENCH" --benchmark_filter="BM_Kepler|BM_Lambert"

# JSON output for external tools
"$BENCH" --benchmark_out=results.json --benchmark_out_format=json
```

### Tracking performance across commits

The `bench/bench_track.sh` script expects the benchmark binary at
`build/bench/cpp/bench_all` (the macOS build path). On Windows, override the
path by setting `BENCH_BINARY` before running:

```bash
export BENCH_BINARY="out/build/x64-Clang-Release/bench/cpp/bench_all.exe"
```

Results are stored in `bench/results/` (gitignored) as JSON files named by
commit hash.

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

```powershell
# 1. On main (or before changes), build and establish baseline
git checkout main
powershell -File scripts/config_and_build.ps1
```

```bash
# (in Git Bash)
export BENCH_BINARY="out/build/x64-Clang-Release/bench/cpp/bench_all.exe"
bench/bench_track.sh baseline
```

```powershell
# 2. Switch to feature branch, rebuild, record
git checkout feature/my-change
powershell -File scripts/config_and_build.ps1
```

```bash
bench/bench_track.sh record

# 3. Compare
bench/bench_track.sh compare
```

### Getting reliable measurements

- **Close other applications** during benchmark runs — browser tabs, Discord, and IDE indexing add noise. Background Windows services (Defender, Windows Update) can also cause variance.
- **Use 5+ repetitions** (the default). The script uses the median, which is robust to occasional outliers but still benefits from more samples.
- **Keep the machine plugged in** and set the power plan to **High Performance** — Windows throttles CPU on battery and on Balanced power plans.
- **Disable hardware P-state management** if possible (via BIOS or `powercfg`). Windows' default frequency scaling adds measurement noise.
- **Let the machine settle** after a rebuild — compiler processes heat up the CPU, which can temporarily throttle clock speeds.
- **Compare only results from the same machine.** Benchmark JSON files are not portable across hardware.
- If uncommitted changes exist, results are tagged `-dirty` in the filename and metadata. Prefer benchmarking on clean commits for reproducibility.
- **ThreadPool benchmarks** (especially `BM_ThreadPool_Dispatch`) have high variance on Windows due to OS scheduler behavior. Treat outliers on single-dispatch benchmarks with skepticism; batch dispatch benchmarks are more stable.

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
