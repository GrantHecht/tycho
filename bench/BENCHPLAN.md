# Benchmark Plan: pybind11 vs nanobind Bindings

## Context

This plan benchmarks the new nanobind-based Python bindings (on `feat/nanobind-migration`) against the old pybind11 bindings (on `main`) to measure:
1. **Compilation time** — how long each takes to build
2. **Binary size** — size of the compiled extension module
3. **Runtime performance** — execution speed of solved optimal control problems

The benchmark will use stripped-down versions of the Brachistochrone problem (the simplest OCP) to isolate binding overhead from computation.

---

## Approach

### Branch Setup

| Branch | Binding | Location |
|--------|---------|----------|
| `main` | pybind11 | Standard build |
| `feat/nanobind-migration` | nanobind | Current feature branch |

### Binary Size Benchmark

**Method:** Measure the size of the compiled `_tycho` extension module.

**Steps:**
1. On each branch, build the project
2. Measure the size of the compiled extension:
   ```bash
   ls -lh build/tycho/*.so
   stat -f "%z bytes" build/tycho/*.so
   ```
3. Also measure the C++ benchmark binary size:
   ```bash
   ls -lh build/examples/cpp_examples/brachistochrone/brachistochrone_bench
   ```

### Compilation Time Benchmark

**Method:** Clean rebuild from scratch with `ninja -t time`.

**Steps:**
1. On each branch, run a clean build:
   ```bash
   # On main (pybind11)
   cd build && rm -rf * && cmake -G Ninja .. && ninja -t time all

   # On feat/nanobind-migration (nanobind)
   cd build && rm -rf * && cmake -G Ninja .. && ninja -t time all
   ```

2. Capture the total build time and per-target breakdown from `ninja -t time`.

3. Run 3 trials per branch and report mean ± std dev.

### Runtime Benchmark

**Method:** Time execution of a stripped-down Brachistochrone solve.

#### C++ Benchmark

Create a minimal benchmark binary `brachistochrone_bench.cpp` based on `examples/cpp_examples/brachistochrone/main.cpp`:

```cpp
// Key additions:
#include <chrono>
auto start = std::chrono::high_resolution_clock::now();
phase->solve_optimize();
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration<double>(end - start).count();
std::cout << "Solve time: " << duration << " s\n";
```

**Files to modify/create:**
- `examples/cpp_examples/brachistochrone/bench.cpp` — stripped-down benchmark
- Update `examples/cpp_examples/CMakeLists.txt` to build `bench` target when `BUILD_CPP_EXAMPLES=ON`

**Measure:**
- Solve time (optimizer convergence)
- Report mean of 5 runs

#### Python Benchmarks

Create stripped-down benchmark scripts (no plotting):

| Script | Problem | Characteristics |
|--------|---------|-----------------|
| `benchmark_brachistochrone.py` | Brachistochrone | Basic, LGL3, 32 nodes |
| `benchmark_bryson_denham.py` | Bryson-Denham | Integral objective, LGL5 |
| `benchmark_vanderpol.py` | Van der Pol | Complex ODE, 256 nodes, threading |
| `benchmark_zermelo.py` | Zermelo's Problem | Time-dependent dynamics, wind model |
| `benchmark_dionysus.py` | Dionysus Low-Thrust | MEE dynamics, 160 nodes, astro models |

Each script prints only the solve time (seconds) to stdout.

**Measure:**
- Solve time from `phase.optimize()` or `phase.solve_optimize()` call
- Report mean of 10 runs per benchmark

---

## Critical Files

| File | Purpose |
|------|---------|
| `config_and_build.sh` | Build configuration |
| `CMakeLists.txt` | Root CMake |
| `src/Bindings/CMakeLists.txt` | Binding build (pybind11 vs nanobind) |
| `examples/cpp_examples/brachistochrone/main.cpp` | C++ reference |
| `examples/Brachistochrone.py` | Python reference |

---

## Implementation Steps

### Step 1: Create C++ Benchmark Binary

1. Create `examples/cpp_examples/brachistochrone/bench.cpp`:
   - Copy from `main.cpp`
   - Remove all I/O except timing output
   - Keep only the solve call and convergence check
   - Wrap solve timing in `std::chrono`

2. Update `examples/cpp_examples/CMakeLists.txt`:
   - Add `bench` executable target
   - Only build when `BUILD_CPP_EXAMPLES=ON`

### Step 2: Create Python Benchmark Scripts

Create the following stripped-down benchmark scripts (no plotting):

1. `examples/benchmark_brachistochrone.py` — Basic Brachistochrone
2. `examples/benchmark_bryson_denham.py` — Integral objective, LGL5
3. `examples/benchmark_vanderpol.py` — Complex ODE, 256 nodes, threading
4. `examples/benchmark_zermelo.py` — Time-dependent dynamics, wind model

Each script should:
- Remove all `matplotlib` imports and plotting code
- Add `time.perf_counter()` around `phase.optimize()`
- Print only the solve time (single float)

### Step 3: Run Compilation Benchmarks

1. **On `main` branch:**
   ```bash
   git checkout main
   rm -rf build && mkdir build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
   ninja -t time all 2>&1 | tee compile_time_main.txt
   ```

2. **On `feat/nanobind-migration` branch:**
   ```bash
   git checkout feat/nanobind-migration
   rm -rf build && mkdir build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
   ninja -t time all 2>&0 | tee compile_time_nb.txt
   ```

3. Record total build time from each output.

### Step 4: Run Runtime Benchmarks

1. **C++ (both branches):**
   ```bash
   # After building on each branch:
   ./build/examples/cpp_examples/brachistochrone/bench
   # Run 5 times, record times
   ```

2. **Python (both branches):**
   ```bash
   # After installing on each branch:
   python examples/benchmark_brachistochrone.py
   # Run 10 times, record times
   ```

---

## Verification

### Pre-Benchmark Checks

- [ ] Both branches build without errors
- [ ] `main` branch produces `_tycho` with pybind11
- [ ] `feat/nanobind-migration` branch produces `_tycho` with nanobind
- [ ] C++ `bench` binary converges (prints "Optimal Solution Found")
- [ ] Python benchmark script converges (no errors)

### Post-Benchmark Verification

- [ ] Record all timing data in a structured format (CSV or JSON)
- [ ] Calculate mean and standard deviation for all measurements
- [ ] Verify solutions are consistent between branches (same optimal time ≈1.80s for Brachistochrone)
- [ ] Compare binary sizes: _tycho.so and C++ benchmark binary

---

## Deliverables

1. **C++ benchmark binary:** `examples/cpp_examples/brachistochrone/bench.cpp`
2. **Python benchmark scripts:**
   - `examples/benchmark_brachistochrone.py` — Brachistochrone (basic, 32 nodes)
   - `examples/benchmark_bryson_denham.py` — Bryson-Denham (integral objective, LGL5)
   - `examples/benchmark_vanderpol.py` — Van der Pol (256 nodes, threading)
   - `examples/benchmark_zermelo.py` — Zermelo's Problem (time-dependent, wind model)
3. **Timing results:**
   - Compilation time: pybind11 vs nanobind (with per-target breakdown)
   - Runtime C++: pybind11 vs nanobind (5-run mean)
   - Runtime Python: pybind11 vs nanobind (10-run mean, 4 benchmarks)
4. **Summary:** Comparison table and analysis

---

## Notes

- **Warm-up runs:** Discard first run for runtime benchmarks (JIT/caching effects)
- **Environment:** Use consistent conda environment (`tycho`) for both branches
- **Isolation:** Reboot or clear caches between branch switches to avoid cross-contamination
- **Thread count:** Use `-j6` for consistency with CLAUDE.md recommendations

### Compilation Benchmark: OS Page-Cache Warming (TODO: improve methodology)

**Observed (2026-03-02, nanobind branch, Apple M-series, `--trials 3`):**

| Trial | Time |
|-------|------|
| 1 (cold) | 737.9 s |
| 2 (warm) | 545.2 s |
| 3 (warm) | 410.4 s |

`ninja -t clean` removes compiled artifacts but leaves source files in the macOS page cache.
Subsequent trials read source files from RAM rather than disk, yielding monotonically
decreasing build times. The std (134 s) is 24% of the mean — too noisy for reliable
cross-branch comparison.

**Current workaround:** Use `--trials 1` (single cold-cache trial) for all branch comparisons.

**Future improvements to consider:**
- Run `sudo purge` between trials for true cold-cache isolation (requires sudo, adds time)
- Use `--trials 1` but run the benchmark 2–3 times on separate occasions and report the range
- Move compilation benchmarking to a dedicated CI environment (Linux, predictable caching)
- Record `ninja -t compdb` per-file times to identify the slowest translation units
