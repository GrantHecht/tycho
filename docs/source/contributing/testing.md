# Testing

Tycho has four layers of automated checks. Every pull request into
`main` must clear all of them.

| Layer                 | Tool                  | Scope                                                                |
| --------------------- | --------------------- | -------------------------------------------------------------------- |
| C++ unit tests        | Google Test + CTest   | Subsystem-level correctness (`tests/cpp/`)                           |
| Python examples       | `scripts/run_examples.py` | 32 example scripts under `examples/python_examples/`             |
| C++ example regression| `brachistochrone_cpp` | End-to-end check that the C++ optimal-control path converges         |
| Benchmarks            | `bench/bench_track.sh`| Performance regressions on the core solve paths                      |

## C++ unit tests (Google Test)

C++ tests are opt-in via `BUILD_CPP_TESTS=ON`. Google Test is fetched
automatically via CMake `FetchContent`.

```bash
cmake --preset <preset> -DBUILD_CPP_TESTS=ON
cd build && ninja -j6 tycho_tests tycho_tests_light
ctest --output-on-failure
```

`tycho_tests_light` is a fast-running subset; `tycho_tests` is the full
suite. CTest will run both. The expectation is **all tests pass** — a
single failure blocks merging.

## Python examples (integration test suite)

The 32 Python scripts under `examples/python_examples/` double as
Tycho's integration test suite. Each script is run by
`scripts/run_examples.py`, which captures stdout / stderr and
verifies a non-zero exit code becomes a test failure.

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Useful flags:

- `--timeout SECONDS` — per-script timeout (default is generous).
- `--filter SUBSTRING` — only run scripts whose path contains the
  substring. Useful when iterating on a single example.

All 32 scripts must exit zero. If your change breaks an example, fix
the example in the same PR (or justify the breakage in the description
and update the example accordingly).

The full example matrix requires a few extra optional dependencies:

```bash
conda run -n tycho pip install seaborn
conda install -n tycho -c conda-forge basemap
```

## C++ brachistochrone regression

The brachistochrone example is the canonical end-to-end check for the
C++ optimal-control path. C++ examples are not built by default; opt
in via `BUILD_CPP_EXAMPLES=ON`.

```bash
cmake --preset <preset> -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j6 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

Expected result — PSIOPT prints its convergence banner and the reported
objective settles near the known optimum (the exact stdout formatting may
differ slightly between platforms):

```
Optimal Solution Found
objective ≈ 1.8013 s
```

A Builder-API variant is also available; `ninja brachistochrone_builder`
builds it under `examples/cpp_examples/builder/brachistochrone/`.

## Benchmarks

Performance regressions are caught by `bench/bench_track.sh`, which
wraps Google Benchmark.

```bash
cmake --preset <preset> -DBUILD_CPP_BENCHMARKS=ON
cd build && ninja -j6 bench_all

bench/bench_track.sh baseline    # record a baseline (one-time)
bench/bench_track.sh record      # record current numbers
bench/bench_track.sh compare     # diff against the baseline
```

Any regression in benchmark numbers requires a written justification in
the PR description. Hardware-specific guidance for reproducible runs
lives in `bench/MACBENCH.md` (macOS) and `bench/WINBENCH.md` (Windows).

## Pre-merge verification sequence

Before opening a PR (or before flipping a draft to ready-for-review),
run the four steps below in order. CI runs the same four checks; doing
them locally first is faster than waiting for CI failures.

1. **C++ unit tests**

   ```bash
   cd build && ctest --output-on-failure
   ```
   All tests must pass.

2. **Python examples**

   ```bash
   conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
   ```
   All 32 examples must exit 0.

3. **C++ brachistochrone**

   ```bash
   cmake --preset <preset> -DBUILD_CPP_EXAMPLES=ON
   cd build && ninja -j6 brachistochrone_cpp
   ./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
   ```
   Must print `Optimal Solution Found`, objective ≈ 1.8013 s.

4. **Benchmarks**

   ```bash
   bench/bench_track.sh compare
   ```
   Any regressions must be justified in the PR description.

## Merge policy

All four steps above must pass before any pull request can be merged
into `main`. Broken examples must be fixed in the same PR; benchmark
regressions must be justified in the PR description.
