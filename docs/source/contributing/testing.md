# Testing

Tycho's automated checks span both the Python API and the C++ core. Run
the layers relevant to your change before opening a PR; the table below
maps each one to its tool and scope.

| Layer              | Tool                      | Scope                                                |
| ------------------ | ------------------------- | ---------------------------------------------------- |
| Python unit tests  | `pytest`                  | Python-API + bindings regression (`tychopy/test/`)   |
| Python examples    | `scripts/run_examples.py` | 32 example scripts under `examples/python_examples/` |
| C++ unit tests     | Google Test + CTest       | Subsystem-level correctness (`tests/cpp/`)           |
| C++ benchmarks     | `bench/bench_track.sh`    | Performance regressions on the core solve paths      |

## Python unit tests (pytest)

The `pytest` suite under `tychopy/test/` exercises the Python API and the
nanobind bindings against a built `_tychopy`. Tests are organised by
subsystem — `test_VectorFunctions/`, `test_OptimalControl/`,
`test_Astro/`, `test_AdaptiveMesh/`, `test_AutoScaling/`,
`test_Bindings/`, and `test_FullProblems/`.

`pytest` is not part of the default environment, so install it first (the
other test dependencies — numpy, scipy, matplotlib, spiceypy — already
ship with the standard `tycho` env):

```bash
conda run -n tycho pip install pytest
```

The full set is also codified as the `[test]` optional-dependency group in
`pyproject.toml` — that is what CI installs against the built wheel. Avoid
`pip install ".[test]"` in a source checkout, though: it would recompile
`_tychopy` from scratch. Install the dependencies directly instead.

Then run the suite from the repo root (the `tycho` env must already have a
built/installed `_tychopy`):

```bash
conda run -n tycho python -m pytest tychopy/test
```

Useful invocations:

- `python -m pytest tychopy/test/test_Bindings` — run a single subsystem.
- `python -m pytest -k Reentry` — run only tests matching an expression.
- `python -m pytest -q` — quieter, dot-style output.

## Python examples (integration suite)

The 32 Python scripts under `examples/python_examples/` double as Tycho's
integration test suite. Each script is run by `scripts/run_examples.py`,
which captures stdout / stderr and treats a non-zero exit code as a
failure.

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Useful flags:

- `--timeout SECONDS` — per-script timeout (default is generous).
- `--filter SUBSTRING` — only run scripts whose path contains the
  substring. Useful when iterating on a single example.

The full example matrix requires a few extra optional dependencies:

```bash
conda run -n tycho pip install seaborn
conda install -n tycho -c conda-forge basemap
```

## C++ unit tests (Google Test)

C++ tests are opt-in via `BUILD_CPP_TESTS=ON`. Google Test is fetched
automatically via CMake `FetchContent`.

```bash
cmake --preset <preset> -DBUILD_CPP_TESTS=ON
cd build && ninja -j6 tycho_tests tycho_tests_light
ctest --output-on-failure
```

`tycho_tests_light` is a fast-running subset; `tycho_tests` is the full
suite. CTest runs both.

## C++ benchmarks

Performance regressions are caught by `bench/bench_track.sh`, which wraps
Google Benchmark.

```bash
cmake --preset <preset> -DBUILD_CPP_BENCHMARKS=ON
cd build && ninja -j6 bench_all

bench/bench_track.sh baseline    # record a baseline (one-time)
bench/bench_track.sh record      # record current numbers
bench/bench_track.sh compare     # diff against the baseline
```

Hardware-specific guidance for reproducible runs lives in
`bench/MACBENCH.md` (macOS) and `bench/WINBENCH.md` (Windows).
