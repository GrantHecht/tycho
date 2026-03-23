# Tycho — Claude Code Memory

## Project Overview

Tycho is an independently maintained fork of [ASSET (AlabamaASRL/asset_asrl)](https://github.com/AlabamaASRL/asset_asrl),
a high-performance C++/Python library for trajectory design and optimal control.
The core use cases are general optimal control problems (solved via direct collocation)
and space trajectory optimization. The built-in optimizer is called **PSIOPT**
(a high-performance interior-point solver).

The Python-facing module is `_tychopy` (nanobind extension) imported via the `tychopy` package.
The C++ namespace is `Tycho`.

## A Word of Caution for Multi-Agent Workloads
Compiling tycho is very computationally expensive, and you MUST therefore be careful about 
how many parallel jobs are being used in any given build, and how many agents are building
simultaneously. 

As a rule of thumb:
- macOS (Apple Silicon): ALWAYS use -j2 for builds
- Linux / Windows: ALWAYS use -j6 for builds (-j2 when building benchmarks)
- DO NOT PERFORM MORE THAN 2 SIMULTANEOUS BUILDS AT ONCE

## Repository Structure

Top-level files of note: `CMakeLists.txt` (root build), `CMakePresets.json`, `CMakeSettings.json` (MSVC),
`setup.py`, `requirements.txt`, `Dockerfile`, `Dockerfile-dev`.

```
include/                Public C++ API headers
  tycho/
    tycho.h             Umbrella header — includes all public modules
    typedefs.h          Eigen type aliases and SIMD detection
    utils.h             Thread pool, type storage, sizing helpers, CRTP base
    vector_functions.h  Core VectorFunction DSL
    integrators.h       Runge-Kutta steppers and coefficients
    optimal_control.h   Phase/ODE transcription, collocation, mesh refinement
    solvers.h           PSIOPT optimizer, NLP layer
    astro.h             Astrodynamics models (Kepler, CR3BP, Lambert, etc.)
    detail/             Template implementation bodies (included automatically)

src/                    C++ source code (private implementation + forwarding headers)
  Tycho.h               Internal aggregate (forwards to include/tycho/tycho.h)
  pch.h / pch.cpp       Precompiled header
  Bindings/             ALL Python binding code — nanobind .cpp files, *Bind.h
                          headers, FunctionRegistry.h (TychoBind<T> trait),
                          TypeCasters.h, TychoModule.cpp
  VectorFunctions/      Forwarding headers → include/tycho/detail/
  OptimalControl/       Forwarding headers + .cpp implementation files
  Solvers/              Forwarding headers + .cpp implementation files
  Astro/                Forwarding headers + .cpp implementation files
  Integrators/          Forwarding headers → include/tycho/detail/
  Utils/                Forwarding headers (public) + private utilities
  TypeDefs/             Forwarding headers → include/tycho/detail/
  PyDocString/          C++-side Python docstring literals

tychopy/                Python package (pure-Python layer over _tychopy extension)
  __init__.py           Package entry point
  VectorFunctions/      Python-side VectorFunction utilities
  OptimalControl/       Pure-Python ODE base class, mesh-error plotting
  Solvers/              Python solver helpers
  Astro/                Astrodynamics helpers — frames, models, constants,
                          SPICE read, date utilities, Lambert wrappers
  Utils/                Python utility helpers
  Extensions/           Python extension helpers
  test/                 Test suite (pytest-style, organised by subsystem)

dep/                    Vendored submodule dependencies
  eigen/                Eigen (MPL-2.0)
  autodiff/             autodiff (MIT)
  fmt/                  {fmt} (MIT)
  nanobind/             nanobind (BSD)

cmake/                  CMake helper modules
  Find*.cmake           Finders for MKL, Accelerate sparse, Python env, Sphinx
  clang-format.cmake    clang-format integration
  cppcheck.cmake        cppcheck integration
  git-submodule-*.cmake Submodule init/update helpers

tests/                  C++ test suite (Google Test)
  cpp/                  Test source files organised by subsystem
    astro/              Kepler, Lambert, CR3BP tests
    integrators/        RK stepper, STM, dense output tests
    optimal_control/    Phase construction, collocation, mesh refinement tests
    solvers/            PSIOPT convergence, NLP structure, Jet tests
    utils/              TypeStorage, ThreadPool, BumpAllocator tests
    vector_functions/   VF DSL, composition, Hessian, generic function tests

bench/                  Benchmark suite and tracking
  MACBENCH.md           macOS benchmarking procedure
  WINBENCH.md           Windows benchmarking procedure
  bench_track.sh        Benchmark tracking script (record, compare, list)
  results/              Stored benchmark results (gitignored)
  cpp/                  Google Benchmark source files
    bench_common.h      Shared ODE definitions and helpers
    kepler/             Kepler/Lambert astrodynamics benchmarks
    vector_functions/   VF DSL evaluation benchmarks
    type_erasure/       GenericFunction VJP dispatch, GFStorage clone
    integrators/        RK stepper throughput benchmarks
    optimal_control/    Phase construction + transcription benchmarks
    solvers/            PSIOPT end-to-end convergence benchmarks
    utils/              TypeStorage, BumpAllocator, ThreadPool benchmarks

extensions/             Optional extension module (Tycho_Extensions.cpp/.h)
examples/               Python example scripts (Brachistochrone, Zermelo, low-thrust, etc.)
  cpp_examples/         C++ example programs
    brachistochrone/    C++ Brachistochrone optimal control example
  MeshRefinement/       Mesh-refinement examples
  UpdatedInterface/     Examples for updated API
  Plots/                Shared plotting helpers
scripts/                Build, test, and packaging helper scripts
dockerfiles/            Dockerfiles for Ubuntu 18.04 / 20.04 CI images
misc/                   Code generation utilities (CodeGen.py, CodeGenExample.py)
pypiwheel/              PyPI wheel packaging (CMakeLists.txt, setup.py.in)
doc/                    Sphinx + Doxygen documentation source
notices/                Third-party license notices — DO NOT modify or delete
```

## Technical Details
**Vector Function Implementation**
- `doc/VectorFunction.md`
**Python Bindings Implementation (nanobind)**
- `doc/Bindings.md`

## Build System

This is a CMake + nanobind project. The output is a nanobind shared library
(`_tychopy.cpython-<ver>-<platform>.so`) plus the pure-Python `tychopy/` package, both
installed directly into the active Python environment's site-packages by the build step.

Each platform has a corresponding CMake preset in `CMakePresets.json` that defines
compiler paths, parallelism, and output directories. **Always use the preset for
your platform** — do not configure manually.

| Platform         | Configure preset        | Build parallelism |
| ---------------- | ----------------------- | ----------------- |
| macOS (Apple Si) | `macos-llvm-release`    | `-j2`             |
| Linux / WSL2     | `linux-clang-release`   | `-j6`             |
| Windows x64      | `x64-Clang-Release`    | `-j6`             |

The `dep/` submodules (eigen, autodiff, fmt, nanobind) must be initialised before the
first build. The cmake helpers in `cmake/git-submodule-*.cmake` do this automatically.

### macOS (Apple Silicon)

**System tools required (install once via Homebrew):**
```
brew install llvm ninja ccache jq
```
Current versions in use: LLVM 22+, Ninja 1.13+.

**Sparse solver:** Apple Accelerate (ships with macOS, detected automatically).

**Python environment — conda env named `tycho` (Python 3.13):**
```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy
```

**First-time build:**
```bash
mkdir build
cmake --preset macos-llvm-release
cd build && ninja -j2 all
```

**Note:** The macOS preset uses the stable Homebrew symlink `/opt/homebrew/opt/llvm`
for the compiler. The libomp paths in `CMakeLists.txt` also use the stable symlink
`/opt/homebrew/opt/libomp` — neither requires manual updating when Homebrew upgrades
LLVM. Presets do not hardcode Python paths; activate your conda/venv before configuring.

### Linux / WSL2 (Ubuntu)

**System tools required:**
```bash
sudo apt install clang llvm llvm-dev libomp-dev lld ninja-build ccache
```

**Sparse solver — Intel MKL (via oneAPI):**
```bash
# Add Intel APT repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-mkl-devel

# Set MKLROOT (add to ~/.bashrc)
source /opt/intel/oneapi/setvars.sh
# or: export MKLROOT=/opt/intel/oneapi/mkl/latest
```

**Python environment:** Use system Python 3.12+ or a conda/venv environment.
If not using conda, install dependencies globally or in a venv:
```bash
pip install numpy scipy matplotlib spiceypy
```

**First-time build:**
```bash
mkdir build
cmake --preset linux-clang-release
cd build && ninja -j6 all
```

**Note:** The Linux preset does not hardcode `Python_EXECUTABLE` — CMake will
auto-detect the active Python from `$PATH`. To target a specific interpreter,
pass `-DPython_EXECUTABLE=/path/to/python` during configure, or set it in the preset.

### Windows x64

Uses LLVM/Clang with clang-cl frontend. Preset: `x64-Clang-Release`.
See `CMakePresets.json` for compiler paths. Sparse solver: Intel MKL.

### Subsequent builds (all platforms)

After C++ source changes, rebuild from the `build` directory:
```bash
cd build && ninja -j<N> all    # N = 2 on macOS, 6 on Linux/Windows
```

### Key CMake variables

| Variable                   | Purpose                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `Python_EXECUTABLE`        | Path to Python interpreter to build against                                                  |
| `PYTHON_LOCAL_INSTALL_DIR` | Site-packages directory to install into; defaults to `python -m site --user-site` if not set |
| `TYCHO_FP_MODE`            | Floating-point mode: `STRICT`, `SAFER_FAST`, or `FAST` (default `SAFER_FAST`)               |
| `BUILD_SPHINX_DOCS`        | `ON` to also build documentation (requires sphinx, breathe, furo, exhale)                    |
| `BUILD_CPP_EXAMPLES`       | `ON` to build C++ example programs under `examples/cpp_examples/`                            |
| `BUILD_CPP_TESTS`          | `ON` to build C++ unit tests via Google Test (fetched via FetchContent)                       |
| `BUILD_CPP_BENCHMARKS`     | `ON` to build C++ benchmarks via Google Benchmark (fetched via FetchContent)                  |
| `BUILD_CPP_BENCHMARKS_LEGACY` | `ON` to build legacy hand-rolled benchmark executables                                    |

## Key Concepts and Domain Language

- **VectorFunction** — the core DSL for defining dynamics, constraints, and objectives.
  Everything in the problem definition layer is expressed as a VectorFunction.
  This is the central abstraction of the library; treat it with care.
- **Phase** — the core optimal control object. Multiple phases can be linked together
  for multi-phase trajectory problems.
- **PSIOPT** — the bundled interior-point nonlinear optimizer.
- **Collocation** — the transcription method used to convert continuous optimal control
  problems into finite-dimensional NLPs. The tool is method-agnostic at the API level
  but collocation is the primary implementation.
- **Astrodynamics** — the primary application domain, though the library is general.

## Binding Architecture

All Python binding code lives exclusively in `src/Bindings/`. Core C++ headers (`src/VectorFunctions/`,
`src/OptimalControl/`, `src/Integrators/`, `src/Astro/`) contain no nanobind code.

**`TychoBind<T>` trait pattern** — the central dispatch mechanism:
- Primary template declared in `src/Bindings/FunctionRegistry.h`
- `FunctionRegistry::Build_Register<T>(m, name)` calls `TychoBind<T>::Build(m, name)`
- Full/partial specializations defined in `*Bind.h` headers or binding `.cpp` files

**`Bind::` free-function helpers** (in `src/Bindings/*Bind.h`):
- `DenseBaseBuild<T>(obj)` — registers standard VectorFunction methods
- `IntegratorBuildConstructors<DODE>(obj)` — registers integrator constructors
- `ODEPhaseBuildImpl<DODE>(phase)` — registers phase constraints/objectives
- `ODESizeBuild<XV,UV,PV,Derived>(obj)` — registers ODE size accessors
- `BuildGenODEModule<BaseType,XV,UV,PV>(name, mod, reg)` — builds a GenericODE submodule

**Key rules for binding `.cpp` files:**
- Must include aggregate headers (`Tycho_Astro.h`, `Tycho_OptimalControl.h`) rather than
  raw core headers — function declarations like `KeplerUtilsBuild`, `BuildKeplerMod`
  live in aggregate headers under `#ifdef TYCHO_PYTHON_BINDINGS`
- `TYCHO_PYTHON_BINDINGS` is defined only for `_tychopy` and `tycho_extensions` targets
  (scoped via `pch_bindings` precompiled header), not globally

**`*Bind.h` files** (included from aggregate headers):
- `DenseFunctionBaseBind.h`, `VectorFunctionBind.h`, `CommonFunctionsBind.h`
- `InterpTableBind.h`, `GenericFunctionBind.h`
- `IntegratorBind.h`, `ODEPhaseBind.h`, `ODEBind.h`, `ODESizesBind.h`
- `PythonFunctions.h`, `PythonArgParsing.h` (moved from core headers)

## Third-Party Dependencies and License Obligations

The project is licensed under **Apache 2.0**. The `notices/` directory contains
copyright and license notices for all bundled third-party libraries. Key ones to
be aware of during development:

- **Eigen** (MPL-2.0) — any Eigen *source files* directly modified must remain MPL-2.0
- **Intel MKL** (Intel Simplified Software License) — redistribution has specific terms;
  flag any changes touching MKL integration for manual review
- **Nanobind** (BSD), **fmt** (MIT), **autodiff** (MIT), **boost-threads** (Boost),
  **rubber_types** (MIT), **kepler propagator** (MIT) — all permissive, just preserve notices

**Never delete or modify files in `notices/`.**
If a new dependency is added, its license notice must be added to `notices/` as well.

## Naming Migration (asset_asrl → tycho)

The naming migration is substantially complete:
- Repository: `tycho` ✅
- C++ namespace: `Tycho` ✅
- Python extension module: `_tychopy` ✅
- Python package: `tychopy` ✅
- PyPI package: not yet published

Do not do a bulk find-and-replace of any remaining `asset_asrl` or `ASSET` identifiers
without explicit instruction — residual uses may be load-bearing (CMake targets, internal
type names, etc.) and require coordinated changes.

## Code Style

All source code is formatted by automated tools. **Always run the formatter before
committing.** Do not introduce new external dependencies without discussion.
Prefer modifying existing files over creating new ones unless the feature is
clearly self-contained.

### C++ — clang-format (LLVM style)

Config: `.clang-format` at the repo root (LLVM base, 4-space indent, 100-column limit).
The CMake `clang-format` target applies it to `src/` and `extensions/`.

**Format all C++ source files (preferred — uses CMake target):**
```bash
cd build && ninja clang-format
```

Or manually (uses whichever `clang-format` is on `$PATH`):
```bash
find src extensions \( -name "*.cpp" -o -name "*.h" \) -print0 \
  | xargs -0 clang-format -style=file -i
```

Do **not** add or modify `.clang-format` files inside `src/` — the root config is
authoritative. The `dep/` vendored dependencies have their own configs and must
not be reformatted.

### Python — ruff (Black-compatible format + isort)

Config: `pyproject.toml` at the repo root.
Required packages: `ruff`, `isort` (`pip install ruff isort`).

**Format all Python files:**
```bash
ruff format .          # Black-compatible formatting
ruff check --select I --fix .  # isort import sorting
```

Line length: 88 (Black default). `dep/` and `build/` are excluded automatically.

## What to Preserve from the Original ASSET

- All content in `notices/` and `LICENSE.txt`
- Copyright headers in original source files

## Commit Conventions

Use descriptive commit messages. Prefix with type where clear:
- `feat:` new functionality
- `fix:` bug fixes
- `refactor:` restructuring without behavior change
- `docs:` documentation only
- `chore:` build system, tooling, dependency updates

## Testing

### C++ unit tests (Google Test)

The project has a C++ unit test suite under `tests/cpp/` using Google Test
(fetched automatically via FetchContent). 42 test files across 6 subdirectories
(astro, integrators, optimal_control, solvers, utils, vector_functions),
compiled into a single `tycho_tests` executable.

**Build and run:**
```bash
# Reconfigure with tests enabled (one-time, use your platform's preset)
cmake --preset <preset> -DBUILD_CPP_TESTS=ON

# Build test executable
cd build && ninja -j<N> tycho_tests

# Run all tests via CTest
ctest --output-on-failure
```

### C++ benchmarks (Google Benchmark)

46 micro-benchmarks live under `bench/cpp/` using Google Benchmark
(fetched automatically via FetchContent), compiled into a single `bench_all`
executable. A local tracking script (`bench/bench_track.sh`) records results
by commit hash and detects regressions. See `bench/<SYS>BENCH.md` for the full
benchmarking procedure on your platform (e.g., `MACBENCH.md` for macOS,
`WINBENCH.md` for Windows).

**Build and run:**
```bash
# Reconfigure with benchmarks enabled (one-time, use your platform's preset)
cmake --preset <preset> -DBUILD_CPP_BENCHMARKS=ON

# Build
cd build && ninja -j2 bench_all    # -j2 required for benchmark builds

# Run all benchmarks
./bench/cpp/bench_all

# Track performance across commits
bench/bench_track.sh baseline   # record baseline
bench/bench_track.sh record     # record after changes
bench/bench_track.sh compare    # compare HEAD vs baseline
```

### Python examples (integration tests)

The 38 Python example scripts under `examples/` serve as the **integration test
suite** and act as the acceptance gate for all changes merged into `main`.

```bash
python scripts/run_examples.py
```

The runner (`scripts/run_examples.py`) executes all 38 example scripts
non-interactively (using the `Agg` matplotlib backend), enforces per-example
timeouts, and exits with code 0 only if every example passes.

Options:
```
--timeout SECONDS   Override the default 300 s per-example limit.
--filter SUBSTRING  Run only examples whose path contains SUBSTRING.
```

### C++ examples

```bash
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

The C++ brachistochrone example must converge to an optimal solution (PSIOPT prints
"Optimal Solution Found") with an objective near 1.8013 s.

### Merge policy

**All C++ unit tests must pass, all 38 Python examples must pass, the C++
brachistochrone example must converge, and benchmarks must show no unexplained
regressions before any pull request can be merged into `main`.** This is the
project's definition of a green build. Reviewers must verify that `ctest`
reports no failures, `python scripts/run_examples.py` exits 0,
`brachistochrone_cpp` reports "Optimal Solution Found", and
`bench/bench_track.sh compare` reports no regressions before approving.

If a change intentionally breaks an example (e.g. an API change that requires
updating an example), the example must be fixed in the same PR. If a change
introduces a benchmark regression, it must be explicitly justified in the PR.

### Required dependencies for the test environment

The following packages must be present in the Python environment for all
examples to run (none will be skipped):

| Package                  | Install                                     |
| ------------------------ | ------------------------------------------- |
| numpy, scipy, matplotlib | `pip install numpy scipy matplotlib`        |
| seaborn                  | `pip install seaborn`                       |
| spiceypy                 | `pip install spiceypy`                      |
| basemap                  | `conda install -c conda-forge basemap`      |

## Development Workflow

Before implementing a change in Tycho, follow this procedure:

1. **Run benchmarks to establish a pre-update baseline if necessary.**
   This is only required if the most recent benchmark result does not correspond
   to the current code. Exercise caution before deciding to skip this step — when
   in doubt, record a fresh baseline. Refer to `bench/<SYS>BENCH.md` for the
   benchmarking procedure on your platform (e.g., `bench/MACBENCH.md` for macOS).

2. **Check out a new branch** from `main` for the work.

3. **Develop the code and iterate on the design as necessary.** Iteration may
   involve running benchmarks and tests during development, but this is not
   strictly required at every step.

4. **Before merging a PR into `main`, you MUST:**
   - Run benchmarks and compare against baseline (see `bench/<SYS>BENCH.md`).
   - Run all C++ unit tests (`ctest --output-on-failure`).
   - Run all 38 Python examples (`python scripts/run_examples.py`).
   - Verify the C++ brachistochrone example converges.
   - Ensure no performance regressions are introduced (or that any regressions
     are explicitly justified and acknowledged in the PR).

## Things to Flag for Human Review

- Any change to the PSIOPT optimizer internals
- Any change touching Intel MKL / Apple Accelerate integration
- Adding new third-party dependencies
- Changes to the public APIs
- Anything affecting PyPI packaging (pypiwheel/, setup.py)
