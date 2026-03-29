# Tycho — Claude Code Memory

## Project Overview

Tycho is an independently maintained fork of [ASSET (AlabamaASRL/asset_asrl)](https://github.com/AlabamaASRL/asset_asrl),
a high-performance C++/Python library for trajectory design and optimal control.
The core use cases are general optimal control problems (solved via direct collocation)
and space trajectory optimization. The built-in optimizer is called **PSIOPT**
(a high-performance interior-point solver).

The Python-facing module is `_tychopy` (nanobind extension) imported via the `tychopy` package.
The top-level C++ namespace is `tycho`.

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
    tycho.h             Master umbrella — includes all public modules
    vector_functions.h  VectorFunction subsystem umbrella
    typedefs.h          Eigen type aliases umbrella
    utils.h             Utilities umbrella
    integrators.h       Integrators umbrella
    optimal_control.h   Optimal control umbrella
    solvers.h           Solvers umbrella
    astro.h             Astrodynamics umbrella
    detail/             Template implementation bodies (included automatically)
      utils/            Threading, math helpers, type utilities, CRTP base
      typedefs/         Eigen type aliases
      vf/               VectorFunction implementation (core/, expressions/, operators/,
                          derivatives/, scaling/, type_erasure/, common/)
      integrators/      RK steppers and coefficients
      optimal_control/  Phase/ODE transcription (core/, phase/, transcription/,
                          interp/, builder/)
      solvers/          PSIOPT and NLP layer (linear/ for sparse solver interfaces)
      astro/            Astrodynamics models

src/                    C++ source code (private implementation)
  tycho_internal.h      Internal aggregate (forwards to include/tycho/tycho.h)
  pch.h / pch.cpp       Precompiled header
  bindings/             ALL Python binding code — nanobind .cpp files, *_bind.h
                          headers, function_registry.h (TychoBind<T> trait),
                          type_casters.h, tycho_module.cpp
  vf/                   tycho_vector_functions.h aggregate + function_domains.cpp
  optimal_control/      tycho_optimal_control.h aggregate + snake_case .cpp files
  solvers/              tycho_solvers.h aggregate + snake_case .cpp files
  astro/                tycho_astro.h aggregate + snake_case .cpp files
  integrators/          tycho_integrators.h aggregate (header-only)
  utils/                tycho_utils.h aggregate + private utility .cpp files
  typedefs/             tycho_typedefs.h aggregate (header-only)

tychopy/                Python package (pure-Python layer over _tychopy extension)
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

tests/                  C++ unit tests (Google Test, organised by subsystem)
bench/                  Benchmark suite and tracking
  MACBENCH.md           macOS benchmarking procedure
  WINBENCH.md           Windows benchmarking procedure
  bench_track.sh        Benchmark tracking script (record, compare, list)
  cpp/                  Google Benchmark source files

extensions/             Optional extension module (Tycho_Extensions.cpp/.h)
examples/               Example programs
  cpp_examples/         C++ example programs
  python_examples/      Python example scripts (38 examples)
notices/                Third-party license notices — DO NOT modify or delete
```

## Key Concepts

- **VectorFunction** — the core DSL for defining dynamics, constraints, and objectives.
  Everything in the problem definition layer is expressed as a VectorFunction.
  This is the central abstraction of the library; treat it with care.
- **Phase** — the core optimal control object. Multiple phases can be linked together
  for multi-phase trajectory problems.
- **PSIOPT** — the bundled interior-point nonlinear optimizer.
- **Collocation** — the transcription method used to convert continuous optimal control
  problems into finite-dimensional NLPs.
- **Astrodynamics** — the primary application domain, though the library is general.

## Technical Details

- **VectorFunction implementation:** `doc/VectorFunction.md`
- **Python bindings (nanobind):** `doc/Bindings.md`

## Build System

This is a CMake + nanobind project. The output is a nanobind shared library
(`_tychopy.cpython-<ver>-<platform>.so`) plus the pure-Python `tychopy/` package, both
installed directly into the active Python environment's site-packages by the build step.

**Always use the `tycho` conda environment** for all Python and build operations.
Activate it before configuring — CMake auto-detects Python from `$PATH`:

```bash
conda activate tycho
```

Each platform has a corresponding CMake preset in `CMakePresets.json`. **Always use the
preset for your platform** — do not configure manually.

| Platform         | Configure preset        | Build parallelism |
| ---------------- | ----------------------- | ----------------- |
| macOS (Apple Si) | `macos-llvm-release`    | `-j2`             |
| Linux / WSL2     | `linux-clang-release`   | `-j6`             |
| Windows x64      | `x64-Clang-Release`     | `-j6`             |

The `dep/` submodules (eigen, autodiff, fmt, nanobind) must be initialised before the
first build. The cmake helpers in `cmake/git-submodule-*.cmake` do this automatically.

**Python environment (all platforms) — conda env named `tycho`:**
```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy
```

### macOS (Apple Silicon)

**System tools:** `brew install llvm ninja ccache jq`

**Sparse solver:** Apple Accelerate (ships with macOS, detected automatically).

**First-time build:**
```bash
mkdir build && conda activate tycho
cmake --preset macos-llvm-release
cd build && ninja -j2 all
```

### Linux / WSL2 (Ubuntu)

**System tools:** `sudo apt install clang llvm llvm-dev libomp-dev lld ninja-build ccache`

**Sparse solver — Intel MKL (via oneAPI):**
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install intel-oneapi-mkl-devel
source /opt/intel/oneapi/setvars.sh   # add to ~/.bashrc
```

**First-time build:**
```bash
mkdir build && conda activate tycho
cmake --preset linux-clang-release
cd build && ninja -j6 all
```

### Windows x64

Uses LLVM/Clang with clang-cl frontend. Preset: `x64-Clang-Release`.
See `CMakePresets.json` for compiler paths. Sparse solver: Intel MKL.

### Subsequent builds (all platforms)

```bash
cd build && ninja -j<N> all    # N = 2 on macOS, 6 on Linux/Windows
```

### Key CMake variables

| Variable                      | Purpose                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `Python_EXECUTABLE`           | Path to Python interpreter to build against                                                  |
| `PYTHON_LOCAL_INSTALL_DIR`    | Site-packages directory to install into; defaults to `python -m site --user-site` if not set |
| `TYCHO_FP_MODE`               | Floating-point mode: `STRICT`, `SAFER_FAST`, or `FAST` (default `SAFER_FAST`)               |
| `BUILD_SPHINX_DOCS`           | `ON` to also build documentation (requires sphinx, breathe, furo, exhale)                    |
| `BUILD_CPP_EXAMPLES`          | `ON` to build C++ example programs under `examples/cpp_examples/`                            |
| `BUILD_CPP_TESTS`             | `ON` to build C++ unit tests via Google Test (fetched via FetchContent)                       |
| `BUILD_CPP_BENCHMARKS`        | `ON` to build C++ benchmarks via Google Benchmark (fetched via FetchContent)                  |
| `BUILD_CPP_BENCHMARKS_LEGACY` | `ON` to build legacy hand-rolled benchmark executables                                        |

## Code Style

All source code is formatted by automated tools. **Always run the formatter before
committing.** Do not introduce new external dependencies without discussion.
Prefer modifying existing files over creating new ones unless the feature is
clearly self-contained.

### Naming Conventions

**C++ namespaces** — a type's namespace is determined by its `include/tycho/detail/`
subdirectory:

| Subdirectory | Namespace |
| ------------ | --------- |
| `detail/vf/` | `tycho::vf` |
| `detail/optimal_control/` (non-builder) | `tycho::oc` |
| `detail/optimal_control/builder/` | `tycho` (root) |
| `detail/solvers/` | `tycho::solvers` |
| `detail/astro/` | `tycho::astro` |
| `detail/integrators/` | `tycho::integrators` |
| `detail/utils/` | `tycho::utils` |
| `detail/typedefs/`, enums, free utilities | `tycho` (root) |
| Python binding helpers | `tycho::bind` |

**C++ identifiers:**
- Types and classes: `PascalCase` (e.g., `DenseFunction`, `ODEPhase`)
- Member functions: `snake_case` (e.g., `set_io_rows()`, `add_equal_con()`)
  > **PSIOPT exception:** The bundled PSIOPT optimizer retains its legacy `set_PascalCase`
  > naming (e.g., `set_MaxIter`, `set_SocIter`, `set_KKTtol`). A dedicated PSIOPT naming
  > migration is planned as a future PR — do not rename these in routine refactor passes.
- Member variables: `snake_case_` with trailing underscore (e.g., `num_defects_`)
- Free functions: `snake_case`

**Python API:** All method and property names exposed via nanobind use `snake_case`,
matching the C++ member function names (e.g., `phase.add_boundary_value()`,
`ocp.return_traj()`). Grandfathered exceptions: `"adjointgradient"`, `"adjointhessian"`,
`"computeall"`.

Do **not** bulk find-and-replace remaining `asset_asrl` or `ASSET` identifiers without
explicit instruction — residual uses may be load-bearing (CMake targets, internal type
names) and require coordinated changes.

### C++ — clang-format (LLVM style)

Config: `.clang-format` at the repo root (LLVM base, 4-space indent, 100-column limit).
The CMake `clang-format` target applies it to `src/` and `extensions/`.

```bash
cd build && ninja clang-format
```

Do **not** add or modify `.clang-format` files inside `src/` — the root config is
authoritative. Do not reformat `dep/` vendored dependencies.

### Python — ruff (Black-compatible format + isort)

Config: `pyproject.toml` at the repo root.

```bash
conda run -n tycho ruff format .
conda run -n tycho ruff check --select I --fix .
```

Line length: 88 (Black default). `dep/` and `build/` are excluded automatically.

### Commit Conventions

Use descriptive commit messages with these prefixes:

| Prefix | Use for |
| ------ | ------- |
| `feat:` | New features or capabilities |
| `fix:` | Bug fixes |
| `refactor:` | Code restructuring without behavior change |
| `docs:` | Documentation-only changes |
| `chore:` | Build system, CI, dependency, or tooling changes |

## License and Notices

The project is licensed under **Apache 2.0**. The `notices/` directory contains
copyright and license notices for all bundled third-party libraries.

**Never delete or modify files in `notices/`.** If a new dependency is added, its
license notice must be added to `notices/` as well. Preserve copyright headers in
original source files.

Key obligations:
- **Eigen** (MPL-2.0) — any Eigen *source files* directly modified must remain MPL-2.0
- **Intel MKL** (Intel Simplified Software License) — redistribution has specific terms;
  flag any changes touching MKL integration for manual review
- **Nanobind** (BSD), **fmt** (MIT), **autodiff** (MIT) — all permissive, just preserve notices
- **boost-threads** (Boost Software License 1.0), **rubber_types** (MIT), **kepler propagator** (MIT),
  **lambert** (MIT), **ctpl** (Apache 2.0) — all permissive; see `notices/` for full list

## Things to Flag for Human Review

Changes in the following areas require explicit human review before merging:

- **PSIOPT optimizer internals** — algorithmic changes can silently degrade convergence
- **Intel MKL / Apple Accelerate integration** — redistribution terms and initialization are sensitive
- **Public API changes** — any Python-facing rename or removal breaks downstream user code
- **New third-party dependencies** — must add license notice to `notices/` and get approval
- **PyPI / packaging** — changes to `setup.py`, `pyproject.toml`, or wheel metadata

## Testing

### C++ unit tests (Google Test)

```bash
cmake --preset <preset> -DBUILD_CPP_TESTS=ON   # one-time reconfigure
cd build && ninja -j<N> tycho_tests
ctest --output-on-failure
```

### Python examples (integration tests)

The 38 Python example scripts under `examples/python_examples/` serve as the **integration
test suite** and acceptance gate for all changes merged into `main`.

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Options: `--timeout SECONDS`, `--filter SUBSTRING`.

Required packages for all 38 examples to run (none skipped):
```bash
conda run -n tycho pip install numpy scipy matplotlib seaborn spiceypy
conda install -n tycho -c conda-forge basemap
```

### C++ brachistochrone example

```bash
./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
# Builder-API variant (equivalent convergence check):
# ./build/examples/cpp_examples/builder/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found", objective ≈ 1.8013 s.

### Pre-Merge Verification Sequence

Run all four steps in order before opening or merging any PR into `main`:

1. **C++ unit tests** — `ctest --output-on-failure` — all must pass
2. **Python examples** — `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"` — all 38 must exit 0
3. **C++ brachistochrone** — see path above — must print "Optimal Solution Found", obj ≈ 1.8013 s
4. **Benchmarks** — `bench/bench_track.sh compare` — justify any regressions in the PR description

### Merge policy

**All C++ unit tests must pass, all 38 Python examples must pass, the C++
brachistochrone example must converge, and benchmarks must show no unexplained
regressions before any pull request can be merged into `main`.** Fix broken examples
or justify benchmark regressions in the same PR.

## Benchmarking

```bash
cmake --preset <preset> -DBUILD_CPP_BENCHMARKS=ON   # one-time reconfigure
cd build && ninja -j2 bench_all                      # -j2 required for benchmark builds
./bench/cpp/bench_all

bench/bench_track.sh baseline   # record baseline
bench/bench_track.sh record     # record after changes
bench/bench_track.sh compare    # compare HEAD vs baseline
```

See `bench/MACBENCH.md` (macOS) or `bench/WINBENCH.md` (Windows) for the full procedure.
