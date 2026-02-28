# Tycho — Claude Code Memory

## Project Overview

Tycho is an independently maintained fork of [ASSET (AlabamaASRL/asset_asrl)](https://github.com/AlabamaASRL/asset_asrl),
a high-performance C++/Python library for trajectory design and optimal control.
The core use cases are general optimal control problems (solved via direct collocation)
and space trajectory optimization. The built-in optimizer is called **PSIOPT**
(a high-performance interior-point solver).

**Important:** The source code currently still builds the `asset_asrl` Python module.
Renaming to `tycho` is an ongoing migration. Do not assume the module name matches
the repository name yet — always check before referencing it.

## Repository Structure

Top-level files of note: `CMakeLists.txt` (root build), `CMakePresets.json`, `CMakeSettings.json` (MSVC),
`setup.py`, `requirements.txt`, `config_and_build.sh`, `Dockerfile`, `Dockerfile-dev`.

```
src/                    C++ source code (core library)
  ASSET.h               Top-level C++ include (aggregates all modules)
  pch.h / pch.cpp       Precompiled header
  VectorFunctions/      Core VectorFunction DSL — autodiff, dense/sparse functions,
                          operator overloads, type erasure, Python bindings build
  OptimalControl/       Phase/ODE transcription — LGL & trapezoidal collocation,
                          FD derivatives, mesh refinement, link functions,
                          OptimalControlProblem, Python bindings build
  Solvers/              PSIOPT interior-point optimizer, NLP layer,
                          Pardiso / MUMPS / Accelerate / MKL interfaces
  Astro/                Astrodynamics models — Kepler, CR3BP, J2, MEE dynamics,
                          Lambert solvers, thruster models
  Integrators/          Runge-Kutta steppers and coefficients
  Utils/                Thread pool, timers, memory management, type erasure
                          helpers, color terminal output, STD extensions
  TypeDefs/             Eigen type aliases (EigenTypes.h, ASSET_TypeDefs.h)
  PyDocString/          C++-side Python docstring literals

asset_asrl/             Python package (pybind11 bindings + pure-Python layer)
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
  pybind11/             pybind11 (BSD)

cmake/                  CMake helper modules
  Find*.cmake           Finders for MKL, Accelerate sparse, Python env, Sphinx
  clang-format.cmake    clang-format integration
  cppcheck.cmake        cppcheck integration
  git-submodule-*.cmake Submodule init/update helpers

extensions/             Optional extension module (ASSET_Extensions.cpp/.h)
examples/               Python example scripts (Brachistochrone, Zermelo, low-thrust, etc.)
  MeshRefinement/       Mesh-refinement examples
  UpdatedInterface/     Examples for updated API
  Plots/                Shared plotting helpers
scripts/                Build helper scripts (Windows + OneAPI installers)
dockerfiles/            Dockerfiles for Ubuntu 18.04 / 20.04 CI images
misc/                   Code generation utilities (CodeGen.py, CodeGenExample.py)
pypiwheel/              PyPI wheel packaging (CMakeLists.txt, setup.py.in)
doc/                    Sphinx + Doxygen documentation source
notices/                Third-party license notices — DO NOT modify or delete
```

## Build System

This is a CMake + pybind11 project. The output is a pybind11 shared library
(`asset.cpython-<ver>-darwin.so`) plus the pure-Python `asset_asrl/` package, both
installed directly into the active Python environment's site-packages by the build step.

### macOS (Apple Silicon) — canonical development setup

**System tools required (install once via Homebrew):**
```
brew install llvm ninja
```
Current versions in use: LLVM 22.1.0, Ninja 1.13+.

**Python environment — conda env named `tycho` (Python 3.13):**
```bash
conda create -n tycho python=3.13
conda activate tycho
pip install numpy scipy matplotlib spiceypy pybind11
```

**First-time build:**
```bash
mkdir build
bash config_and_build.sh
```

**Subsequent builds** (after C++ source changes):
```bash
cd build && ninja -j6 all
```

**Running examples:**
```bash
conda activate tycho
python examples/Brachistochrone.py
```

### Key CMake variables

| Variable | Purpose |
|---|---|
| `Python_EXECUTABLE` | Path to Python interpreter to build against |
| `PYTHON_LOCAL_INSTALL_DIR` | Site-packages directory to install into; defaults to `python -m site --user-site` if not set |
| `STRICT_IEEE_COMPLIANCE` | `ON` to disable fast-math and enforce proper NaN handling (default `OFF`) |
| `BUILD_SPHINX_DOCS` | `ON` to also build documentation (requires sphinx, breathe, furo, exhale) |

`config_and_build.sh` dynamically resolves `Python_EXECUTABLE` and
`PYTHON_LOCAL_INSTALL_DIR` from the `tycho` conda environment, so it always targets
the correct interpreter. When updating LLVM, change `LLVM_VERSION` in that script and
the hardcoded libomp path in `CMakeLists.txt` (search for `/opt/homebrew/Cellar/libomp/`).

The `dep/` submodules (eigen, autodiff, fmt, pybind11) must be initialised before the
first build. The cmake helpers in `cmake/git-submodule-*.cmake` do this automatically.

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

## Third-Party Dependencies and License Obligations

The project is licensed under **Apache 2.0**. The `notices/` directory contains
copyright and license notices for all bundled third-party libraries. Key ones to
be aware of during development:

- **Eigen** (MPL-2.0) — any Eigen *source files* directly modified must remain MPL-2.0
- **Intel MKL** (Intel Simplified Software License) — redistribution has specific terms;
  flag any changes touching MKL integration for manual review
- **Pybind11** (BSD), **fmt** (MIT), **autodiff** (MIT), **boost-threads** (Boost),
  **ctpl** (Apache 2.0), **rubber_types** (MIT), **kepler propagator** (MIT) — all
  permissive, just preserve notices

**Never delete or modify files in `notices/`.**
If a new dependency is added, its license notice must be added to `notices/` as well.

## Naming Migration (asset_asrl → tycho)

The project is in active transition from the ASSET/asset_asrl identity to Tycho.
Current state:
- Repository: `tycho` ✅
- Python module: still `asset_asrl` ⚠️ (not yet renamed)
- PyPI package: not yet published

When making changes, prefer using `tycho` for any *new* identifiers, namespaces,
or documentation. Do not do a bulk find-and-replace of `asset_asrl` without explicit
instruction — this requires coordinated changes across CMake, pybind11 bindings,
Python imports, and documentation.

## Code Style

All source code is formatted by automated tools. **Always run the formatter before
committing.** Do not introduce new external dependencies without discussion.
Prefer modifying existing files over creating new ones unless the feature is
clearly self-contained.

### C++ — clang-format (LLVM style)

Config: `.clang-format` at the repo root (LLVM base, 4-space indent, 100-column limit).
The CMake `clang-format` target applies it to `src/` and `extensions/`.

**Format all C++ source files:**
```bash
find src extensions \( -name "*.cpp" -o -name "*.h" \) -print0 \
  | xargs -0 /opt/homebrew/opt/llvm/bin/clang-format -style=file -i
```

Or via CMake after configuring:
```bash
cd build && ninja clang-format
```

Do **not** add or modify `.clang-format` files inside `src/` — the root config is
authoritative. The `dep/` vendored dependencies have their own configs and must
not be reformatted.

### Python — ruff (Black-compatible format + isort)

Config: `pyproject.toml` at the repo root.
Required packages (in the `tycho` conda env): `ruff`, `isort`.

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

Tycho does not yet have a formal unit test suite. The example scripts under
`examples/` serve as the **integration test suite** and act as the acceptance
gate for all changes merged into `main`.

### Running the tests

```bash
conda activate tycho
python run_examples.py
```

The runner (`run_examples.py` at the repo root) executes all 38 example scripts
non-interactively (using the `Agg` matplotlib backend), enforces per-example
timeouts, and exits with code 0 only if every example passes.

Options:
```
--timeout SECONDS   Override the default 300 s per-example limit.
--filter SUBSTRING  Run only examples whose path contains SUBSTRING.
```

### Merge policy

**All 38 examples must pass before any pull request can be merged into `main`.**
This is the project's definition of a green build.  Reviewers must verify that
`python run_examples.py` exits 0 with no failures before approving a PR.

If a change intentionally breaks an example (e.g. an API change that requires
updating an example), the example must be fixed in the same PR.

### Required dependencies for the test environment

The following packages must be present in the `tycho` conda environment for all
examples to run (none will be skipped):

| Package | Install |
|---|---|
| numpy, scipy, matplotlib | `pip install numpy scipy matplotlib` |
| seaborn | `pip install seaborn` |
| spiceypy | `pip install spiceypy` |
| basemap | `conda install -c conda-forge basemap` |

## Things to Flag for Human Review

- Any change to the PSIOPT optimizer internals
- Any change touching Intel MKL / Apple Accelerate integration
- Any bulk renaming of `asset_asrl` identifiers
- Adding new third-party dependencies
- Changes to the public APIs
- Anything affecting PyPI packaging (pypiwheel/, setup.py)