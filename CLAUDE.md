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

This is a CMake + pybind11 project. `CMakePresets.json` defines the standard configure/build
presets; `config_and_build.sh` is a convenience wrapper for Linux/macOS. The `dep/`
submodules must be initialised before building (`git submodule update --init --recursive`
or the cmake helpers in `cmake/git-submodule-*.cmake` will do it automatically).

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

- C++ does not currently follow a standard convention. We eventually want to format code 
  via clang-format and the LLVM format
- Python does not currently follow a standard convention. We eventually want to format 
  code via Ruff and the black format
- Do not introduce new external dependencies without discussion
- Prefer modifying existing files over creating new ones unless the feature is
  clearly self-contained

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

## Things to Flag for Human Review

- Any change to the PSIOPT optimizer internals
- Any change touching Intel MKL / Apple Accelerate integration
- Any bulk renaming of `asset_asrl` identifiers
- Adding new third-party dependencies
- Changes to the public APIs
- Anything affecting PyPI packaging (pypiwheel/, setup.py)