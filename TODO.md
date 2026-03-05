# Tycho — Modernization TODO

## Status

| PR | Description | Status |
|----|-------------|--------|
| PR 1 | Rename ASSET/asset_asrl → Tycho/tycho | **DONE** (merged) |
| PR 2 | Decouple C++ from Python bindings | **DONE** (merged) |
| PR 3 | Migrate pybind11 → nanobind | **DONE** (merged) |
| PR 4 | Separate extensions C++ from binding code | **SKIPPED** |
| PR 5 | Add per-file provenance headers to all ASSET-derived source files | **DONE** |

---

## PR 3 — Migrate pybind11 → nanobind

### Goal

Replace pybind11 with nanobind in the isolated `src/Bindings/` layer. Core C++ is untouched.

### Scope

- ~84 files across `src/` contain pybind11 usage
- ~46 files have `Build(py::module &m)` binding methods (all in `src/Bindings/` after PR 2)

### Key Migration Challenges

#### 1. `py::implicitly_convertible` — nanobind has no equivalent
- **Location**: `src/VectorFunctions/ASSET_VectorFunctions.cpp`
- **Pattern**: `py::implicitly_convertible<GenS, Gen>()`
- **Fix**: Add explicit conversion constructors or `__init__` overloads in the binding

#### 2. Python callback capture in `PythonFunctions.h`
- **Location**: `src/VectorFunctions/CommonFunctions/PythonFunctions.h`
- **Pattern**: `PyVectorFunction<IRR,ORR>` captures `py::args pyargs` and `py::detail::args_proxy`
- **Fix**: Nanobind callback model differs — requires rewrite of this class (~100 lines)

#### 3. `py::call_guard<py::gil_scoped_release>()` (~40 locations)
- **Locations**: `Integrator.h`, `OptimizationProblemBase.cpp`, `Jet.h`
- **Fix**: Mechanical — `py::call_guard<py::gil_scoped_release>()` → `nb::call_guard<nb::gil_scoped_release>()`

#### 4. `py::overload_cast` (~474 occurrences)
- **Fix**: Mostly mechanical — `py::overload_cast<Args>` → `nb::overload_cast<Args>`
  (verify resolution order in `ODEPhaseBaseBind.cpp` which has 74 overloads)

#### 5. `py::is_operator()` (~59 occurrences)
- **Fix**: Mechanical — `py::is_operator()` → `nb::is_operator()`

#### 6. `std::shared_ptr` holders
- **Pattern**: `py::class_<T, std::shared_ptr<T>, Base>(m, "T")`
- **Fix**: Nanobind uses `nb::class_<T>(m, "T")` with `nb::intrusive_ptr` or `std::shared_ptr`
  via `NANOBIND_OVERRIDE_HASH`. Verify per class.

#### 7. Dep submodule swap
- Remove `dep/pybind11` reference; add `dep/nanobind` as git submodule
- Update `CMakeLists.txt`: `find_package(pybind11)` → `find_package(nanobind)`
- `src/Bindings/CMakeLists.txt`: `pybind11_add_module` → `nanobind_add_module`

#### 8. Header replacements (mechanical)

```
#include <pybind11/pybind11.h>   → #include <nanobind/nanobind.h>
#include <pybind11/eigen.h>      → #include <nanobind/eigen/dense.h>
#include <pybind11/stl.h>        → #include <nanobind/stl/vector.h> (etc.)
#include <pybind11/functional.h> → #include <nanobind/stl/function.h>
#include <pybind11/operators.h>  → (built into nanobind core)
namespace py = pybind11          → namespace nb = nanobind
```

### Recommended Migration Order

1. Swap dep submodule; update CMake
2. Replace headers + namespace alias mechanically
3. Fix `PYBIND11_MODULE` → `NB_MODULE` in `src/Bindings/TychoModule.cpp`
4. Fix `py::overload_cast`, `py::arg`, `py::init`, `py::is_operator` (mechanical)
5. Fix `py::call_guard` → `nb::call_guard` (mechanical)
6. Fix `std::shared_ptr` holder classes
7. Rewrite `PythonFunctions.h` callback mechanism
8. Fix `py::implicitly_convertible` → explicit conversion

### Verification

- `ninja -j6 all` — clean build with nanobind
- `python -c "import tycho"` — module loads
- `python run_examples.py` — all 38 examples pass

---

## PR 4 — Separate Extensions C++ from Binding Code

### Goal

Mirror the architecture established in PR 2 for the core library: split `extensions/` into
a pure C++ layer (no Python dependency) and a separate binding layer (compiled with
`TYCHO_PYTHON_BINDINGS`). This allows users to write extension ODE models usable from both
C++ programs and Python without dragging in the Python C API.

### Problem

`extensions/Tycho_Extensions.cpp` currently mixes pure C++ ODE model definitions with
pybind11 `Build()` calls in the same file/target. The `tycho_extensions` static lib is
compiled with `TYCHO_PYTHON_BINDINGS` and `pch_bindings`, making it Python-dependent.
A standalone C++ binary that wants to use an extension ODE model would pull in Python —
the same anti-pattern fixed in PR 2 for the core.

### Changes

#### New directory: `extensions/Bindings/`
- Move all `Build(py::module &m)` calls out of `Tycho_Extensions.cpp` into
  `extensions/Bindings/TychoExtensionsBind.cpp`

#### `tycho_extensions` target (pure C++)
- Compiled WITHOUT `TYCHO_PYTHON_BINDINGS`
- Uses `REUSE_FROM pch` (not `pch_bindings`)
- No pybind11 dependency

#### New `tycho_extensions_bindings` target
- Compiled WITH `TYCHO_PYTHON_BINDINGS`; uses `REUSE_FROM pch_bindings`
- Contains `TychoExtensionsBind.cpp`
- Links `tycho_extensions` (pure C++) and `tycho_core`
- `_tycho` links `tycho_extensions_bindings` instead of `tycho_extensions` directly

#### `extensions/CMakeLists.txt` Refactor

```cmake
# Pure C++ extension library
add_library(tycho_extensions STATIC Tycho_Extensions.cpp)
target_precompile_headers(tycho_extensions REUSE_FROM pch)
target_link_libraries(tycho_extensions PUBLIC tycho_core)

# Python binding layer for extensions
add_library(tycho_extensions_bindings STATIC Bindings/TychoExtensionsBind.cpp)
target_compile_definitions(tycho_extensions_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
target_precompile_headers(tycho_extensions_bindings REUSE_FROM pch_bindings)
target_link_libraries(tycho_extensions_bindings PUBLIC tycho_extensions)
```

### Verification

- `cmake -DENABLE_PYTHON_BINDINGS=OFF ... -DBUILD_CPP_EXAMPLES=ON` + `ninja tycho_extensions` — builds with no Python
- `cmake -DENABLE_PYTHON_BINDINGS=ON ...` + `ninja -j6 all` — full build works
- `python run_examples.py` — all 38 examples pass

---

## PR 5 — Add Per-File Provenance Headers to ASSET-Derived Source Files

### Goal

Every source file that originated from the ASSET repository should carry a
per-file header that (a) attributes the original authors, (b) states what
license the original code was distributed under, and (c) briefly describes
what changes were made in the Tycho fork. This makes the boundary between
inherited and new code explicit at the file level, consistent with the
promise made in `notices/asset-apache2.txt`.

Files that are entirely new (no ASSET origin) require no header. However, we
need to be careful, because many new files contain lots of code stripped 
from existing asset_asrl files (i.e., all binding code, which still require
a header).

Note that some files include technical descriptions of what occurs 
in the relevant code. We want to preserve this information.

### Header Format

For C++ files (`.h` / `.cpp`):

```cpp
// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: <insert developer> **Obtain from existing comments, git history, or assume James B. Pezent in this order**
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - <brief description of changes, e.g. "renamed namespace asset → Tycho">
//   - <additional bullet per logical change group>
//   - <add as many bullet points as needed based on commit history>
// =============================================================================
```

For Python files (`.py`):

```python
# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0..
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: <insert developer> **Obtain from existing comments, git history, or assume James B. Pezent in this order**
#
# Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
#   Apache 2.0 — see LICENSE.txt):
#   - <brief description of changes>
# =============================================================================
```

### Scope

All files under `src/`, `tycho/`, `examples/`, and `extensions/` that have a
direct counterpart in the ASSET repository. Use `git log --follow` and manual
inspection against the upstream ASSET commit history to identify which files
originated from ASSET vs. were added fresh in the Tycho fork.

### Approach

1. Enumerate ASSET-derived files by diffing the Tycho file tree against the
   last known upstream ASSET commit.
2. For each file, draft a short change summary (namespace rename, binding
   removal, nanobind migration, etc.).
3. Prepend the header; do not alter any other content.
4. Open as a single PR — review confirms correctness of the provenance
   statements before merging.

### Verification

- All 38 Python examples still pass after header insertion.
- `ninja -j6 all` still succeeds (headers are comments; no code change).
