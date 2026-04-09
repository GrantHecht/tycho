# Phase 1: Python Package Rename (`tycho` -> `tychopy`) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Rename the Python package from `tycho` to `tychopy` and the nanobind module from `_tycho` to `_tychopy`, establishing a clean naming boundary where "tycho" refers to the C++ library alone.

**Architecture:** This is a mechanical rename touching ~80+ Python files, CMake build files, and the nanobind module declaration. The standard import convention becomes `import tychopy as typy`. No C++ namespace or library target changes (those happen in Phase 2).

**Tech Stack:** Python, CMake, nanobind, scikit-build-core

---

### Task 1: Create feature branch

**Files:**
- None (git operation only)

**Step 1: Create and check out branch**

```bash
git checkout -b feat/tychopy-rename main
```

**Step 2: Verify clean state**

```bash
git status
```

Expected: clean working tree on `feat/tychopy-rename`

---

### Task 2: Rename nanobind module `_tycho` -> `_tychopy`

**Files:**
- Modify: `src/Bindings/TychoModule.cpp:149` — `NB_MODULE(_tycho, m)` -> `NB_MODULE(_tychopy, m)`
- Modify: `src/Bindings/CMakeLists.txt:27` — `nanobind_add_module(_tycho` -> `nanobind_add_module(_tychopy`
- Modify: `src/Bindings/CMakeLists.txt:72-105` — all `_tycho` target references -> `_tychopy`
- Modify: `CMakeLists.txt:87` — option description referencing `_tycho`
- Modify: `extensions/CMakeLists.txt` — if it references `_tycho` target

**Step 1: Update `NB_MODULE` in TychoModule.cpp**

In `src/Bindings/TychoModule.cpp`, change line 149:
```cpp
// Before:
NB_MODULE(_tycho, m) {
// After:
NB_MODULE(_tychopy, m) {
```

**Step 2: Update `src/Bindings/CMakeLists.txt`**

Replace all `_tycho` references with `_tychopy`. Key lines:
- Line 27: `nanobind_add_module(_tychopy`
- Line 72: `target_compile_options(_tychopy PRIVATE ${BINDING_COMPILE_FLAGS})`
- Line 73: `target_compile_definitions(_tychopy PUBLIC TYCHO_PYTHON_BINDINGS)`
- Line 74-76: `set_target_properties(_tychopy ...)`
- Lines 77-84: `target_link_libraries(_tychopy ...)`
- Line 87: `target_link_options(_tychopy ...)`
- Line 89: `target_link_libraries(_tychopy ...)`
- Line 93: `install(TARGETS _tychopy LIBRARY DESTINATION .)`
- Lines 98-104: Post-build copy commands — update both target name and directory references:
  ```cmake
  add_custom_command(TARGET _tychopy POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tychopy> ${PYTHON_LOCAL_INSTALL_DIR}/
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/tychopy ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
      COMMENT "Installed _tychopy and tychopy package to ${PYTHON_LOCAL_INSTALL_DIR}"
  )
  ```

**Step 3: Update `CMakeLists.txt` root**

Line 87: Update option description:
```cmake
option(ENABLE_PYTHON_BINDINGS "Build Python bindings (_tychopy nanobind module)" ON)
```

Search for any other `_tycho` references in the root `CMakeLists.txt` and update them.

**Step 4: Check `extensions/CMakeLists.txt`**

Search for `_tycho` references. The extensions target links to `tycho_extensions_bindings`, not `_tycho` directly, so this likely needs no change. Verify.

**Step 5: Commit**

```bash
git add src/Bindings/TychoModule.cpp src/Bindings/CMakeLists.txt CMakeLists.txt extensions/CMakeLists.txt
git commit -m "feat: rename nanobind module _tycho -> _tychopy"
```

---

### Task 3: Rename Python package directory `tycho/` -> `tychopy/`

**Files:**
- Rename: `tycho/` -> `tychopy/` (entire directory tree)

**Step 1: Rename the directory**

```bash
git mv tycho tychopy
```

**Step 2: Verify the rename**

```bash
ls tychopy/
ls tychopy/Astro/
ls tychopy/OptimalControl/
```

Expected: all subpackages present in new location.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: rename Python package directory tycho/ -> tychopy/"
```

---

### Task 4: Update imports in `tychopy/` package files

**Files:**
- Modify: `tychopy/__init__.py`
- Modify: `tychopy/Astro/__init__.py`
- Modify: `tychopy/VectorFunctions/__init__.py`
- Modify: `tychopy/OptimalControl/__init__.py`
- Modify: `tychopy/Solvers/__init__.py`
- Modify: `tychopy/Utils/__init__.py`
- Modify: `tychopy/Extensions/__init__.py`
- Modify: `tychopy/test/__init__.py`
- Modify: all `Extensions/__init__.py` files in subpackages
- Modify: all `.py` module files that do `import tycho` or `from tycho` or `import _tycho`

**Step 1: Update `tychopy/__init__.py`**

```python
# Before:
import _tycho as _tycho

import tycho.Astro
import tycho.OptimalControl
import tycho.Solvers
import tycho.Utils
import tycho.VectorFunctions

# After:
import _tychopy as _tychopy

import tychopy.Astro
import tychopy.OptimalControl
import tychopy.Solvers
import tychopy.Utils
import tychopy.VectorFunctions
```

Also update the `__main__` block at the bottom:
```python
if __name__ == "__main__":
    _tychopy.SoftwareInfo()
    mlist = inspect.getmembers(_tychopy)
    for m in mlist:
        print(m[0], "= _tychopy." + str(m[0]))
```

And update the `SoftwareInfo` binding:
```python
SoftwareInfo = _tychopy.SoftwareInfo
```

**Step 2: Bulk-update all internal imports**

In all files under `tychopy/`, replace:
- `import _tycho` -> `import _tychopy`
- `_tycho.` -> `_tychopy.`
- `from tycho.` -> `from tychopy.`
- `import tycho.` -> `import tychopy.`
- `from tycho ` -> `from tychopy `

Use the following commands to find all files needing changes:
```bash
grep -rl "_tycho" tychopy/
grep -rl "import tycho" tychopy/
grep -rl "from tycho" tychopy/
```

Key files with `_tycho` references (all `__init__.py` files that import from the nanobind module):
- `tychopy/__init__.py`
- `tychopy/Astro/__init__.py`
- `tychopy/VectorFunctions/__init__.py`
- `tychopy/OptimalControl/__init__.py`
- `tychopy/Solvers/__init__.py`
- `tychopy/Utils/__init__.py`
- `tychopy/Extensions/__init__.py`

Key module files with `tycho` package imports:
- `tychopy/OptimalControl/ODEBaseClass.py`
- `tychopy/OptimalControl/MeshErrorPlots.py`
- `tychopy/Astro/AstroConstraints.py`
- `tychopy/Astro/AstroFrames.py`
- `tychopy/Astro/AstroModels.py`
- `tychopy/Astro/SpiceRead.py`
- `tychopy/VectorFunctions/Extensions/DerivChecker.py`
- All `Extensions/*.py` files in subpackages

**Step 3: Verify no remaining references**

```bash
grep -r "import _tycho" tychopy/
grep -r "import tycho\b" tychopy/
grep -r "from tycho\b" tychopy/
```

Expected: no matches (all should reference `_tychopy` and `tychopy`).

**Step 4: Commit**

```bash
git add tychopy/
git commit -m "feat: update all internal imports _tycho -> _tychopy, tycho -> tychopy"
```

---

### Task 5: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update package name and wheel config**

```toml
# Before:
[project]
name = "tycho"

# After:
[project]
name = "tychopy"
```

```toml
# Before:
wheel.packages = ["tycho"]

# After:
wheel.packages = ["tychopy"]
```

```toml
# Before:
Homepage = "https://github.com/GrantHecht/tycho"

# After (keep as-is — repo name stays tycho):
Homepage = "https://github.com/GrantHecht/tycho"
```

Update the comment on line 29:
```toml
# Before:
# Pure-Python package collected from source tree; _tycho extension installed via CMake install()
# After:
# Pure-Python package collected from source tree; _tychopy extension installed via CMake install()
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: update pyproject.toml package name tycho -> tychopy"
```

---

### Task 6: Update all 38 example scripts

**Files:**
- Modify: all `.py` files under `examples/`

**Step 1: Update import statements**

In every example file, replace:
- `import tycho as ast` -> `import tychopy as typy`
- `import tycho.OptimalControl as oc` -> `import tychopy.OptimalControl as oc`
- `import tycho.VectorFunctions as vf` -> `import tychopy.VectorFunctions as vf`
- `from tycho.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot` -> `from tychopy.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot`
- `from tycho.VectorFunctions import Arguments as Args` -> `from tychopy.VectorFunctions import Arguments as Args`
- `import tycho` -> `import tychopy as typy`
- `from tycho` -> `from tychopy`

Where examples previously used `ast` as the alias (e.g., `ast.VectorFunctions`, `ast.OptimalControl`, `ast.Astro`, `ast.SoftwareInfo()`), update to `typy`:
- `ast.VectorFunctions` -> `typy.VectorFunctions`
- `ast.OptimalControl` -> `typy.OptimalControl`
- `ast.Astro` -> `typy.Astro`
- `ast.SoftwareInfo()` -> `typy.SoftwareInfo()`

The full list of example files:

**Root examples/ (27 files):**
```
examples/AnalyticExample.py
examples/BettsLowThrust.py
examples/BikeObstacle.py
examples/Brachistochrone.py
examples/BrysonDenham.py
examples/CartPole.py
examples/Delta3Launch.py
examples/DionysusLowThrust.py
examples/FreeFlyingRobotExample.py
examples/GoddardRocket.py
examples/HangingChain.py
examples/Heteroclinic.py
examples/HyperSens.py
examples/MinimumTimeToClimb.py
examples/MinimumTimeToClimbTables.py
examples/MountainCar.py
examples/MultiPhaseCannon.py
examples/MultiPhaseZermelo.py
examples/MultiSpacecraftOptimization.py
examples/OptimalDocking.py
examples/OrbitContinuation.py
examples/ParallelParking.py
examples/Reentry.py
examples/SimpleLowThrust.py
examples/TopputtoLowThrust.py
examples/VanDerPol.py
examples/Zermelo.py
```

**examples/MeshRefinement/ (4 files):**
```
examples/MeshRefinement/CartPole.py
examples/MeshRefinement/Delta3Launch.py
examples/MeshRefinement/HyperSensLong.py
examples/MeshRefinement/Reentry.py
```

**examples/UpdatedInterface/ (7 files):**
```
examples/UpdatedInterface/BettsLowThrust.py
examples/UpdatedInterface/Delta3Launch.py
examples/UpdatedInterface/GoddardRocket.py
examples/UpdatedInterface/MinimumTimeToClimb.py
examples/UpdatedInterface/MinimumTimeToClimbTables.py
examples/UpdatedInterface/MultiPhaseCannon.py
examples/UpdatedInterface/Reentry.py
```

**Step 2: Verify no remaining tycho imports in examples**

```bash
grep -r "import tycho" examples/
grep -r "from tycho" examples/
grep -r "as ast$" examples/
```

Expected: no matches.

**Step 3: Commit**

```bash
git add examples/
git commit -m "feat: update all example imports tycho -> tychopy (import tychopy as typy)"
```

---

### Task 7: Update test files

**Files:**
- Modify: all `.py` files under `tychopy/test/`

**Step 1: Update imports in all test files**

Same replacement pattern as Task 6:
- `import tycho` -> `import tychopy as typy`
- `from tycho` -> `from tychopy`
- `_tycho` -> `_tychopy`
- `ast.` alias usage -> `typy.`

Test files to update:
```
tychopy/test/test_AdaptiveMesh/test_CartPole.py
tychopy/test/test_AdaptiveMesh/test_Delta3Launch.py
tychopy/test/test_AdaptiveMesh/test_Reentry.py
tychopy/test/test_AutoScaling/test_ObjScaling.py
tychopy/test/test_AutoScaling/test_Reentry.py
tychopy/test/test_Bindings/test_bindings.py
tychopy/test/test_FullProblems/test_CartPole.py
tychopy/test/test_FullProblems/test_Delta3Launch.py
tychopy/test/test_FullProblems/test_FreeFlyingRobot.py
tychopy/test/test_FullProblems/test_MultiPhaseCannon.py
tychopy/test/test_FullProblems/test_Reentry.py
tychopy/test/test_FullProblems/test_RosenBrock.py
tychopy/test/test_OptimalControl/test_Integrators.py
tychopy/test/test_VectorFunctions/test_Tables.py
```

**Step 2: Verify**

```bash
grep -r "import tycho\b" tychopy/test/
grep -r "_tycho\b" tychopy/test/
```

Expected: no matches.

**Step 3: Commit**

```bash
git add tychopy/test/
git commit -m "feat: update test imports tycho -> tychopy"
```

---

### Task 8: Update documentation guide scripts

**Files:**
- Modify: `doc/docguides/LGLInterpTableGuide.py`
- Modify: `doc/docguides/OCPGuide.py`
- Modify: `doc/docguides/ODEPhaseGuide.py`
- Modify: `doc/docguides/OptimalControlProblemGuide.py`
- Modify: `doc/docguides/PhaseGuide.py`

**Step 1: Update imports**

Same pattern: `import tycho` -> `import tychopy as typy`, update alias usage.

**Step 2: Verify**

```bash
grep -r "import tycho" doc/
```

Expected: no matches.

**Step 3: Commit**

```bash
git add doc/
git commit -m "feat: update doc guide imports tycho -> tychopy"
```

---

### Task 9: Update legacy benchmark scripts

**Files:**
- Modify: all `.py` files under `bench/legacy/`

**Step 1: Update imports**

Same pattern as Task 6. Files:
```
bench/legacy/python/basic/bench_*.py (13 files)
bench/legacy/python/multi_phase/bench_*.py (4 files)
bench/legacy/python/mesh_refinement/bench_*.py (3 files)
bench/legacy/analysis/pr8_breakdown.py
bench/legacy/run_benchmarks.py
```

**Step 2: Verify**

```bash
grep -r "import tycho" bench/legacy/
grep -r "_tycho" bench/legacy/
```

Expected: no matches.

**Step 3: Commit**

```bash
git add bench/legacy/
git commit -m "feat: update legacy benchmark imports tycho -> tychopy"
```

---

### Task 10: Update shell scripts

**Files:**
- Modify: `build_wheel.sh` — update glob patterns `dist/tycho-*.whl` -> `dist/tychopy-*.whl`
- Modify: `config_and_build.sh` — update references if they mention the Python package
- Modify: `scripts/run_examples.py` — update docstring referencing conda env

**Step 1: Update `build_wheel.sh`**

Replace `dist/tycho-*.whl` with `dist/tychopy-*.whl` in glob patterns.

**Step 2: Update `scripts/run_examples.py`**

Update the docstring at the top (conda env name stays `tycho` — it's the env name, not the package):
```python
"""
run_examples.py — Tychopy example test runner
...
"""
```

The runner script itself doesn't import tycho, so no import changes needed.

**Step 3: Update `config_and_build.sh`**

The conda env name (`tycho`) is the environment name, not the package name. It can stay as-is unless the user wants to rename the conda env too. Leave as-is for now.

**Step 4: Commit**

```bash
git add build_wheel.sh config_and_build.sh scripts/run_examples.py
git commit -m "feat: update shell scripts for tychopy rename"
```

---

### Task 11: Update documentation and markdown files

**Files:**
- Modify: `CLAUDE.md` — update Python package references
- Modify: `TYCHO_API_DESIGN_PLAN.md` — already references tychopy
- Modify: `README.md` — update import examples
- Modify: `doc/BuildAndPackaging.md` — update package references
- Modify: `doc/Bindings.md` — update module references
- Modify: `AGENTS.md` — update if it references the Python package

**Step 1: Update `CLAUDE.md`**

Key sections to update:
- "The Python-facing module is `_tycho`" -> "The Python-facing module is `_tychopy`"
- "imported via the `tycho` package" -> "imported via the `tychopy` package"
- References to `tycho/` directory -> `tychopy/`
- `import tycho` -> `import tychopy as typy`
- Update the repository structure section
- Update example commands

**Step 2: Update `README.md`**

Update any `import tycho` examples and package name references.

**Step 3: Update other docs**

Update `doc/BuildAndPackaging.md`, `doc/Bindings.md`, and any other docs referencing the Python package name.

**Step 4: Commit**

```bash
git add CLAUDE.md README.md AGENTS.md doc/ TYCHO_API_DESIGN_PLAN.md
git commit -m "docs: update documentation for tychopy rename"
```

---

### Task 12: Update GitHub workflows

**Files:**
- Modify: `.github/workflows/pypi-wheels.yml` — update package name references
- Modify: `.github/workflows/pip-ubuntu-22.yml` — update if needed
- Modify: `.github/install_ubuntu_dependencies.sh` — update if needed

**Step 1: Search and update**

```bash
grep -r "tycho" .github/
```

Update any references to the Python package name (not the repo name or C++ library name).

**Step 2: Commit**

```bash
git add .github/
git commit -m "ci: update GitHub workflows for tychopy rename"
```

---

### Task 13: Build and verify

**Step 1: Clean build**

```bash
conda activate tycho
cd build
rm -rf *
cmake --preset macos-llvm-release
ninja -j2 all
```

**Step 2: Verify the nanobind module builds with new name**

```bash
python -c "import _tychopy; print('nanobind module loaded')"
```

Expected: `nanobind module loaded`

**Step 3: Verify the Python package imports**

```bash
python -c "import tychopy as typy; typy.SoftwareInfo()"
```

Expected: Tycho software info printed.

```bash
python -c "import tychopy as typy; print(typy.VectorFunctions); print(typy.OptimalControl)"
```

Expected: module objects printed.

**Step 4: Run all 38 examples**

```bash
python scripts/run_examples.py
```

Expected: all 38 examples PASS (some may SKIP due to optional deps like basemap).

**Step 5: Run C++ brachistochrone example**

```bash
./examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found" with objective ~1.8013 s. (C++ example should be unaffected by Python rename.)

---

### Task 14: Final verification and cleanup

**Step 1: Search for any remaining `_tycho` or `import tycho` references**

```bash
# Should find NO matches in Python/CMake files (C++ namespace Tycho is expected and OK)
grep -r "import _tycho\b" --include="*.py" .
grep -r "import tycho\b" --include="*.py" .
grep -r "from tycho\b" --include="*.py" .
grep -rn "_tycho\b" --include="CMakeLists.txt" .
```

Expected: no matches. Any straggler references must be fixed.

Note: `_tycho` may still appear in:
- C++ comments/docstrings (update these to `_tychopy`)
- `CLAUDE.md` memory files (update relevant ones)
- Git history (expected, leave as-is)

**Step 2: Update memory files**

Update the Claude memory files that reference `_tycho` or the `tycho` Python package:
- `/Users/granthec/.claude/projects/-Users-granthec-Source-tycho/memory/MEMORY.md`

**Step 3: Format all changed Python files**

```bash
ruff format .
ruff check --select I --fix .
```

**Step 4: Final commit if formatting changed anything**

```bash
git add -A
git commit -m "chore: apply ruff format after tychopy rename"
```

---

## Validation Checklist

Before marking Phase 1 complete:

- [ ] `NB_MODULE(_tychopy, m)` in TychoModule.cpp
- [ ] `nanobind_add_module(_tychopy ...)` in src/Bindings/CMakeLists.txt
- [ ] Python package directory is `tychopy/` (not `tycho/`)
- [ ] `pyproject.toml` has `name = "tychopy"` and `wheel.packages = ["tychopy"]`
- [ ] All examples use `import tychopy as typy`
- [ ] All internal package imports use `_tychopy` and `tychopy`
- [ ] `python -c "import tychopy as typy; typy.SoftwareInfo()"` works
- [ ] All 38 Python examples pass (`python scripts/run_examples.py` exits 0)
- [ ] C++ brachistochrone example converges
- [ ] `grep -r "import tycho\b" --include="*.py" .` returns no matches
- [ ] `grep -r "_tycho\b" --include="CMakeLists.txt" .` returns no matches
- [ ] All documentation updated
