# Python Bindings Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Example count note:** This plan was written before Task 4's
> `examples/python_examples/UpdatedInterface/` merge collapsed seven
> duplicate examples into the parent directory. References to "38
> examples" / "38-example suite" / "all 38" below were accurate at
> plan-authoring time; the post-merge count is **32**
> (`find examples/python_examples -name "*.py" -not -path
> "*/UpdatedInterface/*"`). Trust the actual count over the plan body.

**Goal:** Rename Python-facing submodules, packages, and files from PascalCase to snake_case (hard break, no aliases); fully remove the legacy OCP link API (Python bindings + C++ class overloads + builder migration + C++ test migration); drop duplicate PascalCase factory aliases; rename internal C++ binding helper functions for consistency with the project's snake_case convention.

**Architecture:** Six sequential tasks ordered to keep the build green at every step. Task 1 lands the foundational snake_case rename so all subsequent tasks see canonical paths. Tasks 3 and 4 prepare callers (C++ builder, C++ tests, Python examples, Python tests) for the legacy OCP link API removal in Task 5. Task 6 cleans up internal binding helper names last so it doesn't conflict with Task 5's binding edits.

**Tech Stack:** C++20 + nanobind for the Python bindings; Python 3.13 + numpy/scipy/matplotlib for the wrapper layer; CMake + Ninja for the build; Google Test for C++ tests; pytest-style scripts for Python tests; `scripts/run_examples.py` for the 38-example integration suite.

**Branch:** `cleanup/python-bindings-snakecase-and-ocp-link` (already created, has the spec commit).

**Build/test conventions** (per CLAUDE.md):
- Always activate conda env: `conda activate tycho`
- Linux build parallelism: `-j6` (this dev machine has 32 GB RAM)
- Configure preset: `cmake --preset linux-clang-release`
- Build command: `cd build && ninja -j6 all`
- C++ tests: `cmake --preset linux-clang-release -DBUILD_CPP_TESTS=ON` then `ctest --test-dir build --output-on-failure`
- Python examples: `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"`
- C++ brachistochrone: build with `-DBUILD_CPP_EXAMPLES=ON` and run `./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp`; expect "Optimal Solution Found", obj ≈ 1.8013 s

---

## Task 1: Snake_case rename — C++ binding submodules, Python wrapper packages and files, and all references

**Goal:** Rename all PascalCase Python-facing modules, subpackages, and `.py` files to snake_case in one mechanical sweep, updating every reference (binding source, pure-Python wrappers, examples, tests). Foundational task — every subsequent task uses the renamed paths.

**Files:**
- Modify: `src/bindings/function_registry.h` (lines 64-71: 4 submodule names)
- Modify: `src/bindings/utils_bind.cpp` (line 23: "Utils" → "utils")
- Modify: `src/bindings/astro/tycho_astro.cpp` (line 31: "Astro" → "astro")
- Modify: `src/bindings/astro/kepler_model.cpp` (line 82: "Kepler" → "kepler")
- Move (git mv): `tychopy/Astro/` → `tychopy/astro/`
- Move: `tychopy/OptimalControl/` → `tychopy/optimal_control/`
- Move: `tychopy/Solvers/` → `tychopy/solvers/`
- Move: `tychopy/Utils/` → `tychopy/utils/`
- Move: `tychopy/VectorFunctions/` → `tychopy/vector_functions/`
- Move: `tychopy/Extensions/` → `tychopy/extensions/`
- Move: `tychopy/astro/AstroFrames.py` → `astro_frames.py`
- Move: `tychopy/astro/AstroModels.py` → `astro_models.py`
- Move: `tychopy/astro/AstroConstraints.py` → `astro_constraints.py`
- Move: `tychopy/astro/Constants.py` → `constants.py`
- Move: `tychopy/astro/Date.py` → `date.py`
- Move: `tychopy/astro/DataReadWrite.py` → `data_read_write.py`
- Move: `tychopy/astro/FramePlot.py` → `frame_plot.py`
- Move: `tychopy/astro/SpiceRead.py` → `spice_read.py`
- Move: `tychopy/optimal_control/MeshErrorPlots.py` → `mesh_error_plots.py`
- Move: `tychopy/optimal_control/ODEBaseClass.py` → `ode_base_class.py`
- Modify: `tychopy/__init__.py` (top-level imports)
- Modify: every `tychopy/<sub>/__init__.py` (`from _tychopy.<Sub> import *` → `from _tychopy.<sub> import *`; all `_tychopy.<Sub>.X` references; relative imports of renamed files)
- Modify: every `tychopy/astro/Extensions/*.py` (relative imports of renamed parent files like `from ..AstroModels import …`)
- Modify: every `tychopy/<sub>/Extensions/__init__.py`
- Modify: all 38 examples in `examples/python_examples/` (import paths)
- Modify: all Python tests under `tychopy/test/test_*/` (import paths)

**Acceptance Criteria:**
- [ ] `_tychopy.{vector_functions,optimal_control,solvers,utils,astro,extensions}` (and nested `_tychopy.astro.kepler`) all importable from Python; PascalCase forms raise `ImportError`/`AttributeError`.
- [ ] `tychopy.{astro,optimal_control,solvers,utils,vector_functions,extensions}` all importable; PascalCase forms raise `ImportError`.
- [ ] `from tychopy.astro.astro_models import CR3BP` works; `from tychopy.optimal_control.mesh_error_plots import PhaseMeshErrorPlot` works; `from tychopy.optimal_control.ode_base_class import ODEBase` works.
- [ ] Build succeeds: `cd build && ninja -j6 all`.
- [ ] All 38 Python examples exit 0: `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"`.
- [ ] All Python tests pass: `conda run -n tycho python -m pytest tychopy/test/`.
- [ ] All C++ tests pass: `ctest --test-dir build --output-on-failure`.
- [ ] `grep -rn "_tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\|Kepler\)" src tychopy examples tests bench` returns 0 hits.
- [ ] `grep -rn "tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\)" examples tychopy/test` returns 0 hits.

**Verify:** `cd /home/ghecht/Projects/tycho && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py" && ctest --test-dir build --output-on-failure` → all 38 examples PASS, all C++ tests PASS.

**Steps:**

- [ ] **Step 1: Rename the 4 C++ binding submodule string literals.**

Edit `src/bindings/function_registry.h` lines 63-73 (the `FunctionRegistry` constructor):

```cpp
    FunctionRegistry(nb::module_ &m)
        : mod(m), vfmod(m.def_submodule(
                      "vector_functions",
                      "SubModule Containing Vector and Scalar Function Types and Functions")),
          ocmod(
              m.def_submodule("optimal_control",
                              "SubModule Containing Optimal Control ODEs, Phases, and Utilities")),
          solmod(m.def_submodule("solvers", "SubModule Containing PSIOPT,NLP, and Solver Flags")),
          extmod(m.def_submodule("extensions", "User Defined Extensions")),
          vfuncx(nb::class_<VectorFunctionalX>(this->vfmod, "VectorFunction")),
          sfuncx(nb::class_<ScalarFunctionalX>(this->vfmod, "ScalarFunction")) {}
```

Edit `src/bindings/utils_bind.cpp` line 23:

```cpp
    auto um = m.def_submodule("utils", "Contains miscilanaeous utilities");
```

Edit `src/bindings/astro/tycho_astro.cpp` line 31:

```cpp
    auto mod = m.def_submodule("astro");
```

Edit `src/bindings/astro/kepler_model.cpp` line 82:

```cpp
    auto odemod = m.def_submodule("kepler");
```

- [ ] **Step 2: git mv the 6 Python wrapper subpackage directories.**

```bash
cd /home/ghecht/Projects/tycho
git mv tychopy/Astro tychopy/astro
git mv tychopy/OptimalControl tychopy/optimal_control
git mv tychopy/Solvers tychopy/solvers
git mv tychopy/Utils tychopy/utils
git mv tychopy/VectorFunctions tychopy/vector_functions
git mv tychopy/Extensions tychopy/extensions
```

- [ ] **Step 3: git mv the 10 Python files inside renamed subpackages.**

```bash
cd /home/ghecht/Projects/tycho
git mv tychopy/astro/AstroFrames.py tychopy/astro/astro_frames.py
git mv tychopy/astro/AstroModels.py tychopy/astro/astro_models.py
git mv tychopy/astro/AstroConstraints.py tychopy/astro/astro_constraints.py
git mv tychopy/astro/Constants.py tychopy/astro/constants.py
git mv tychopy/astro/Date.py tychopy/astro/date.py
git mv tychopy/astro/DataReadWrite.py tychopy/astro/data_read_write.py
git mv tychopy/astro/FramePlot.py tychopy/astro/frame_plot.py
git mv tychopy/astro/SpiceRead.py tychopy/astro/spice_read.py
git mv tychopy/optimal_control/MeshErrorPlots.py tychopy/optimal_control/mesh_error_plots.py
git mv tychopy/optimal_control/ODEBaseClass.py tychopy/optimal_control/ode_base_class.py
```

- [ ] **Step 4: Rewrite `tychopy/__init__.py` to use snake_case imports.**

Replace the import block (lines 16-22) so the file reads:

```python
import inspect

import _tychopy as _tychopy

import tychopy.astro
import tychopy.extensions
import tychopy.optimal_control
import tychopy.solvers
import tychopy.utils
import tychopy.vector_functions

software_info = _tychopy.software_info

if __name__ == "__main__":
    _tychopy.software_info()
    mlist = inspect.getmembers(_tychopy)
    for m in mlist:
        print(m[0], "= _tychopy." + str(m[0]))
```

- [ ] **Step 5: Rewrite the per-subpackage `__init__.py` files.**

Mechanical text replacement in every `tychopy/<sub>/__init__.py`:
- `from _tychopy.<Sub> import *` → `from _tychopy.<sub> import *`
- `_tychopy.<Sub>.X` → `_tychopy.<sub>.X`
- `_tychopy.Astro.Kepler` → `_tychopy.astro.kepler` (only in `tychopy/astro/__init__.py`)
- Relative imports of renamed files (`from .AstroFrames import …` → `from .astro_frames import …`, `from .ODEBaseClass import ODEBase` → `from .ode_base_class import ODEBase`).
- Update `inspect.getmembers(_tychopy.<Sub>)` and the `print("…= _tychopy.<Sub>." + …)` strings in each `if __name__ == "__main__":` block.

Run this sed-style sweep across all 6 init files:

```bash
cd /home/ghecht/Projects/tycho
python <<'EOF'
import re
from pathlib import Path

mapping = {
    "VectorFunctions": "vector_functions",
    "OptimalControl": "optimal_control",
    "Solvers": "solvers",
    "Utils": "utils",
    "Astro": "astro",
    "Extensions": "extensions",
    "Kepler": "kepler",
}
file_renames = {
    "AstroFrames": "astro_frames",
    "AstroModels": "astro_models",
    "AstroConstraints": "astro_constraints",
    "Constants": "constants",
    "Date": "date",
    "DataReadWrite": "data_read_write",
    "FramePlot": "frame_plot",
    "SpiceRead": "spice_read",
    "MeshErrorPlots": "mesh_error_plots",
    "ODEBaseClass": "ode_base_class",
}

def rewrite(text: str) -> str:
    # _tychopy.<PascalSub>  (and nested _tychopy.<Sub>.<Sub>)
    for old, new in mapping.items():
        text = re.sub(rf"_tychopy\.{old}\b", f"_tychopy.{new}", text)
        text = re.sub(rf"\btychopy\.{old}\b", f"tychopy.{new}", text)
    # Relative imports of renamed files: from .OldName import …
    for old, new in file_renames.items():
        text = re.sub(rf"from \.{old}\b", f"from .{new}", text)
        text = re.sub(rf"from \.\.{old}\b", f"from ..{new}", text)
    return text

for p in Path("tychopy").rglob("*.py"):
    if "__pycache__" in p.parts:
        continue
    new = rewrite(p.read_text())
    if new != p.read_text():
        p.write_text(new)
        print(f"rewrote {p}")
EOF
```

Manually inspect each `tychopy/<sub>/__init__.py` to confirm the rewrite landed cleanly, especially the `inspect.getmembers(_tychopy.<sub>)` line and the `print(... + str(m[0]))` strings.

- [ ] **Step 6: Update `tychopy/astro/Extensions/`, `tychopy/optimal_control/Extensions/`, etc. — relative imports.**

The Step 5 sweep should have already rewritten these. Spot-check `tychopy/astro/Extensions/EPPRFrame.py` and other files in Extensions subdirectories for any `from ..AstroModels` style imports that the sweep missed (e.g. fully-qualified `tychopy.Astro.AstroModels` strings).

```bash
cd /home/ghecht/Projects/tycho
grep -rn "Astro\|OptimalControl\|VectorFunctions\|Solvers\|Utils\|Kepler\|AstroFrames\|AstroModels\|AstroConstraints\|MeshErrorPlots\|ODEBaseClass\|FramePlot\|SpiceRead\|DataReadWrite" tychopy/ | grep -v __pycache__ | grep -v ".pyc"
```

Inspect the output: any remaining occurrences of PascalCase module/file names should either be class names (which we keep — e.g. `CR3BP`, `OptimalControlProblem`) or comments. If any import lines remain with PascalCase, fix them by hand.

- [ ] **Step 7: Sweep examples and Python tests for snake_case import paths.**

```bash
cd /home/ghecht/Projects/tycho
python <<'EOF'
import re
from pathlib import Path

mapping = {
    "VectorFunctions": "vector_functions",
    "OptimalControl": "optimal_control",
    "Solvers": "solvers",
    "Utils": "utils",
    "Astro": "astro",
    "Extensions": "extensions",
    "Kepler": "kepler",
}
file_renames = {
    "AstroFrames": "astro_frames",
    "AstroModels": "astro_models",
    "AstroConstraints": "astro_constraints",
    "Constants": "constants",
    "Date": "date",
    "DataReadWrite": "data_read_write",
    "FramePlot": "frame_plot",
    "SpiceRead": "spice_read",
    "MeshErrorPlots": "mesh_error_plots",
    "ODEBaseClass": "ode_base_class",
}

# Only rewrite tychopy.<Sub> patterns and tychopy.<Sub>.<File> patterns —
# do NOT touch Sub names appearing as e.g. "Astro.Constants" without
# tychopy. prefix (could be unrelated code).
def rewrite(text: str) -> str:
    for old, new in mapping.items():
        text = re.sub(rf"\btychopy\.{old}\b", f"tychopy.{new}", text)
        text = re.sub(rf"\b_tychopy\.{old}\b", f"_tychopy.{new}", text)
    for old, new in file_renames.items():
        # Only when preceded by a snake_case subpackage path
        text = re.sub(rf"\btychopy\.([a-z_]+)\.{old}\b", rf"tychopy.\1.{new}", text)
    return text

for root in ["examples/python_examples", "tychopy/test", "scripts"]:
    for p in Path(root).rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        original = p.read_text()
        new = rewrite(original)
        if new != original:
            p.write_text(new)
            print(f"rewrote {p}")
EOF
```

- [ ] **Step 8: Build and verify.**

```bash
cd /home/ghecht/Projects/tycho/build
ninja -j6 all
```

Expected: build succeeds (the heavy_compile job pool auto-throttles).

```bash
cd /home/ghecht/Projects/tycho
conda run -n tycho python -c "
import _tychopy
print('vector_functions' in dir(_tychopy))
print('optimal_control' in dir(_tychopy))
print('VectorFunctions' not in dir(_tychopy))
print('OptimalControl' not in dir(_tychopy))
import tychopy.astro
import tychopy.optimal_control
import tychopy.vector_functions
import tychopy.solvers
import tychopy.utils
import tychopy.extensions
from tychopy.astro.astro_models import CR3BP
from tychopy.optimal_control.ode_base_class import ODEBase
from tychopy.optimal_control.mesh_error_plots import PhaseMeshErrorPlot
print('all imports work')
"
```

Expected: prints `True\nTrue\nTrue\nTrue\nall imports work`.

- [ ] **Step 9: Run the full Python example suite.**

```bash
cd /home/ghecht/Projects/tycho
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: all 38 examples exit 0.

- [ ] **Step 10: Run C++ tests and Python tests.**

```bash
cd /home/ghecht/Projects/tycho
ctest --test-dir build --output-on-failure
conda run -n tycho python -m pytest tychopy/test/
```

Expected: all C++ tests PASS; all Python tests PASS.

- [ ] **Step 11: Verify zero PascalCase submodule references remain.**

```bash
cd /home/ghecht/Projects/tycho
grep -rn "_tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\|Kepler\)" src tychopy examples tests bench scripts 2>/dev/null | grep -v __pycache__ | grep -v ".pyc"
grep -rn "\btychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\)" examples tychopy/test 2>/dev/null | grep -v __pycache__
```

Expected: both commands print nothing.

- [ ] **Step 12: Commit.**

```bash
cd /home/ghecht/Projects/tycho
git add -A
git commit -m "$(cat <<'EOF'
refactor(bindings): rename Python submodules and files to snake_case

Hard break: PascalCase forms (`_tychopy.OptimalControl`, `tychopy.Astro`,
`tychopy/astro/AstroModels.py`, etc.) no longer exist. All call sites in
the bindings, pure-Python wrappers, examples, and tests updated to the
snake_case canonicals (`_tychopy.optimal_control`, `tychopy.astro`,
`tychopy/astro/astro_models.py`, etc.).

Class names (`OptimalControlProblem`, `PSIOPT`, `Arguments`, `Segment`,
`CR3BPFrame`, etc.) are unchanged — they were already idiomatic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Drop duplicate PascalCase factory aliases

**Goal:** Remove the PascalCase `Stack`/`StackScalar`/`Sum`/`SumElems`/`SumScalar` aliases in `tychopy/vector_functions/__init__.py` and the `ModifiedDynamics` alias in `tychopy/astro/__init__.py`. These are PascalCase mirrors of canonical snake_case factory functions; users go through the snake_case form. Update the in-tree call sites (`tychopy/astro/AstroModels.py`, `tychopy/astro/Extensions/EPPRFrame.py`, `tychopy/test/test_FullProblems/test_FreeFlyingRobot.py`, `examples/python_examples/MultiPhaseZermelo.py`, `examples/python_examples/Zermelo.py`, `tychopy/test/test_Astro/test_vec6_dispatch.py`) to use the snake_case factories.

**Files:**
- Modify: `tychopy/vector_functions/__init__.py` (drop 5 alias lines)
- Modify: `tychopy/astro/__init__.py` (drop `ModifiedDynamics` alias)
- Modify: `tychopy/astro/astro_models.py` (1 site: `vf.Stack` → `vf.stack`)
- Modify: `tychopy/astro/Extensions/EPPRFrame.py` (5 sites: `vf.Stack`, `vf.Sum`)
- Modify: `tychopy/test/test_FullProblems/test_FreeFlyingRobot.py` (2 sites: `vf.SumElems`)
- Modify: `tychopy/test/test_Astro/test_vec6_dispatch.py` (1 site: `vf.Stack`)
- Modify: `examples/python_examples/MultiPhaseZermelo.py` (1 site: `vf.Stack`)
- Modify: `examples/python_examples/Zermelo.py` (1 site: `vf.Stack`)

**Acceptance Criteria:**
- [ ] `from tychopy.vector_functions import stack, stack_scalar, sum, sum_elems, sum_scalar` works.
- [ ] `from tychopy.vector_functions import Stack` raises `ImportError`. Same for `StackScalar`, `Sum`, `SumElems`, `SumScalar`.
- [ ] `from tychopy.astro import modified_dynamics` works.
- [ ] `from tychopy.astro import ModifiedDynamics` raises `ImportError`.
- [ ] All 38 Python examples still pass.
- [ ] All Python tests still pass.

**Verify:** `cd /home/ghecht/Projects/tycho && conda run -n tycho python -c "from tychopy.vector_functions import stack, sum, sum_elems; from tychopy.astro import modified_dynamics; print('ok')"` prints `ok`. `conda run -n tycho python -c "from tychopy.vector_functions import Stack" 2>&1 | grep ImportError` matches.

**Steps:**

- [ ] **Step 1: Drop the 5 PascalCase factory aliases from `tychopy/vector_functions/__init__.py`.**

Delete the lines:

```python
Stack = _tychopy.vector_functions.stack
StackScalar = _tychopy.vector_functions.stack_scalar
Sum = _tychopy.vector_functions.sum
SumElems = _tychopy.vector_functions.sum_elems
SumScalar = _tychopy.vector_functions.sum_scalar
```

(The snake_case forms `stack`, `stack_scalar`, `sum`, `sum_elems`, `sum_scalar` already exist on the next lines and remain.)

- [ ] **Step 2: Drop the `ModifiedDynamics` alias from `tychopy/astro/__init__.py`.**

Delete the line:

```python
ModifiedDynamics = _tychopy.astro.modified_dynamics
```

- [ ] **Step 3: Migrate call sites — `vf.Stack` → `vf.stack`, `vf.Sum` → `vf.sum`, `vf.SumElems` → `vf.sum_elems`.**

Edit each call site listed in Files. Example for `tychopy/astro/Extensions/EPPRFrame.py`:

```python
# before:
state = vf.Stack([Xrot, Vrot, t])
# after:
state = vf.stack([Xrot, Vrot, t])
```

Same pattern for the other sites — purely a casing change, no semantic change.

- [ ] **Step 4: Verify.**

```bash
cd /home/ghecht/Projects/tycho
conda run -n tycho python -c "
from tychopy.vector_functions import stack, stack_scalar, sum, sum_elems, sum_scalar
from tychopy.astro import modified_dynamics
print('snake_case ok')
try:
    from tychopy.vector_functions import Stack
    print('FAIL: Stack still importable')
except ImportError:
    print('Stack ImportError ok')
try:
    from tychopy.astro import ModifiedDynamics
    print('FAIL: ModifiedDynamics still importable')
except ImportError:
    print('ModifiedDynamics ImportError ok')
"
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
conda run -n tycho python -m pytest tychopy/test/
```

Expected: prints `snake_case ok\nStack ImportError ok\nModifiedDynamics ImportError ok`; all 38 examples pass; all Python tests pass.

- [ ] **Step 5: Commit.**

```bash
cd /home/ghecht/Projects/tycho
git add -A
git commit -m "$(cat <<'EOF'
refactor(bindings): drop duplicate PascalCase factory aliases

Remove the PascalCase mirrors of snake_case factory functions
(`Stack`/`StackScalar`/`Sum`/`SumElems`/`SumScalar` in vector_functions,
`ModifiedDynamics` in astro). These were autocomplete helpers from the
ASSET upstream that conflated factory functions with classes. The
snake_case canonicals are the only public form going forward.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Migrate C++ builder and C++ tests off the legacy OCP link API

**Goal:** Update `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` and the three C++ tests (`test_ocp_direct_link.cpp`, `test_ocp_phase_lifetime.cpp`, `test_builder_brachistochrone.cpp`) to call the new `add_forward_link_equal_con(PhaseRefType, PhaseRefType, VarIndexType, ScaleType)` and `add_direct_link_equal_con(PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType, VarIndexType, ScaleType)` overloads instead of the legacy ones. Precondition for Task 5 (which deletes the legacy C++ overloads).

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` (lines 53, 89, 94, 101, 109, 121 — six call sites to migrate)
- Modify: `tests/cpp/optimal_control/test_ocp_direct_link.cpp` (~5 sites)
- Modify: `tests/cpp/optimal_control/test_ocp_phase_lifetime.cpp` (~3 sites)
- Modify: `tests/cpp/optimal_control/test_builder_brachistochrone.cpp` (no source changes — passes through after wrapper migration; only verify it still builds + passes)

**Acceptance Criteria:**
- [ ] `ocp_wrapper.h` no longer calls any legacy `add_forward_link_equal_con` or `add_direct_link_equal_con` overload (the surviving calls go through the new `(PhaseRefType, …, ScaleType)` signatures).
- [ ] `tests/cpp/optimal_control/test_ocp_direct_link.cpp` and `test_ocp_phase_lifetime.cpp` use the new 7-arg `add_direct_link_equal_con` form (with `ScaleType`).
- [ ] Build succeeds: `cd build && ninja -j6 all`.
- [ ] All C++ tests pass: `ctest --test-dir build --output-on-failure`.
- [ ] All 38 Python examples still pass (no change expected; sanity check).

**Verify:** `cd /home/ghecht/Projects/tycho && cd build && ninja -j6 all && ctest --test-dir build --output-on-failure` → all C++ tests PASS.

**Steps:**

- [ ] **Step 1: Read the new C++ method signatures.**

In `include/tycho/detail/optimal_control/phase/optimal_control_problem.h`:

```cpp
std::vector<int> add_forward_link_equal_con(PhaseRefType iphase_t, PhaseRefType fphase_t,
                                            VarIndexType vars, ScaleType scale_t);

int add_direct_link_equal_con(PhaseRefType p0, RegionType reg0_t, VarIndexType v0,
                              PhaseRefType p1, RegionType reg1_t, VarIndexType v1,
                              ScaleType scale_t);
```

`PhaseRefType = std::variant<int, std::shared_ptr<ODEPhaseBase>, std::string>` — accepts `int`, `PhasePtr`, or string name.
`RegionType = std::variant<int, PhaseRegionFlags, std::string>` — accepts `PhaseRegionFlags::Back` etc., string `"Back"`, or int.
`VarIndexType = std::variant<Eigen::VectorXi, std::vector<std::string>>` — accepts `Eigen::VectorXi` or list of variable names.
`ScaleType` — variant; the canonical default is `std::string("auto")` (matches the Python binding default).

- [ ] **Step 2: Migrate `ocp_wrapper.h` `add_forward_link_equal_con` call sites (lines 53, 89).**

Edit `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` lines 53-91. The wrapper signatures stay identical; only the delegated calls change. Replace:

```cpp
return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), idx1);
```

with:

```cpp
return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), idx1,
                                       std::string("auto"));
```

And replace:

```cpp
return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), vars);
```

with:

```cpp
return ocp_.add_forward_link_equal_con(p1.base_ptr(), p2.base_ptr(), vars,
                                       std::string("auto"));
```

- [ ] **Step 3: Migrate `ocp_wrapper.h` `add_direct_link_equal_con` call sites (lines 94, 101, 109, 121).**

Replace each call of the form:

```cpp
return ocp_.add_direct_link_equal_con(phase_a, region_a, vars_a, phase_b, region_b, vars_b);
```

with:

```cpp
return ocp_.add_direct_link_equal_con(phase_a, region_a, vars_a, phase_b, region_b, vars_b,
                                      std::string("auto"));
```

Apply this pattern to all four overloads in the wrapper (lines 94, 101, 109, 121 in the original file). The wrapper's exposed signatures remain unchanged — only the delegated `ocp_.…` call gains the `ScaleType` arg.

- [ ] **Step 4: Migrate `tests/cpp/optimal_control/test_ocp_direct_link.cpp`.**

Open the file. For each `ocp.add_direct_link_equal_con(...)` call (5 sites at lines 69, 87, 103, 117, 130), append `, std::string("auto")` as the trailing argument. Example:

```cpp
// before:
EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, vars_a, 1,
                                              PhaseRegionFlags::Back, vars_b));
// after:
EXPECT_NO_THROW(ocp.add_direct_link_equal_con(0, PhaseRegionFlags::Back, vars_a, 1,
                                              PhaseRegionFlags::Back, vars_b,
                                              std::string("auto")));
```

Apply the same edit to the `EXPECT_THROW` call at line 130.

- [ ] **Step 5: Migrate `tests/cpp/optimal_control/test_ocp_phase_lifetime.cpp`.**

Same pattern: append `, std::string("auto")` to every `add_direct_link_equal_con(...)` call (3 sites at lines 80, 82, 119).

- [ ] **Step 6: Build and run C++ tests.**

```bash
cd /home/ghecht/Projects/tycho/build
ninja -j6 all
ctest --test-dir . --output-on-failure
```

Expected: build succeeds; all C++ tests PASS, including `test_ocp_direct_link`, `test_ocp_phase_lifetime`, and `test_builder_brachistochrone`.

- [ ] **Step 7: Sanity-check Python examples.**

```bash
cd /home/ghecht/Projects/tycho
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: all 38 examples pass (no behavior change expected from this task — the wrapper edits are surface-level).

- [ ] **Step 8: Commit.**

```bash
cd /home/ghecht/Projects/tycho
git add -A
git commit -m "$(cat <<'EOF'
refactor(astro,oc): migrate C++ builder and tests to new OCP link API

Update ocp_wrapper.h and the C++ test suite to call the new
add_forward_link_equal_con(PhaseRefType, PhaseRefType, VarIndexType,
ScaleType) and add_direct_link_equal_con(PhaseRefType, RegionType,
VarIndexType, PhaseRefType, RegionType, VarIndexType, ScaleType)
overloads. ScaleType defaults to std::string("auto") to match the
Python binding default. Precondition for the legacy C++ overload
deletion in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migrate Python examples and tests off the legacy OCP link API; merge `UpdatedInterface/` into parent

**Goal:** Eliminate every Python call to a legacy OCP link API form. For files duplicated in `examples/python_examples/UpdatedInterface/`, replace the parent (old-API) version with the `UpdatedInterface/` (new-API) version. For parent-only files using the legacy API, migrate inline. Migrate Python tests using `add_forward_link_equal_con(int, int, list)` to the new form. Delete `UpdatedInterface/` after merging. Precondition for Task 5 (which deletes the legacy Python bindings).

**Files:**
- Replace (mv from UpdatedInterface to parent): `examples/python_examples/UpdatedInterface/{Delta3Launch,GoddardRocket,MinimumTimeToClimb,MinimumTimeToClimbTables,MultiPhaseCannon,Reentry}.py` → overwrite the same-named files in the parent directory.
- Modify (inline migration): `examples/python_examples/MultiSpacecraftOptimization.py`, `examples/python_examples/Heteroclinic.py`, `examples/python_examples/MultiPhaseZermelo.py`, `examples/python_examples/MeshRefinement/Delta3Launch.py`
- Modify (inline migration of tests): `tychopy/test/test_AdaptiveMesh/test_Delta3Launch.py`, `tychopy/test/test_AutoScaling/test_ObjScaling.py`, `tychopy/test/test_FullProblems/test_MultiPhaseCannon.py`, `tychopy/test/test_FullProblems/test_Delta3Launch.py`
- Delete: `examples/python_examples/UpdatedInterface/` (empty directory after the moves)

**Acceptance Criteria:**
- [ ] `examples/python_examples/UpdatedInterface/` no longer exists.
- [ ] `grep -rn "add_forward_link_equal_con\|add_direct_link_equal_con\|add_link_equal_con\|add_link_inequal_con\|add_link_objective" examples/python_examples tychopy/test` shows no remaining call that omits the `auto_scale` argument when the example/test was previously using the legacy API.
- [ ] All 38 Python examples exit 0.
- [ ] All Python tests pass.

**Verify:** `cd /home/ghecht/Projects/tycho && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py" && conda run -n tycho python -m pytest tychopy/test/` → all 38 examples PASS, all Python tests PASS.

**Steps:**

- [ ] **Step 1: Replace duplicated parent examples with their `UpdatedInterface/` versions.**

```bash
cd /home/ghecht/Projects/tycho/examples/python_examples
for f in Delta3Launch GoddardRocket MinimumTimeToClimb MinimumTimeToClimbTables MultiPhaseCannon Reentry; do
    git mv -f UpdatedInterface/${f}.py ${f}.py
done
```

(Note: `git mv -f` overwrites; the parent old-API file is replaced by the new-API one.)

- [ ] **Step 2: Inline-migrate parent-only old-API examples.**

For `examples/python_examples/MultiSpacecraftOptimization.py`: the `add_link_equal_con(LinkFun, [(i, "Back", range(0, 7), [], [])], range(0, 7))` call is already in the new PhasePack form. Just append `auto_scale='auto'`:

```python
# before (line ~133):
ocp.add_link_equal_con(LinkFun, [(i, "Back", range(0, 7), [], [])], range(0, 7))
# after:
ocp.add_link_equal_con(LinkFun, [(i, "Back", range(0, 7), [], [])], range(0, 7),
                       auto_scale='auto')
```

For `examples/python_examples/Heteroclinic.py` line 201, `examples/python_examples/MultiPhaseZermelo.py` line 162, and `examples/python_examples/MeshRefinement/Delta3Launch.py` line 449: append `, 'auto'` (positional) or `, auto_scale='auto'` (kwarg) to each `add_forward_link_equal_con(...)` call. Example:

```python
# before:
ocp.add_forward_link_equal_con(0, -1, [0, 1, 2])
# after:
ocp.add_forward_link_equal_con(0, -1, [0, 1, 2], auto_scale='auto')
```

- [ ] **Step 3: Inline-migrate Python tests.**

For each of `tychopy/test/test_AdaptiveMesh/test_Delta3Launch.py`, `tychopy/test/test_AutoScaling/test_ObjScaling.py`, `tychopy/test/test_FullProblems/test_MultiPhaseCannon.py`, `tychopy/test/test_FullProblems/test_Delta3Launch.py`: append `auto_scale='auto'` (kwarg form) to each `add_forward_link_equal_con(...)` and `add_direct_link_equal_con(...)` call. Same pattern as Step 2.

- [ ] **Step 4: Delete `UpdatedInterface/` directory.**

```bash
cd /home/ghecht/Projects/tycho
git rm -r examples/python_examples/UpdatedInterface
```

(`git mv` in Step 1 should have already emptied the directory; this removes the now-empty directory entry.)

- [ ] **Step 5: Verify.**

```bash
cd /home/ghecht/Projects/tycho
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
conda run -n tycho python -m pytest tychopy/test/
ls examples/python_examples/UpdatedInterface 2>&1 | head -1
```

Expected: all 38 examples pass; all Python tests pass; `ls` reports `No such file or directory`.

- [ ] **Step 6: Commit.**

```bash
cd /home/ghecht/Projects/tycho
git add -A
git commit -m "$(cat <<'EOF'
refactor(examples): migrate Python examples and tests to new OCP link API

Replace the parent-directory old-API examples with their UpdatedInterface
equivalents (Delta3Launch, GoddardRocket, MinimumTimeToClimb,
MinimumTimeToClimbTables, MultiPhaseCannon, Reentry). Inline-migrate
parent-only old-API examples (MultiSpacecraftOptimization,
Heteroclinic, MultiPhaseZermelo, MeshRefinement/Delta3Launch) and the
Python tests that called legacy add_forward_link_equal_con /
add_direct_link_equal_con. Delete the now-redundant UpdatedInterface/
subdirectory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Remove the legacy OCP link API (Python bindings + C++ class overloads + Python aliases)

**Goal:** With every caller migrated (Tasks 3 and 4), delete the legacy OCP link API. Remove `BuildOldLinkIterface()` from the Python bindings, rename `build_new_link_interface()` → `build_link_interface()`, delete the `LinkConstraint`/`LinkObjective` `nb::class_` registrations, drop the corresponding aliases in `tychopy/optimal_control/__init__.py`, and delete the legacy overloads from `optimal_control_problem.h`.

**Files:**
- Modify: `src/bindings/optimal_control/optimal_control_problem_bind.cpp` (delete `BuildOldLinkIterface()` body and forward declaration; rename `build_new_link_interface()` → `build_link_interface()`; remove the call to `BuildOldLinkIterface(obj)` in `Build()`)
- Modify: `src/bindings/optimal_control/tycho_optimal_control.cpp` (delete `nb::class_<LinkFunction<GenericFunction<-1, -1>>>::Build(oc, "LinkConstraint")` and the same for `LinkObjective` at lines 104-105)
- Modify: `tychopy/optimal_control/__init__.py` (drop `LinkConstraint`, `LinkObjective`, `LinkFlags` aliases)
- Modify: `include/tycho/detail/optimal_control/phase/optimal_control_problem.h` (delete the legacy overloads listed in the spec — lines 546, 690-785, 802-827, and the `add_direct_link_equal_con` legacy block at line 830 onward; apply mirror deletions to the `add_link_inequal_con` and `add_link_objective` sections in the same file)

**Acceptance Criteria:**
- [ ] `src/bindings/optimal_control/optimal_control_problem_bind.cpp` no longer contains `BuildOldLinkIterface` or `build_new_link_interface`. The new `build_link_interface()` is its only link-binding helper.
- [ ] `tycho.optimal_control.LinkConstraint`, `tycho.optimal_control.LinkObjective`, and `tycho.optimal_control.LinkFlags` are no longer importable.
- [ ] `optimal_control_problem.h` no longer declares any of the legacy overloads (no `add_link_equal_con(LinkConstraint)`, no `add_link_equal_con(…, RegVec, …)`-style 7-arg parallel-list forms, no `add_forward_link_equal_con(int, int, VectorXi)`, no `add_direct_link_equal_con(LinkFlags, …)`, etc.).
- [ ] `LinkConstraint` and `LinkObjective` types remain in the C++ source as internal implementation details (used by `make_func_impl<LinkConstraint, …>`).
- [ ] Build succeeds: `ninja -j6 all`.
- [ ] All C++ tests pass.
- [ ] All 38 Python examples pass.
- [ ] All Python tests pass.
- [ ] `grep -rn "BuildOldLinkIterface\|build_new_link_interface" src` returns 0 hits.
- [ ] `grep -rn "LinkConstraint\|LinkObjective\|LinkFlags" tychopy examples` returns 0 hits.

**Verify:** `cd /home/ghecht/Projects/tycho && cd build && ninja -j6 all && ctest --test-dir . --output-on-failure && cd .. && conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"` → build succeeds, all C++ tests PASS, all 38 examples PASS.

**Steps:**

- [ ] **Step 1: Edit `src/bindings/optimal_control/optimal_control_problem_bind.cpp` — remove legacy block.**

Delete:
- The forward declaration `static void BuildOldLinkIterface(...)` at lines 37-38.
- The function body `static void BuildOldLinkIterface(...) { ... }` at lines 315-625 (~310 lines).
- The call `BuildOldLinkIterface(obj);` at line 46.

Rename `build_new_link_interface` → `build_link_interface` at the forward declaration (line 35), the call site (line 45), and the function definition (line 170).

- [ ] **Step 2: Edit `src/bindings/optimal_control/tycho_optimal_control.cpp` — remove `LinkConstraint`/`LinkObjective` registrations.**

Delete lines 104-105:

```cpp
TychoBind<LinkFunction<GenericFunction<-1, -1>>>::Build(oc, "LinkConstraint");
TychoBind<LinkFunction<GenericFunction<-1, 1>>>::Build(oc, "LinkObjective");
```

If there are forward declarations of these `TychoBind<LinkFunction<...>>::Build` specializations elsewhere in the file or in headers, also delete them. Check by grepping after the edit:

```bash
grep -rn "TychoBind<LinkFunction" src/bindings
```

Expected: zero hits after the edit.

- [ ] **Step 3: Edit `tychopy/optimal_control/__init__.py` — drop `LinkConstraint`, `LinkObjective`, `LinkFlags` aliases.**

Delete the lines:

```python
LinkConstraint = _tychopy.optimal_control.LinkConstraint
LinkFlags = _tychopy.optimal_control.LinkFlags
LinkObjective = _tychopy.optimal_control.LinkObjective
```

(These will fail at import time after Step 2 anyway; this just keeps the file clean.)

- [ ] **Step 4: Delete legacy overloads from `include/tycho/detail/optimal_control/phase/optimal_control_problem.h`.**

Delete the following overloads of `add_link_equal_con` (and apply the same deletion pattern to the `add_link_inequal_con` and `add_link_objective` mirror sections):

- `add_link_equal_con(LinkConstraint)` (line 546).
- All 8 7-arg parallel-list overloads at lines 690-749 (`RegVec`/`std::vector<std::string>`/`LinkFlags`/`std::string` × `std::vector<VectorXi>`/`std::vector<std::vector<PhasePtr>>`).
- All 5-arg parallel-list overloads at lines 753-765.
- All 4-arg parallel-list overloads at lines 767-785.
- `add_forward_link_equal_con(int, int, VectorXi)` (line 802).
- `add_forward_link_equal_con(PhasePtr, PhasePtr, VectorXi)` (line 825).
- All `add_direct_link_equal_con` legacy overloads starting at line 830 (the `LinkFlags`-pair form at 830, the four `(VectorFunctionalX, int/PhasePtr, PhaseRegionFlags/string, …)` forms following it).

Search-and-mirror for `add_link_inequal_con` and `add_link_objective` legacy overloads further down the same file, deleting the equivalent set.

Keep the new-API overloads:
- `add_link_equal_con(VectorFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv, ScaleType scale_t)` (line 553).
- `add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType XtUV0, …)` (line 559).
- `add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0, PhaseRefType p1, …)` (lines 569 and 577).
- `add_forward_link_equal_con(PhaseRefType, PhaseRefType, VarIndexType, ScaleType)` (line 585).
- `add_direct_link_equal_con(PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType, VarIndexType, ScaleType)` (line 618).
- `add_param_link_equal_con(PhaseRefType, PhaseRefType, RegionType, VarIndexType, ScaleType)` (line 642).
- `add_link_param_equal_con` overloads.

`LinkConstraint` and `LinkObjective` types stay — they are used internally by `make_func_impl<LinkConstraint, …>` (line 555 and similar). Do not delete those types.

`LinkFlags` enum: after the deletions, grep for any remaining `LinkFlags::` reference inside `optimal_control_problem.h`. If zero hits, the enum can be removed from the header where it is declared. If any hits remain (e.g. surviving overloads still pass `LinkFlags::LinkParams` to `LinkConstraint(...)`), keep the enum but remove its Python binding — already accomplished in Step 3.

- [ ] **Step 5: Build and verify all gates.**

```bash
cd /home/ghecht/Projects/tycho/build
ninja -j6 all
ctest --test-dir . --output-on-failure
cd ..
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
conda run -n tycho python -m pytest tychopy/test/
```

Expected: build succeeds; all C++ tests PASS; all 38 examples PASS; all Python tests PASS.

- [ ] **Step 6: Verify Python bindings no longer expose legacy types.**

```bash
cd /home/ghecht/Projects/tycho
conda run -n tycho python -c "
import tychopy.optimal_control as oc
for name in ('LinkConstraint', 'LinkObjective', 'LinkFlags'):
    if hasattr(oc, name):
        raise AssertionError(f'{name} still exposed')
print('all legacy link types removed')
"
```

Expected: prints `all legacy link types removed`.

- [ ] **Step 7: Verify zero in-tree references to deleted symbols.**

```bash
cd /home/ghecht/Projects/tycho
grep -rn "BuildOldLinkIterface\|build_new_link_interface" src 2>/dev/null
grep -rn "LinkConstraint\|LinkObjective\|LinkFlags" tychopy examples 2>/dev/null | grep -v __pycache__
```

Expected: first command prints nothing; second command prints nothing (or only matches inside docstrings/comments that explicitly reference the removal).

- [ ] **Step 8: Commit.**

```bash
cd /home/ghecht/Projects/tycho
git add -A
git commit -m "$(cat <<'EOF'
refactor(oc): remove legacy OCP link API from C++ class and Python bindings

Delete BuildOldLinkIterface() from the Python bindings and rename
build_new_link_interface() -> build_link_interface(). Delete the
LinkConstraint / LinkObjective / LinkFlags Python aliases and their
nb::class_ registrations. Delete the legacy add_link_equal_con /
add_link_inequal_con / add_link_objective overloads (parallel-list
and LinkConstraint object forms) plus the legacy
add_forward_link_equal_con(int,int,VectorXi) and
add_direct_link_equal_con(LinkFlags,...) overloads from the C++ class.

LinkConstraint and LinkObjective types remain as internal
implementation details (used by make_func_impl<LinkConstraint,...>).

Capability preservation audit confirmed every legacy form maps to a
new-API call before deletion (see
doc/plans/2026-05-06-python-bindings-cleanup-design.md §
"Capability preservation audit").

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Internal C++ binding helper rename

**Goal:** Rename the internal binding helper functions to snake_case to match the project's "member functions and free functions are snake_case" convention. `TychoBind<T>::Build` → `build`; `VectorFunctionBuild`, `SolversBuild`, `OptimalControlBuild`, `UtilsBuild`, `AstroBuild`, `ExtensionsBuild`, `KeplerUtilsBuild`, `LambertSolversBuild` → snake_case (`vector_functions_build`, `solvers_build`, `optimal_control_build`, `utils_build`, `astro_build`, `extensions_build`, `kepler_utils_build`, `lambert_solvers_build`). Internal-only — does not affect Python.

**Files:**
- Modify: `src/bindings/function_registry.h` (lines 98-114: 6 references to `TychoBind<Derived>::Build`; rename to `::build`)
- Modify: `src/bindings/tycho_module.cpp` (forward declarations at lines 26-37 and call sites at lines 172-180)
- Modify: every `*_bind.cpp` and `*_bind.h` containing `TychoBind<X>::Build(...)` definition or `XxxBuild(...)` definition
- Files known to need touching:
  - `src/bindings/bump_allocator_bind.cpp` (line 26: `TychoBind<BumpAllocator>::Build`)
  - `src/bindings/utils_bind.cpp` (line 22: `UtilsBuild`; line 30: `TychoBind<BumpAllocator>::Build` call)
  - `src/bindings/astro/tycho_astro.cpp` (lines 23, 26, 27, 30, 35, 36)
  - `src/bindings/astro/kepler_utils.cpp` (lines 47, 50)
  - `src/bindings/astro/kepler_model.cpp` (line 85)
  - `src/bindings/astro/lambert_solvers_bind.cpp` (lines 27, 30)
  - `src/bindings/optimal_control/tycho_optimal_control.cpp` (forward decl + body of `OptimalControlBuild`)
  - `src/bindings/optimal_control/mesh_iterate_info_bind.cpp` (line 25)
  - `src/bindings/optimal_control/lgl_interp_table_bind.cpp` (line 26)
  - `src/bindings/optimal_control/ode_phase_base_bind.cpp` (line 31)
  - `src/bindings/optimal_control/ode_bind.h` (lines 47, 65)
  - `src/bindings/solvers/jet_bind.cpp` (line 27)
  - `src/bindings/solvers/optimization_problem_bind.cpp` (line 26)
  - `src/bindings/solvers/optimization_problem_base.cpp` (line 26)
  - `src/bindings/solvers/tycho_solvers.cpp` (forward decl + body of `SolversBuild`)
  - `src/bindings/vf/tycho_vector_functions.cpp` (forward decl + body of `VectorFunctionBuild`)
  - any other `*_bind.cpp` with a `TychoBind<X>::Build` definition (find via grep)

**Acceptance Criteria:**
- [ ] `grep -rn "TychoBind<.*>::Build\b" src/bindings` returns 0 hits (everything migrated to `::build`).
- [ ] `grep -rnE "VectorFunctionBuild|SolversBuild|OptimalControlBuild|UtilsBuild|AstroBuild|ExtensionsBuild|KeplerUtilsBuild|LambertSolversBuild" src/bindings` returns 0 hits.
- [ ] Build succeeds.
- [ ] Python module loads (`import _tychopy; _tychopy.software_info()` runs).
- [ ] All 38 Python examples pass; all C++ tests pass.

**Verify:** `cd /home/ghecht/Projects/tycho && cd build && ninja -j6 all && cd .. && conda run -n tycho python -c "import _tychopy; _tychopy.software_info()"` → build succeeds, software_info banner prints.

**Steps:**

- [ ] **Step 1: Find every `TychoBind<X>::Build` definition.**

```bash
cd /home/ghecht/Projects/tycho
grep -rn "TychoBind<.*>::Build\b" src/bindings
```

Capture the list. For each file in the list, edit the function definition signature to use lowercase `build`:

```cpp
// before:
void TychoBind<X>::Build(nb::module_ &m) { ... }
// after:
void TychoBind<X>::build(nb::module_ &m) { ... }
```

(Including the two-arg overload with `const char *name` where it appears.)

- [ ] **Step 2: Update `src/bindings/function_registry.h` — `TychoBind` primary template.**

The primary template at line 40 is `template <class T> struct TychoBind;` and specializations supply `Build`. The `Build_Register` template at lines 97-114 calls `TychoBind<Derived>::Build(...)`. Update all 6 `Build` references in this file to `build`:

```cpp
template <class Derived>
    requires requires(nb::module_ &m) { TychoBind<Derived>::build(m); }
void Build_Register(nb::module_ &m) {
    TychoBind<Derived>::build(m);
    RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
}
template <class Derived>
    requires requires(nb::module_ &m, const char *name) { TychoBind<Derived>::build(m, name); }
void Build_Register(const char *name) {
    TychoBind<Derived>::build(this->mod, name);
    RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
}
template <class Derived>
    requires requires(nb::module_ &m, const char *name) { TychoBind<Derived>::build(m, name); }
void Build_Register(nb::module_ &m, const char *name) {
    TychoBind<Derived>::build(m, name);
    RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
}
```

(Note: `Build_Register` itself is a member template — leaving its name PascalCase or renaming it to `build_register` is a style choice. Per the project convention rename it to `build_register` for consistency.)

If `Build_Register` is renamed to `build_register`, find every call site of `Build_Register` and update:

```bash
grep -rn "Build_Register\b" src/bindings
```

Update each.

- [ ] **Step 3: Find and rename `*Build` free functions.**

```bash
cd /home/ghecht/Projects/tycho
grep -rnE "\b(VectorFunctionBuild|SolversBuild|OptimalControlBuild|UtilsBuild|AstroBuild|ExtensionsBuild|KeplerUtilsBuild|LambertSolversBuild)\b" src/bindings
```

Apply this rename map to every match (forward declarations, definitions, and call sites):

| Old | New |
|---|---|
| `VectorFunctionBuild` | `vector_functions_build` |
| `SolversBuild` | `solvers_build` |
| `OptimalControlBuild` | `optimal_control_build` |
| `UtilsBuild` | `utils_build` |
| `AstroBuild` | `astro_build` |
| `ExtensionsBuild` | `extensions_build` |
| `KeplerUtilsBuild` | `kepler_utils_build` |
| `LambertSolversBuild` | `lambert_solvers_build` |

Use `sed -i` per file or a single-shot Python script:

```bash
cd /home/ghecht/Projects/tycho
python <<'EOF'
import re
from pathlib import Path

renames = {
    "VectorFunctionBuild": "vector_functions_build",
    "SolversBuild": "solvers_build",
    "OptimalControlBuild": "optimal_control_build",
    "UtilsBuild": "utils_build",
    "AstroBuild": "astro_build",
    "ExtensionsBuild": "extensions_build",
    "KeplerUtilsBuild": "kepler_utils_build",
    "LambertSolversBuild": "lambert_solvers_build",
}

for p in Path("src/bindings").rglob("*"):
    if p.is_dir() or p.suffix not in {".cpp", ".h"}:
        continue
    text = p.read_text()
    new = text
    for old, sub in renames.items():
        new = re.sub(rf"\b{old}\b", sub, new)
    if new != text:
        p.write_text(new)
        print(f"rewrote {p}")
EOF
```

- [ ] **Step 4: Build and verify.**

```bash
cd /home/ghecht/Projects/tycho/build
ninja -j6 all
cd ..
conda run -n tycho python -c "import _tychopy; _tychopy.software_info()"
ctest --test-dir build --output-on-failure
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: build succeeds; banner prints; all C++ tests PASS; all 38 examples PASS.

- [ ] **Step 5: Verify zero PascalCase helper references remain.**

```bash
cd /home/ghecht/Projects/tycho
grep -rn "TychoBind<.*>::Build\b" src/bindings
grep -rnE "\b(VectorFunctionBuild|SolversBuild|OptimalControlBuild|UtilsBuild|AstroBuild|ExtensionsBuild|KeplerUtilsBuild|LambertSolversBuild)\b" src/bindings
```

Expected: both commands print nothing.

- [ ] **Step 6: Run clang-format on the touched bindings.**

```bash
cd /home/ghecht/Projects/tycho/build
ninja clang-format
```

Expected: no errors. The clang-format target reformats `src/`, `extensions/`, and `include/tycho/detail/`.

- [ ] **Step 7: Commit.**

```bash
cd /home/ghecht/Projects/tycho
git add -A
git commit -m "$(cat <<'EOF'
refactor(bindings): rename internal binding helper functions to snake_case

Per the project convention (member functions and free functions are
snake_case), rename TychoBind<T>::Build -> ::build,
Build_Register -> build_register, and the per-subsystem aggregate free
functions (VectorFunctionBuild, SolversBuild, OptimalControlBuild,
UtilsBuild, AstroBuild, ExtensionsBuild, KeplerUtilsBuild,
LambertSolversBuild) to snake_case.

Internal-only change — no Python-facing API impact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Pre-PR Verification

After all 6 tasks land on the feature branch, run the full pre-merge sequence per `CLAUDE.md`:

```bash
cd /home/ghecht/Projects/tycho

# 1. C++ unit tests
ctest --test-dir build --output-on-failure

# 2. Python examples (all 38 must pass)
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"

# 3. C++ brachistochrone (must converge to obj ≈ 1.8013 s)
cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
cd build && ninja -j6 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp

# 4. Benchmarks (no unexplained regressions)
cd /home/ghecht/Projects/tycho/build
cmake --preset linux-clang-release -DBUILD_CPP_BENCHMARKS=ON
ninja -j6 bench_all
cd /home/ghecht/Projects/tycho
bench/bench_track.sh compare

# 5. Static checks (zero hits expected)
grep -rn "_tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\|Kepler\)" src tychopy examples tests bench scripts | grep -v __pycache__
grep -rn "tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\)" examples tychopy/test | grep -v __pycache__
grep -rn "BuildOldLinkIterface\|build_new_link_interface" src
grep -rn "LinkConstraint\|LinkObjective\|LinkFlags" tychopy examples | grep -v __pycache__
grep -rn "TychoBind<.*>::Build\b" src/bindings
grep -rnE "\b(VectorFunctionBuild|SolversBuild|OptimalControlBuild|UtilsBuild|AstroBuild|ExtensionsBuild|KeplerUtilsBuild|LambertSolversBuild)\b" src/bindings
```

All gates must pass / return 0 hits before opening the PR.
