# Python Bindings Cleanup — Snake_Case Modules and Legacy OCP Link API Removal

**Date:** 2026-05-06
**Status:** Design approved, ready for implementation plan
**Author:** Grant Hecht (with Claude Opus 4.7)

> **Example count note:** This design was written before Task 4's
> `examples/python_examples/UpdatedInterface/` merge collapsed seven
> duplicate examples into the parent directory. References to "38
> Python examples" below were accurate at design-authoring time; the
> post-merge count is **32**.

## Motivation

The Python bindings for Tycho carry two layers of legacy state from the
upstream ASSET project:

1. **PascalCase submodule and file names.** The C++ binding submodules
   (`_tychopy.VectorFunctions`, `_tychopy.OptimalControl`, etc.) and the
   pure-Python wrapper packages mirror C++ class casing rather than the
   Pythonic `snake_case` convention. Python file names inside those
   packages (`AstroFrames.py`, `MeshErrorPlots.py`, `ODEBaseClass.py`,
   etc.) carry the same anti-pattern.
2. **Old vs. new OCP link interface.** `OptimalControlProblemBase`
   exposes two parallel APIs for `add_link_equal_con` /
   `add_link_inequal_con` / `add_link_objective` plus
   `add_forward_link_equal_con` / `add_direct_link_equal_con`:
   - The new interface (`build_new_link_interface()` in
     `optimal_control_problem_bind.cpp`) uses `PhasePack` tuples,
     `PhaseRefType` / `RegionType` / `VarIndexType` variants, and a
     first-class `auto_scale` argument.
   - The old interface (`BuildOldLinkIterface()`, ~40 overloads) uses
     parallel lists (`ptl`, `xtuvs`, `opvs`, `spvs`, `lpvs`) the user
     must keep aligned, an opaque `LinkFlags` enum that couples both
     phases at once, and explicit `LinkConstraint` / `LinkObjective`
     object construction.

   The new interface is strictly superior in every dimension audited
   (per-phase data grouping, per-phase region specification, variable
   names, auto-scaling consistency, overload count, helper-function
   factoring). All capabilities of the old interface are reachable
   from the new one.

Per the project's "no users — break API freely" policy and the
"prefer full principled fixes that remove the design smell"
guidance, this PR removes both layers of legacy state in one shot.

## Scope

One PR. Mechanical work, bundled because the pieces interact (legacy
OCP removal touches the same files we are renaming, and Python example
migration is gated on legacy removal).

1. Rename the 6 C++ binding submodules + the nested `Astro.Kepler`
   submodule to `snake_case` (hard break — no aliases, no deprecation
   period).
2. Rename the 6 pure-Python wrapper packages and their contained `.py`
   files to `snake_case`.
3. Drop duplicated PascalCase factory aliases in `tychopy/.../__init__.py`.
4. Full removal of the legacy OCP link API: Python bindings, C++ class
   overloads, C++ builder migration, C++ test migration.
5. Migrate Python examples off the old link API and merge
   `examples/python_examples/UpdatedInterface/` back into its parent.
6. Internal C++ binding helper rename (`TychoBind<T>::Build` → `build`,
   `VectorFunctionBuild`/etc. → `vector_functions_build`/etc.) for
   consistency with the project's "member functions and free functions
   are snake_case" convention.

Class names already conform (PascalCase) and are not touched. Enum
member casing keeps the existing C++ casing (out of scope).

## Naming Map

### C++ binding submodules

Touched in `function_registry.h:64-71`, `utils_bind.cpp:23`,
`astro/tycho_astro.cpp:31`, `astro/kepler_model.cpp:82`.

| Old | New |
|---|---|
| `_tychopy.VectorFunctions` | `_tychopy.vector_functions` |
| `_tychopy.OptimalControl` | `_tychopy.optimal_control` |
| `_tychopy.Solvers` | `_tychopy.solvers` |
| `_tychopy.Utils` | `_tychopy.utils` |
| `_tychopy.Astro` | `_tychopy.astro` |
| `_tychopy.Astro.Kepler` | `_tychopy.astro.kepler` |
| `_tychopy.Extensions` | `_tychopy.extensions` |

### Pure-Python subpackage directories

| Old | New |
|---|---|
| `tychopy/VectorFunctions/` | `tychopy/vector_functions/` |
| `tychopy/OptimalControl/` | `tychopy/optimal_control/` |
| `tychopy/Solvers/` | `tychopy/solvers/` |
| `tychopy/Utils/` | `tychopy/utils/` |
| `tychopy/Astro/` | `tychopy/astro/` |
| `tychopy/Extensions/` | `tychopy/extensions/` |

### Python file renames

`tychopy/astro/`:

| Old | New |
|---|---|
| `AstroFrames.py` | `astro_frames.py` |
| `AstroModels.py` | `astro_models.py` |
| `AstroConstraints.py` | `astro_constraints.py` |
| `Constants.py` | `constants.py` |
| `Date.py` | `date.py` |
| `DataReadWrite.py` | `data_read_write.py` |
| `FramePlot.py` | `frame_plot.py` |
| `SpiceRead.py` | `spice_read.py` |

`tychopy/optimal_control/`:

| Old | New |
|---|---|
| `MeshErrorPlots.py` | `mesh_error_plots.py` |
| `ODEBaseClass.py` | `ode_base_class.py` |

### Symbol drops

PascalCase aliases of snake_case factory functions (these are not
classes — they are factory functions whose canonical name is already
snake_case). Drop the duplicate alias; consumers use the snake_case
form.

`tychopy/vector_functions/__init__.py`:
- Drop `Stack`, `StackScalar`, `Sum`, `SumElems`, `SumScalar`. Canonical
  forms `stack`, `stack_scalar`, `sum`, `sum_elems`, `sum_scalar`
  remain.

`tychopy/astro/__init__.py`:
- Drop `ModifiedDynamics`. Canonical form `modified_dynamics` remains.

`tychopy/optimal_control/__init__.py` (also follows from §4 below):
- Drop `LinkConstraint`, `LinkObjective`, `LinkFlags` aliases (legacy
  OCP types that lose their public binding).

### Class names preserved

These already conform to the Pythonic convention (PascalCase for
classes) and are NOT touched: `OptimalControlProblem`, `PSIOPT`,
`Jet`, `LGLInterpTable`, `Arguments`, `Segment`, `Segment2`,
`Segment3`, `Element`, `ScalarFunction`, `VectorFunction`,
`PyScalarFunction`, `PyVectorFunction`, `ColMatrix`, `RowMatrix`,
`Comparative`, `Conditional`, `ConstantScalar`, `ConstantVector`,
`InterpFunction`, `InterpFunction_1/3/6`, `FiniteDiffTable`,
`ODEArguments`, `PhaseInterface`, `IVPAlg`, `StateConstraint`,
`StateObjective`, `OptimizationProblem`, `OptimizationProblemBase`,
all `*Modes`/`*Flags` enum types (`AlgorithmModes`, `BarrierModes`,
`ConvergenceFlags`, `JetJobModes`, `LineSearchModes`,
`PDStepStrategies`, `QPOrderingModes`, `QPPivotModes`, `ControlModes`,
`IntegralModes`, `PhaseRegionFlags`, `TranscriptionModes`),
`CR3BPFrame`, `ODEBase`, `PhaseMeshErrorPlot`.

### Internal C++ binding helpers (not Python-facing)

Per the project's CLAUDE.md convention (member functions and free
functions are `snake_case`), the internal binding helpers are also
renamed:

| Old | New |
|---|---|
| `TychoBind<T>::Build(...)` | `TychoBind<T>::build(...)` |
| `VectorFunctionBuild` | `vector_functions_build` |
| `SolversBuild` | `solvers_build` |
| `OptimalControlBuild` | `optimal_control_build` |
| `UtilsBuild` | `utils_build` |
| `AstroBuild` | `astro_build` |
| `ExtensionsBuild` | `extensions_build` |
| `BuildOldLinkIterface` | (deleted; see §4) |
| `build_new_link_interface` | `build_link_interface` |

These appear in `tycho_module.cpp` and the per-subsystem
`tycho_<subsystem>.cpp` aggregate files.

## Legacy OCP Link API Removal

### Python bindings

`src/bindings/optimal_control/optimal_control_problem_bind.cpp`:

- Delete `BuildOldLinkIterface()` (lines 315-625). All bindings inside
  it disappear: `add_link_equal_con(LinkConstraint)`, the parallel-list
  forms, the `LinkFlags`-pair `add_direct_link_equal_con`, and the
  `(int, int, VectorXi)` / `(PhasePtr, PhasePtr, VectorXi)` forms of
  `add_forward_link_equal_con`.
- Rename `build_new_link_interface()` → `build_link_interface()`. No
  more "old/new" distinction.
- Delete the call to `BuildOldLinkIterface(obj)` in the `Build()` (now
  `build()`) entry point.

`src/bindings/optimal_control/tycho_optimal_control.cpp:104-105`:

- Delete the `nb::class_<LinkConstraint>` and `nb::class_<LinkObjective>`
  registrations. These types stop being Python-visible.

### C++ class

`include/tycho/detail/optimal_control/phase/optimal_control_problem.h`:

Delete the following overloads of `add_link_equal_con` (and their
mirror `add_link_inequal_con` and `add_link_objective` overloads in
the same file):

- `add_link_equal_con(LinkConstraint)` (line 546).
- The 7-arg parallel-list overloads (lines 690-749): all 8 combinations
  of `RegVec`/`std::vector<std::string>`/`LinkFlags`/`std::string`
  region type × `std::vector<VectorXi>`/`std::vector<std::vector<PhasePtr>>`
  phase-list type.
- The 5-arg parallel-list overloads (lines 753-765).
- The 4-arg parallel-list overloads (lines 767-785).

Delete the legacy `add_forward_link_equal_con`:

- `add_forward_link_equal_con(int, int, VectorXi)` (line 802).
- `add_forward_link_equal_con(PhasePtr, PhasePtr, VectorXi)` (line 825).

Delete the legacy `add_direct_link_equal_con`:

- `add_direct_link_equal_con(LinkFlags, int, VectorXi, int, VectorXi)`
  (line 830).
- `add_direct_link_equal_con(VectorFunctionalX, int, PhaseRegionFlags,
  VectorXi, int, PhaseRegionFlags, VectorXi)`.
- `add_direct_link_equal_con(VectorFunctionalX, PhasePtr,
  PhaseRegionFlags, VectorXi, PhasePtr, PhaseRegionFlags, VectorXi)`.
- `add_direct_link_equal_con(VectorFunctionalX, int, std::string,
  VectorXi, int, std::string, VectorXi)`.
- `add_direct_link_equal_con(VectorFunctionalX, PhasePtr, std::string,
  VectorXi, PhasePtr, std::string, VectorXi)`.

`LinkConstraint` and `LinkObjective` types remain as internal
implementation detail — `make_func_impl<LinkConstraint, ...>` is still
used by the surviving new-API overloads. They are simply no longer
bound to Python.

`LinkFlags` enum: delete only if no in-tree caller remains after the
overload deletions above. If `make_func_impl<LinkConstraint, ...>` or
the surviving overloads still reference any `LinkFlags::*` value
internally, the enum stays in the C++ surface but is no longer Python
bound.

### C++ builder migration

`include/tycho/detail/optimal_control/builder/ocp_wrapper.h`:

- Migrate `add_forward_link_equal_con(Phase&, Phase&, ...)` overloads
  (lines 53, 89) to call the new
  `add_forward_link_equal_con(PhaseRefType, PhaseRefType, VarIndexType,
  ScaleType)` form. The wrapper signatures remain the same; only the
  delegated call changes. Pass `std::string("auto")` (the canonical
  default for the `ScaleType` variant — matches the Python binding
  default `nb::arg("auto_scale").none() = std::string("auto")`).
- Migrate the four `add_direct_link_equal_con` overloads (lines 94,
  101, 109, 121) to the new
  `(PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType,
  VarIndexType, ScaleType)` form. Same pattern: wrapper signatures
  unchanged, delegated call changes.

### C++ test migration

- `tests/cpp/optimal_control/test_ocp_direct_link.cpp` — extend the
  existing 6-arg `add_direct_link_equal_con` calls with a `ScaleType`
  argument. ~5 sites; mechanical.
- `tests/cpp/optimal_control/test_ocp_phase_lifetime.cpp` — same
  pattern. ~3 sites.
- `tests/cpp/optimal_control/test_builder_brachistochrone.cpp` — uses
  the wrapper API; passes through unchanged after wrapper migration.

### Capability preservation audit

Every legacy Python overload that was *exercised* by any in-tree caller
(tests, examples, benches, extensions) maps to a concrete new-API call.
This table is the receipt:

| Legacy form | New equivalent |
|---|---|
| `add_link_equal_con(LinkConstraint)` | Any `add_link_equal_con(func, ...)` form constructs `LinkConstraint` internally |
| `add_link_equal_con(func, RegVec/list[str]/LinkFlags/str, ptl, xtuvs, opvs, spvs, lpvs)` | `add_link_equal_con(func, packs=[(phase, region, xt_u_vars, op_vars, sp_vars), ...], linkparams)` |
| `add_link_equal_con(func, regs, vec<vec<PhasePtr>>, ...)` | Same — `PhaseRefType` is `variant<int, PhasePtr, string>` |
| `add_link_equal_con(func, regs, ptl, xtuvs, lpvs)` (5-arg, no opvs/spvs) | New 12-arg `add_link_equal_con(func, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, ...)` with empty `OPV/SPV` |
| `add_link_equal_con(func, LinkFlags/str, ptl, xtuvs)` (4-arg) | New 9-arg `add_link_equal_con(func, p0, reg0, v0, p1, reg1, v1, lps, scale)` |
| `add_forward_link_equal_con(int, int, VectorXi)` and `(PhasePtr, ...)` | `add_forward_link_equal_con(PhaseRefType, PhaseRefType, VarIndexType, ScaleType)` |
| `add_direct_link_equal_con(LinkFlags, int, VectorXi, int, VectorXi)` | `add_direct_link_equal_con(p0, reg0, v0, p1, reg1, v1, scale)` (caller splits the symmetric `LinkFlags` into two regions) — see footnote |
| `add_direct_link_equal_con(func, int/PhasePtr, PhaseRegionFlags/str, VectorXi, ...)` | Subsumed by `add_link_equal_con(func, p0, reg0, v0, p1, reg1, v1, ...)` |

**Footnote on `LinkFlags` enum coverage.** Three legacy `LinkFlags` values
warrant individual mention because they fall outside the symmetric
"split into two `PhaseRegionFlags`" pattern:

- `LinkFlags::LinkParams` is preserved. The new-API `add_link_param_*_con`
  family (which still constructs `LinkConstraint(..., LinkFlags::LinkParams,
  ...)` directly at `optimal_control_problem.h` `add_link_param_equal_con`
  / `add_link_param_inequal_con` / `add_link_param_objective`) carries
  the semantics; `link_function.h`'s `make_phase_reg_flags()` switch
  resizes `RegFlags` to 0 for this case (line ~165). Callers do not
  construct it explicitly via the new public API.
- `LinkFlags::BackTwoToTwoFront` and `LinkFlags::FrontTwoToTwoBack` are
  **dead enum values on both `main` and HEAD**. They have no
  `make_phase_reg_flags()` cases in `link_function.h` (they hit the
  `default:` empty branch, producing an empty `RegFlags`) and are not
  referenced by any test, example, benchmark, or extension. The new
  `RegionType = variant<int, PhaseRegionFlags, std::string>` cannot
  express their intended "two contiguous points" semantics, so the
  capability gap is real but theoretical: no in-tree caller exercises it.
  These enum values are candidates for deletion in the same follow-up
  PR that prunes the dead 2-arg `add_link_param_*` wrappers.

## Examples Folder Reorganisation

After legacy removal, `examples/python_examples/UpdatedInterface/`
(which holds new-API rewrites of files that exist in old-API form in
the parent) becomes redundant.

Plan:

- For each file in `UpdatedInterface/`, replace the same-named parent
  file with it. Files involved: `Delta3Launch.py`, `GoddardRocket.py`,
  `MinimumTimeToClimb.py`, `MinimumTimeToClimbTables.py`,
  `MultiPhaseCannon.py`, `Reentry.py`.
- Migrate parent-only old-API examples inline to the new API:
  - `examples/python_examples/MultiSpacecraftOptimization.py`
  - `examples/python_examples/Heteroclinic.py`
  - `examples/python_examples/MultiPhaseZermelo.py`
  - `examples/python_examples/MeshRefinement/Delta3Launch.py`
  - `tychopy/test/test_AdaptiveMesh/test_Delta3Launch.py`
  - `tychopy/test/test_AutoScaling/test_ObjScaling.py`
  - `tychopy/test/test_FullProblems/test_MultiPhaseCannon.py`
  - `tychopy/test/test_FullProblems/test_Delta3Launch.py`

  Each calls `add_forward_link_equal_con` with int/list args — already
  supported by the new API; the only change is the new mandatory
  `ScaleType` arg (use `'auto'` as the canonical default).
- Delete `examples/python_examples/UpdatedInterface/` after merging.
- Update all import statements throughout examples and tests to the
  new snake_case module paths.

## Verification

Pre-merge sequence per CLAUDE.md, plus extras specific to this work:

1. **C++ unit tests** — `ctest --output-on-failure`. Must pass after
   wrapper + test migration.
2. **All 38 Python examples** — `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"`.
   Must exit 0 for all 38. This is the primary proof that snake_case
   rename + legacy removal + folder reorg preserves all example
   functionality.
3. **C++ brachistochrone** — must converge, objective ≈ 1.8013 s.
4. **Benchmarks** — `bench/bench_track.sh compare` shows no
   unexplained regressions. This work is not on the perf hot path;
   noise-level variation expected.
5. **Static checks** (no occurrences in working tree after the rename):
   - `grep -rn "_tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\)" src tychopy examples tests bench` → 0 hits.
   - `grep -rn "tychopy\.\(VectorFunctions\|OptimalControl\|Solvers\|Utils\|Astro\|Extensions\)" examples tychopy/test` → 0 hits.
   - `grep -rn "BuildOldLinkIterface\|build_new_link_interface" src` → 0 hits.
   - `grep -rn "LinkFlags\|LinkConstraint\|LinkObjective" tychopy examples` → 0 hits (the three types disappear from Python).
6. **Builder API smoke** — `cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON && ninja -j6 && ./examples/cpp_examples/builder/multi_phase_cannon/multi_phase_cannon` runs to completion.

## Out of Scope

- Enum member casing (`LinkFlags.BackToFront` → `BACK_TO_FRONT`). Out
  of scope per Q2 — the existing casing matches C++ and is not strictly
  un-Pythonic.
- Renaming binding C++ source files (e.g. `vf/` → `vector_functions/`
  for the binding subdirectory). Out of scope — the directory is
  internal-only and the cost of renaming exceeds the consistency win.
- Public API of `OptimalControlProblem` beyond the link interface (e.g.
  `add_state_objective`, mesh APIs, etc.). Out of scope — those are
  not the documented design smell.
- Submodule docstrings (currently say things like "SubModule
  Containing..."). Cosmetic; can be improved opportunistically but is
  not required.
- Nested `Extensions/` subpackages — `tychopy/astro/Extensions/`,
  `tychopy/optimal_control/Extensions/`, `tychopy/solvers/Extensions/`,
  `tychopy/utils/Extensions/`, `tychopy/vector_functions/Extensions/`
  remain PascalCase. Internal callers in
  `tychopy/astro/{astro_frames.py,astro_models.py}` and
  `tychopy/vector_functions/__init__.py` still use
  `from .Extensions.X import …`. Deferred to a follow-up PR alongside
  the sub-aggregate-helper rename so the snake_case sweep is one
  coherent change rather than spread across multiple PRs. The
  top-level `tychopy/Extensions/` (now `tychopy/extensions/`) WAS
  renamed in this PR — only the nested per-submodule `Extensions/`
  directories are deferred.

## Risks

- **Import breakage in third-party scripts.** Per CLAUDE.md memory
  (`feedback_break_api_freely.md`), the project has no external users
  and breaks API freely. Mitigated by making the rename mechanical and
  catchable: `ImportError: No module named 'tychopy.OptimalControl'`
  is unambiguous.
- **C++ builder regression.** The builder API migration is mechanical
  but touches a heavily-used internal interface. Mitigated by the C++
  test suite (already covers the relevant builder paths).
- **Hidden capability gap.** A legacy form not represented in the
  audit table above could surface as a missing capability. Mitigated
  by running all 38 Python examples + C++ tests; any gap surfaces as a
  failure.
