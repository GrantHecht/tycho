# PSIOPT Readability Refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor PSIOPT from a public struct with ~60 fields to a proper class with encapsulated state, grouped configuration/results, deduplicated code, and no algorithm changes.

**Architecture:** Introduce nested `Settings`, `SolveResult`, and `KKTVector` types. Deduplicate 5 entry points via `run_phase_sequence` and 3 line search variants via shared helpers. Extract printing to a separate TU. Convert `struct` to `class` with private fields and public accessors. Update Python bindings to use property-backed access.

**Tech Stack:** C++20, Eigen, nanobind, CMake/Ninja, fmt

**Spec:** `docs/superpowers/specs/2026-04-04-psiopt-refactor-design.md`

**Build command:** `cd build && ninja -j4 all` (from project root with `conda activate tycho`)

**Test commands:**
- C++ unit tests: `cd build && ctest --output-on-failure`
- C++ brachistochrone: `./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp`
- Python examples: `conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"`
- Benchmarks: `bench/bench_track.sh compare`

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `include/tycho/detail/solvers/psiopt_fwd.h` | Create | Forward decls + `ConvergenceFlags` enum |
| `include/tycho/detail/solvers/psiopt.h` | Rewrite | PSIOPT class with nested Settings/SolveResult/KKTVector |
| `src/solvers/psiopt.cpp` | Rewrite | Algorithm core, setters, barrier math, NLP eval wrappers |
| `src/solvers/psiopt_print.cpp` | Create | Printing methods (same PSIOPT class, separate TU) |
| `src/bindings/solvers/psiopt_bind.cpp` | Modify | `def_rw` → `def_prop_rw`/`def_prop_ro`, remove dead bindings |
| `src/CMakeLists.txt` | Modify | Add `solvers/psiopt_print.cpp` to `psiopt` target |
| `include/tycho/solvers.h` | Modify | Add `#include "tycho/detail/solvers/psiopt_fwd.h"` |
| `include/tycho/detail/solvers/optimization_problem.h` | Modify | Update `optimizer_->` field accesses to use `result()` / `settings()` |
| `include/tycho/detail/solvers/optimization_problem_base.h` | Modify | Update `optimizer_->qp_threads_` to use `settings()` |
| `include/tycho/detail/optimal_control/phase/ode_phase_base.h` | Modify | Update `optimizer_->print_level_` to use `set_print_level()` |
| `include/tycho/detail/optimal_control/phase/optimal_control_problem.h` | Modify | Update `optimizer_->print_level_` to use `set_print_level()` |
| `src/optimal_control/optimal_control_problem.cpp` | Modify | Update `optimizer_->last_*` / `converge_flag_` to use `result()` |
| `src/optimal_control/ode_phase_base.cpp` | Modify | Update `optimizer_->last_*` / `converge_flag_` to use `result()` |
| `tests/cpp/solvers/solver_test_utils.h` | Modify | Update `optimizer_->print_level_` to `set_print_level()` |
| `tests/cpp/solvers/test_psiopt_convergence.cpp` | Modify | Update direct field access to use setters/getters |
| `tests/cpp/optimal_control/test_ocp_multiphase.cpp` | Modify | Update `optimizer_->print_level_` |
| `tests/cpp/optimal_control/test_mesh_refinement.cpp` | Modify | Update `optimizer_->print_level_` |
| `bench/cpp/bench_common.h` | Modify | Update `optimizer_->print_level_` |
| `bench/legacy/brachistochrone/bench.cpp` | Modify | Update `optimizer_->print_level_` |
| `examples/cpp_examples/static/zermelo/main.cpp` | Modify | Already uses setters, no change needed |
| `examples/cpp_examples/static/delta3_launch/main.cpp` | Modify | Already uses setters (lines 404-407), no change needed |
| `examples/cpp_examples/static/hypersens/main.cpp` | Modify | Update `optimizer_->print_level_` (line 118) |

---

## External Caller Migration Reference

All direct PSIOPT member accesses from outside the class, and how they map to the new API:

| Old access | New access | Locations |
|---|---|---|
| `optimizer_->print_level_ = N` | `optimizer_->set_print_level(N)` | 11 locations across tests, benchmarks, examples, OC headers |
| `optimizer_->qp_threads_ = N` | `optimizer_->settings().qp_threads_ = N` | `optimization_problem_base.h:77,85` |
| `optimizer_->last_eq_lmults_` | `optimizer_->result().eq_lmults_` | `optimization_problem.h` (5x), `ode_phase_base.cpp`, `optimal_control_problem.cpp` |
| `optimizer_->last_iq_lmults_` | `optimizer_->result().iq_lmults_` | Same locations as eq_lmults |
| `optimizer_->last_eq_cons_` | `optimizer_->result().eq_cons_` | `ode_phase_base.cpp`, `optimal_control_problem.cpp` |
| `optimizer_->last_iq_cons_` | `optimizer_->result().iq_cons_` | Same locations as eq_cons |
| `optimizer_->converge_flag_` | `optimizer_->result().converge_flag_` | `optimization_problem.h` (5x), `ode_phase_base.cpp`, `optimal_control_problem.cpp` |

---

### Task 1: Dead Code Removal

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`

Remove all dead code identified in the spec. This reduces noise for subsequent tasks.

- [ ] **Step 1: Remove dead fields and enum values from `psiopt.h`**

  Remove the following from `include/tycho/detail/solvers/psiopt.h`:
  - `diagnostic_` field (line 198)
  - `store_sp_mat_` field (line 393)
  - `spmat` field (line 394)
  - `ex_obj_val_` field (line 320)
  - `max_cpu_time_` field (line 252)
  - `min_ls_step_` field (line 342)
  - `print_matrixinfo()` declaration (line 712 area)
  - `get_sp_mat()` and `get_sp_mat2()` declarations (lines 470-471)
  - `BarrierModes::FIACCO` and `BarrierModes::BARDISABLED` from the enum (line 75)
  - `LineSearchModes::L2` from the enum (line 76)

- [ ] **Step 2: Remove dead code from `psiopt.cpp`**

  Remove:
  - Empty `if (diagnostic_) {}` block (around line 636-637)
  - Empty `print_matrixinfo()` definition (line 255)
  - `wide` variable in `print_last_iterate` (line 285)
  - `tst` variable in `print_last_iterate` (line 301)
  - `BarrierModes::FIACCO` empty case in `alg_impl` (around line 618-619)
  - The `if (store_sp_mat_) spmat = ...` line in `set_nlp` (around line 72-73)

- [ ] **Step 3: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```
  Expected: all tests pass, no behavioral change.

- [ ] **Step 4: Commit**

  ```bash
  git add include/tycho/detail/solvers/psiopt.h src/solvers/psiopt.cpp
  git commit -m "refactor(psiopt): remove dead code — fields, empty methods, unused enum values"
  ```

---

### Task 2: Create `psiopt_fwd.h`

**Files:**
- Create: `include/tycho/detail/solvers/psiopt_fwd.h`
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `include/tycho/solvers.h`

Extract `ConvergenceFlags` enum and its comparison operator into a lightweight forward-declaration header.

- [ ] **Step 1: Create `psiopt_fwd.h`**

  Create `include/tycho/detail/solvers/psiopt_fwd.h` containing:
  - `#pragma once`
  - Copyright header (same as psiopt.h, updated modification note)
  - `namespace tycho { ... }` with the `ConvergenceFlags` enum and `operator<=>`
  - Forward declaration: `namespace tycho::solvers { class PSIOPT; }`

- [ ] **Step 2: Update `psiopt.h` to include `psiopt_fwd.h`**

  Replace the inline `ConvergenceFlags` enum definition in `psiopt.h` with `#include "tycho/detail/solvers/psiopt_fwd.h"`. Remove the duplicated enum and operator.

- [ ] **Step 3: Add include to `solvers.h`**

  Add `#include "tycho/detail/solvers/psiopt_fwd.h"` to `include/tycho/solvers.h` (before the `psiopt.h` include).

- [ ] **Step 4: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add include/tycho/detail/solvers/psiopt_fwd.h include/tycho/detail/solvers/psiopt.h include/tycho/solvers.h
  git commit -m "refactor(psiopt): extract ConvergenceFlags to psiopt_fwd.h"
  ```

---

### Task 3: Extract Printing to `psiopt_print.cpp`

**Files:**
- Create: `src/solvers/psiopt_print.cpp`
- Modify: `src/solvers/psiopt.cpp`
- Modify: `src/CMakeLists.txt`

Move all printing/formatting methods to a separate TU. Methods remain on PSIOPT — this is a build-organization split only.

- [ ] **Step 1: Create `psiopt_print.cpp`**

  Create `src/solvers/psiopt_print.cpp` with:
  - Copyright header
  - `#include "tycho/detail/solvers/psiopt.h"`
  - Move these function definitions from `psiopt.cpp`:
    - `print_psiopt()`
    - `print_timing_summary()`
    - `print_settings()`
    - `print_stats()`
    - `print_last_iterate()`
    - `print_beginning()`
    - `print_finished()`
    - `print_exit_stats()`
    - `calculate_color()`

- [ ] **Step 2: Remove moved functions from `psiopt.cpp`**

  Delete the function definitions that were moved to `psiopt_print.cpp`. Keep only the `#include` and algorithm code.

- [ ] **Step 3: Update `src/CMakeLists.txt`**

  Add `solvers/psiopt_print.cpp` to the `psiopt` static library target:
  ```cmake
  add_library(psiopt STATIC
      solvers/psiopt.cpp
      solvers/psiopt_print.cpp
      solvers/non_linear_program.cpp
      solvers/optimization_problem.cpp
      solvers/solver_init.cpp
  )
  ```

- [ ] **Step 4: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/solvers/psiopt_print.cpp src/solvers/psiopt.cpp src/CMakeLists.txt
  git commit -m "refactor(psiopt): extract printing methods to psiopt_print.cpp"
  ```

---

### Task 4: Introduce `Settings` Struct + Migrate Config Fields

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`
- Modify: `src/solvers/psiopt_print.cpp`
- Modify: `include/tycho/detail/solvers/optimization_problem_base.h`
- Modify: `include/tycho/detail/solvers/optimization_problem.h`
- Modify: `include/tycho/detail/optimal_control/phase/ode_phase_base.h`
- Modify: `include/tycho/detail/optimal_control/phase/optimal_control_problem.h`
- Modify: `tests/cpp/solvers/solver_test_utils.h`
- Modify: `tests/cpp/solvers/test_psiopt_convergence.cpp`
- Modify: `tests/cpp/optimal_control/test_ocp_multiphase.cpp`
- Modify: `tests/cpp/optimal_control/test_mesh_refinement.cpp`
- Modify: `bench/cpp/bench_common.h`
- Modify: `bench/legacy/brachistochrone/bench.cpp`
- Modify: `examples/cpp_examples/static/hypersens/main.cpp`

Add `Settings` as a nested struct within PSIOPT. Move all ~40 configuration fields into it. Update all internal references from `this->field_` to `settings_.field_`. Add `settings()` accessor. Update ALL external callers that directly access config fields (`print_level_`, `qp_threads_`).

- [ ] **Step 1: Add `Settings` struct definition to `psiopt.h`**

  Add the `PSIOPT::Settings` struct as defined in the spec (all fields with trailing underscores, grouped by section). Place it after the enum declarations, before the existing field declarations.

- [ ] **Step 2: Replace individual config fields with `Settings settings_`**

  Remove all ~40 individual config field declarations from PSIOPT and replace with a single `Settings settings_;` member. Add inline accessor:
  ```cpp
  Settings& settings() { return settings_; }
  const Settings& settings() const { return settings_; }
  ```

- [ ] **Step 3: Update all internal references in `psiopt.h`**

  Update all inline method bodies and setter methods in the header:
  - `this->kkt_tol_` → `this->settings_.kkt_tol_`
  - `this->max_iters_` → `this->settings_.max_iters_`
  - etc. for all ~40 fields

  Update setter methods (e.g., `set_kkt_tol`) to write to `settings_.kkt_tol_`.

- [ ] **Step 4: Update all internal references in `psiopt.cpp`**

  Global find-and-replace within `psiopt.cpp` for each config field:
  - `this->kkt_tol_` → `this->settings_.kkt_tol_`
  - `this->max_iters_` → `this->settings_.max_iters_`
  - etc. Also update bare references without `this->` (e.g., `econ_tol_` → `settings_.econ_tol_`).

- [ ] **Step 5: Update all internal references in `psiopt_print.cpp`**

  Same replacement as Step 4 for the printing code.

- [ ] **Step 6: Update external callers — `qp_threads_`**

  In `include/tycho/detail/solvers/optimization_problem_base.h`:
  - Line 77: `this->optimizer_->qp_threads_ = ...` → `this->optimizer_->settings().qp_threads_ = ...`
  - Line 85: same pattern

- [ ] **Step 7: Update external callers — `print_level_` (all 16 locations)**

  All direct writes to `optimizer_->print_level_` must change to `optimizer_->set_print_level(N)` (setter already exists and will write to `settings_.print_level_`):
  - `tests/cpp/solvers/solver_test_utils.h:37`
  - `tests/cpp/solvers/test_psiopt_convergence.cpp:40`
  - `tests/cpp/optimal_control/test_ocp_multiphase.cpp:24,38`
  - `tests/cpp/optimal_control/test_mesh_refinement.cpp:13,25`
  - `bench/cpp/bench_common.h:149,211,377`
  - `bench/legacy/brachistochrone/bench.cpp:95`
  - `examples/cpp_examples/static/hypersens/main.cpp:118`
  - `include/tycho/detail/solvers/optimization_problem.h:151,157`
  - `include/tycho/detail/optimal_control/phase/ode_phase_base.h:1176,1185`
  - `include/tycho/detail/optimal_control/phase/optimal_control_problem.h:1570,1577`

- [ ] **Step 8: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 9: Commit**

  ```bash
  git add -A
  git commit -m "refactor(psiopt): introduce Settings struct and migrate config fields"
  ```

---

### Task 5: Introduce `SolveResult` Struct + Migrate Result Fields

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`
- Modify: `src/solvers/psiopt_print.cpp`
- Modify: `include/tycho/detail/solvers/optimization_problem.h`
- Modify: `src/optimal_control/optimal_control_problem.cpp`
- Modify: `src/optimal_control/ode_phase_base.cpp`

Add `SolveResult` as a nested struct. Move timing, convergence, multiplier, and constraint fields into it. Update all internal and external references.

- [ ] **Step 1: Add `SolveResult` struct definition to `psiopt.h`**

  Add the `PSIOPT::SolveResult` struct as defined in the spec (primals_, iter_num_, obj_val_, converge_flag_, multipliers, timing, factor stats, zero_timing()). Place it after `Settings`.

- [ ] **Step 2: Replace individual result fields with `SolveResult result_`**

  Remove individual result field declarations:
  - `converge_flag_`, `last_obj_val_`, `last_total_time_`, `last_pre_time_`, `last_misc_time_`, `last_func_time_`, `last_kkt_time_`, `last_print_time_`, `last_solver_init_time_`, `last_iter_num_`, `factor_mem_`, `factor_flops_`, `last_eq_lmults_`, `last_iq_lmults_`, `last_eq_cons_`, `last_iq_cons_`

  Replace with `SolveResult result_;` and add accessor:
  ```cpp
  const SolveResult& result() const { return result_; }
  ```
  Replace `zero_timing_stats()` with calls to `result_.zero_timing()`.
  Update `get_convergence_flag()` inline body: `return this->converge_flag_` → `return result_.converge_flag_`.

- [ ] **Step 3: Update internal references in `psiopt.cpp`**

  - `this->converge_flag_` → `this->result_.converge_flag_`
  - `this->last_obj_val_` → `this->result_.obj_val_`
  - `this->last_total_time_` → `this->result_.total_time_`
  - `this->last_eq_lmults_` → `this->result_.eq_lmults_`
  - etc. for all result fields
  - `this->zero_timing_stats()` → `this->result_.zero_timing()`

- [ ] **Step 4: Update internal references in `psiopt_print.cpp`**

  Same replacements as Step 3 for the printing methods.

- [ ] **Step 5: Update external callers — `optimization_problem.h`**

  In `include/tycho/detail/solvers/optimization_problem.h`, update all 5 solve method bodies:
  - `this->optimizer_->last_eq_lmults_` → `this->optimizer_->result().eq_lmults_`
  - `this->optimizer_->last_iq_lmults_` → `this->optimizer_->result().iq_lmults_`
  - `this->optimizer_->converge_flag_` → `this->optimizer_->result().converge_flag_`

- [ ] **Step 6: Update external callers — `optimal_control_problem.cpp`**

  In `src/optimal_control/optimal_control_problem.cpp`:
  - `this->optimizer_->last_eq_cons_` → `this->optimizer_->result().eq_cons_`
  - `this->optimizer_->last_eq_lmults_` → `this->optimizer_->result().eq_lmults_`
  - `this->optimizer_->last_iq_cons_` → `this->optimizer_->result().iq_cons_`
  - `this->optimizer_->last_iq_lmults_` → `this->optimizer_->result().iq_lmults_`
  - `this->optimizer_->converge_flag_` → `this->optimizer_->result().converge_flag_`

- [ ] **Step 7: Update external callers — `ode_phase_base.cpp`**

  In `src/optimal_control/ode_phase_base.cpp`:
  - Same replacements as Step 6.

- [ ] **Step 8: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 9: Commit**

  ```bash
  git add include/tycho/detail/solvers/psiopt.h src/solvers/psiopt.cpp src/solvers/psiopt_print.cpp \
      include/tycho/detail/solvers/optimization_problem.h \
      src/optimal_control/optimal_control_problem.cpp src/optimal_control/ode_phase_base.cpp
  git commit -m "refactor(psiopt): introduce SolveResult struct and migrate result fields"
  ```

---

### Task 6: Introduce `KKTVector` + Replace `get_*` Helpers

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`

Add `KKTVector` class. Refactor `alg_impl`, `init_impl`, `ls_impl`, and helper methods to use `KKTVector` instead of the 14 `get_*` methods. Remove old `get_*` helpers.

- [ ] **Step 1: Add `KKTVector` class to `psiopt.h`**

  Add `PSIOPT::KKTVector` as defined in the spec — non-owning view with `primals()`, `slacks()`, `eq_lmults()`, `iq_lmults()`, `prim_grad()`, `dual_grad()`, `eq_cons()`, `iq_cons()`, `all_cons()`, etc. All inline, returning Eigen segment expressions.

- [ ] **Step 2: Refactor `alg_impl` to use `KKTVector`**

  At the top of `alg_impl`, create `KKTVector` views:
  ```cpp
  KKTVector xsl(XSL, primal_vars_, slack_vars_, equal_cons_, inequal_cons_);
  KKTVector rhs(RHS, primal_vars_, slack_vars_, equal_cons_, inequal_cons_);
  KKTVector dxsl(DXSL, primal_vars_, slack_vars_, equal_cons_, inequal_cons_);
  ```
  Replace all `this->get_slacks(XSL)` → `xsl.slacks()`, `this->get_eq_lmults(XSL)` → `xsl.eq_lmults()`, etc. throughout the function.

- [ ] **Step 3: Refactor `init_impl` to use `KKTVector`**

  Same pattern as Step 2.

- [ ] **Step 4: Refactor `ls_impl` to use `KKTVector`**

  Same pattern. Create views for XSL, DXSL, XSL2, RHS, RHS2.

- [ ] **Step 5: Refactor barrier/step helper methods**

  Update `max_primal_dual_step` and `fill_iter_info` to accept `KKTVector&` parameters — they are called from `alg_impl` which already has views, so pass them through. For `eval_nlp` and the `eval_kkt`/`eval_kkt_no`/`eval_aug`/`eval_soe`/`eval_rhs` wrappers, create local `KKTVector` views inside each method from their existing `EigenRef<VectorXd>` parameters — these methods have their own parameter names and the overhead of a view construction is negligible.

- [ ] **Step 6: Remove old `get_*` helper methods from `psiopt.h`**

  Remove: `get_primals`, `get_slacks`, `get_primals_slacks`, `get_lmults`, `get_eq_lmults`, `get_iq_lmults`, `get_prim_grad`, `get_dual_grad`, `get_prim_dual_grad`, `get_eq_cons`, `get_iq_cons`, `get_all_cons`.

- [ ] **Step 7: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add include/tycho/detail/solvers/psiopt.h src/solvers/psiopt.cpp
  git commit -m "refactor(psiopt): introduce KKTVector view, replace get_* helpers"
  ```

---

### Task 7: Entry Point Deduplication

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`

Implement `PhaseStep` struct and `run_phase_sequence` method. Rewrite 5 entry points as thin wrappers.

- [ ] **Step 1: Add `PhaseStep` struct and `run_phase_sequence` declaration to `psiopt.h`**

  Add the private `PhaseStep` struct and `run_phase_sequence` declaration as specified in the design.

- [ ] **Step 2: Implement `run_phase_sequence` in `psiopt.cpp`**

  Implement the shared scaffolding:
  1. `result_.zero_timing()`, conditionally print stats, `ensure_solver_initialized()`, start timer
  2. `analyze_kkt_matrix()`, `init_impl()`
  3. For each step: if conditional and already converged, skip. Otherwise `print_beginning(label)`, `alg_impl()`, `print_finished(label)`. Between steps: extract primals, re-run `init_impl`.
  4. Stop timer, compute misc time, print timing summary, populate `result_`, return primals.

  **Critical:** Carefully trace each current entry point to ensure the scaffolding logic matches exactly. Pay attention to:
  - `solve_optimize_solve`'s conditional third phase
  - `optimize_solve`'s conditional second phase
  - Print level checks (`settings_.print_level_ < 2`)
  - `result_.obj_val_` is set by `alg_impl` internally — do NOT duplicate this in `run_phase_sequence`. Just ensure multi-phase sequences properly accumulate timings across `alg_impl` calls.
  - **Visible output change (spec-approved):** `run_phase_sequence` uniformly prints `print_beginning`/`print_finished` around every phase step. The current code has inconsistent printing (e.g., `solve_optimize` prints `print_finished` after SOE but not after OPT). The unified behavior is an intentional improvement.

- [ ] **Step 3: Rewrite 5 entry points as thin wrappers**

  Replace the body of `optimize`, `solve`, `solve_optimize`, `optimize_solve`, `solve_optimize_solve` with single calls to `run_phase_sequence` with the appropriate `PhaseStep` list (see spec table).

- [ ] **Step 4: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 5: Run C++ brachistochrone to verify convergence**

  ```bash
  ./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
  ```
  Expected: "Optimal Solution Found", objective ~ 1.8013 s.

- [ ] **Step 6: Commit**

  ```bash
  git add include/tycho/detail/solvers/psiopt.h src/solvers/psiopt.cpp
  git commit -m "refactor(psiopt): deduplicate entry points via run_phase_sequence"
  ```

---

### Task 8: Line Search Deduplication

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`

Split `ls_impl` into a dispatcher + 3 private methods + shared helpers.

- [ ] **Step 1: Add helper declarations to `psiopt.h`**

  Add private declarations:
  - `double ls_lang(...)`
  - `double ls_l1(...)`
  - `double ls_auglang(...)`
  - `double eval_trial_point(...)` — takes trial alpha, XSL, DXSL, evaluates NLP at trial point, returns barrier objective, fills XSL2 and RHS2
  - `struct PenaltyTerms { double l1, l2, linf; };`
  - `PenaltyTerms compute_penalties(...)` — computes penalty norms from constraint residuals
  - `bool secondary_accept(...)` — checks `(ptest < prim_obj) && (test_l2 < init_l2 || test_linf < init_linf)`

- [ ] **Step 2: Extract shared helpers in `psiopt.cpp`**

  Implement `eval_trial_point`, `compute_penalties`, `secondary_accept` by extracting the common code from the three line search variants.

- [ ] **Step 3: Implement `ls_lang`, `ls_l1`, `ls_auglang`**

  Rewrite each as a self-contained method calling the shared helpers. Each should be ~20-30 lines.

  **Critical:** Preserve exact numerical behavior:
  - LANG: merit = obj + barrier + Lagrangian. No secondary acceptance.
  - L1: `sc` bias = 0.1. Uses `RHS.head(pv + sv).dot(DXSL.head(pv + sv))` for `vv`. Simple L1 penalty via `cwiseAbs().dot()`.
  - AUGLANG: `sc` bias = 0.01. Uses `prim_dual_grad().dot(primals_slacks())` for `vv`. Tolerance-filtered L1 penalty. Zeros out L2 when within tolerance.
  - LANG sets `citer.ls_iters_ = j` (not `j+1` on rejection). L1/AUGLANG set `citer.ls_iters_ = j+1` on rejection.

- [ ] **Step 4: Rewrite `ls_impl` as dispatcher**

  ```cpp
  switch (lsmode) {
  case LineSearchModes::LANG:    return ls_lang(...);
  case LineSearchModes::L1:      return ls_l1(...);
  case LineSearchModes::AUGLANG: return ls_auglang(...);
  case LineSearchModes::NOLS:    citer.ls_iters_ = 0; return 1.0;
  default: throw std::invalid_argument("Unknown LineSearchMode");
  }
  ```

- [ ] **Step 5: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 6: Run C++ brachistochrone**

  ```bash
  ./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
  ```
  Expected: identical convergence, objective ~ 1.8013 s.

- [ ] **Step 7: Commit**

  ```bash
  git add include/tycho/detail/solvers/psiopt.h src/solvers/psiopt.cpp
  git commit -m "refactor(psiopt): deduplicate line search into ls_lang/ls_l1/ls_auglang + helpers"
  ```

---

### Task 9: `struct` → `class` + Move Implementations to `.cpp`

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h`
- Modify: `src/solvers/psiopt.cpp`
- Modify: All external caller files listed in the File Map

Convert PSIOPT from `struct` to `class`. Make fields private. Add public API surface. Move non-trivial inline method bodies to `.cpp`. Update all remaining external callers.

- [ ] **Step 1: Convert `struct PSIOPT` to `class PSIOPT`**

  Change `struct PSIOPT {` to `class PSIOPT {` and add `public:` / `private:` sections as specified in the class shape from the spec.

- [ ] **Step 2: Move non-trivial inline methods from header to `.cpp`**

  Move these method bodies from `psiopt.h` to `psiopt.cpp`, leaving only declarations in the header:
  - String-to-enum converters: `strto_OrderingMode`, `strto_LineSearchMode`, `strto_BarrierMode`, `strto_BestCriteriaMode`
  - All validated setters: `set_max_iters`, `set_max_acc_iters`, `set_max_ls_iters`, `set_all_max_iters`, `set_kkt_tol`, `set_bar_tol`, `set_econ_tol`, `set_icon_tol`, `set_tols`, `set_acc_kkt_tol`, `set_acc_bar_tol`, `set_acc_econ_tol`, `set_acc_icon_tol`, `set_acc_tols`, `set_unacc_tols`, `set_div_kkt_tol`, `set_div_bar_tol`, `set_div_econ_tol`, `set_div_icon_tol`, `set_div_tols`, `set_bound_fraction`, `set_bound_push`, `set_alpha_red`, `set_delta_h`, `set_incr_h`, `set_decr_h`, `set_hpert_params`, `set_print_level`, `set_qp_ordering_mode` (both overloads), `set_opt_bar_mode` (both), `set_soe_bar_mode` (both), `set_opt_ls_mode` (both), `set_soe_ls_mode` (both), `set_best_criteria` (both)
  - `set_qp_params()`
  - `analyze_kkt_matrix()`
  - Barrier math: `apply_reset_slacks`, `max_step_to_boundary`, `complementarity`, `barrier_objective`, `barrier_gradient` (both overloads), `barrier_hessian`, `loqo_mu`, `mpc_mu`
  - NLP eval wrappers: `eval_kkt`, `eval_kkt_no`, `eval_aug`, `eval_soe`, `eval_rhs`
  - `release()`

  Keep inline in header: `settings()`, `result()`, `KKTVector` accessors, callback setters (one-liners), `SolveResult::zero_timing()`, `print_header()`.

- [ ] **Step 3: Update `print_beginning`/`print_finished` signatures**

  Change parameter type from `std::string` to `std::string_view` in both declaration and definition.

- [ ] **Step 4: Make `calculate_color` static**

  Change declaration from member to `static` (it doesn't access `this`).

- [ ] **Step 5: Build and test**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  ```

- [ ] **Step 6: Run C++ brachistochrone**

  ```bash
  ./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add -A
  git commit -m "refactor(psiopt): convert struct to class, make fields private, move implementations to .cpp"
  ```

---

### Task 10: Update Python Bindings

**Files:**
- Modify: `src/bindings/solvers/psiopt_bind.cpp`

Convert `def_rw` to `def_prop_rw` for config fields (routing through `settings()`), `def_prop_ro` for result fields (routing through `result()`). Remove dead bindings.

- [ ] **Step 1: Remove dead bindings**

  Remove:
  - `obj.def_rw("store_sp_mat", ...)`
  - `obj.def("get_sp_mat", ...)`
  - `obj.def("get_sp_mat2", ...)`
  - `obj.def_rw("diagnostic", ...)`
  - `obj.def("set_qp_params", ...)` — now private after struct→class conversion

- [ ] **Step 2: Convert config field bindings to `def_prop_rw`**

  For each config field currently using `def_rw`, replace with `def_prop_rw` using getter/setter lambdas. Example:
  ```cpp
  obj.def_prop_rw(
      "max_iters",
      [](const PSIOPT& self) { return self.settings().max_iters_; },
      [](PSIOPT& self, int v) { self.settings().max_iters_ = v; });
  ```

  Do this for ALL config `def_rw` entries:
  `max_iters`, `max_acc_iters`, `max_ls_iters`, `alpha_red`, `wide_console`, `fast_factor_alg`, `obj_scale`, `print_level`, `kkt_tol`, `bar_tol`, `eq_con_tol`, `ineq_con_tol`, `acc_kkt_tol`, `acc_bar_tol`, `acc_eq_con_tol`, `acc_ineq_con_tol`, `div_kkt_tol`, `div_bar_tol`, `div_eq_con_tol`, `div_ineq_con_tol`, `neg_slack_reset`, `bound_fraction`, `bound_push`, `delta_h`, `incr_h`, `decr_h`, `init_mu`, `min_mu`, `max_mu`, `max_soc`, `pd_step_strategy`, `soe_bound_relax`, `qp_par_solve`, `soe_mode`, `opt_bar_mode`, `soe_bar_mode`, `opt_ls_mode`, `soe_ls_mode`, `force_qp_analysis`, `qp_ref_steps`, `qp_pivot_perturb`, `qp_threads`, `qp_pivot_strategy`, `qp_ordering_mode`, `qp_print`, `return_best`, `cnr_mode`

  For `best_criteria`, preserve the existing `def_prop_rw` lambda pattern (already uses one).

  Platform-specific: `accel_pivot_tolerance`, `accel_zero_tolerance` (inside `#ifdef USE_ACCELERATE_SPARSE`).

- [ ] **Step 3: Convert result field bindings to `def_prop_ro`**

  Replace `def_rw` with `def_prop_ro` for result fields:
  ```cpp
  obj.def_prop_ro("last_total_time",
      [](const PSIOPT& self) { return self.result().total_time_; });
  obj.def_prop_ro("last_pre_time",
      [](const PSIOPT& self) { return self.result().pre_time_; });
  // ... etc. for all timing fields, last_iter_num, last_obj_val, converge_flag
  ```

- [ ] **Step 4: Build and test C++ + Python**

  ```bash
  cd build && ninja -j4 all && ctest --output-on-failure
  conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
  ```
  Expected: all C++ tests pass AND all 38 Python examples pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/bindings/solvers/psiopt_bind.cpp
  git commit -m "refactor(psiopt): update Python bindings — def_prop_rw/ro, remove dead bindings"
  ```

---

### Task 11: Final Verification

Run the full pre-merge verification sequence.

- [ ] **Step 1: C++ unit tests**

  ```bash
  cd build && ctest --output-on-failure
  ```
  Expected: all pass.

- [ ] **Step 2: Python examples**

  ```bash
  conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
  ```
  Expected: all 38 pass.

- [ ] **Step 3: C++ brachistochrone**

  ```bash
  ./build/examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
  ```
  Expected: "Optimal Solution Found", objective ~ 1.8013 s.

- [ ] **Step 4: Benchmarks**

  ```bash
  cd build && ninja -j4 bench_all && ./bench/cpp/bench_all
  bench/bench_track.sh compare
  ```
  Expected: no unexplained regressions.

- [ ] **Step 5: clang-format**

  ```bash
  cd build && ninja clang-format
  ```

- [ ] **Step 6: Final commit (formatting only, if needed)**

  ```bash
  git add -A && git commit -m "chore: apply clang-format after PSIOPT refactor"
  ```
