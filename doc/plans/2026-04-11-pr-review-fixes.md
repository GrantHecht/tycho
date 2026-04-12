# PR Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all findings from the comprehensive PR review of `feat/cpp-example-suite` before merge.

**Architecture:** Batch independent fixes by category — safety fixes, example convergence standardization, code quality improvements, documentation. Each batch can be committed independently.

**Tech Stack:** C++17, CMake/ninja, clang-format

---

## File Map

### Critical safety & cleanup
- Modify: `include/tycho/detail/integrators/integrator.h:555-556` — fix empty `default` in `stepper_compute`
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h:177-186` — remove dead `find_phase_index`
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h:205-206` — document `Phase*` lifetime

### Rename `RowMatrix` -> `row_matrix`
- Modify: `include/tycho/detail/vf/operators/matrix_function.h:43` — rename free function
- Modify: 4 example `main.cpp` files that call `RowMatrix()`

### Example convergence standardization
- Modify: `examples/cpp_examples/builder/reentry/main.cpp`
- Modify: `examples/cpp_examples/builder/topputto_low_thrust/main.cpp`
- Modify: `examples/cpp_examples/builder/simple_low_thrust/main.cpp`
- Modify: `examples/cpp_examples/builder/dionysus_low_thrust/main.cpp`
- Modify: `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp`
- Modify: `examples/cpp_examples/builder/optimal_docking/main.cpp`
- Modify: `examples/cpp_examples/builder/orbit_continuation/main.cpp`

### Comment & doc fixes
- Modify: `include/tycho/detail/vf/operators/operator_overloads.h:426` — fix `quat_rotate` comment
- Modify: `src/optimal_control/runtime_ode.cpp:52-54` — explain `lerp_ig` double `set_traj`
- Modify: `doc/cpp_example_api_gaps.md` — strike through resolved entries

### Code quality: InterpTable dedup
- Modify: `include/tycho/detail/optimal_control/interp/interp_type.h` — add `parse_interp_type()`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_1d.h:70-143` — deduplicate constructors
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_2d.h` — use `parse_interp_type()`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_3d.h` — use `parse_interp_type()`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_4d.h` — use `parse_interp_type()`

### Debug-only asserts
- Modify: `include/tycho/detail/vf/operators/matrix_function.h:29-32` — assert rows*cols == orows
- Modify: `include/tycho/detail/vf/operators/interp_compose.h` — assert table not null

---

## Review Findings Not Addressed (and why)

- **`set_method` missing `default` case** (`integrator.h:217-234`): Intentionally omitted. The compiler `-Wswitch` warning on unhandled enum values is the preferred C++ idiom for catching future additions — a `default` case would silence this.
- **Full runtime dimension checks in VF `operator*`** (`operator_overloads.h:380-391`): Inconsistent with existing VF DSL style. No existing operators validate dimensions at expression-construction time; all validation happens at evaluation time. Adding checks here would be both inconsistent and a performance concern.
- **`interp_compose.h` accesses `table->vlen_` directly**: `vlen_` is a public member, and all existing InterpTable access patterns in the codebase use public members directly. Adding an accessor would be inconsistent.
- **`sp_names_` range validation** (`phase_wrapper.h`): Valid suggestion but requires adding a `num_static_params()` accessor to `ODEPhaseBase` that doesn't currently exist. Out of scope for review fixes.

---

## Task 1: Fix `stepper_compute` empty `default`

**Files:**
- Modify: `include/tycho/detail/integrators/integrator.h:555-556`

The empty `default: {}` case in `stepper_compute` is a silent no-op. If any code path reaches it (e.g., `rk_method_` is set to an unsupported value), the integrator produces garbage output with no error. While `set_method` currently guards against this, the defensive throw is essential in numerical code.

- [ ] **Step 1: Add throw to empty default**

Replace the empty body at lines 555-556:
```cpp
// Before:
default: {
}

// After:
default:
    throw std::logic_error("stepper_compute: unsupported IVPAlg for adaptive stepping");
```

- [ ] **Step 2: Build and verify**

Run: `cd /home/ghecht/Projects/tycho/build && ninja -j8 _tychopy`
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/integrators/integrator.h
git commit -m "fix: add throw to empty default in stepper_compute"
```

---

## Task 2: OCP wrapper cleanup — remove dead code, document lifetime

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h:177-186, 205-206`

Two issues in one file: (a) `find_phase_index` is declared and defined but never called,
and (b) the `phases_` vector stores raw `Phase*` pointers with no documented lifetime
requirement.

- [ ] **Step 1: Remove `find_phase_index` (lines 177-186)**

Delete the entire method.

- [ ] **Step 2: Add lifetime documentation to `phases_`**

Above the `phases_` member declaration (line 206), add:
```cpp
    /// Raw Phase pointers for int-indexed name resolution in add_direct_link_equal_con.
    /// Lifetime requirement: all Phase objects added via add_phase() must outlive
    /// this OptimalControlProblem.  Typical usage (stack-local phases) satisfies this.
    std::vector<Phase *> phases_;
```

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/optimal_control/builder/ocp_wrapper.h
git commit -m "refactor: remove dead find_phase_index, document Phase* lifetime"
```

---

## Task 3: Rename `RowMatrix` -> `row_matrix`

**Files:**
- Modify: `include/tycho/detail/vf/operators/matrix_function.h:43`
- Modify: all examples calling `RowMatrix()`

- [ ] **Step 1: Find all call sites**

```bash
grep -rn 'RowMatrix(' examples/ include/ src/ --include='*.cpp' --include='*.h'
```

Expected C++ call sites: `matrix_function.h` (definition), `betts_low_thrust/main.cpp`
(2 calls), `cart_pole/main.cpp` (1 call + 1 comment reference).

**Important:** The Python class `vf.RowMatrix` (registered in the nanobind bindings) is
a *type* and correctly uses PascalCase per CLAUDE.md — do NOT rename it. Only the C++
free function is affected.

- [ ] **Step 2: Rename function definition**

In `include/tycho/detail/vf/operators/matrix_function.h:43`:
```cpp
// Before:
auto RowMatrix(const DenseFunctionBase<Func, IR, OR> &f, int rows, int cols) {

// After:
auto row_matrix(const DenseFunctionBase<Func, IR, OR> &f, int rows, int cols) {
```

- [ ] **Step 3: Update all call sites**

Replace `RowMatrix(` with `row_matrix(` in every example and header found in Step 1.

- [ ] **Step 4: Build and verify**

Run: `cd /home/ghecht/Projects/tycho/build && ninja -j8 _tychopy`
Expected: Clean compilation. If examples are enabled: `ninja -j2 <affected_example_targets>`

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/vf/operators/matrix_function.h
git add examples/cpp_examples/builder/*/main.cpp
git commit -m "refactor: rename RowMatrix -> row_matrix (snake_case convention)"
```

---

## Task 4: Fix example convergence — `reentry`

**Files:**
- Modify: `examples/cpp_examples/builder/reentry/main.cpp:184-211`

This example has three solve calls (`solve_optimize`, `optimize`, `optimize`) with ALL
return values discarded. It unconditionally prints "PASSED" and returns `EXIT_SUCCESS`.
No trajectory verification is performed.

- [ ] **Step 1: Capture convergence flags and verify trajectory**

Replace lines 184-212 with code that:
1. Captures the return value of all three solve calls
2. Checks cross-range values are positive (sanity check)
3. Returns `EXIT_FAILURE` if any solve fails or values are unreasonable

Pattern to follow (from `analytic_example/main.cpp:71-74`):
```cpp
auto flag = phase.solve_optimize();
if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
    std::cerr << "Reentry (builder): FAILED at step 1 (status "
              << static_cast<int>(flag) << ")\n";
    return EXIT_FAILURE;
}
```

Apply this pattern after each of the three solve calls (`solve_optimize` at line 184,
`optimize` at line 189, `optimize` at line 197). Keep the existing cross-range output.
Add a trajectory plausibility check: cross-range values should be positive.

- [ ] **Step 2: Commit**

```bash
git add examples/cpp_examples/builder/reentry/main.cpp
git commit -m "fix: reentry example checks convergence flags and returns non-zero on failure"
```

---

## Task 5: Fix example convergence — `topputto_low_thrust`

**Files:**
- Modify: `examples/cpp_examples/builder/topputto_low_thrust/main.cpp:111-166`

Three problems: (a) three solve calls (`solve_optimize_solve`, `optimize_solve`,
`optimize_solve`) have return values discarded, (b) the final `else` branch at lines
163-166 prints "PASSED (with convergence notes)" and returns 0 even when both boundary
checks fail, (c) the "partial convergence" branch at lines 160-162 also always returns 0.

- [ ] **Step 1: Fix convergence checking and exit code**

1. Capture return value of `solve_optimize_solve()` at line 111 — check flag, print
   warning if not ACCEPTABLE (but continue since the example re-solves)
2. For lines 157-166, fix the exit code logic:
   - `ok1 && ok2` -> PASSED, return 0
   - `ok1 || ok2` -> PASSED (partial), return 0 (one phase is acceptable)
   - neither -> FAILED, return 1

```cpp
    if (ok1 && ok2) {
        std::cout << "\nTopputtoLowThrust: PASSED\n";
        return 0;
    } else if (ok1 || ok2) {
        std::cout << "\nTopputtoLowThrust: PASSED (partial convergence)\n";
        return 0;
    } else {
        std::cerr << "\nTopputtoLowThrust: FAILED\n";
        return 1;
    }
```

- [ ] **Step 2: Commit**

```bash
git add examples/cpp_examples/builder/topputto_low_thrust/main.cpp
git commit -m "fix: topputto_low_thrust returns non-zero when both phases fail"
```

---

## Task 6: Fix example convergence — `simple_low_thrust`

**Files:**
- Modify: `examples/cpp_examples/builder/simple_low_thrust/main.cpp:122-192`

Three solve calls (`solve_optimize`, `optimize`, `optimize`) have return values discarded.
The example DOES check boundary errors (good), but always returns 0 even when errors are
elevated.

- [ ] **Step 1: Propagate boundary-error failures to exit code**

The existing `check_traj` lambda correctly sets `ok = false` on elevated errors. Fix
lines 187-192 to return non-zero:

```cpp
    if (ok) {
        std::cout << "\nSimpleLowThrust: PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSimpleLowThrust: FAILED (elevated boundary errors)\n";
        return 1;
    }
```

- [ ] **Step 2: Commit**

```bash
git add examples/cpp_examples/builder/simple_low_thrust/main.cpp
git commit -m "fix: simple_low_thrust returns non-zero on elevated boundary errors"
```

---

## Task 7: Fix example convergence — `dionysus_low_thrust`

**Files:**
- Modify: `examples/cpp_examples/builder/dionysus_low_thrust/main.cpp:197-223`

`phase.optimize()` return value is discarded. The example checks boundary errors but the
non-converged branch (lines 220-222) prints a "PASSED" message with a note and returns 0.

- [ ] **Step 1: Capture convergence flag and fix exit logic**

Capture the return value of `optimize()`. Keep the boundary error check. Fix the exit
logic so that non-convergence returns 1:

```cpp
    auto flag = phase.optimize();

    // ... (existing trajectory extraction and output) ...

    if (converged) {
        std::cout << "\nDionysusLowThrust: PASSED\n";
        return 0;
    } else {
        std::cerr << "\nDionysusLowThrust: FAILED (boundary errors elevated; "
                  << "MEE dynamics inline, convergence may differ from Python)\n";
        return 1;
    }
```

- [ ] **Step 2: Commit**

```bash
git add examples/cpp_examples/builder/dionysus_low_thrust/main.cpp
git commit -m "fix: dionysus_low_thrust returns non-zero on convergence failure"
```

---

## Task 8: Fix example convergence — `multi_spacecraft_opt`

**Files:**
- Modify: `examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp:190-241`

`ocp.solve()` and `ocp.optimize()` return values discarded (lines 190-191). The
continuation loop captures `flag` (line 213) but never checks it. The example checks
boundary errors but always returns 0.

- [ ] **Step 1: Fix exit code propagation**

Lines 235-241: Change the non-converged branch to return 1:

```cpp
    if (ok) {
        std::cout << "  All spacecraft rendezvous at common point\n";
        std::cout << "\nMultiSpacecraftOpt: PASSED\n";
        return 0;
    } else {
        std::cerr << "\nMultiSpacecraftOpt: FAILED (elevated position errors)\n";
        return 1;
    }
```

- [ ] **Step 2: Commit**

```bash
git add examples/cpp_examples/builder/multi_spacecraft_opt/main.cpp
git commit -m "fix: multi_spacecraft_opt returns non-zero on convergence failure"
```

---

## Task 9: Fix example convergence — `optimal_docking`

**Files:**
- Modify: `examples/cpp_examples/builder/optimal_docking/main.cpp:298-435`

`run_form1()` and `run_form2()` both check convergence flags internally and print
"FAILED" with the flag value, but they return `void`. `main()` calls both and
unconditionally returns 0.

- [ ] **Step 1: Make `run_form1` and `run_form2` return bool**

Change signatures from `static void run_form1()` / `static void run_form2()` to
`static bool run_form1()` / `static bool run_form2()`.

Return the convergence check result:
```cpp
    // At end of run_form2 (line 332-337):
    if (flag <= PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cout << "  Form2: PASSED\n\n";
        return true;
    } else {
        std::cerr << "  Form2: FAILED (convergence flag = " << static_cast<int>(flag)
                  << ")\n\n";
        return false;
    }
```

Same pattern for `run_form1`.

- [ ] **Step 2: Propagate results in main()**

```cpp
int main() {
    bool ok2 = run_form2();
    bool ok1 = run_form1();
    return (ok1 && ok2) ? 0 : 1;
}
```

- [ ] **Step 3: Commit**

```bash
git add examples/cpp_examples/builder/optimal_docking/main.cpp
git commit -m "fix: optimal_docking propagates convergence failures to exit code"
```

---

## Task 10: Fix example convergence — `orbit_continuation`

**Files:**
- Modify: `examples/cpp_examples/builder/orbit_continuation/main.cpp:198-204`

The example has practical failure detection (empty trajectory stops continuation), but
the exit logic at lines 198-204 returns 0 even when `ok` is false (the "computed N
orbits" branch).

- [ ] **Step 1: Fix exit code**

```cpp
    if (ok) {
        std::cout << "\nOrbitContinuation: PASSED\n";
        return 0;
    } else {
        std::cerr << "\nOrbitContinuation: FAILED (periodicity or orbit count insufficient)\n";
        return 1;
    }
```

- [ ] **Step 2: Commit**

```bash
git add examples/cpp_examples/builder/orbit_continuation/main.cpp
git commit -m "fix: orbit_continuation returns non-zero on convergence failure"
```

---

## Task 11: Fix comments — `quat_rotate` and `ODE::phase()`

**Files:**
- Modify: `include/tycho/detail/vf/operators/operator_overloads.h:425-426`
- Modify: `src/optimal_control/runtime_ode.cpp:52-54`

Two comment accuracy fixes.

- [ ] **Step 1: Fix `quat_rotate` comment**

At line 425-426 of `operator_overloads.h`:
```cpp
// Before:
/// Rotate a 3-element vector by a 4-element quaternion (scalar-last).
/// Computes: q * (v, 0) * q_inv, then extracts the 3-element vector part.

// After:
/// Rotate a 3-element vector by a 4-element quaternion (scalar-last).
/// Computes: q * (v, 0) * conj(q), extracting the vector part (assumes unit quaternion).
```

- [ ] **Step 2: Add comment to `ODE::phase()` explaining `lerp_ig`**

At line 52-54 of `runtime_ode.cpp`:
```cpp
    if (lerp_ig) {
        // Re-set trajectory with linear interpolation between waypoints.
        // The constructor distributes traj normally; set_traj(..., true)
        // replaces that with a lerp IG (sparse waypoint interpolation).
        phase_ptr->set_traj(traj, num_segments, true);
    }
```

- [ ] **Step 3: Commit**

```bash
git add include/tycho/detail/vf/operators/operator_overloads.h
git add src/optimal_control/runtime_ode.cpp
git commit -m "docs: fix quat_rotate comment, explain lerp_ig double set_traj"
```

---

## Task 12: Update stale entries in `cpp_example_api_gaps.md`

**Files:**
- Modify: `doc/cpp_example_api_gaps.md`

12+ entries are listed as open but have been resolved in this PR. Strike them through
using the `~~text~~` format that the existing resolved entries use.

- [ ] **Step 1: Identify resolved entries and strike through**

Entries to resolve (confirmed by examining code):

| Line | Entry | Resolved by |
|------|-------|-------------|
| 7 | CartPole RowMatrix/inverse | `matrix_function.h` + `operator_overloads.h` |
| 8 | ParallelParking mixed var sources | `phase_wrapper.h` mixed-var overloads |
| 9 | ParallelParking refine_traj_manual | `phase_wrapper.h:486` |
| 10 | ParallelParking sub_variable | `phase_wrapper.h:493` |
| 11 | ParallelParking LerpIG | `runtime_ode.h:83` `bool lerp_ig` param |
| 15 | MultiPhaseCannon add_direct_link_equal_con | `ocp_wrapper.h:93-129` |
| 16 | MultiPhaseCannon ODEParams + named vars | `phase_wrapper.h` region resolution |
| 17 | MultiPhaseCannon mixed inequality | `phase_wrapper.h` mixed-var overloads |
| 19 | DionysusLowThrust ODE::integrator() | `runtime_ode.h:87-116` |
| 22 | OrbitContinuation ODE::integrator() | Same |
| 26 | OptimalDocking quat_rotate | `operator_overloads.h` |
| 27 | OptimalDocking cwise_product(Eigen) | `operator_overloads.h` |
| 30 | MultiSpacecraftOpt sub_variables | `phase_wrapper.h` |
| 32 | GoddardRocket control-law integration | `runtime_ode.cpp` integrator overloads |
| 33 | MultiPhaseCannon event detection | `integrator.h` integrate_dense with stop_fn |

For each, add `~~` strikethrough and "RESOLVED" note matching the existing format
(e.g., lines 12, 14, 20-21, 23-25, 28, 31).

- [ ] **Step 2: Commit**

```bash
git add doc/cpp_example_api_gaps.md
git commit -m "docs: mark resolved API gaps in cpp_example_api_gaps.md"
```

---

## Task 13: InterpTable constructor dedup + centralize InterpType parsing

**Files:**
- Modify: `include/tycho/detail/optimal_control/interp/interp_type.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_1d.h:70-143, 191-200`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_2d.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_3d.h`
- Modify: `include/tycho/detail/optimal_control/interp/interp_table_4d.h`

Two related issues: (a) the `InterpTable1D` `vector<VectorXd>` constructor is duplicated
for `InterpType` and `std::string` parameters (40 identical lines), and (b) the string-
to-`InterpType` conversion is duplicated across all 4 table types.

- [ ] **Step 1: Add `parse_interp_type` to `interp_type.h`**

```cpp
#pragma once

#include <stdexcept>
#include <string>

namespace tycho {

enum class InterpType { Cubic, Linear };

/// Parse a string to InterpType. Accepts "cubic"/"Cubic" and "linear"/"Linear".
inline InterpType parse_interp_type(const std::string &kind) {
    if (kind == "cubic" || kind == "Cubic")
        return InterpType::Cubic;
    if (kind == "linear" || kind == "Linear")
        return InterpType::Linear;
    throw std::invalid_argument("Unknown interp type: '" + kind +
                                "' (expected 'cubic' or 'linear')");
}

} // namespace tycho
```

- [ ] **Step 2: Deduplicate InterpTable1D constructors**

Replace the `vector<VectorXd>` + `string` constructor (lines 107-143) to delegate:
```cpp
    InterpTable1D(const std::vector<Eigen::VectorXd> &Vts, int tvar, std::string kind)
        : InterpTable1D(Vts, tvar, parse_interp_type(kind)) {}
```

- [ ] **Step 3: Replace inline parsing in all 4 table types' string-accepting overloads**

The parsing duplication appears in different forms across the tables:

**InterpTable1D** (`set_data(string)` at line 191) and **InterpTable2D** (`set_data(string)`
at line 132): Replace the inline parsing block with a one-line delegation:
```cpp
void set_data(/* same args */, std::string kind) {
    set_data(/* same args */, parse_interp_type(kind));
}
```

**InterpTable3D** (constructor at line 161) and **InterpTable4D** (constructor at line 181):
These use `*this = InterpTableXD(...)` delegation. Replace the inline parsing with:
```cpp
// InterpTable3D:
InterpTable3D(const Eigen::VectorXd &Xs, const Eigen::VectorXd &Ys,
              const Eigen::VectorXd &Zs, const Eigen::Tensor<double, 3> &Fs,
              std::string kind, bool cache)
    : InterpTable3D(Xs, Ys, Zs, Fs, parse_interp_type(kind), cache) {}

// InterpTable4D:
InterpTable4D(const Eigen::VectorXd &Xs, const Eigen::VectorXd &Ys,
              const Eigen::VectorXd &Zs, const Eigen::VectorXd &Ws,
              const Eigen::Tensor<double, 4> &Fs, std::string kind, bool cache)
    : InterpTable4D(Xs, Ys, Zs, Ws, Fs, parse_interp_type(kind), cache) {}
```

This eliminates the inline parsing in all 4 table types.

- [ ] **Step 4: Build and verify**

Run: `cd /home/ghecht/Projects/tycho/build && ninja -j8 _tychopy`
Expected: Clean compilation.

- [ ] **Step 5: Commit**

```bash
git add include/tycho/detail/optimal_control/interp/interp_type.h
git add include/tycho/detail/optimal_control/interp/interp_table_1d.h
git add include/tycho/detail/optimal_control/interp/interp_table_2d.h
git add include/tycho/detail/optimal_control/interp/interp_table_3d.h
git add include/tycho/detail/optimal_control/interp/interp_table_4d.h
git commit -m "refactor: centralize InterpType parsing, deduplicate InterpTable1D constructors"
```

---

## Task 14: Debug-only asserts in VF expression layer

**Files:**
- Modify: `include/tycho/detail/vf/operators/matrix_function.h:29-32`
- Modify: `include/tycho/detail/vf/operators/interp_compose.h`

Add `assert()` calls for invariants that should hold but aren't currently validated.
These run in debug builds and compile out in release — consistent with VF DSL performance
expectations.

- [ ] **Step 1: Add assert in MatrixFunctionView constructor**

In `matrix_function.h`, the constructor at line 29-32:
```cpp
    MatrixFunctionView(Func f, int rows, int cols)
        : Func(f), MatrixRowsCols<MRows, MCols>(rows, cols) {
        assert(rows > 0 && cols > 0 && "MatrixFunctionView: rows and cols must be positive");
        assert((f.output_rows() < 0 || rows * cols == f.output_rows())
               && "MatrixFunctionView: rows * cols must match function output size");
    }
```

Note: `f.output_rows() < 0` handles the dynamic-size case (`ORC == -1`) where
output size isn't known at construction time.

- [ ] **Step 2: Add assert for null table pointers in `interp_compose.h`**

At the top of each `interp()` and `lgl_interp()` overload (8 overloads total), add:
```cpp
    assert(table && "interp: table pointer must not be null");
```

Add `#include <cassert>` at the top of the file.

- [ ] **Step 3: Build and verify**

Run: `cd /home/ghecht/Projects/tycho/build && ninja -j8 _tychopy`
Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/vf/operators/matrix_function.h
git add include/tycho/detail/vf/operators/interp_compose.h
git commit -m "fix: add debug-only asserts in MatrixFunctionView and interp_compose"
```

---

## Task 15: Build, format, and verify

**Files:** All modified files from Tasks 1-14.

- [ ] **Step 1: Format all C++ changes**

```bash
cd /home/ghecht/Projects/tycho/build && ninja clang-format
```

- [ ] **Step 2: Full build**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 all
```

Expected: Clean build with no errors or warnings.

- [ ] **Step 3: Run C++ unit tests**

```bash
ctest --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 4: Run Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: All 38 examples pass.

- [ ] **Step 5: Build and run C++ brachistochrone**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 brachistochrone_cpp
./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found", objective ~1.8013 s.

- [ ] **Step 6: Build and run affected C++ builder examples**

```bash
cd /home/ghecht/Projects/tycho/build
ninja -j2 reentry_builder topputto_builder simple_low_thrust_builder \
         dionysus_builder multi_spacecraft_builder optimal_docking_builder \
         orbit_continuation_builder
```

Run each and verify exit codes are 0 (convergence) or expected non-zero (known issue).

- [ ] **Step 7: Final commit if formatting changed anything**

```bash
git diff --name-only  # check if clang-format changed anything
# If yes:
git add include/ src/ extensions/ examples/ && git commit -m "chore: apply clang-format"
```
