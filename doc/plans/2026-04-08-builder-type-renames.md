# Builder API Type Renames — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename builder API types to primary names (`RuntimeODE` -> `ODE`, `OCP` -> `OptimalControlProblem`) and rename static DSL types to secondary names (`ODE<>` -> `StaticODE<>`, `OptimalControlProblem` -> `OptimalControlProblemBase`).

**Architecture:** Mechanical find-and-replace in a specific order — rename definitions first, then update all references. No behavioral changes. The key constraint is that `ODE` and `OptimalControlProblem` names must be freed by the static DSL renames before the builder types can claim them.

**Tech Stack:** C++ (templates, macros), nanobind Python bindings, CMake

**Branch:** Create `builder-type-renames` from `main` before any changes.

**Spec:** `doc/plans/TYCHO_BUILDER_TYPE_RENAMES_SPEC.md`

---

### Task 0: Create feature branch

**Files:** None

- [ ] **Step 1: Create and checkout branch**

```bash
git checkout main && git pull
git checkout -b builder-type-renames
```

- [ ] **Step 2: Verify clean state**

```bash
git status  # should be clean
```

---

### Task 1: Rename static DSL ODE types (ODE -> StaticODE)

Rename the struct templates in `ode.h` and update the macro internals.
This frees the `ODE` name for the builder class.

**Files:**
- Modify: `include/tycho/detail/optimal_control/phase/ode.h`

- [ ] **Step 1: Rename struct declarations**

In `include/tycho/detail/optimal_control/phase/ode.h`:
- `struct ODE :` -> `struct StaticODE :`
- `struct ODE_Expression :` -> `struct StaticODE_Expression :`
- `struct ODE_DerivModeWrapper :` -> `struct StaticODE_DerivModeWrapper :`

Also rename the constructor: `ODE_DerivModeWrapper(Args &&...args)` -> `StaticODE_DerivModeWrapper(Args &&...args)`

- [ ] **Step 2: Update macro internal expansions**

In the same file, update the three macro bodies:
- `BUILD_ODE_FROM_EXPRESSION`: internal `ODE_Expression<NAME, IMPL, __VA_ARGS__>` -> `StaticODE_Expression<NAME, IMPL, __VA_ARGS__>`
- `BUILD_ODE_FROM_EXPRESSION_FD`: internal `ODE_Expression<NAME##_AD, ...>` -> `StaticODE_Expression<NAME##_AD, ...>`, and `ODE_DerivModeWrapper<NAME, ...>` -> `StaticODE_DerivModeWrapper<NAME, ...>`
- `BUILD_ODE_FROM_EXPRESSION_FWAD`: same pattern

Do NOT rename the macro names themselves — only the types they expand to.

- [ ] **Step 3: Verify the file compiles in isolation**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 tycho_tests 2>&1 | head -50
```

Expected: Failures from downstream files that still reference `ODE<>` — this is expected and fixed in the next steps.

- [ ] **Step 4: Commit**

```bash
git add include/tycho/detail/optimal_control/phase/ode.h
git commit -m "refactor: rename static DSL ODE -> StaticODE in ode.h"
```

---

### Task 2: Update all references to old static DSL ODE names

Update every file that references `ODE<`, `ODE_Expression<`, or `ODE_DerivModeWrapper<` to use the `StaticODE` prefix.

**Files:**
- Modify: `include/tycho/detail/optimal_control/phase/ode_phase.h`
- Modify: `include/tycho/detail/astro/kepler_model.h` (line 24: `using oc::ODE_Expression;`, line 43: inherits from `ODE_Expression<>`)
- Modify: `include/tycho/detail/astro/cr3bp_model.h` (if it directly references `ODE_Expression` outside macros)
- Modify: `src/bindings/optimal_control/ode_bind.h` (line 33: `ODE_ExpressionBuild` function name and comment references)
- Modify: `tests/cpp/vector_functions/test_ode_syntax_prototypes.cpp` (lines 86, 104-115: `ODE<>`, `ODE_DerivModeWrapper<>`)
- Modify: `doc/Bindings.md` (references to `ODE_ExpressionBuild`, `ODE_Expression`)

- [ ] **Step 1: Update kepler_model.h**

```
using oc::ODE_Expression; -> using oc::StaticODE_Expression;
struct Kepler : ODE_Expression<...> -> struct Kepler : StaticODE_Expression<...>
```

- [ ] **Step 2: Update ode_phase.h**

Any direct references to `ODE<>` template (check for type aliases or template parameters).

- [ ] **Step 3: Update ode_bind.h**

Rename `ODE_ExpressionBuild` -> `StaticODE_ExpressionBuild` (function name and all call sites in bindings). Update comments.

- [ ] **Step 4: Update test_ode_syntax_prototypes.cpp**

```
struct DynamicBrach : ODE<DynamicBrach, ...> -> struct DynamicBrach : StaticODE<DynamicBrach, ...>
ODE_DerivModeWrapper<...> -> StaticODE_DerivModeWrapper<...>
```

- [ ] **Step 5: Update doc/Bindings.md**

Replace `ODE_ExpressionBuild` with `StaticODE_ExpressionBuild`, `ODE_Expression` with `StaticODE_Expression`.

- [ ] **Step 6: Build and verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 all 2>&1 | tail -5
```

Expected: Clean build (all static DSL references updated).

- [ ] **Step 7: Run tests**

```bash
cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add -u
git commit -m "refactor: update all references ODE -> StaticODE, ODE_Expression -> StaticODE_Expression"
```

---

### Task 3: Rename RuntimeODE -> ODE

Now that the `ODE` name is free, rename the builder class.

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/runtime_ode.h` (class declaration + deprecated alias)
- Modify: `include/tycho/detail/optimal_control/builder/ode_builder.h` (return type, comments)
- Modify: `include/tycho/detail/optimal_control/builder/var_registry.h` (comments)
- Modify: `include/tycho/detail/optimal_control/builder/phase_wrapper.h` (comments)
- Modify: `src/optimal_control/runtime_ode.cpp` (method definitions, error messages)
- Modify: `examples/cpp_examples/builder/zermelo/main.cpp`
- Modify: `examples/cpp_examples/builder/delta3_launch/main.cpp`
- Modify: `tests/cpp/optimal_control/test_runtime_ode.cpp`
- Modify: `tests/cpp/optimal_control/test_control_modes.cpp`
- Modify: `tests/cpp/optimal_control/test_phase_wrapper.cpp`

- [ ] **Step 1: Rename class in runtime_ode.h**

`class RuntimeODE {` -> `class ODE {`

Update all constructors, method signatures. Update error message prefixes from `"RuntimeODE:"` to `"ODE:"`.

Add deprecated alias at the bottom of the file:
```cpp
[[deprecated("Use ODE instead")]] typedef ODE RuntimeODE;
```

- [ ] **Step 2: Update ode_builder.h**

Return type `RuntimeODE` -> `ODE`. Update comments.

- [ ] **Step 3: Update runtime_ode.cpp**

All `RuntimeODE::` method qualifiers -> `ODE::`. Error message strings: `"RuntimeODE:"` -> `"ODE:"`. Comments.

- [ ] **Step 4: Update phase_wrapper.h and var_registry.h comments**

Replace `RuntimeODE` with `ODE` in comments.

- [ ] **Step 5: Update all example files**

- `zermelo/main.cpp`: `RuntimeODE` -> `ODE` in function signatures, comments
- `delta3_launch/main.cpp`: `RuntimeODE` -> `ODE` in function signature

- [ ] **Step 6: Update all test files**

- `test_runtime_ode.cpp`: class name `RuntimeODETest`, all `RuntimeODE ode(...)` instantiations -> `ODE ode(...)`
- `test_control_modes.cpp`: function signature `RuntimeODE` -> `ODE`
- `test_phase_wrapper.cpp`: function return type `RuntimeODE` -> `ODE`

- [ ] **Step 7: Build and verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 all 2>&1 | tail -5
```

- [ ] **Step 8: Run tests**

```bash
cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure 2>&1 | tail -10
```

- [ ] **Step 9: Commit**

```bash
git add -u
git commit -m "refactor: rename RuntimeODE -> ODE (builder API primary name)"
```

---

### Task 4: Rename OptimalControlProblem -> OptimalControlProblemBase

Rename the base struct. This frees the name for the builder wrapper.

**Files:**
- Modify: `include/tycho/detail/optimal_control/phase/optimal_control_problem.h` (struct declaration)
- Modify: `include/tycho/detail/optimal_control/phase/ode_phase_base.h` (forward declaration)
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` (using declaration, member type)
- Modify: `src/optimal_control/optimal_control_problem.cpp` (method definitions)
- Modify: `src/bindings/optimal_control/optimal_control_problem_bind.cpp` (C++ type, keep Python string `"OptimalControlProblem"`)
- Modify: `src/bindings/optimal_control/tycho_optimal_control.cpp` (`TychoBind<OptimalControlProblem>` -> `TychoBind<OptimalControlProblemBase>`)
- Modify: `src/bindings/optimal_control/ode_phase_bind.h` (type references)
- Modify: `tests/cpp/optimal_control/test_ocp_multiphase.cpp` (`OptimalControlProblem ocp;` -> `OptimalControlProblemBase ocp;`)
- Modify: `examples/cpp_examples/static/delta3_launch/main.cpp`

- [ ] **Step 1: Rename struct in optimal_control_problem.h**

`struct OptimalControlProblem :` -> `struct OptimalControlProblemBase :`

Update all method signatures in the struct body.

- [ ] **Step 2: Update forward declaration in ode_phase_base.h**

`struct OptimalControlProblem;` -> `struct OptimalControlProblemBase;`

- [ ] **Step 3: Update ocp_wrapper.h**

```
using oc::OptimalControlProblem; -> using oc::OptimalControlProblemBase;
```
Member variable: `OptimalControlProblem ocp_;` -> `OptimalControlProblemBase ocp_;`
Method returns: `OptimalControlProblem &base()` -> `OptimalControlProblemBase &base()`
Comments referencing `OptimalControlProblem` in context of the base struct.

- [ ] **Step 4: Update optimal_control_problem.cpp**

All `OptimalControlProblem::` method qualifiers -> `OptimalControlProblemBase::`.

- [ ] **Step 5: Update Python bindings — keep Python name unchanged**

In `optimal_control_problem_bind.cpp`:
```cpp
nb::class_<OptimalControlProblemBase, OptimizationProblemBase>(m, "OptimalControlProblem")
```
The C++ type changes to `OptimalControlProblemBase` but the Python-facing string stays `"OptimalControlProblem"`.

Update `ode_phase_bind.h` similarly.

- [ ] **Step 6: Update tycho_optimal_control.cpp**

`TychoBind<OptimalControlProblem>::Build(oc);` -> `TychoBind<OptimalControlProblemBase>::Build(oc);`

- [ ] **Step 7: Update test_ocp_multiphase.cpp**

`OptimalControlProblem ocp;` -> `OptimalControlProblemBase ocp;` (and any other direct references).

- [ ] **Step 8: Update static DSL delta3_launch example**

Any direct references to `OptimalControlProblem` -> `OptimalControlProblemBase`.

- [ ] **Step 9: Build and verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 all 2>&1 | tail -5
```

- [ ] **Step 10: Run tests and Python import**

```bash
cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure 2>&1 | tail -10
conda run -n tycho python -c "from tychopy import OptimalControl as oc; print(oc.OptimalControlProblem)"
```

Expected: Tests pass. Python import succeeds and prints the class.

- [ ] **Step 11: Commit**

```bash
git add -u
git commit -m "refactor: rename OptimalControlProblem -> OptimalControlProblemBase"
```

---

### Task 5: Rename OCP -> OptimalControlProblem

Now that the name is free, promote the builder wrapper.

**Files:**
- Modify: `include/tycho/detail/optimal_control/builder/ocp_wrapper.h` (class declaration + deprecated alias)
- Modify: `examples/cpp_examples/builder/delta3_launch/main.cpp` (`OCP ocp;`)
- Modify: `tests/cpp/optimal_control/test_builder_brachistochrone.cpp` (`OCP ocp;` x5)

- [ ] **Step 1: Rename class in ocp_wrapper.h**

`class OCP {` -> `class OptimalControlProblem {`

Update constructors, error messages (`"OCP::"` -> `"OptimalControlProblem::"`).

Add deprecated alias:
```cpp
[[deprecated("Use OptimalControlProblem instead")]] typedef OptimalControlProblem OCP;
```

- [ ] **Step 2: Update delta3_launch example**

`OCP ocp;` -> `OptimalControlProblem ocp;`

- [ ] **Step 3: Update test_builder_brachistochrone.cpp**

All `OCP ocp;` -> `OptimalControlProblem ocp;` (5 occurrences).

- [ ] **Step 4: Build and verify**

```bash
cd /home/ghecht/Projects/tycho/build && ninja -j8 all 2>&1 | tail -5
```

- [ ] **Step 5: Run tests**

```bash
cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor: rename OCP -> OptimalControlProblem (builder API primary name)"
```

---

### Task 6: Full verification and cleanup

Run the complete pre-merge verification sequence.

**Files:** None (verification only)

- [ ] **Step 1: C++ unit tests**

```bash
cd /home/ghecht/Projects/tycho/build && ctest --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 2: C++ examples (builder + static)**

```bash
cd /home/ghecht/Projects/tycho/build && cmake --preset linux-clang-release -DBUILD_CPP_EXAMPLES=ON
ninja -j8 brachistochrone_cpp brachistochrone_builder zermelo_builder hypersens_builder delta3_launch_builder
```

Run each and verify convergence.

- [ ] **Step 3: Python examples**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
```

Expected: All 38 Python examples pass.

- [ ] **Step 4: Push branch**

```bash
git push -u origin builder-type-renames
```

- [ ] **Step 5: Commit**

No commit needed — this task is verification only.
