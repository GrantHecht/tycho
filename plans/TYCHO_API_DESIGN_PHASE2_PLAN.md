# Phase 2: Public Header Separation + Namespace Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Separate Tycho's public C++ API headers into `include/tycho/` and rename the `tycho_core` library target to `tycho`. The `namespace Tycho` is kept as-is (no lowercase rename).

**Architecture:** Create a clean `include/tycho/` public header tree with `detail/` subdirectories for template implementation bodies. Public headers contain API declarations and type definitions; private implementation stays in `src/`. The `detail/` convention signals "included automatically, don't use directly." Template-heavy modules (VectorFunctions, Integrators) require most of their implementation in `detail/` headers since they must be visible at compile time. Non-template code (PSIOPT internals, linear solver backends) stays entirely in `src/`.

**Tech Stack:** C++20, CMake, Eigen, nanobind

---

### Task 1: Create feature branch

**Step 1: Create and check out branch**

```bash
git checkout -b feat/public-headers main
```

---

### Task 2: Create `include/tycho/` directory structure

**Files:**
- Create: `include/tycho/` directory tree

**Step 1: Create the directory layout**

```bash
mkdir -p include/tycho/detail
```

**Step 2: Create placeholder umbrella header**

Create `include/tycho/tycho.h`:
```cpp
#pragma once

// Tycho — High-performance trajectory design and optimal control
// Public API umbrella header

#include "tycho/typedefs.h"
#include "tycho/utils.h"
#include "tycho/vector_functions.h"
#include "tycho/integrators.h"
#include "tycho/optimal_control.h"
#include "tycho/solvers.h"
#include "tycho/astro.h"
```

This is a skeleton — each module header will be populated in subsequent tasks.

**Step 3: Commit**

```bash
git add include/
git commit -m "feat: create include/tycho/ public header directory structure"
```

---

### ~~Task 3: Namespace migration~~ — REMOVED

> **Decision (2026-03-22):** Keep `namespace Tycho` (uppercase). The namespace rename
> has been removed from this phase. Rationale: the churn of touching every C++ file
> for a purely aesthetic change is not justified. Uppercase namespaces are common
> (Eigen::, Qt::). The include path `<tycho/...>` is lowercase per filesystem convention;
> the C++ namespace inside remains `Tycho`.

---

### Task 4: Public headers — TypeDefs module

**Files:**
- Create: `include/tycho/typedefs.h`
- Create: `include/tycho/detail/eigen_types.h`

**Step 1: Create `include/tycho/typedefs.h`**

This is the simplest module — just Eigen type aliases. The content of
`src/TypeDefs/EigenTypes.h` moves to `include/tycho/detail/eigen_types.h`
and the public header includes it:

```cpp
#pragma once

// Tycho — Eigen type aliases and SIMD-width detection
#include "tycho/detail/eigen_types.h"
```

**Step 2: Move `EigenTypes.h` content**

Copy `src/TypeDefs/EigenTypes.h` to `include/tycho/detail/eigen_types.h`.
Update the namespace from `Tycho` to `tycho` (should already be done from Task 3).
Update `src/TypeDefs/EigenTypes.h` to simply forward-include:
```cpp
#pragma once
#include "tycho/detail/eigen_types.h"
```

This ensures existing `src/` code that includes `EigenTypes.h` still works.

**Step 3: Update `src/TypeDefs/Tycho_TypeDefs.h`**

```cpp
#pragma once
#include "tycho/typedefs.h"
```

**Step 4: Build and verify**

```bash
cd build && ninja -j2 all
```

**Step 5: Commit**

```bash
git add include/tycho/typedefs.h include/tycho/detail/eigen_types.h src/TypeDefs/
git commit -m "feat: public header for TypeDefs module"
```

---

### Task 5: Public headers — Utils module

**Files:**
- Create: `include/tycho/utils.h`
- Create: `include/tycho/detail/type_storage.h`
- Create: `include/tycho/detail/sizing_helpers.h`
- Create: `include/tycho/detail/crtp_base.h`
- Create: `include/tycho/detail/thread_pool.h`
- Create: `include/tycho/detail/get_core_count.h`

**Step 1: Identify public Utils API**

Public (used by consumers):
- `TypeStorage.h` — SBO container, used in VF type erasure
- `SizingHelpers.h` — compile-time size arithmetic (`SZ_SUM`, etc.)
- `CRTPBase.h` — CRTP base traits
- `ThreadPool.h` — work-stealing pool, `set_num_threads()`, `get_num_threads()`, `parallel_blocks()`, `parallel_sequence()`, `parallel_task()` dispatch helpers, `DispatchContext`
- `GetCoreCount.h` — CPU core detection
- `MathFunctions.h` — basic math utilities
- `STDExtensions.h` — STL extensions used in public API
- `FunctionReturnType.h` — return type deduction (used in VF)
- `TypeName.h` — type name printing (debug utility)
- `FlatMap.h` — fixed-capacity sorted map (used by `ODESize`)

Private (stays in `src/`):
- `ColorText.h` — terminal output
- `MemoryManagement.h` — internal memory tracking
- `Timer.h` — internal timing
- `fmtlib.h` — fmt wrapper
- `EigenSTL.h` — Eigen/STL glue
- `LambdaJumpTable.h` — internal dispatch
- `TupleIterator.h` — internal meta-programming

**Step 2: Move public headers to `include/tycho/detail/`**

For each public Utils header, copy content to `include/tycho/detail/` and
leave a forwarding include in `src/Utils/`. Update namespace references.

**Step 3: Create `include/tycho/utils.h`**

```cpp
#pragma once

#include "tycho/detail/type_storage.h"
#include "tycho/detail/sizing_helpers.h"
#include "tycho/detail/crtp_base.h"
#include "tycho/detail/thread_pool.h"
#include "tycho/detail/get_core_count.h"
#include "tycho/detail/math_functions.h"
#include "tycho/detail/std_extensions.h"
#include "tycho/detail/function_return_type.h"
#include "tycho/detail/type_name.h"
#include "tycho/detail/flat_map.h"
```

**Step 4: Update `src/Utils/Tycho_Utils.h` to forward**

```cpp
#pragma once
#include "tycho/utils.h"
// Private headers (not in public API)
#include "ColorText.h"
#include "MemoryManagement.h"
// etc.
```

**Step 5: Build and verify**

```bash
cd build && ninja -j2 all
```

**Step 6: Commit**

```bash
git add include/tycho/utils.h include/tycho/detail/ src/Utils/
git commit -m "feat: public headers for Utils module"
```

---

### Task 6: Public headers — VectorFunctions module

**Files:**
- Create: `include/tycho/vector_functions.h`
- Create: multiple `include/tycho/detail/` headers for VF types

**This is the largest module (78 headers). Strategy:**

The VectorFunction DSL is almost entirely template-based, so most implementation
must be visible in headers. The split is:

**Public (`include/tycho/detail/vf_*`):**
- `VectorFunction.h` — CRTP base
- `DenseFunction.h`, `DenseFunctionBase.h`, `DenseScalarFunctionBase.h` — core bases
- `OperatorOverloads.h`, `MathOverloads.h` — DSL operators
- All CommonFunctions headers — user-facing function library
- `GenericFunction.h`, `GFTypeErasure.h` — type erasure
- `GenericConditional.h`, `GenericComparative.h` — conditional VF types
- `VectorFunctionConcepts.h` — C++20 concepts
- Solver interface specs (`DenseFunctionSpecs.h`, `SizingSpecs.h`, `SolverInterfaceSpecs.h`)

**Private (stays in `src/VectorFunctions/`):**
- `FunctionDomains.h`, `ObjectDetectors.h`, `IndexingData.h` — internal infrastructure
- `DetectDiagonal.h`, `DetectSuperScalar.h` — type detection
- `ComputableBase.h`, `Computable.h` — below-base computation
- `DenseDerivatives.h` — derivative CRTP chain
- `AssigmentTypes.h`, `FunctionTypeDefMacros.h` — internal macros
- `FunctionTemplates.h`, `FunctionalFlags.h` — metadata
- `SparseFunctionBase.h` — sparse functions (unused publicly)

**Important note on interdependencies:** Many "private" headers are included by
"public" headers transitively. We may need to keep some private headers in
`include/tycho/detail/` even though they're not user-facing, simply because
the public templates require them. The `detail/` prefix signals this.

**Step 1: Move VF core headers to `include/tycho/detail/`**

Start with the foundational types and work up. Each moved header:
1. Gets copied to `include/tycho/detail/`
2. Has its internal `#include` paths updated to use `tycho/detail/` paths
3. The original in `src/VectorFunctions/` becomes a forwarding include

This is a large mechanical task. Process module by module:
1. Foundation: `InputOutputSize.h`, `AssigmentTypes.h`, `FunctionTypeDefMacros.h`
2. CRTP chain: `ComputableBase.h`, `DenseDerivatives.h`, `DenseFunctionBase.h`, `VectorFunction.h`
3. Type erasure: `GFTypeErasure.h`, `GenericFunction.h`, specs
4. CommonFunctions: all 30+ function headers
5. Operators: `OperatorOverloads.h`, `MathOverloads.h`

**Step 2: Create `include/tycho/vector_functions.h`**

```cpp
#pragma once

#include "tycho/typedefs.h"
#include "tycho/utils.h"

// Core VF types
#include "tycho/detail/vector_function.h"
#include "tycho/detail/dense_function_base.h"
#include "tycho/detail/operator_overloads.h"
#include "tycho/detail/math_overloads.h"

// Type erasure
#include "tycho/detail/generic_function.h"
#include "tycho/detail/generic_conditional.h"
#include "tycho/detail/generic_comparative.h"

// Common functions library
#include "tycho/detail/common_functions.h"

// Concepts
#include "tycho/detail/vector_function_concepts.h"
```

**Step 3: Update `src/VectorFunctions/Tycho_VectorFunctions.h` to forward**

```cpp
#pragma once
#include "tycho/vector_functions.h"
```

**Step 4: Build and verify iteratively**

Build after each sub-group of headers is moved. The VF module has deep include
chains, so incremental verification is essential.

```bash
cd build && ninja -j2 all
```

**Step 5: Commit**

```bash
git add include/tycho/ src/VectorFunctions/
git commit -m "feat: public headers for VectorFunctions module"
```

---

### Task 7: Public headers — OptimalControl module

**Files:**
- Create: `include/tycho/optimal_control.h`
- Create: multiple `include/tycho/detail/` headers

**Public API types:**
- `ODE.h` — ODE definition base + `BUILD_ODE_FROM_EXPRESSION` macro
- `ODESizes.h` — `ODESize<XV, UV, PV>` trait
- `ODEArguments.h` — argument packing
- `ODEPhaseBase.h` — phase base class (48KB, template-free — can be declaration-only in public header)
- `ODEPhase.h` — template specialization (must be in detail/)
- `OptimalControlProblem.h` — multi-phase coordinator (70KB, partially template-free)
- `OptimalControlFlags.h` — `TranscriptionModes`, `PhaseRegionFlags`, `ControlModes` enums
- `InterfaceTypes.h` — constraint/objective interface types
- `LGLInterpTable.h` — interpolation table (used by consumers)
- `MeshIterateInfo.h` — mesh refinement iteration info

**Private (stays in `src/OptimalControl/`):**
- `LGLCoeffs.h`, `LGLControlSplines.h`, `LGLDefects.h`, `LGLIntegrals.h`
- `TrapezoidalDefects.h`, `TrapezoidalIntegrals.h`
- `ShootingDefects.h`
- `FDCoeffs.h`, `FDDerivArbitrary.h`, `FDDerivUniform.h`
- `PhaseIndexer.h`, `TranscriptionSizing.h`
- `ValueLock.h`, `AutoScalingUtils.h`
- `MeshSpacingConstraints.h`
- `Blocked_ODE_Wrapper.h`

**Key challenge:** `ODEPhaseBase.h` (48KB) and `OptimalControlProblem.h` (70KB)
are partially template-free. The non-template portions (class declarations with
virtual methods) can go in a public header with implementation in `src/`. The
template portions (methods that depend on `DODE` type) go in `detail/`.

**Step 1: Move public OC headers to `include/tycho/detail/`**

Process in dependency order:
1. Flags and sizes: `OptimalControlFlags.h`, `ODESizes.h`, `ODEArguments.h`
2. Interface types: `InterfaceTypes.h`, `StateFunction.h`, `LinkFunction.h`
3. ODE definition: `ODE.h`
4. Phase: `ODEPhaseBase.h`, `ODEPhase.h`
5. Multi-phase: `OptimalControlProblem.h`
6. Interpolation: `LGLInterpTable.h`, `LGLInterpFunctions.h`
7. Mesh info: `MeshIterateInfo.h`

**Step 2: Create `include/tycho/optimal_control.h`**

```cpp
#pragma once

#include "tycho/vector_functions.h"

#include "tycho/detail/optimal_control_flags.h"
#include "tycho/detail/ode_sizes.h"
#include "tycho/detail/ode_arguments.h"
#include "tycho/detail/ode.h"
#include "tycho/detail/ode_phase_base.h"
#include "tycho/detail/ode_phase.h"
#include "tycho/detail/optimal_control_problem.h"
#include "tycho/detail/lgl_interp_table.h"
#include "tycho/detail/mesh_iterate_info.h"
```

**Step 3: Update `src/OptimalControl/Tycho_OptimalControl.h` to forward**

**Step 4: Build and verify**

**Step 5: Commit**

```bash
git add include/tycho/ src/OptimalControl/
git commit -m "feat: public headers for OptimalControl module"
```

---

### Task 8: Public headers — Solvers module

**Files:**
- Create: `include/tycho/solvers.h`
- Create: `include/tycho/detail/` headers for solver types

**Public:**
- `PSIOPT.h` — solver class with configuration enums
- `OptimizationProblem.h` — problem definition
- `OptimizationProblemBase.h` — base class
- `ConstraintFunction.h`, `ObjectiveFunction.h` — solver function wrappers
- `SolverFunctionBase.h` — base interface
- `SolverFlags.h` — solver flags
- `SolverInit.h` — initialization helper
- `Jet.h` — AD forward mode type (used in solver interfaces)
- `IterateInfo.h` — iteration history

**Private (stays in `src/Solvers/`):**
- `NonLinearProgram.h` — internal NLP structure
- `PardisoInterface.h` — Intel MKL backend
- `AccelerateInterface.h`, `AccelerateUtils.h` — Apple Accelerate backend
- `MumpsInterface.h` — MUMPS backend

**Step 1: Move public solver headers**

**Step 2: Create `include/tycho/solvers.h`**

```cpp
#pragma once

#include "tycho/vector_functions.h"

#include "tycho/detail/solver_flags.h"
#include "tycho/detail/solver_init.h"
#include "tycho/detail/solver_function_base.h"
#include "tycho/detail/constraint_function.h"
#include "tycho/detail/objective_function.h"
#include "tycho/detail/optimization_problem_base.h"
#include "tycho/detail/optimization_problem.h"
#include "tycho/detail/psiopt.h"
#include "tycho/detail/jet.h"
#include "tycho/detail/iterate_info.h"
```

**Step 3-5: Update forwarding, build, commit**

---

### Task 9: Public headers — Integrators module

**Files:**
- Create: `include/tycho/integrators.h`
- Create: `include/tycho/detail/integrator.h`
- Create: `include/tycho/detail/rk_steppers.h`
- Create: `include/tycho/detail/rk_coeffs.h`

**All integrator headers are public** — the module is entirely template-based
and consumers need all of it:
- `Integrator.h` (80KB) — main integrator template
- `RKSteppers.h` (55KB) — stepper implementations
- `RKCoeffs.h` (18KB) — coefficient tables

**Step 1: Move all integrator headers to `include/tycho/detail/`**

**Step 2: Create `include/tycho/integrators.h`**

```cpp
#pragma once

#include "tycho/vector_functions.h"

#include "tycho/detail/rk_coeffs.h"
#include "tycho/detail/rk_steppers.h"
#include "tycho/detail/integrator.h"
```

**Step 3-5: Update forwarding, build, commit**

---

### Task 10: Public headers — Astro module

**Files:**
- Create: `include/tycho/astro.h`
- Create: `include/tycho/detail/` headers for astro types

**All astro headers are public** — they define ODE models and utilities
that consumers use directly:
- `KeplerModel.h`, `CR3BPModel.h`, `J2.h`, `MEEDynamics.h` — ODE models
- `KeplerPropagator.h`, `KeplerUtils.h` — propagation utilities
- `LambertSolvers.h` — Lambert problem
- `ThrusterModels.h` — thruster models
- `CR3BPUtils.h` — CR3BP utilities

**Step 1: Move all astro headers to `include/tycho/detail/`**

**Step 2: Create `include/tycho/astro.h`**

```cpp
#pragma once

#include "tycho/optimal_control.h"

#include "tycho/detail/kepler_model.h"
#include "tycho/detail/kepler_propagator.h"
#include "tycho/detail/kepler_utils.h"
#include "tycho/detail/cr3bp_model.h"
#include "tycho/detail/cr3bp_utils.h"
#include "tycho/detail/j2.h"
#include "tycho/detail/mee_dynamics.h"
#include "tycho/detail/lambert_solvers.h"
#include "tycho/detail/thruster_models.h"
```

**Step 3-5: Update forwarding, build, commit**

---

### Task 11: Update CMake for public headers

**Files:**
- Modify: `src/CMakeLists.txt` — rename `tycho_core` to `tycho`, update include paths
- Modify: `CMakeLists.txt` (root) — update target references

**Step 1: Rename `tycho_core` to `tycho`**

In `src/CMakeLists.txt`, change the INTERFACE library:
```cmake
# Before:
add_library(tycho_core INTERFACE)
target_link_libraries(tycho_core INTERFACE psiopt utils optimalcontrol vectorfunctions astro)

# After:
add_library(tycho INTERFACE)
target_link_libraries(tycho INTERFACE psiopt utils optimalcontrol vectorfunctions astro)
add_library(tycho::tycho ALIAS tycho)
```

**Step 2: Add public include path**

```cmake
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
```

**Step 3: Update all references to `tycho_core`**

Search for `tycho_core` in all CMakeLists.txt files and replace with `tycho`.
Key locations:
- `src/Bindings/CMakeLists.txt` — `_tychopy` links `tycho` instead of individual libs
- `extensions/CMakeLists.txt` — `tycho_extensions` links `tycho`
- `tests/cpp/CMakeLists.txt` — test target links `tycho`
- `bench/cpp/CMakeLists.txt` — benchmark target links `tycho`
- `examples/cpp_examples/*/CMakeLists.txt` — example links `tycho`

**Step 4: Build and verify**

```bash
cd build && rm -rf * && cmake --preset macos-llvm-release && ninja -j2 all
```

**Step 5: Commit**

```bash
git add src/CMakeLists.txt CMakeLists.txt extensions/ tests/ bench/ examples/
git commit -m "feat: rename tycho_core -> tycho, add public include path"
```

---

### Task 12: Update precompiled headers

**Files:**
- Modify: `src/pch.h` — update includes to use new public header paths
- Modify: `src/Bindings/pch_nb.h` — update if needed

**Step 1: Update `src/pch.h`**

The PCH should include the public headers so that internal code benefits from
precompilation of the public API. Update include paths from `src/` internal paths
to `include/tycho/` public paths where appropriate.

Note: The PCH compilation itself needs both the public `include/` path and the
private `src/` path. The CMake `target_include_directories` already provides both.

**Step 2: Update PCH entries in `src/Bindings/CMakeLists.txt`**

The `pch_bindings` target precompiles several headers. Update paths:
```cmake
target_precompile_headers(pch_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pch_nb.h
    # These may need path updates depending on whether they moved:
    ${CMAKE_SOURCE_DIR}/src/VectorFunctions/Tycho_VectorFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctionsBind.h
)
```

If `Tycho_VectorFunctions.h` now just forwards to `include/tycho/vector_functions.h`,
this should still work.

**Step 3: Build and verify**

```bash
cd build && rm -rf * && cmake --preset macos-llvm-release && ninja -j2 all
```

**Step 4: Commit**

```bash
git add src/pch.h src/Bindings/
git commit -m "feat: update PCH for public header paths"
```

---

### Task 13: Update C++ examples and tests

**Files:**
- Modify: `examples/cpp_examples/brachistochrone/main.cpp`
- Modify: `tests/cpp/**/*.cpp` — update includes
- Modify: `bench/cpp/**/*.cpp` — update includes

**Step 1: Update C++ example includes**

```cpp
// Before:
#include "Tycho.h"
// After:
#include <tycho/tycho.h>
```

And update namespace usage:
```cpp
// Before:
using namespace Tycho;
// After:
using namespace Tycho;
```

(The namespace change should already be done from Task 3, but verify the include
path change.)

**Step 2: Update test includes**

All test files in `tests/cpp/` that include Tycho headers should use the new
public paths: `#include <tycho/tycho.h>` or specific module headers.

**Step 3: Update benchmark includes**

Same pattern for `bench/cpp/` files.

**Step 4: Build and verify**

```bash
cd build && rm -rf * && cmake --preset macos-llvm-release \
    -DBUILD_CPP_TESTS=ON -DBUILD_CPP_BENCHMARKS=ON && ninja -j2 all
ctest --output-on-failure
```

**Step 5: Commit**

```bash
git add examples/ tests/ bench/
git commit -m "feat: update C++ examples/tests/benchmarks for public headers"
```

---

### Task 14: Update documentation

**Files:**
- Modify: `CLAUDE.md` — update include paths, namespace references, library target name
- Modify: `TYCHO_API_DESIGN_PLAN.md` — should already reference new conventions
- Modify: `doc/*.md` — update code examples
- Modify: `TODO.md` — update code examples

**Step 1: Update `CLAUDE.md`**

Key changes:
- `#include "Tycho.h"` -> `#include <tycho/tycho.h>`
- `tycho_core` -> `tycho` (library target)
- Update repository structure section with `include/` directory
- Update any code examples

**Step 2: Update other docs**

**Step 3: Commit**

```bash
git add CLAUDE.md TYCHO_API_DESIGN_PLAN.md TODO.md doc/
git commit -m "docs: update documentation for public headers"
```

---

### Task 15: Full validation

**Step 1: Clean build with all targets**

```bash
cd build && rm -rf *
cmake --preset macos-llvm-release -DBUILD_CPP_TESTS=ON -DBUILD_CPP_BENCHMARKS=ON
ninja -j2 all
```

**Step 2: Run C++ tests**

```bash
ctest --output-on-failure
```

Expected: all tests pass.

**Step 3: Run C++ brachistochrone**

```bash
./examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: "Optimal Solution Found", objective ~1.8013 s.

**Step 4: Run all Python examples**

```bash
python scripts/run_examples.py
```

Expected: all 38 pass.

**Step 5: Verify public header isolation**

Create a minimal test file outside the source tree:
```cpp
// /tmp/test_tycho.cpp
#include <tycho/tycho.h>
int main() {
    auto args = Tycho::Arguments<3>();
    return 0;
}
```

Compile against the build tree:
```bash
clang++ -std=c++20 -I/path/to/tycho/include -I/path/to/tycho/dep/eigen \
    /tmp/test_tycho.cpp -c -o /tmp/test_tycho.o
```

This verifies the public headers are self-contained.

**Step 6: Run clang-format on changed files**

```bash
find src extensions include -name "*.cpp" -o -name "*.h" -print0 | \
    xargs -0 /opt/homebrew/opt/llvm/bin/clang-format -style=file -i
```

**Step 7: Run ruff on Python files**

```bash
ruff format .
ruff check --select I --fix .
```

**Step 8: Final commit if formatting changed anything**

```bash
git add -A
git commit -m "chore: apply formatting after public header migration"
```

---

## Validation Checklist

Before marking Phase 2 complete:

- [ ] `include/tycho/tycho.h` exists and includes all module headers
- [ ] `include/tycho/detail/` contains template implementation bodies
- [ ] `namespace Tycho` used consistently (no lowercase `namespace tycho`)
- [ ] `tycho_core` renamed to `tycho` in all CMake files
- [ ] `tycho::tycho` alias target works
- [ ] Public include path: `#include <tycho/tycho.h>` works from outside `src/`
- [ ] `src/` headers forward-include from `include/tycho/` (no duplication)
- [ ] All C++ tests pass (`ctest --output-on-failure`)
- [ ] All 38 Python examples pass
- [ ] C++ brachistochrone converges
- [ ] External compilation test passes (public headers are self-contained)
- [ ] Formatting applied (clang-format + ruff)

## Notes

**This is the most labor-intensive phase.** The VectorFunctions module alone has
78 headers with deep include chains. The header move must be done incrementally
with builds after each module to catch breakages early.

**Include path strategy:** During the migration, both `src/` and `include/tycho/`
paths will be on the include path. The forwarding headers in `src/` ensure
backward compatibility. Once all modules are migrated, internal `src/` code can
gradually shift to using `#include <tycho/...>` paths, but this is not required
in this phase.

**PCH interaction:** The precompiled header system is sensitive to include path
changes. If PCH mismatches occur, a clean rebuild (`rm -rf build/*`) resolves them.
