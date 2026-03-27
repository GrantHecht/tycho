# Tycho Project Reorganization — Design Spec

**Date:** 2026-03-26
**Scope:** Phase 1 of a broader cleanup — structural reorganization only, no naming convention changes to code identifiers.
**Delivery:** Single PR covering all changes.

---

## Overview

This spec covers two workstreams delivered as one atomic PR:

1. **Move Python examples** into `examples/python_examples/`
2. **Restructure all headers and source files** — remove forwarding headers, rename to snake_case, reorganize `detail/` into domain subdirectories (done atomically to avoid touching include paths multiple times)

A second phase (not covered here) will migrate code identifiers (classes, functions, variables, namespaces) to the agreed naming conventions, module by module.

---

## Agreed Conventions (for reference — applied in Phase 2)

| Category | Convention | Example |
|---|---|---|
| Files | snake_case | `dense_function_base.h` |
| Classes/structs | PascalCase | `DenseFunctionBase` |
| Member functions | snake_case | `compute()`, `input_rows()` |
| Member variables | snake_case + trailing `_` | `is_linear_`, `num_defects_` |
| Local variables | snake_case | `x_dual`, `sin_e` |
| Free functions | snake_case | `classic_to_cartesian()` |
| Template params | PascalCase | `Derived`, `Scalar`, `InputRows` |
| Enum classes | PascalCase (singular) | `BarrierMode` |
| Enum values | PascalCase | `BarrierMode::Probe` |
| Constants/constexpr | UPPER_CASE | `MAX_ITERS`, `TOL` |
| Macros | UPPER_CASE | `DENSE_FUNCTION_BASE_TYPES` |
| Root namespace | lowercase | `tycho` |
| Sub-namespaces | snake_case | `tycho::vf`, `tycho::solvers` |

---

## Step 1: Move Python Examples

### Current layout

```
examples/
├── <27 Python scripts at root>
├── MeshRefinement/          (4 scripts)
├── UpdatedInterface/        (7 scripts)
├── cpp_examples/
│   ├── static/
│   └── builder/
└── Plots/
    ├── Zermelo/
    ├── OrbitContinuation/
    └── MultiSpacecraftOptimization/
```

### Target layout

```
examples/
├── python_examples/
│   ├── <27 scripts>
│   ├── MeshRefinement/      (preserved)
│   ├── UpdatedInterface/    (preserved)
│   └── Plots/               (empty subdirs, gitignored contents)
│       ├── Zermelo/.gitkeep
│       ├── OrbitContinuation/.gitkeep
│       └── MultiSpacecraftOptimization/.gitkeep
└── cpp_examples/
    ├── static/
    └── builder/
```

### Changes required

- `git mv` the 27 root Python scripts into `examples/python_examples/`
- `git mv` `MeshRefinement/` and `UpdatedInterface/` into `examples/python_examples/`
- `git mv` `Plots/` into `examples/python_examples/Plots/`
- Delete any existing plot output files from `Plots/` subdirectories
- Add `.gitkeep` files to each `Plots/` subdirectory so git tracks the empty dirs
- Add `examples/python_examples/Plots/**` (excluding `.gitkeep`) to `.gitignore`
- Update `scripts/run_examples.py`:
  - Change `EXAMPLES_DIR` to point to `examples/python_examples/`
  - Update `REQUIRED_DIRS` paths for the new `Plots/` location
  - Update `PYTHONPATH` setup for the new directory
  - Update `TIMEOUT_OVERRIDES` keys if any paths changed

---

## Step 2: Restructure Headers and Source Files

This step is executed atomically — forwarding header removal, snake_case renaming, and `detail/` reorganization happen together so each `#include` path is rewritten exactly once to its final form.

### 2A: Forwarding Headers to DELETE

These are one-line `#include "tycho/detail/..."` shims in `src/` subdirectories. All are deleted.

**CRITICAL: The following `src/Utils/` headers are NOT forwarding headers — they are real private implementation headers with no `detail/` counterpart. DO NOT DELETE:**
- `src/Utils/ColorText.h` (+ `ColorText.cpp`) — terminal color output
- `src/Utils/EigenSTL.h` — Eigen/STL interop helpers
- `src/Utils/fmtlib.h` (+ `fmtlib.cpp`) — fmt library configuration

**Complete list of forwarding headers to delete:**

<details>
<summary>src/Astro/ (8 files)</summary>

| File | Forwards to |
|------|-------------|
| `src/Astro/CR3BPModel.h` | `tycho/detail/CR3BPModel.h` |
| `src/Astro/J2.h` | `tycho/detail/J2.h` |
| `src/Astro/KeplerModel.h` | `tycho/detail/KeplerModel.h` |
| `src/Astro/KeplerPropagator.h` | `tycho/detail/KeplerPropagator.h` |
| `src/Astro/KeplerUtils.h` | `tycho/detail/KeplerUtils.h` |
| `src/Astro/LambertSolvers.h` | `tycho/detail/LambertSolvers.h` |
| `src/Astro/MEEDynamics.h` | `tycho/detail/MEEDynamics.h` |
| `src/Astro/ThrusterModels.h` | `tycho/detail/ThrusterModels.h` |
</details>

<details>
<summary>src/Integrators/ (3 files)</summary>

| File | Forwards to |
|------|-------------|
| `src/Integrators/Integrator.h` | `tycho/detail/Integrator.h` |
| `src/Integrators/RKCoeffs.h` | `tycho/detail/RKCoeffs.h` |
| `src/Integrators/RKSteppers.h` | `tycho/detail/RKSteppers.h` |
</details>

<details>
<summary>src/OptimalControl/ (28 files)</summary>

| File | Forwards to |
|------|-------------|
| `src/OptimalControl/AutoScalingUtils.h` | `tycho/detail/AutoScalingUtils.h` |
| `src/OptimalControl/Blocked_ODE_Wrapper.h` | `tycho/detail/Blocked_ODE_Wrapper.h` |
| `src/OptimalControl/FDCoeffs.h` | `tycho/detail/FDCoeffs.h` |
| `src/OptimalControl/FDDerivArbitrary.h` | `tycho/detail/FDDerivArbitrary.h` |
| `src/OptimalControl/FDDerivUniform.h` | `tycho/detail/FDDerivUniform.h` |
| `src/OptimalControl/InterfaceTypes.h` | `tycho/detail/InterfaceTypes.h` |
| `src/OptimalControl/LGLCoeffs.h` | `tycho/detail/LGLCoeffs.h` |
| `src/OptimalControl/LGLControlSplines.h` | `tycho/detail/LGLControlSplines.h` |
| `src/OptimalControl/LGLDefects.h` | `tycho/detail/LGLDefects.h` |
| `src/OptimalControl/LGLIntegrals.h` | `tycho/detail/LGLIntegrals.h` |
| `src/OptimalControl/LGLInterpFunctions.h` | `tycho/detail/LGLInterpFunctions.h` |
| `src/OptimalControl/LGLInterpTable.h` | `tycho/detail/LGLInterpTable.h` |
| `src/OptimalControl/LinkFunction.h` | `tycho/detail/LinkFunction.h` |
| `src/OptimalControl/MeshIterateInfo.h` | `tycho/detail/MeshIterateInfo.h` |
| `src/OptimalControl/MeshSpacingConstraints.h` | `tycho/detail/MeshSpacingConstraints.h` |
| `src/OptimalControl/ODE.h` | `tycho/detail/ODE.h` |
| `src/OptimalControl/ODEArguments.h` | `tycho/detail/ODEArguments.h` |
| `src/OptimalControl/ODEPhase.h` | `tycho/detail/ODEPhase.h` |
| `src/OptimalControl/ODEPhaseBase.h` | `tycho/detail/ODEPhaseBase.h` |
| `src/OptimalControl/ODESizes.h` | `tycho/detail/ODESizes.h` |
| `src/OptimalControl/OptimalControlFlags.h` | `tycho/detail/OptimalControlFlags.h` |
| `src/OptimalControl/OptimalControlProblem.h` | `tycho/detail/OptimalControlProblem.h` |
| `src/OptimalControl/PhaseIndexer.h` | `tycho/detail/PhaseIndexer.h` |
| `src/OptimalControl/ShootingDefects.h` | `tycho/detail/ShootingDefects.h` |
| `src/OptimalControl/StateFunction.h` | `tycho/detail/StateFunction.h` |
| `src/OptimalControl/TranscriptionSizing.h` | `tycho/detail/TranscriptionSizing.h` |
| `src/OptimalControl/TrapezoidalDefects.h` | `tycho/detail/TrapezoidalDefects.h` |
| `src/OptimalControl/TrapezoidalIntegrals.h` | `tycho/detail/TrapezoidalIntegrals.h` |
| `src/OptimalControl/ValueLock.h` | `tycho/detail/ValueLock.h` |
</details>

<details>
<summary>src/Solvers/ (14 files)</summary>

| File | Forwards to |
|------|-------------|
| `src/Solvers/AccelerateInterface.h` | `tycho/detail/AccelerateInterface.h` |
| `src/Solvers/AccelerateUtils.h` | `tycho/detail/AccelerateUtils.h` |
| `src/Solvers/ConstraintFunction.h` | `tycho/detail/ConstraintFunction.h` |
| `src/Solvers/IterateInfo.h` | `tycho/detail/IterateInfo.h` |
| `src/Solvers/Jet.h` | `tycho/detail/Jet.h` |
| `src/Solvers/NonLinearProgram.h` | `tycho/detail/NonLinearProgram.h` |
| `src/Solvers/ObjectiveFunction.h` | `tycho/detail/ObjectiveFunction.h` |
| `src/Solvers/OptimizationProblem.h` | `tycho/detail/OptimizationProblem.h` |
| `src/Solvers/OptimizationProblemBase.h` | `tycho/detail/OptimizationProblemBase.h` |
| `src/Solvers/PardisoInterface.h` | `tycho/detail/PardisoInterface.h` |
| `src/Solvers/PSIOPT.h` | `tycho/detail/PSIOPT.h` |
| `src/Solvers/SolverFlags.h` | `tycho/detail/SolverFlags.h` |
| `src/Solvers/SolverFunctionBase.h` | `tycho/detail/SolverFunctionBase.h` |
| `src/Solvers/SolverInit.h` | `tycho/detail/SolverInit.h` |
</details>

<details>
<summary>src/TypeDefs/ (1 file)</summary>

| File | Forwards to |
|------|-------------|
| `src/TypeDefs/EigenTypes.h` | `tycho/detail/eigen_types.h` |
</details>

<details>
<summary>src/Utils/ (13 files — excludes 3 real private headers)</summary>

| File | Forwards to |
|------|-------------|
| `src/Utils/CRTPBase.h` | `tycho/detail/crtp_base.h` |
| `src/Utils/FlatMap.h` | `tycho/detail/flat_map.h` |
| `src/Utils/FunctionReturnType.h` | `tycho/detail/function_return_type.h` |
| `src/Utils/GetCoreCount.h` | `tycho/detail/get_core_count.h` |
| `src/Utils/LambdaJumpTable.h` | `tycho/detail/LambdaJumpTable.h` |
| `src/Utils/MathFunctions.h` | `tycho/detail/math_functions.h` |
| `src/Utils/MemoryManagement.h` | `tycho/detail/memory_management.h` |
| `src/Utils/SizingHelpers.h` | `tycho/detail/sizing_helpers.h` |
| `src/Utils/STDExtensions.h` | `tycho/detail/std_extensions.h` |
| `src/Utils/ThreadPool.h` | `tycho/detail/thread_pool.h` |
| `src/Utils/Timer.h` | `tycho/detail/timer.h` |
| `src/Utils/TupleIterator.h` | `tycho/detail/tuple_iterator.h` |
| `src/Utils/TypeName.h` | `tycho/detail/type_name.h` |
| `src/Utils/TypeStorage.h` | `tycho/detail/type_storage.h` |
</details>

<details>
<summary>src/VectorFunctions/ (75 files — excludes Tycho_VectorFunctions.h aggregate)</summary>

| File | Forwards to |
|------|-------------|
| `src/VectorFunctions/AssigmentTypes.h` | `tycho/detail/AssigmentTypes.h` |
| `src/VectorFunctions/BinaryMath.h` | `tycho/detail/BinaryMath.h` |
| `src/VectorFunctions/ComputableBase.h` | `tycho/detail/ComputableBase.h` |
| `src/VectorFunctions/Computable.h` | `tycho/detail/Computable.h` |
| `src/VectorFunctions/DenseDerivatives.h` | `tycho/detail/DenseDerivatives.h` |
| `src/VectorFunctions/DenseFunctionBase.h` | `tycho/detail/DenseFunctionBase.h` |
| `src/VectorFunctions/DenseFunction.h` | `tycho/detail/DenseFunction.h` |
| `src/VectorFunctions/DenseFunctionOperations.h` | `tycho/detail/DenseFunctionOperations.h` |
| `src/VectorFunctions/DenseScalarFunctionBase.h` | `tycho/detail/DenseScalarFunctionBase.h` |
| `src/VectorFunctions/DetectDiagonal.h` | `tycho/detail/DetectDiagonal.h` |
| `src/VectorFunctions/DetectSuperScalar.h` | `tycho/detail/DetectSuperScalar.h` |
| `src/VectorFunctions/FunctionalFlags.h` | `tycho/detail/FunctionalFlags.h` |
| `src/VectorFunctions/FunctionDomains.h` | `tycho/detail/FunctionDomains.h` |
| `src/VectorFunctions/FunctionTemplates.h` | `tycho/detail/FunctionTemplates.h` |
| `src/VectorFunctions/FunctionTypeDefMacros.h` | `tycho/detail/FunctionTypeDefMacros.h` |
| `src/VectorFunctions/IndexingData.h` | `tycho/detail/IndexingData.h` |
| `src/VectorFunctions/InputOuputSize.h` | `tycho/detail/InputOuputSize.h` |
| `src/VectorFunctions/MathOverloads.h` | `tycho/detail/MathOverloads.h` |
| `src/VectorFunctions/ObjectDetectors.h` | `tycho/detail/ObjectDetectors.h` |
| `src/VectorFunctions/OperatorOverloads.h` | `tycho/detail/OperatorOverloads.h` |
| `src/VectorFunctions/PythonArgParsing.h` | `tycho/detail/PythonArgParsing.h` |
| `src/VectorFunctions/SparseFunctionBase.h` | `tycho/detail/SparseFunctionBase.h` |
| `src/VectorFunctions/VectorFunctionConcepts.h` | `tycho/detail/VectorFunctionConcepts.h` |
| `src/VectorFunctions/VectorFunction.h` | `tycho/detail/VectorFunction.h` |
| `src/VectorFunctions/CommonFunctions/ArcTan2.h` | `tycho/detail/ArcTan2.h` |
| `src/VectorFunctions/CommonFunctions/AutoDiffFunction.h` | `tycho/detail/AutoDiffFunction.h` |
| `src/VectorFunctions/CommonFunctions/CallAndAppend.h` | `tycho/detail/CallAndAppend.h` |
| `src/VectorFunctions/CommonFunctions/CommonFunctions.h` | `tycho/detail/CommonFunctions.h` |
| `src/VectorFunctions/CommonFunctions/Comparative.h` | `tycho/detail/Comparative.h` |
| `src/VectorFunctions/CommonFunctions/Conditional.h` | `tycho/detail/Conditional.h` |
| `src/VectorFunctions/CommonFunctions/Constant.h` | `tycho/detail/Constant.h` |
| `src/VectorFunctions/CommonFunctions/CrossProduct.h` | `tycho/detail/CrossProduct.h` |
| `src/VectorFunctions/CommonFunctions/CwiseOperators.h` | `tycho/detail/CwiseOperators.h` |
| `src/VectorFunctions/CommonFunctions/CwiseProduct.h` | `tycho/detail/CwiseProduct.h` |
| `src/VectorFunctions/CommonFunctions/CwiseSum.h` | `tycho/detail/CwiseSum.h` |
| `src/VectorFunctions/CommonFunctions/DotProduct.h` | `tycho/detail/DotProduct.h` |
| `src/VectorFunctions/CommonFunctions/Elements.h` | `tycho/detail/Elements.h` |
| `src/VectorFunctions/CommonFunctions/ExpressionFwdDeclarations.h` | `tycho/detail/ExpressionFwdDeclarations.h` |
| `src/VectorFunctions/CommonFunctions/For.h` | `tycho/detail/For.h` |
| `src/VectorFunctions/CommonFunctions/FunctionHolder.h` | `tycho/detail/FunctionHolder.h` |
| `src/VectorFunctions/CommonFunctions/FunctionVectorSums.h` | `tycho/detail/FunctionVectorSums.h` |
| `src/VectorFunctions/CommonFunctions/InterpTable1D.h` | `tycho/detail/InterpTable1D.h` |
| `src/VectorFunctions/CommonFunctions/InterpTable2D.h` | `tycho/detail/InterpTable2D.h` |
| `src/VectorFunctions/CommonFunctions/InterpTable3D.h` | `tycho/detail/InterpTable3D.h` |
| `src/VectorFunctions/CommonFunctions/InterpTable4D.h` | `tycho/detail/InterpTable4D.h` |
| `src/VectorFunctions/CommonFunctions/IOScaled.h` | `tycho/detail/IOScaled.h` |
| `src/VectorFunctions/CommonFunctions/LambdaFunction.h` | `tycho/detail/LambdaFunction.h` |
| `src/VectorFunctions/CommonFunctions/MatrixFunction.h` | `tycho/detail/MatrixFunction.h` |
| `src/VectorFunctions/CommonFunctions/MatrixInverse.h` | `tycho/detail/MatrixInverse.h` |
| `src/VectorFunctions/CommonFunctions/MatrixProduct.h` | `tycho/detail/MatrixProduct.h` |
| `src/VectorFunctions/CommonFunctions/NestedFunction.h` | `tycho/detail/NestedFunction.h` |
| `src/VectorFunctions/CommonFunctions/Normalized.h` | `tycho/detail/Normalized.h` |
| `src/VectorFunctions/CommonFunctions/Norms.h` | `tycho/detail/Norms.h` |
| `src/VectorFunctions/CommonFunctions/Padded.h` | `tycho/detail/Padded.h` |
| `src/VectorFunctions/CommonFunctions/ParsedInput.h` | `tycho/detail/ParsedInput.h` |
| `src/VectorFunctions/CommonFunctions/RootFinder.h` | `tycho/detail/RootFinder.h` |
| `src/VectorFunctions/CommonFunctions/Scaled.h` | `tycho/detail/Scaled.h` |
| `src/VectorFunctions/CommonFunctions/Segment.h` | `tycho/detail/Segment.h` |
| `src/VectorFunctions/CommonFunctions/SignFunction.h` | `tycho/detail/SignFunction.h` |
| `src/VectorFunctions/CommonFunctions/Stacked.h` | `tycho/detail/Stacked.h` |
| `src/VectorFunctions/CommonFunctions/Summation.h` | `tycho/detail/Summation.h` |
| `src/VectorFunctions/CommonFunctions/Value.h` | `tycho/detail/Value.h` |
| `src/VectorFunctions/CommonFunctions/VectorProducts.h` | `tycho/detail/VectorProducts.h` |
| `src/VectorFunctions/CommonFunctions/VectorScalarFunctionDivision.h` | `tycho/detail/VectorScalarFunctionDivision.h` |
| `src/VectorFunctions/CommonFunctions/VectorScalarFunctionProduct.h` | `tycho/detail/VectorScalarFunctionProduct.h` |
| `src/VectorFunctions/DenseDifferentiation/DenseAutodiffFwd.h` | `tycho/detail/DenseAutodiffFwd.h` |
| `src/VectorFunctions/DenseDifferentiation/DenseFDiffCentArray.h` | `tycho/detail/DenseFDiffCentArray.h` |
| `src/VectorFunctions/DenseDifferentiation/DenseFDiffFwd.h` | `tycho/detail/DenseFDiffFwd.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h` | `tycho/detail/ConditionalTypeErasure.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h` | `tycho/detail/DenseFunctionSpecs.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h` | `tycho/detail/GenericComparative.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h` | `tycho/detail/GenericConditional.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h` | `tycho/detail/GenericFunction.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h` | `tycho/detail/GFTypeErasure.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h` | `tycho/detail/SizingSpecs.h` |
| `src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h` | `tycho/detail/SolverInterfaceSpecs.h` |
</details>

**Total forwarding headers to delete: ~143**

### 2B: Aggregate Headers to KEEP and RENAME

These contain real declarations or multi-header aggregation. Keep in their current `src/` locations, rename to snake_case:

| Current path | New path |
|---|---|
| `src/Tycho.h` | `src/tycho_internal.h` |
| `src/VectorFunctions/Tycho_VectorFunctions.h` | `src/VectorFunctions/tycho_vector_functions.h` |
| `src/OptimalControl/Tycho_OptimalControl.h` | `src/OptimalControl/tycho_optimal_control.h` |
| `src/Solvers/Tycho_Solvers.h` | `src/Solvers/tycho_solvers.h` |
| `src/Utils/Tycho_Utils.h` | `src/Utils/tycho_utils.h` |
| `src/Astro/Tycho_Astro.h` | `src/Astro/tycho_astro.h` |
| `src/Integrators/Tycho_Integrators.h` | `src/Integrators/tycho_integrators.h` |
| `src/TypeDefs/Tycho_TypeDefs.h` | `src/TypeDefs/tycho_typedefs.h` |

These aggregate headers must also have their internal `#include` directives updated to use the new `detail/` paths (since they currently include forwarding headers that are being deleted).

### 2C: Real Private Headers to KEEP and RENAME

These are implementation headers in `src/` with no `detail/` counterpart:

| Current path | New path |
|---|---|
| `src/Utils/ColorText.h` | `src/Utils/color_text.h` |
| `src/Utils/EigenSTL.h` | `src/Utils/eigen_stl.h` |
| `src/Utils/fmtlib.h` | `src/Utils/fmtlib.h` (already snake_case) |

### 2D: `PythonArgParsing.h` — Move to Bindings

`include/tycho/detail/PythonArgParsing.h` is a binding-only header that does not belong in the public API. Move it to `src/Bindings/VectorFunctions/PythonArgParsing.h` (which already exists as a forwarding header — replace it with the real content). Rename to `src/Bindings/VectorFunctions/python_arg_parsing.h`.

### 2E: `src/*.cpp` File Renames

| Current path | New path |
|---|---|
| `src/pch.cpp` | `src/pch.cpp` (unchanged) |
| `src/Astro/AccellerationModels.cpp` | `src/Astro/accelleration_models.cpp` |
| `src/Astro/LambertSolvers.cpp` | `src/Astro/lambert_solvers.cpp` |
| `src/OptimalControl/LGLInterpTable.cpp` | `src/OptimalControl/lgl_interp_table.cpp` |
| `src/OptimalControl/ODEPhaseBase.cpp` | `src/OptimalControl/ode_phase_base.cpp` |
| `src/OptimalControl/OptimalControlProblem.cpp` | `src/OptimalControl/optimal_control_problem.cpp` |
| `src/OptimalControl/PhaseIndexer.cpp` | `src/OptimalControl/phase_indexer.cpp` |
| `src/OptimalControl/RuntimeODE.cpp` | `src/OptimalControl/runtime_ode.cpp` |
| `src/Solvers/NonLinearProgram.cpp` | `src/Solvers/non_linear_program.cpp` |
| `src/Solvers/OptimizationProblem.cpp` | `src/Solvers/optimization_problem.cpp` |
| `src/Solvers/PSIOPT.cpp` | `src/Solvers/psiopt.cpp` |
| `src/Solvers/SolverInit.cpp` | `src/Solvers/solver_init.cpp` |
| `src/Utils/ColorText.cpp` | `src/Utils/color_text.cpp` |
| `src/Utils/fmtlib.cpp` | `src/Utils/fmtlib.cpp` (unchanged) |
| `src/Utils/GetCoreCount.cpp` | `src/Utils/get_core_count.cpp` |
| `src/Utils/MemoryManagement.cpp` | `src/Utils/memory_management.cpp` |
| `src/Utils/ThreadPool.cpp` | `src/Utils/thread_pool.cpp` |
| `src/VectorFunctions/FunctionDomains.cpp` | `src/VectorFunctions/function_domains.cpp` |

### 2F: Complete `detail/` Header Mapping

**The target directory tree below is AUTHORITATIVE.** Where the snake_case name differs from a mechanical PascalCase→snake_case transform, it is intentional (typo fixes, semantic renames).

**Non-mechanical renames (called out explicitly):**
- `AssigmentTypes.h` → `assignment_types.h` (typo fix: "Assigment" → "assignment")
- `InputOuputSize.h` → `input_output_size.h` (typo fix: "Ouput" → "output")
- `ConditionalTypeErasure.h` → `conditional_base.h` (semantic rename — this is the base class for conditional type erasure)
- `AutoDiffFunction.h` → `autodiff_function.h` (the class is `ADFun`, this wraps autodiff library functions)
- `Blocked_ODE_Wrapper.h` → `blocked_ode_wrapper.h` (already had underscores, just lowercase)
- `DenseFunctionSpecs.h` → `dense_function_specs.h` (moved from solvers context to vf/core — it defines VF specs)
- `SizingSpecs.h` → `sizing_specs.h` (stays in solvers)
- `SolverInterfaceSpecs.h` → `solver_interface_specs.h` (stays in solvers)
- `IndexingData.h` → `indexing_data.h` (stays in solvers)

<details>
<summary>detail/utils/ (14 headers)</summary>

| Old path (`detail/`) | New path (`detail/utils/`) |
|---|---|
| `crtp_base.h` | `utils/crtp_base.h` |
| `flat_map.h` | `utils/flat_map.h` |
| `function_return_type.h` | `utils/function_return_type.h` |
| `get_core_count.h` | `utils/get_core_count.h` |
| `LambdaJumpTable.h` | `utils/lambda_jump_table.h` |
| `math_functions.h` | `utils/math_functions.h` |
| `memory_management.h` | `utils/memory_management.h` |
| `sizing_helpers.h` | `utils/sizing_helpers.h` |
| `std_extensions.h` | `utils/std_extensions.h` |
| `thread_pool.h` | `utils/thread_pool.h` |
| `timer.h` | `utils/timer.h` |
| `tuple_iterator.h` | `utils/tuple_iterator.h` |
| `type_name.h` | `utils/type_name.h` |
| `type_storage.h` | `utils/type_storage.h` |
</details>

<details>
<summary>detail/typedefs/ (1 header)</summary>

| Old path (`detail/`) | New path (`detail/typedefs/`) |
|---|---|
| `eigen_types.h` | `typedefs/eigen_types.h` |
</details>

<details>
<summary>detail/vf/core/ (18 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/core/`) |
|---|---|
| `AssigmentTypes.h` | `vf/core/assignment_types.h` |
| `ComputableBase.h` | `vf/core/computable_base.h` |
| `Computable.h` | `vf/core/computable.h` |
| `DenseFunctionBase.h` | `vf/core/dense_function_base.h` |
| `DenseFunction.h` | `vf/core/dense_function.h` |
| `DenseFunctionOperations.h` | `vf/core/dense_function_operations.h` |
| `DenseFunctionSpecs.h` | `vf/core/dense_function_specs.h` |
| `DenseScalarFunctionBase.h` | `vf/core/dense_scalar_function_base.h` |
| `ExpressionFwdDeclarations.h` | `vf/core/expression_fwd_declarations.h` |
| `FunctionalFlags.h` | `vf/core/functional_flags.h` |
| `FunctionDomains.h` | `vf/core/function_domains.h` |
| `FunctionTemplates.h` | `vf/core/function_templates.h` |
| `FunctionTypeDefMacros.h` | `vf/core/function_type_def_macros.h` |
| `InputOuputSize.h` | `vf/core/input_output_size.h` |
| `ObjectDetectors.h` | `vf/core/object_detectors.h` |
| `SparseFunctionBase.h` | `vf/core/sparse_function_base.h` |
| `VectorFunctionConcepts.h` | `vf/core/vector_function_concepts.h` |
| `VectorFunction.h` | `vf/core/vector_function.h` |
</details>

<details>
<summary>detail/vf/expressions/ (9 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/expressions/`) |
|---|---|
| `CallAndAppend.h` | `vf/expressions/call_and_append.h` |
| `For.h` | `vf/expressions/for.h` |
| `FunctionHolder.h` | `vf/expressions/function_holder.h` |
| `LambdaFunction.h` | `vf/expressions/lambda_function.h` |
| `NestedFunction.h` | `vf/expressions/nested_function.h` |
| `ParsedInput.h` | `vf/expressions/parsed_input.h` |
| `Segment.h` | `vf/expressions/segment.h` |
| `Stacked.h` | `vf/expressions/stacked.h` |
| `Summation.h` | `vf/expressions/summation.h` |
</details>

<details>
<summary>detail/vf/operators/ (20 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/operators/`) |
|---|---|
| `ArcTan2.h` | `vf/operators/arc_tan2.h` |
| `BinaryMath.h` | `vf/operators/binary_math.h` |
| `CrossProduct.h` | `vf/operators/cross_product.h` |
| `CwiseOperators.h` | `vf/operators/cwise_operators.h` |
| `CwiseProduct.h` | `vf/operators/cwise_product.h` |
| `CwiseSum.h` | `vf/operators/cwise_sum.h` |
| `DotProduct.h` | `vf/operators/dot_product.h` |
| `FunctionVectorSums.h` | `vf/operators/function_vector_sums.h` |
| `MathOverloads.h` | `vf/operators/math_overloads.h` |
| `MatrixFunction.h` | `vf/operators/matrix_function.h` |
| `MatrixInverse.h` | `vf/operators/matrix_inverse.h` |
| `MatrixProduct.h` | `vf/operators/matrix_product.h` |
| `Normalized.h` | `vf/operators/normalized.h` |
| `Norms.h` | `vf/operators/norms.h` |
| `OperatorOverloads.h` | `vf/operators/operator_overloads.h` |
| `RootFinder.h` | `vf/operators/root_finder.h` |
| `SignFunction.h` | `vf/operators/sign_function.h` |
| `VectorProducts.h` | `vf/operators/vector_products.h` |
| `VectorScalarFunctionDivision.h` | `vf/operators/vector_scalar_function_division.h` |
| `VectorScalarFunctionProduct.h` | `vf/operators/vector_scalar_function_product.h` |
</details>

<details>
<summary>detail/vf/derivatives/ (9 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/derivatives/`) |
|---|---|
| `DenseAutodiffFwd.h` | `vf/derivatives/dense_autodiff_fwd.h` |
| `DenseDerivatives.h` | `vf/derivatives/dense_derivatives.h` |
| `DenseFDiffCentArray.h` | `vf/derivatives/dense_fdiff_cent_array.h` |
| `DenseFDiffFwd.h` | `vf/derivatives/dense_fdiff_fwd.h` |
| `DetectDiagonal.h` | `vf/derivatives/detect_diagonal.h` |
| `DetectSuperScalar.h` | `vf/derivatives/detect_super_scalar.h` |
| `FDCoeffs.h` | `vf/derivatives/fd_coeffs.h` |
| `FDDerivArbitrary.h` | `vf/derivatives/fd_deriv_arbitrary.h` |
| `FDDerivUniform.h` | `vf/derivatives/fd_deriv_uniform.h` |
</details>

<details>
<summary>detail/vf/scaling/ (4 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/scaling/`) |
|---|---|
| `AutoScalingUtils.h` | `vf/scaling/auto_scaling_utils.h` |
| `IOScaled.h` | `vf/scaling/io_scaled.h` |
| `Padded.h` | `vf/scaling/padded.h` |
| `Scaled.h` | `vf/scaling/scaled.h` |
</details>

<details>
<summary>detail/vf/type_erasure/ (8 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/type_erasure/`) |
|---|---|
| `AutoDiffFunction.h` | `vf/type_erasure/autodiff_function.h` |
| `Comparative.h` | `vf/type_erasure/comparative.h` |
| `Conditional.h` | `vf/type_erasure/conditional.h` |
| `ConditionalTypeErasure.h` | `vf/type_erasure/conditional_base.h` |
| `GenericComparative.h` | `vf/type_erasure/generic_comparative.h` |
| `GenericConditional.h` | `vf/type_erasure/generic_conditional.h` |
| `GenericFunction.h` | `vf/type_erasure/generic_function.h` |
| `GFTypeErasure.h` | `vf/type_erasure/gf_type_erasure.h` |
</details>

<details>
<summary>detail/vf/common/ (5 headers)</summary>

| Old path (`detail/`) | New path (`detail/vf/common/`) |
|---|---|
| `CommonFunctions.h` | `vf/common/common_functions.h` |
| `Constant.h` | `vf/common/constant.h` |
| `Elements.h` | `vf/common/elements.h` |
| `Value.h` | `vf/common/value.h` |
| `ValueLock.h` | `vf/common/value_lock.h` |
</details>

<details>
<summary>detail/integrators/ (3 headers)</summary>

| Old path (`detail/`) | New path (`detail/integrators/`) |
|---|---|
| `Integrator.h` | `integrators/integrator.h` |
| `RKCoeffs.h` | `integrators/rk_coeffs.h` |
| `RKSteppers.h` | `integrators/rk_steppers.h` |
</details>

<details>
<summary>detail/optimal_control/core/ (6 headers)</summary>

| Old path (`detail/`) | New path (`detail/optimal_control/core/`) |
|---|---|
| `InterfaceTypes.h` | `optimal_control/core/interface_types.h` |
| `LinkFunction.h` | `optimal_control/core/link_function.h` |
| `ODEArguments.h` | `optimal_control/core/ode_arguments.h` |
| `ODESizes.h` | `optimal_control/core/ode_sizes.h` |
| `OptimalControlFlags.h` | `optimal_control/core/optimal_control_flags.h` |
| `StateFunction.h` | `optimal_control/core/state_function.h` |
</details>

<details>
<summary>detail/optimal_control/phase/ (6 headers)</summary>

| Old path (`detail/`) | New path (`detail/optimal_control/phase/`) |
|---|---|
| `MeshIterateInfo.h` | `optimal_control/phase/mesh_iterate_info.h` |
| `ODE.h` | `optimal_control/phase/ode.h` |
| `ODEPhase.h` | `optimal_control/phase/ode_phase.h` |
| `ODEPhaseBase.h` | `optimal_control/phase/ode_phase_base.h` |
| `OptimalControlProblem.h` | `optimal_control/phase/optimal_control_problem.h` |
| `PhaseIndexer.h` | `optimal_control/phase/phase_indexer.h` |
</details>

<details>
<summary>detail/optimal_control/transcription/ (12 headers)</summary>

| Old path (`detail/`) | New path (`detail/optimal_control/transcription/`) |
|---|---|
| `Blocked_ODE_Wrapper.h` | `optimal_control/transcription/blocked_ode_wrapper.h` |
| `LGLCoeffs.h` | `optimal_control/transcription/lgl_coeffs.h` |
| `LGLControlSplines.h` | `optimal_control/transcription/lgl_control_splines.h` |
| `LGLDefects.h` | `optimal_control/transcription/lgl_defects.h` |
| `LGLIntegrals.h` | `optimal_control/transcription/lgl_integrals.h` |
| `LGLInterpFunctions.h` | `optimal_control/transcription/lgl_interp_functions.h` |
| `LGLInterpTable.h` | `optimal_control/transcription/lgl_interp_table.h` |
| `MeshSpacingConstraints.h` | `optimal_control/transcription/mesh_spacing_constraints.h` |
| `ShootingDefects.h` | `optimal_control/transcription/shooting_defects.h` |
| `TranscriptionSizing.h` | `optimal_control/transcription/transcription_sizing.h` |
| `TrapezoidalDefects.h` | `optimal_control/transcription/trapezoidal_defects.h` |
| `TrapezoidalIntegrals.h` | `optimal_control/transcription/trapezoidal_integrals.h` |
</details>

<details>
<summary>detail/optimal_control/interp/ (4 headers)</summary>

| Old path (`detail/`) | New path (`detail/optimal_control/interp/`) |
|---|---|
| `InterpTable1D.h` | `optimal_control/interp/interp_table_1d.h` |
| `InterpTable2D.h` | `optimal_control/interp/interp_table_2d.h` |
| `InterpTable3D.h` | `optimal_control/interp/interp_table_3d.h` |
| `InterpTable4D.h` | `optimal_control/interp/interp_table_4d.h` |
</details>

<details>
<summary>detail/optimal_control/builder/ (5 headers)</summary>

| Old path (`detail/`) | New path (`detail/optimal_control/builder/`) |
|---|---|
| `ocp_wrapper.h` | `optimal_control/builder/ocp_wrapper.h` |
| `ode_builder.h` | `optimal_control/builder/ode_builder.h` |
| `phase_wrapper.h` | `optimal_control/builder/phase_wrapper.h` |
| `runtime_ode.h` | `optimal_control/builder/runtime_ode.h` |
| `var_registry.h` | `optimal_control/builder/var_registry.h` |
</details>

<details>
<summary>detail/solvers/ (14 headers at root + 3 in linear/)</summary>

| Old path (`detail/`) | New path (`detail/solvers/`) |
|---|---|
| `ConstraintFunction.h` | `solvers/constraint_function.h` |
| `IndexingData.h` | `solvers/indexing_data.h` |
| `IterateInfo.h` | `solvers/iterate_info.h` |
| `Jet.h` | `solvers/jet.h` |
| `NonLinearProgram.h` | `solvers/non_linear_program.h` |
| `ObjectiveFunction.h` | `solvers/objective_function.h` |
| `OptimizationProblem.h` | `solvers/optimization_problem.h` |
| `OptimizationProblemBase.h` | `solvers/optimization_problem_base.h` |
| `PSIOPT.h` | `solvers/psiopt.h` |
| `SizingSpecs.h` | `solvers/sizing_specs.h` |
| `SolverFlags.h` | `solvers/solver_flags.h` |
| `SolverFunctionBase.h` | `solvers/solver_function_base.h` |
| `SolverInit.h` | `solvers/solver_init.h` |
| `SolverInterfaceSpecs.h` | `solvers/solver_interface_specs.h` |
| `AccelerateInterface.h` | `solvers/linear/accelerate_interface.h` |
| `AccelerateUtils.h` | `solvers/linear/accelerate_utils.h` |
| `PardisoInterface.h` | `solvers/linear/pardiso_interface.h` |
</details>

<details>
<summary>detail/astro/ (8 headers)</summary>

| Old path (`detail/`) | New path (`detail/astro/`) |
|---|---|
| `CR3BPModel.h` | `astro/cr3bp_model.h` |
| `J2.h` | `astro/j2.h` |
| `KeplerModel.h` | `astro/kepler_model.h` |
| `KeplerPropagator.h` | `astro/kepler_propagator.h` |
| `KeplerUtils.h` | `astro/kepler_utils.h` |
| `LambertSolvers.h` | `astro/lambert_solvers.h` |
| `MEEDynamics.h` | `astro/mee_dynamics.h` |
| `ThrusterModels.h` | `astro/thruster_models.h` |
</details>

**Total: 151 headers mapped** (150 original + 1 `LambdaJumpTable.h` that was missing from initial count)

### 2F-notes: Design Decisions for Header Placement

- **`vf/` mirrors the `tycho::vf` namespace** agreed upon for Phase 2.
- **LGL headers in `optimal_control/transcription/`** — these are collocation transcription machinery included by `optimal_control.h`, not `integrators.h`.
- **Interpolation tables (`InterpTable1D-4D`) in `optimal_control/interp/`** — although their forwarding headers lived in `src/VectorFunctions/CommonFunctions/`, the tables are included by `optimal_control.h` (not `vector_functions.h`) and are primarily used for table-driven dynamics within the OC framework (drag tables, thrust profiles, etc.).
- **`typedefs/` gets its own directory** for symmetry, even with one file.
- **`PythonArgParsing.h`** belongs in `src/Bindings/`, not `detail/`. Moved there as part of this cleanup.

### 2G: `#include` Path Updates

Every file that contains an `#include` referencing an old path must be updated to use the new final path. This is the bulk of the mechanical work.

**Files requiring `#include` updates (by category):**

1. **Top-level public headers** (`include/tycho/*.h`) — update `#include "tycho/detail/..."` to new subdirectory paths
2. **All `detail/` headers** — internal cross-includes must use new paths
3. **`src/pch.h`** — currently includes forwarding headers by bare name. Must be rewritten. Concrete target state for the Tycho-specific includes block:
   ```cpp
   // Typedefs (was: "TypeDefs/EigenTypes.h" forwarding header)
   #include "tycho/detail/typedefs/eigen_types.h"

   // Real private headers (no detail/ counterpart — use relative paths)
   #include "Utils/eigen_stl.h"
   #include "Utils/fmtlib.h"

   // Utils (were: forwarding headers like "Utils/ThreadPool.h")
   #include "tycho/detail/utils/get_core_count.h"
   #include "tycho/detail/utils/lambda_jump_table.h"
   #include "tycho/detail/utils/math_functions.h"
   #include "tycho/detail/utils/std_extensions.h"
   #include "tycho/detail/utils/thread_pool.h"
   #include "tycho/detail/utils/timer.h"
   #include "tycho/detail/utils/tuple_iterator.h"
   #include "tycho/detail/utils/type_name.h"
   #include "tycho/detail/utils/type_storage.h"
   ```
   Note: `ColorText.h` is not in `pch.h` (it has its own `.cpp` — included only where needed).
4. **Aggregate headers** (`src/*/Tycho_*.h` → `src/*/tycho_*.h`) — update internal includes
5. **All `src/*.cpp` files** — currently use bare includes like `#include "PSIOPT.h"` that resolve via `_TYCHO_SRC_INCLUDES` to forwarding headers. Must be rewritten to `#include "tycho/detail/solvers/psiopt.h"` etc.
6. **`src/Bindings/pch_nb.h`** — update includes of aggregate headers
7. **All `src/Bindings/*.cpp` and `src/Bindings/*.h` files** — update includes
8. **`tests/cpp/` test files** — any that include `detail/` headers directly
9. **`bench/cpp/` benchmark files** — any that include `detail/` headers or forwarding headers (e.g., `bench/cpp/utils/bench_utils.cpp` includes `"Utils/ThreadPool.h"`)
10. **`src/PyDocString/` files** — these are binding docstring headers; update if they include `detail/` paths
11. **`extensions/` files** — `extensions/Tycho_Extensions.h` includes `"Tycho_VectorFunctions.h"` via PCH reuse; must be updated to use the renamed aggregate header path

### 2H: CMakeLists.txt Updates

**`src/CMakeLists.txt`:**

1. **`_TYCHO_SRC_INCLUDES`** — all 7 subdirectory paths (`OptimalControl`, `Solvers`, `TypeDefs`, `Utils`, `VectorFunctions`, `Integrators`, and the `src/` root) must be **retained** for now. The `src/` subdirectories still contain aggregate headers, real private headers, and `.cpp` files that use relative includes. These paths will be revisited in Phase 2 if `src/` subdirectory renaming happens. The only change needed is ensuring `${PROJECT_SOURCE_DIR}/include` remains first (it already is).

2. **`add_library()` source lists** — remove deleted forwarding `.h` files, update renamed `.cpp` and aggregate `.h` file references:

   `pch`:
   - `pch.h pch.cpp` (unchanged)
   - Update `target_precompile_headers` reference: `VectorFunctions/Tycho_VectorFunctions.h` → `VectorFunctions/tycho_vector_functions.h`

   `optimalcontrol`:
   - Remove forwarding `.h` entries: `OptimalControl/ODEPhaseBase.h`, `OptimalControl/PhaseIndexer.h`, `OptimalControl/OptimalControlProblem.h`, `OptimalControl/LGLInterpTable.h`
   - Rename `.cpp` entries: `ODEPhaseBase.cpp` → `ode_phase_base.cpp`, `PhaseIndexer.cpp` → `phase_indexer.cpp`, etc.
   - Rename aggregate: `Tycho_OptimalControl.h` → `tycho_optimal_control.h`

   `psiopt`:
   - Remove forwarding `.h` entries: `Solvers/PSIOPT.h`, `Solvers/NonLinearProgram.h`, `Solvers/Jet.h`, `Solvers/OptimizationProblemBase.h`, `Solvers/OptimizationProblem.h`, `Solvers/SolverInit.h`
   - Rename `.cpp` entries accordingly

   `utils`:
   - Remove forwarding `.h` entries: `Utils/GetCoreCount.h`, `Utils/MemoryManagement.h`, `Utils/ThreadPool.h`
   - Rename real `.h` entries: `Utils/ColorText.h` → `Utils/color_text.h`
   - Rename `.cpp` entries accordingly

   `astro`:
   - Remove forwarding `.h` entries: `Astro/KeplerUtils.h`, `Astro/KeplerModel.h`, `Astro/LambertSolvers.h`
   - Rename aggregate: `Tycho_Astro.h` → `tycho_astro.h`
   - Rename `.cpp` entries accordingly

**`src/Bindings/CMakeLists.txt`:**
- Update `_TYCHO_BINDINGS_INCLUDES` if it mirrors `_TYCHO_SRC_INCLUDES`
- Update any explicit source file references

**`examples/cpp_examples/CMakeLists.txt`** and subdirectory CMakeLists — likely no changes needed (they link `tycho::tycho` and include public headers only), but verify.

**`tests/CMakeLists.txt`** and `bench/CMakeLists.txt` — update if they reference `detail/` paths or forwarding header directories in include paths.

**`extensions/CMakeLists.txt`** — uses `TYCHO_INCLUDES` which propagates `_TYCHO_SRC_INCLUDES`. Verify extension builds still work after aggregate header renames.

### 2I: PyDocString Headers

The `src/PyDocString/` directory contains docstring headers organized by module:
- `Integrators/Integrators_doc.h`
- `OptimalControl/ODEPhaseBase_doc.h`, `OptimalControlProblem_doc.h`
- `Solvers/PSIOPT_doc.h`
- `VectorFunctions/DenseFunctionBase_doc.h`
- `VectorFunctions/VectorFunctionTypeErasure/GenericFunction_doc.h`

These stay in `src/PyDocString/` but are renamed to snake_case:
- `integrators_doc.h`, `ode_phase_base_doc.h`, `optimal_control_problem_doc.h`, `psiopt_doc.h`, `dense_function_base_doc.h`, `generic_function_doc.h`

### 2J: Documentation Updates

The following documentation files reference internal paths that will become stale:

- **`CLAUDE.md`** — repository structure section, `detail/` references, forwarding header descriptions, aggregate header names. Must be updated to reflect the new directory layout.
- **`doc/VectorFunction.md`** — references to `src/VectorFunctions/` and `detail/` paths.
- **`doc/Bindings.md`** — references to `src/` paths, forwarding headers, and `*Bind.h` include patterns.

---

## Verification Criteria

Before merging this PR:

1. **Build:** `cd build && ninja -j6 all` — clean build succeeds (reconfigure first: `cmake --preset linux-clang-release`)
2. **C++ tests:** `ctest --output-on-failure` — all tests pass
3. **Python examples:** `python scripts/run_examples.py` — all 38 examples pass
4. **C++ example:** `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` — reports "Optimal Solution Found"
5. **No stale paths:** `grep -r "tycho/detail/[A-Z]" include/ src/ tests/ bench/` returns empty (no PascalCase detail includes remain)
6. **No stale paths:** `grep -r '#include "tycho/detail/[a-z_]*\.h"' include/ src/` where the path doesn't contain a subdirectory returns empty (no flat `detail/foo.h` includes remain — all should be `detail/domain/foo.h`)
7. **No forwarding headers:** No one-line `#include "tycho/detail/..."` shims remain in `src/` subdirectories (excluding aggregate headers)
8. **clang-format:** `cd build && ninja clang-format` — no formatting violations in `src/`

---

## Out of Scope

- **Naming convention migration** (class names, function names, variables, namespaces) — deferred to Phase 2, done module by module
- **`src/` directory renaming** (e.g., `src/VectorFunctions/` → `src/vf/`) — may be addressed in Phase 2 alongside namespace changes
- **`src/Bindings/` file renaming** — binding files keep their current names for now; renaming is lower priority and can be done in Phase 2
- **Python package restructuring** (`tychopy/`) — not part of this effort
- **Build system modernization** — CMake changes are limited to updating file references
