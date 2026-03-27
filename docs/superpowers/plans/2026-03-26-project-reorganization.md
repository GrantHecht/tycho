# Project Reorganization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize Tycho's file structure — move Python examples, remove forwarding headers, rename all files to snake_case, and reorganize `include/tycho/detail/` into domain subdirectories.

**Architecture:** Two sequential phases: (1) move Python examples independently verifiable; (2) atomic restructuring of all headers and source files — file moves, deletions, and include-path updates done together. The `detail/` directory gains domain subdirs (`utils/`, `typedefs/`, `vf/`, `integrators/`, `optimal_control/`, `solvers/`, `astro/`). All 143 forwarding headers in `src/` are deleted. All `.cpp` and `.h` files renamed to snake_case.

**Tech Stack:** C++20, CMake + Ninja, nanobind, git, Python 3.12+

**Spec:** `docs/superpowers/specs/2026-03-26-project-reorganization-design.md` — the complete file mapping tables are authoritative. Refer to the spec constantly during implementation.

---

## Background: Include Path Transformation Rules

After all file moves and renames, every `#include` referencing an old path must be updated. The transformation is mechanical:

| Old pattern | New pattern |
|---|---|
| `"tycho/detail/FooBar.h"` | `"tycho/detail/<domain>/foo_bar.h"` |
| `"FooBar.h"` (bare, in src/*.cpp) | `"tycho/detail/<domain>/foo_bar.h"` |
| `"SubDir/FooBar.h"` (forwarding path, in src/*.cpp) | `"tycho/detail/<domain>/foo_bar.h"` |

The domain and snake_case name for every file is in the spec's section 2F mapping tables.

---

## Task 1: Create Feature Branch

**Files:** none

- [ ] **Step 1: Create and check out a new branch**

```bash
git checkout -b feat/project-reorganization
```

- [ ] **Step 2: Confirm clean starting state**

```bash
git status
```
Expected: working tree clean (no uncommitted changes).

---

## Task 2: Move Python Examples

**Files:**
- Move: `examples/*.py` → `examples/python_examples/*.py`
- Move: `examples/MeshRefinement/` → `examples/python_examples/MeshRefinement/`
- Move: `examples/UpdatedInterface/` → `examples/python_examples/UpdatedInterface/`
- Move: `examples/Plots/` → `examples/python_examples/Plots/`
- Modify: `scripts/run_examples.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create python_examples directory and move all root Python scripts**

```bash
mkdir -p examples/python_examples
git mv examples/AnalyticExample.py examples/python_examples/
git mv examples/BettsLowThrust.py examples/python_examples/
git mv examples/BikeObstacle.py examples/python_examples/
git mv examples/Brachistochrone.py examples/python_examples/
git mv examples/BrysonDenham.py examples/python_examples/
git mv examples/CartPole.py examples/python_examples/
git mv examples/Delta3Launch.py examples/python_examples/
git mv examples/DionysusLowThrust.py examples/python_examples/
git mv examples/FreeFlyingRobotExample.py examples/python_examples/
git mv examples/GoddardRocket.py examples/python_examples/
git mv examples/HangingChain.py examples/python_examples/
git mv examples/Heteroclinic.py examples/python_examples/
git mv examples/HyperSens.py examples/python_examples/
git mv examples/MinimumTimeToClimb.py examples/python_examples/
git mv examples/MinimumTimeToClimbTables.py examples/python_examples/
git mv examples/MountainCar.py examples/python_examples/
git mv examples/MultiPhaseCannon.py examples/python_examples/
git mv examples/MultiPhaseZermelo.py examples/python_examples/
git mv examples/MultiSpacecraftOptimization.py examples/python_examples/
git mv examples/OptimalDocking.py examples/python_examples/
git mv examples/OrbitContinuation.py examples/python_examples/
git mv examples/ParallelParking.py examples/python_examples/
git mv examples/Reentry.py examples/python_examples/
git mv examples/SimpleLowThrust.py examples/python_examples/
git mv examples/TopputtoLowThrust.py examples/python_examples/
git mv examples/VanDerPol.py examples/python_examples/
git mv examples/Zermelo.py examples/python_examples/
```

- [ ] **Step 2: Move subdirectories**

```bash
git mv examples/MeshRefinement examples/python_examples/MeshRefinement
git mv examples/UpdatedInterface examples/python_examples/UpdatedInterface
git mv examples/Plots examples/python_examples/Plots
```

- [ ] **Step 3: Clean up any existing plot output files and add .gitkeep files**

```bash
# Remove any actual plot files (the dirs should already be empty or have only subdirs)
find examples/python_examples/Plots -type f ! -name '.gitkeep' -delete

# Ensure subdirectories exist with .gitkeep
touch examples/python_examples/Plots/Zermelo/.gitkeep
touch examples/python_examples/Plots/OrbitContinuation/.gitkeep
touch examples/python_examples/Plots/MultiSpacecraftOptimization/.gitkeep

git add examples/python_examples/Plots/
```

- [ ] **Step 4: Update .gitignore to ignore plot output files**

Open `.gitignore` and add the following entry (after any existing examples-related entries):

```
# Python example plot outputs (keep .gitkeep, ignore generated files)
examples/python_examples/Plots/**
!examples/python_examples/Plots/**/.gitkeep
```

- [ ] **Step 5: Update scripts/run_examples.py**

Make these changes to `scripts/run_examples.py`:

Change line 34 from:
```python
EXAMPLES_DIR = REPO_ROOT / "examples"
```
To:
```python
EXAMPLES_DIR = REPO_ROOT / "examples" / "python_examples"
```

The `REQUIRED_DIRS`, `ALL_EXAMPLES`, `TIMEOUT_OVERRIDES`, and all other paths are relative to `EXAMPLES_DIR`, so they need no changes — they already use relative paths like `"MeshRefinement/CartPole.py"`.

- [ ] **Step 6: Verify the examples runner still works**

```bash
conda activate tycho
python scripts/run_examples.py --filter Brachistochrone
```

Expected: `PASS` for `Brachistochrone.py`.

- [ ] **Step 7: Commit**

```bash
git add examples/python_examples/ .gitignore scripts/run_examples.py
git commit -m "refactor: move Python examples into examples/python_examples/"
```

---

## Task 3: Create New detail/ Subdirectory Structure

**Files:** Create directories only.

- [ ] **Step 1: Create all new subdirectories under include/tycho/detail/**

```bash
mkdir -p include/tycho/detail/utils
mkdir -p include/tycho/detail/typedefs
mkdir -p include/tycho/detail/vf/core
mkdir -p include/tycho/detail/vf/expressions
mkdir -p include/tycho/detail/vf/operators
mkdir -p include/tycho/detail/vf/derivatives
mkdir -p include/tycho/detail/vf/scaling
mkdir -p include/tycho/detail/vf/type_erasure
mkdir -p include/tycho/detail/vf/common
mkdir -p include/tycho/detail/integrators
mkdir -p include/tycho/detail/optimal_control/core
mkdir -p include/tycho/detail/optimal_control/phase
mkdir -p include/tycho/detail/optimal_control/transcription
mkdir -p include/tycho/detail/optimal_control/interp
mkdir -p include/tycho/detail/optimal_control/builder
mkdir -p include/tycho/detail/solvers/linear
mkdir -p include/tycho/detail/astro
```

---

## Task 4: Move and Rename All detail/ Headers

**Files:** All 151 files in `include/tycho/detail/` moved to their new subdirectory paths with snake_case names.

Use `git mv` for every file so git tracks renames properly. Work through each subdomain systematically.

> **Reference:** The complete mapping is in spec section 2F. The tables below mirror it exactly.

- [ ] **Step 1: Move detail/utils/ headers (14 files)**

```bash
git mv include/tycho/detail/crtp_base.h          include/tycho/detail/utils/crtp_base.h
git mv include/tycho/detail/flat_map.h            include/tycho/detail/utils/flat_map.h
git mv include/tycho/detail/function_return_type.h include/tycho/detail/utils/function_return_type.h
git mv include/tycho/detail/get_core_count.h      include/tycho/detail/utils/get_core_count.h
git mv include/tycho/detail/LambdaJumpTable.h     include/tycho/detail/utils/lambda_jump_table.h
git mv include/tycho/detail/math_functions.h      include/tycho/detail/utils/math_functions.h
git mv include/tycho/detail/memory_management.h   include/tycho/detail/utils/memory_management.h
git mv include/tycho/detail/sizing_helpers.h      include/tycho/detail/utils/sizing_helpers.h
git mv include/tycho/detail/std_extensions.h      include/tycho/detail/utils/std_extensions.h
git mv include/tycho/detail/thread_pool.h         include/tycho/detail/utils/thread_pool.h
git mv include/tycho/detail/timer.h               include/tycho/detail/utils/timer.h
git mv include/tycho/detail/tuple_iterator.h      include/tycho/detail/utils/tuple_iterator.h
git mv include/tycho/detail/type_name.h           include/tycho/detail/utils/type_name.h
git mv include/tycho/detail/type_storage.h        include/tycho/detail/utils/type_storage.h
```

- [ ] **Step 2: Move detail/typedefs/ headers (1 file)**

```bash
git mv include/tycho/detail/eigen_types.h include/tycho/detail/typedefs/eigen_types.h
```

- [ ] **Step 3: Move detail/vf/core/ headers (18 files)**

```bash
git mv include/tycho/detail/AssigmentTypes.h       include/tycho/detail/vf/core/assignment_types.h
git mv include/tycho/detail/ComputableBase.h        include/tycho/detail/vf/core/computable_base.h
git mv include/tycho/detail/Computable.h            include/tycho/detail/vf/core/computable.h
git mv include/tycho/detail/DenseFunctionBase.h     include/tycho/detail/vf/core/dense_function_base.h
git mv include/tycho/detail/DenseFunction.h         include/tycho/detail/vf/core/dense_function.h
git mv include/tycho/detail/DenseFunctionOperations.h include/tycho/detail/vf/core/dense_function_operations.h
git mv include/tycho/detail/DenseFunctionSpecs.h    include/tycho/detail/vf/core/dense_function_specs.h
git mv include/tycho/detail/DenseScalarFunctionBase.h include/tycho/detail/vf/core/dense_scalar_function_base.h
git mv include/tycho/detail/ExpressionFwdDeclarations.h include/tycho/detail/vf/core/expression_fwd_declarations.h
git mv include/tycho/detail/FunctionalFlags.h       include/tycho/detail/vf/core/functional_flags.h
git mv include/tycho/detail/FunctionDomains.h       include/tycho/detail/vf/core/function_domains.h
git mv include/tycho/detail/FunctionTemplates.h     include/tycho/detail/vf/core/function_templates.h
git mv include/tycho/detail/FunctionTypeDefMacros.h include/tycho/detail/vf/core/function_type_def_macros.h
git mv include/tycho/detail/InputOuputSize.h        include/tycho/detail/vf/core/input_output_size.h
git mv include/tycho/detail/ObjectDetectors.h       include/tycho/detail/vf/core/object_detectors.h
git mv include/tycho/detail/SparseFunctionBase.h    include/tycho/detail/vf/core/sparse_function_base.h
git mv include/tycho/detail/VectorFunctionConcepts.h include/tycho/detail/vf/core/vector_function_concepts.h
git mv include/tycho/detail/VectorFunction.h        include/tycho/detail/vf/core/vector_function.h
```

- [ ] **Step 4: Move detail/vf/expressions/ headers (9 files)**

```bash
git mv include/tycho/detail/CallAndAppend.h    include/tycho/detail/vf/expressions/call_and_append.h
git mv include/tycho/detail/For.h              include/tycho/detail/vf/expressions/for.h
git mv include/tycho/detail/FunctionHolder.h   include/tycho/detail/vf/expressions/function_holder.h
git mv include/tycho/detail/LambdaFunction.h   include/tycho/detail/vf/expressions/lambda_function.h
git mv include/tycho/detail/NestedFunction.h   include/tycho/detail/vf/expressions/nested_function.h
git mv include/tycho/detail/ParsedInput.h      include/tycho/detail/vf/expressions/parsed_input.h
git mv include/tycho/detail/Segment.h          include/tycho/detail/vf/expressions/segment.h
git mv include/tycho/detail/Stacked.h          include/tycho/detail/vf/expressions/stacked.h
git mv include/tycho/detail/Summation.h        include/tycho/detail/vf/expressions/summation.h
```

- [ ] **Step 5: Move detail/vf/operators/ headers (20 files)**

```bash
git mv include/tycho/detail/ArcTan2.h                    include/tycho/detail/vf/operators/arc_tan2.h
git mv include/tycho/detail/BinaryMath.h                 include/tycho/detail/vf/operators/binary_math.h
git mv include/tycho/detail/CrossProduct.h               include/tycho/detail/vf/operators/cross_product.h
git mv include/tycho/detail/CwiseOperators.h             include/tycho/detail/vf/operators/cwise_operators.h
git mv include/tycho/detail/CwiseProduct.h               include/tycho/detail/vf/operators/cwise_product.h
git mv include/tycho/detail/CwiseSum.h                   include/tycho/detail/vf/operators/cwise_sum.h
git mv include/tycho/detail/DotProduct.h                 include/tycho/detail/vf/operators/dot_product.h
git mv include/tycho/detail/FunctionVectorSums.h         include/tycho/detail/vf/operators/function_vector_sums.h
git mv include/tycho/detail/MathOverloads.h              include/tycho/detail/vf/operators/math_overloads.h
git mv include/tycho/detail/MatrixFunction.h             include/tycho/detail/vf/operators/matrix_function.h
git mv include/tycho/detail/MatrixInverse.h              include/tycho/detail/vf/operators/matrix_inverse.h
git mv include/tycho/detail/MatrixProduct.h              include/tycho/detail/vf/operators/matrix_product.h
git mv include/tycho/detail/Normalized.h                 include/tycho/detail/vf/operators/normalized.h
git mv include/tycho/detail/Norms.h                      include/tycho/detail/vf/operators/norms.h
git mv include/tycho/detail/OperatorOverloads.h          include/tycho/detail/vf/operators/operator_overloads.h
git mv include/tycho/detail/RootFinder.h                 include/tycho/detail/vf/operators/root_finder.h
git mv include/tycho/detail/SignFunction.h               include/tycho/detail/vf/operators/sign_function.h
git mv include/tycho/detail/VectorProducts.h             include/tycho/detail/vf/operators/vector_products.h
git mv include/tycho/detail/VectorScalarFunctionDivision.h include/tycho/detail/vf/operators/vector_scalar_function_division.h
git mv include/tycho/detail/VectorScalarFunctionProduct.h  include/tycho/detail/vf/operators/vector_scalar_function_product.h
```

- [ ] **Step 6: Move detail/vf/derivatives/ headers (9 files)**

```bash
git mv include/tycho/detail/DenseAutodiffFwd.h    include/tycho/detail/vf/derivatives/dense_autodiff_fwd.h
git mv include/tycho/detail/DenseDerivatives.h    include/tycho/detail/vf/derivatives/dense_derivatives.h
git mv include/tycho/detail/DenseFDiffCentArray.h include/tycho/detail/vf/derivatives/dense_fdiff_cent_array.h
git mv include/tycho/detail/DenseFDiffFwd.h       include/tycho/detail/vf/derivatives/dense_fdiff_fwd.h
git mv include/tycho/detail/DetectDiagonal.h      include/tycho/detail/vf/derivatives/detect_diagonal.h
git mv include/tycho/detail/DetectSuperScalar.h   include/tycho/detail/vf/derivatives/detect_super_scalar.h
git mv include/tycho/detail/FDCoeffs.h            include/tycho/detail/vf/derivatives/fd_coeffs.h
git mv include/tycho/detail/FDDerivArbitrary.h    include/tycho/detail/vf/derivatives/fd_deriv_arbitrary.h
git mv include/tycho/detail/FDDerivUniform.h      include/tycho/detail/vf/derivatives/fd_deriv_uniform.h
```

- [ ] **Step 7: Move detail/vf/scaling/ headers (4 files)**

```bash
git mv include/tycho/detail/AutoScalingUtils.h include/tycho/detail/vf/scaling/auto_scaling_utils.h
git mv include/tycho/detail/IOScaled.h         include/tycho/detail/vf/scaling/io_scaled.h
git mv include/tycho/detail/Padded.h           include/tycho/detail/vf/scaling/padded.h
git mv include/tycho/detail/Scaled.h           include/tycho/detail/vf/scaling/scaled.h
```

- [ ] **Step 8: Move detail/vf/type_erasure/ headers (8 files)**

```bash
git mv include/tycho/detail/AutoDiffFunction.h       include/tycho/detail/vf/type_erasure/autodiff_function.h
git mv include/tycho/detail/Comparative.h            include/tycho/detail/vf/type_erasure/comparative.h
git mv include/tycho/detail/Conditional.h            include/tycho/detail/vf/type_erasure/conditional.h
git mv include/tycho/detail/ConditionalTypeErasure.h include/tycho/detail/vf/type_erasure/conditional_base.h
git mv include/tycho/detail/GenericComparative.h     include/tycho/detail/vf/type_erasure/generic_comparative.h
git mv include/tycho/detail/GenericConditional.h     include/tycho/detail/vf/type_erasure/generic_conditional.h
git mv include/tycho/detail/GenericFunction.h        include/tycho/detail/vf/type_erasure/generic_function.h
git mv include/tycho/detail/GFTypeErasure.h          include/tycho/detail/vf/type_erasure/gf_type_erasure.h
```

- [ ] **Step 9: Move detail/vf/common/ headers (5 files)**

```bash
git mv include/tycho/detail/CommonFunctions.h include/tycho/detail/vf/common/common_functions.h
git mv include/tycho/detail/Constant.h        include/tycho/detail/vf/common/constant.h
git mv include/tycho/detail/Elements.h        include/tycho/detail/vf/common/elements.h
git mv include/tycho/detail/Value.h           include/tycho/detail/vf/common/value.h
git mv include/tycho/detail/ValueLock.h       include/tycho/detail/vf/common/value_lock.h
```

- [ ] **Step 10: Move detail/integrators/ headers (3 files)**

```bash
git mv include/tycho/detail/Integrator.h include/tycho/detail/integrators/integrator.h
git mv include/tycho/detail/RKCoeffs.h   include/tycho/detail/integrators/rk_coeffs.h
git mv include/tycho/detail/RKSteppers.h include/tycho/detail/integrators/rk_steppers.h
```

- [ ] **Step 11: Move detail/optimal_control/core/ headers (6 files)**

```bash
git mv include/tycho/detail/InterfaceTypes.h      include/tycho/detail/optimal_control/core/interface_types.h
git mv include/tycho/detail/LinkFunction.h        include/tycho/detail/optimal_control/core/link_function.h
git mv include/tycho/detail/ODEArguments.h        include/tycho/detail/optimal_control/core/ode_arguments.h
git mv include/tycho/detail/ODESizes.h            include/tycho/detail/optimal_control/core/ode_sizes.h
git mv include/tycho/detail/OptimalControlFlags.h include/tycho/detail/optimal_control/core/optimal_control_flags.h
git mv include/tycho/detail/StateFunction.h       include/tycho/detail/optimal_control/core/state_function.h
```

- [ ] **Step 12: Move detail/optimal_control/phase/ headers (6 files)**

```bash
git mv include/tycho/detail/MeshIterateInfo.h      include/tycho/detail/optimal_control/phase/mesh_iterate_info.h
git mv include/tycho/detail/ODE.h                  include/tycho/detail/optimal_control/phase/ode.h
git mv include/tycho/detail/ODEPhase.h             include/tycho/detail/optimal_control/phase/ode_phase.h
git mv include/tycho/detail/ODEPhaseBase.h         include/tycho/detail/optimal_control/phase/ode_phase_base.h
git mv include/tycho/detail/OptimalControlProblem.h include/tycho/detail/optimal_control/phase/optimal_control_problem.h
git mv include/tycho/detail/PhaseIndexer.h         include/tycho/detail/optimal_control/phase/phase_indexer.h
```

- [ ] **Step 13: Move detail/optimal_control/transcription/ headers (12 files)**

```bash
git mv include/tycho/detail/Blocked_ODE_Wrapper.h    include/tycho/detail/optimal_control/transcription/blocked_ode_wrapper.h
git mv include/tycho/detail/LGLCoeffs.h              include/tycho/detail/optimal_control/transcription/lgl_coeffs.h
git mv include/tycho/detail/LGLControlSplines.h      include/tycho/detail/optimal_control/transcription/lgl_control_splines.h
git mv include/tycho/detail/LGLDefects.h             include/tycho/detail/optimal_control/transcription/lgl_defects.h
git mv include/tycho/detail/LGLIntegrals.h           include/tycho/detail/optimal_control/transcription/lgl_integrals.h
git mv include/tycho/detail/LGLInterpFunctions.h     include/tycho/detail/optimal_control/transcription/lgl_interp_functions.h
git mv include/tycho/detail/LGLInterpTable.h         include/tycho/detail/optimal_control/transcription/lgl_interp_table.h
git mv include/tycho/detail/MeshSpacingConstraints.h include/tycho/detail/optimal_control/transcription/mesh_spacing_constraints.h
git mv include/tycho/detail/ShootingDefects.h        include/tycho/detail/optimal_control/transcription/shooting_defects.h
git mv include/tycho/detail/TranscriptionSizing.h    include/tycho/detail/optimal_control/transcription/transcription_sizing.h
git mv include/tycho/detail/TrapezoidalDefects.h     include/tycho/detail/optimal_control/transcription/trapezoidal_defects.h
git mv include/tycho/detail/TrapezoidalIntegrals.h   include/tycho/detail/optimal_control/transcription/trapezoidal_integrals.h
```

- [ ] **Step 14: Move detail/optimal_control/interp/ headers (4 files)**

```bash
git mv include/tycho/detail/InterpTable1D.h include/tycho/detail/optimal_control/interp/interp_table_1d.h
git mv include/tycho/detail/InterpTable2D.h include/tycho/detail/optimal_control/interp/interp_table_2d.h
git mv include/tycho/detail/InterpTable3D.h include/tycho/detail/optimal_control/interp/interp_table_3d.h
git mv include/tycho/detail/InterpTable4D.h include/tycho/detail/optimal_control/interp/interp_table_4d.h
```

- [ ] **Step 15: Move detail/optimal_control/builder/ headers (5 files)**

```bash
git mv include/tycho/detail/ocp_wrapper.h   include/tycho/detail/optimal_control/builder/ocp_wrapper.h
git mv include/tycho/detail/ode_builder.h   include/tycho/detail/optimal_control/builder/ode_builder.h
git mv include/tycho/detail/phase_wrapper.h include/tycho/detail/optimal_control/builder/phase_wrapper.h
git mv include/tycho/detail/runtime_ode.h   include/tycho/detail/optimal_control/builder/runtime_ode.h
git mv include/tycho/detail/var_registry.h  include/tycho/detail/optimal_control/builder/var_registry.h
```

- [ ] **Step 16: Move detail/solvers/ headers (17 files including linear/ subdir)**

```bash
git mv include/tycho/detail/ConstraintFunction.h    include/tycho/detail/solvers/constraint_function.h
git mv include/tycho/detail/IndexingData.h          include/tycho/detail/solvers/indexing_data.h
git mv include/tycho/detail/IterateInfo.h           include/tycho/detail/solvers/iterate_info.h
git mv include/tycho/detail/Jet.h                   include/tycho/detail/solvers/jet.h
git mv include/tycho/detail/NonLinearProgram.h      include/tycho/detail/solvers/non_linear_program.h
git mv include/tycho/detail/ObjectiveFunction.h     include/tycho/detail/solvers/objective_function.h
git mv include/tycho/detail/OptimizationProblem.h   include/tycho/detail/solvers/optimization_problem.h
git mv include/tycho/detail/OptimizationProblemBase.h include/tycho/detail/solvers/optimization_problem_base.h
git mv include/tycho/detail/PSIOPT.h                include/tycho/detail/solvers/psiopt.h
git mv include/tycho/detail/SizingSpecs.h           include/tycho/detail/solvers/sizing_specs.h
git mv include/tycho/detail/SolverFlags.h           include/tycho/detail/solvers/solver_flags.h
git mv include/tycho/detail/SolverFunctionBase.h    include/tycho/detail/solvers/solver_function_base.h
git mv include/tycho/detail/SolverInit.h            include/tycho/detail/solvers/solver_init.h
git mv include/tycho/detail/SolverInterfaceSpecs.h  include/tycho/detail/solvers/solver_interface_specs.h
git mv include/tycho/detail/AccelerateInterface.h   include/tycho/detail/solvers/linear/accelerate_interface.h
git mv include/tycho/detail/AccelerateUtils.h       include/tycho/detail/solvers/linear/accelerate_utils.h
git mv include/tycho/detail/PardisoInterface.h      include/tycho/detail/solvers/linear/pardiso_interface.h
```

- [ ] **Step 17: Move detail/astro/ headers (8 files)**

```bash
git mv include/tycho/detail/CR3BPModel.h    include/tycho/detail/astro/cr3bp_model.h
git mv include/tycho/detail/J2.h            include/tycho/detail/astro/j2.h
git mv include/tycho/detail/KeplerModel.h   include/tycho/detail/astro/kepler_model.h
git mv include/tycho/detail/KeplerPropagator.h include/tycho/detail/astro/kepler_propagator.h
git mv include/tycho/detail/KeplerUtils.h   include/tycho/detail/astro/kepler_utils.h
git mv include/tycho/detail/LambertSolvers.h include/tycho/detail/astro/lambert_solvers.h
git mv include/tycho/detail/MEEDynamics.h   include/tycho/detail/astro/mee_dynamics.h
git mv include/tycho/detail/ThrusterModels.h include/tycho/detail/astro/thruster_models.h
```

- [ ] **Step 18: Handle PythonArgParsing.h — move to Bindings**

The file `include/tycho/detail/PythonArgParsing.h` is a binding-only header. The file `src/Bindings/VectorFunctions/PythonArgParsing.h` is a forwarding header pointing to it. The real file should live in Bindings.

```bash
# Copy real content to Bindings (replacing the forwarding header)
cp include/tycho/detail/PythonArgParsing.h src/Bindings/VectorFunctions/PythonArgParsing.h

# Rename the Bindings copy to snake_case
git mv src/Bindings/VectorFunctions/PythonArgParsing.h src/Bindings/VectorFunctions/python_arg_parsing.h

# Remove from detail/
git rm include/tycho/detail/PythonArgParsing.h
```

- [ ] **Step 19: Verify all 151 detail/ headers are moved (no files remain in the flat detail/ directory)**

```bash
ls include/tycho/detail/*.h 2>/dev/null && echo "ERROR: files remain in detail/" || echo "OK: detail/ root is empty"
```

Expected: `OK: detail/ root is empty`

---

## Task 5: Delete Forwarding Headers from src/

**Files:** ~143 files deleted from `src/` subdirectories.

> **CRITICAL:** Do NOT delete `src/Utils/ColorText.h`, `src/Utils/EigenSTL.h`, or `src/Utils/fmtlib.h` — these are real private implementation headers.

- [ ] **Step 1: Delete all src/Astro/ forwarding headers (8 files)**

```bash
git rm src/Astro/CR3BPModel.h src/Astro/J2.h src/Astro/KeplerModel.h \
       src/Astro/KeplerPropagator.h src/Astro/KeplerUtils.h \
       src/Astro/LambertSolvers.h src/Astro/MEEDynamics.h src/Astro/ThrusterModels.h
```

- [ ] **Step 2: Delete all src/Integrators/ forwarding headers (3 files)**

```bash
git rm src/Integrators/Integrator.h src/Integrators/RKCoeffs.h src/Integrators/RKSteppers.h
```

- [ ] **Step 3: Delete all src/OptimalControl/ forwarding headers (29 files)**

```bash
git rm src/OptimalControl/AutoScalingUtils.h \
       src/OptimalControl/Blocked_ODE_Wrapper.h \
       src/OptimalControl/FDCoeffs.h \
       src/OptimalControl/FDDerivArbitrary.h \
       src/OptimalControl/FDDerivUniform.h \
       src/OptimalControl/InterfaceTypes.h \
       src/OptimalControl/LGLCoeffs.h \
       src/OptimalControl/LGLControlSplines.h \
       src/OptimalControl/LGLDefects.h \
       src/OptimalControl/LGLIntegrals.h \
       src/OptimalControl/LGLInterpFunctions.h \
       src/OptimalControl/LGLInterpTable.h \
       src/OptimalControl/LinkFunction.h \
       src/OptimalControl/MeshIterateInfo.h \
       src/OptimalControl/MeshSpacingConstraints.h \
       src/OptimalControl/ODE.h \
       src/OptimalControl/ODEArguments.h \
       src/OptimalControl/ODEPhase.h \
       src/OptimalControl/ODEPhaseBase.h \
       src/OptimalControl/ODESizes.h \
       src/OptimalControl/OptimalControlFlags.h \
       src/OptimalControl/OptimalControlProblem.h \
       src/OptimalControl/PhaseIndexer.h \
       src/OptimalControl/ShootingDefects.h \
       src/OptimalControl/StateFunction.h \
       src/OptimalControl/TranscriptionSizing.h \
       src/OptimalControl/TrapezoidalDefects.h \
       src/OptimalControl/TrapezoidalIntegrals.h \
       src/OptimalControl/ValueLock.h
```

- [ ] **Step 4: Delete all src/Solvers/ forwarding headers (14 files)**

```bash
git rm src/Solvers/AccelerateInterface.h \
       src/Solvers/AccelerateUtils.h \
       src/Solvers/ConstraintFunction.h \
       src/Solvers/IterateInfo.h \
       src/Solvers/Jet.h \
       src/Solvers/NonLinearProgram.h \
       src/Solvers/ObjectiveFunction.h \
       src/Solvers/OptimizationProblem.h \
       src/Solvers/OptimizationProblemBase.h \
       src/Solvers/PardisoInterface.h \
       src/Solvers/PSIOPT.h \
       src/Solvers/SolverFlags.h \
       src/Solvers/SolverFunctionBase.h \
       src/Solvers/SolverInit.h
```

- [ ] **Step 5: Delete src/TypeDefs/ forwarding header (1 file)**

```bash
git rm src/TypeDefs/EigenTypes.h
```

- [ ] **Step 6: Delete src/Utils/ forwarding headers (14 files — NOT ColorText.h, EigenSTL.h, fmtlib.h)**

```bash
git rm src/Utils/CRTPBase.h \
       src/Utils/FlatMap.h \
       src/Utils/FunctionReturnType.h \
       src/Utils/GetCoreCount.h \
       src/Utils/LambdaJumpTable.h \
       src/Utils/MathFunctions.h \
       src/Utils/MemoryManagement.h \
       src/Utils/SizingHelpers.h \
       src/Utils/STDExtensions.h \
       src/Utils/ThreadPool.h \
       src/Utils/Timer.h \
       src/Utils/TupleIterator.h \
       src/Utils/TypeName.h \
       src/Utils/TypeStorage.h
```

- [ ] **Step 7: Delete src/VectorFunctions/ forwarding headers (75 files)**

```bash
git rm src/VectorFunctions/AssigmentTypes.h \
       src/VectorFunctions/BinaryMath.h \
       src/VectorFunctions/ComputableBase.h \
       src/VectorFunctions/Computable.h \
       src/VectorFunctions/DenseDerivatives.h \
       src/VectorFunctions/DenseFunctionBase.h \
       src/VectorFunctions/DenseFunction.h \
       src/VectorFunctions/DenseFunctionOperations.h \
       src/VectorFunctions/DenseScalarFunctionBase.h \
       src/VectorFunctions/DetectDiagonal.h \
       src/VectorFunctions/DetectSuperScalar.h \
       src/VectorFunctions/FunctionalFlags.h \
       src/VectorFunctions/FunctionDomains.h \
       src/VectorFunctions/FunctionTemplates.h \
       src/VectorFunctions/FunctionTypeDefMacros.h \
       src/VectorFunctions/IndexingData.h \
       src/VectorFunctions/InputOuputSize.h \
       src/VectorFunctions/MathOverloads.h \
       src/VectorFunctions/ObjectDetectors.h \
       src/VectorFunctions/OperatorOverloads.h \
       src/VectorFunctions/PythonArgParsing.h \
       src/VectorFunctions/SparseFunctionBase.h \
       src/VectorFunctions/VectorFunctionConcepts.h \
       src/VectorFunctions/VectorFunction.h \
       src/VectorFunctions/CommonFunctions/ArcTan2.h \
       src/VectorFunctions/CommonFunctions/AutoDiffFunction.h \
       src/VectorFunctions/CommonFunctions/CallAndAppend.h \
       src/VectorFunctions/CommonFunctions/CommonFunctions.h \
       src/VectorFunctions/CommonFunctions/Comparative.h \
       src/VectorFunctions/CommonFunctions/Conditional.h \
       src/VectorFunctions/CommonFunctions/Constant.h \
       src/VectorFunctions/CommonFunctions/CrossProduct.h \
       src/VectorFunctions/CommonFunctions/CwiseOperators.h \
       src/VectorFunctions/CommonFunctions/CwiseProduct.h \
       src/VectorFunctions/CommonFunctions/CwiseSum.h \
       src/VectorFunctions/CommonFunctions/DotProduct.h \
       src/VectorFunctions/CommonFunctions/Elements.h \
       src/VectorFunctions/CommonFunctions/ExpressionFwdDeclarations.h \
       src/VectorFunctions/CommonFunctions/For.h \
       src/VectorFunctions/CommonFunctions/FunctionHolder.h \
       src/VectorFunctions/CommonFunctions/FunctionVectorSums.h \
       src/VectorFunctions/CommonFunctions/InterpTable1D.h \
       src/VectorFunctions/CommonFunctions/InterpTable2D.h \
       src/VectorFunctions/CommonFunctions/InterpTable3D.h \
       src/VectorFunctions/CommonFunctions/InterpTable4D.h \
       src/VectorFunctions/CommonFunctions/IOScaled.h \
       src/VectorFunctions/CommonFunctions/LambdaFunction.h \
       src/VectorFunctions/CommonFunctions/MatrixFunction.h \
       src/VectorFunctions/CommonFunctions/MatrixInverse.h \
       src/VectorFunctions/CommonFunctions/MatrixProduct.h \
       src/VectorFunctions/CommonFunctions/NestedFunction.h \
       src/VectorFunctions/CommonFunctions/Normalized.h \
       src/VectorFunctions/CommonFunctions/Norms.h \
       src/VectorFunctions/CommonFunctions/Padded.h \
       src/VectorFunctions/CommonFunctions/ParsedInput.h \
       src/VectorFunctions/CommonFunctions/RootFinder.h \
       src/VectorFunctions/CommonFunctions/Scaled.h \
       src/VectorFunctions/CommonFunctions/Segment.h \
       src/VectorFunctions/CommonFunctions/SignFunction.h \
       src/VectorFunctions/CommonFunctions/Stacked.h \
       src/VectorFunctions/CommonFunctions/Summation.h \
       src/VectorFunctions/CommonFunctions/Value.h \
       src/VectorFunctions/CommonFunctions/VectorProducts.h \
       src/VectorFunctions/CommonFunctions/VectorScalarFunctionDivision.h \
       src/VectorFunctions/CommonFunctions/VectorScalarFunctionProduct.h \
       src/VectorFunctions/DenseDifferentiation/DenseAutodiffFwd.h \
       src/VectorFunctions/DenseDifferentiation/DenseFDiffCentArray.h \
       src/VectorFunctions/DenseDifferentiation/DenseFDiffFwd.h \
       src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h \
       src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h \
       src/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h \
       src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h \
       src/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h \
       src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h \
       src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h \
       src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h
```

---

## Task 6: Rename src/ Files (Aggregate Headers, Private Headers, .cpp files)

**Files:** 8 aggregate headers renamed, 2 private headers renamed, 16 .cpp files renamed.

- [ ] **Step 1: Rename aggregate headers to snake_case**

```bash
git mv src/Tycho.h                                    src/tycho_internal.h
git mv src/VectorFunctions/Tycho_VectorFunctions.h    src/VectorFunctions/tycho_vector_functions.h
git mv src/OptimalControl/Tycho_OptimalControl.h      src/OptimalControl/tycho_optimal_control.h
git mv src/Solvers/Tycho_Solvers.h                    src/Solvers/tycho_solvers.h
git mv src/Utils/Tycho_Utils.h                        src/Utils/tycho_utils.h
git mv src/Astro/Tycho_Astro.h                        src/Astro/tycho_astro.h
git mv src/Integrators/Tycho_Integrators.h            src/Integrators/tycho_integrators.h
git mv src/TypeDefs/Tycho_TypeDefs.h                  src/TypeDefs/tycho_typedefs.h
```

- [ ] **Step 2: Rename real private headers in src/Utils/**

```bash
git mv src/Utils/ColorText.h src/Utils/color_text.h
git mv src/Utils/EigenSTL.h  src/Utils/eigen_stl.h
# src/Utils/fmtlib.h is already snake_case — no rename needed
```

- [ ] **Step 3: Rename src/*.cpp files**

```bash
git mv src/Astro/AccellerationModels.cpp          src/Astro/accelleration_models.cpp
git mv src/Astro/LambertSolvers.cpp               src/Astro/lambert_solvers.cpp
git mv src/OptimalControl/LGLInterpTable.cpp      src/OptimalControl/lgl_interp_table.cpp
git mv src/OptimalControl/ODEPhaseBase.cpp        src/OptimalControl/ode_phase_base.cpp
git mv src/OptimalControl/OptimalControlProblem.cpp src/OptimalControl/optimal_control_problem.cpp
git mv src/OptimalControl/PhaseIndexer.cpp        src/OptimalControl/phase_indexer.cpp
git mv src/OptimalControl/RuntimeODE.cpp          src/OptimalControl/runtime_ode.cpp
git mv src/Solvers/NonLinearProgram.cpp           src/Solvers/non_linear_program.cpp
git mv src/Solvers/OptimizationProblem.cpp        src/Solvers/optimization_problem.cpp
git mv src/Solvers/PSIOPT.cpp                     src/Solvers/psiopt.cpp
git mv src/Solvers/SolverInit.cpp                 src/Solvers/solver_init.cpp
git mv src/Utils/ColorText.cpp                    src/Utils/color_text.cpp
git mv src/Utils/GetCoreCount.cpp                 src/Utils/get_core_count.cpp
git mv src/Utils/MemoryManagement.cpp             src/Utils/memory_management.cpp
git mv src/Utils/ThreadPool.cpp                   src/Utils/thread_pool.cpp
git mv src/VectorFunctions/FunctionDomains.cpp    src/VectorFunctions/function_domains.cpp
# src/Utils/fmtlib.cpp is already snake_case — no rename needed
```

- [ ] **Step 4: Rename PyDocString headers**

```bash
git mv src/PyDocString/Integrators/Integrators_doc.h                                 src/PyDocString/Integrators/integrators_doc.h
git mv src/PyDocString/OptimalControl/ODEPhaseBase_doc.h                              src/PyDocString/OptimalControl/ode_phase_base_doc.h
git mv src/PyDocString/OptimalControl/OptimalControlProblem_doc.h                     src/PyDocString/OptimalControl/optimal_control_problem_doc.h
git mv src/PyDocString/Solvers/PSIOPT_doc.h                                           src/PyDocString/Solvers/psiopt_doc.h
git mv src/PyDocString/VectorFunctions/DenseFunctionBase_doc.h                        src/PyDocString/VectorFunctions/dense_function_base_doc.h
git mv src/PyDocString/VectorFunctions/VectorFunctionTypeErasure/GenericFunction_doc.h src/PyDocString/VectorFunctions/VectorFunctionTypeErasure/generic_function_doc.h
```

- [ ] **Step 5: Commit all moves, renames, and deletions so far**

```bash
git add -A
git commit -m "refactor: move/rename/delete headers and source files (no content changes)"
```

This commit represents the pure structural changes. The project will not build at this point — include paths are all still pointing to old locations.

---

## Task 7: Write the Include-Path Migration Script

**Files:**
- Create: `scripts/migrate_includes.py` (temporary helper, will be deleted after use)

This script reads the complete old→new mapping and performs all `#include` path substitutions across the codebase in one pass.

- [ ] **Step 1: Create the migration script**

Create `scripts/migrate_includes.py`:

```python
#!/usr/bin/env python3
"""
migrate_includes.py — Rewrite #include paths for the Tycho project reorganization.

Applies all old->new path substitutions across the codebase atomically.
Run from the repo root. Safe to run multiple times (idempotent).

Usage:
    python scripts/migrate_includes.py [--dry-run]
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Complete mapping: old #include path -> new #include path
# Keys are the exact strings that appear inside #include "..."
# ALL paths are relative to the include root or are bare names.
INCLUDE_MAP = {
    # ---- detail/ headers: old flat path -> new subdirectory path ----

    # utils/
    "tycho/detail/crtp_base.h":             "tycho/detail/utils/crtp_base.h",
    "tycho/detail/flat_map.h":              "tycho/detail/utils/flat_map.h",
    "tycho/detail/function_return_type.h":  "tycho/detail/utils/function_return_type.h",
    "tycho/detail/get_core_count.h":        "tycho/detail/utils/get_core_count.h",
    "tycho/detail/LambdaJumpTable.h":       "tycho/detail/utils/lambda_jump_table.h",
    "tycho/detail/math_functions.h":        "tycho/detail/utils/math_functions.h",
    "tycho/detail/memory_management.h":     "tycho/detail/utils/memory_management.h",
    "tycho/detail/sizing_helpers.h":        "tycho/detail/utils/sizing_helpers.h",
    "tycho/detail/std_extensions.h":        "tycho/detail/utils/std_extensions.h",
    "tycho/detail/thread_pool.h":           "tycho/detail/utils/thread_pool.h",
    "tycho/detail/timer.h":                 "tycho/detail/utils/timer.h",
    "tycho/detail/tuple_iterator.h":        "tycho/detail/utils/tuple_iterator.h",
    "tycho/detail/type_name.h":             "tycho/detail/utils/type_name.h",
    "tycho/detail/type_storage.h":          "tycho/detail/utils/type_storage.h",

    # typedefs/
    "tycho/detail/eigen_types.h":           "tycho/detail/typedefs/eigen_types.h",

    # vf/core/
    "tycho/detail/AssigmentTypes.h":        "tycho/detail/vf/core/assignment_types.h",
    "tycho/detail/ComputableBase.h":        "tycho/detail/vf/core/computable_base.h",
    "tycho/detail/Computable.h":            "tycho/detail/vf/core/computable.h",
    "tycho/detail/DenseFunctionBase.h":     "tycho/detail/vf/core/dense_function_base.h",
    "tycho/detail/DenseFunction.h":         "tycho/detail/vf/core/dense_function.h",
    "tycho/detail/DenseFunctionOperations.h": "tycho/detail/vf/core/dense_function_operations.h",
    "tycho/detail/DenseFunctionSpecs.h":    "tycho/detail/vf/core/dense_function_specs.h",
    "tycho/detail/DenseScalarFunctionBase.h": "tycho/detail/vf/core/dense_scalar_function_base.h",
    "tycho/detail/ExpressionFwdDeclarations.h": "tycho/detail/vf/core/expression_fwd_declarations.h",
    "tycho/detail/FunctionalFlags.h":       "tycho/detail/vf/core/functional_flags.h",
    "tycho/detail/FunctionDomains.h":       "tycho/detail/vf/core/function_domains.h",
    "tycho/detail/FunctionTemplates.h":     "tycho/detail/vf/core/function_templates.h",
    "tycho/detail/FunctionTypeDefMacros.h": "tycho/detail/vf/core/function_type_def_macros.h",
    "tycho/detail/InputOuputSize.h":        "tycho/detail/vf/core/input_output_size.h",
    "tycho/detail/ObjectDetectors.h":       "tycho/detail/vf/core/object_detectors.h",
    "tycho/detail/SparseFunctionBase.h":    "tycho/detail/vf/core/sparse_function_base.h",
    "tycho/detail/VectorFunctionConcepts.h": "tycho/detail/vf/core/vector_function_concepts.h",
    "tycho/detail/VectorFunction.h":        "tycho/detail/vf/core/vector_function.h",

    # vf/expressions/
    "tycho/detail/CallAndAppend.h":         "tycho/detail/vf/expressions/call_and_append.h",
    "tycho/detail/For.h":                   "tycho/detail/vf/expressions/for.h",
    "tycho/detail/FunctionHolder.h":        "tycho/detail/vf/expressions/function_holder.h",
    "tycho/detail/LambdaFunction.h":        "tycho/detail/vf/expressions/lambda_function.h",
    "tycho/detail/NestedFunction.h":        "tycho/detail/vf/expressions/nested_function.h",
    "tycho/detail/ParsedInput.h":           "tycho/detail/vf/expressions/parsed_input.h",
    "tycho/detail/Segment.h":               "tycho/detail/vf/expressions/segment.h",
    "tycho/detail/Stacked.h":               "tycho/detail/vf/expressions/stacked.h",
    "tycho/detail/Summation.h":             "tycho/detail/vf/expressions/summation.h",

    # vf/operators/
    "tycho/detail/ArcTan2.h":              "tycho/detail/vf/operators/arc_tan2.h",
    "tycho/detail/BinaryMath.h":           "tycho/detail/vf/operators/binary_math.h",
    "tycho/detail/CrossProduct.h":         "tycho/detail/vf/operators/cross_product.h",
    "tycho/detail/CwiseOperators.h":       "tycho/detail/vf/operators/cwise_operators.h",
    "tycho/detail/CwiseProduct.h":         "tycho/detail/vf/operators/cwise_product.h",
    "tycho/detail/CwiseSum.h":             "tycho/detail/vf/operators/cwise_sum.h",
    "tycho/detail/DotProduct.h":           "tycho/detail/vf/operators/dot_product.h",
    "tycho/detail/FunctionVectorSums.h":   "tycho/detail/vf/operators/function_vector_sums.h",
    "tycho/detail/MathOverloads.h":        "tycho/detail/vf/operators/math_overloads.h",
    "tycho/detail/MatrixFunction.h":       "tycho/detail/vf/operators/matrix_function.h",
    "tycho/detail/MatrixInverse.h":        "tycho/detail/vf/operators/matrix_inverse.h",
    "tycho/detail/MatrixProduct.h":        "tycho/detail/vf/operators/matrix_product.h",
    "tycho/detail/Normalized.h":           "tycho/detail/vf/operators/normalized.h",
    "tycho/detail/Norms.h":               "tycho/detail/vf/operators/norms.h",
    "tycho/detail/OperatorOverloads.h":    "tycho/detail/vf/operators/operator_overloads.h",
    "tycho/detail/RootFinder.h":           "tycho/detail/vf/operators/root_finder.h",
    "tycho/detail/SignFunction.h":         "tycho/detail/vf/operators/sign_function.h",
    "tycho/detail/VectorProducts.h":       "tycho/detail/vf/operators/vector_products.h",
    "tycho/detail/VectorScalarFunctionDivision.h": "tycho/detail/vf/operators/vector_scalar_function_division.h",
    "tycho/detail/VectorScalarFunctionProduct.h":  "tycho/detail/vf/operators/vector_scalar_function_product.h",

    # vf/derivatives/
    "tycho/detail/DenseAutodiffFwd.h":     "tycho/detail/vf/derivatives/dense_autodiff_fwd.h",
    "tycho/detail/DenseDerivatives.h":     "tycho/detail/vf/derivatives/dense_derivatives.h",
    "tycho/detail/DenseFDiffCentArray.h":  "tycho/detail/vf/derivatives/dense_fdiff_cent_array.h",
    "tycho/detail/DenseFDiffFwd.h":        "tycho/detail/vf/derivatives/dense_fdiff_fwd.h",
    "tycho/detail/DetectDiagonal.h":       "tycho/detail/vf/derivatives/detect_diagonal.h",
    "tycho/detail/DetectSuperScalar.h":    "tycho/detail/vf/derivatives/detect_super_scalar.h",
    "tycho/detail/FDCoeffs.h":             "tycho/detail/vf/derivatives/fd_coeffs.h",
    "tycho/detail/FDDerivArbitrary.h":     "tycho/detail/vf/derivatives/fd_deriv_arbitrary.h",
    "tycho/detail/FDDerivUniform.h":       "tycho/detail/vf/derivatives/fd_deriv_uniform.h",

    # vf/scaling/
    "tycho/detail/AutoScalingUtils.h":     "tycho/detail/vf/scaling/auto_scaling_utils.h",
    "tycho/detail/IOScaled.h":             "tycho/detail/vf/scaling/io_scaled.h",
    "tycho/detail/Padded.h":               "tycho/detail/vf/scaling/padded.h",
    "tycho/detail/Scaled.h":               "tycho/detail/vf/scaling/scaled.h",

    # vf/type_erasure/
    "tycho/detail/AutoDiffFunction.h":     "tycho/detail/vf/type_erasure/autodiff_function.h",
    "tycho/detail/Comparative.h":          "tycho/detail/vf/type_erasure/comparative.h",
    "tycho/detail/Conditional.h":          "tycho/detail/vf/type_erasure/conditional.h",
    "tycho/detail/ConditionalTypeErasure.h": "tycho/detail/vf/type_erasure/conditional_base.h",
    "tycho/detail/GenericComparative.h":   "tycho/detail/vf/type_erasure/generic_comparative.h",
    "tycho/detail/GenericConditional.h":   "tycho/detail/vf/type_erasure/generic_conditional.h",
    "tycho/detail/GenericFunction.h":      "tycho/detail/vf/type_erasure/generic_function.h",
    "tycho/detail/GFTypeErasure.h":        "tycho/detail/vf/type_erasure/gf_type_erasure.h",

    # vf/common/
    "tycho/detail/CommonFunctions.h":      "tycho/detail/vf/common/common_functions.h",
    "tycho/detail/Constant.h":             "tycho/detail/vf/common/constant.h",
    "tycho/detail/Elements.h":             "tycho/detail/vf/common/elements.h",
    "tycho/detail/Value.h":                "tycho/detail/vf/common/value.h",
    "tycho/detail/ValueLock.h":            "tycho/detail/vf/common/value_lock.h",

    # integrators/
    "tycho/detail/Integrator.h":           "tycho/detail/integrators/integrator.h",
    "tycho/detail/RKCoeffs.h":             "tycho/detail/integrators/rk_coeffs.h",
    "tycho/detail/RKSteppers.h":           "tycho/detail/integrators/rk_steppers.h",

    # optimal_control/core/
    "tycho/detail/InterfaceTypes.h":       "tycho/detail/optimal_control/core/interface_types.h",
    "tycho/detail/LinkFunction.h":         "tycho/detail/optimal_control/core/link_function.h",
    "tycho/detail/ODEArguments.h":         "tycho/detail/optimal_control/core/ode_arguments.h",
    "tycho/detail/ODESizes.h":             "tycho/detail/optimal_control/core/ode_sizes.h",
    "tycho/detail/OptimalControlFlags.h":  "tycho/detail/optimal_control/core/optimal_control_flags.h",
    "tycho/detail/StateFunction.h":        "tycho/detail/optimal_control/core/state_function.h",

    # optimal_control/phase/
    "tycho/detail/MeshIterateInfo.h":      "tycho/detail/optimal_control/phase/mesh_iterate_info.h",
    "tycho/detail/ODE.h":                  "tycho/detail/optimal_control/phase/ode.h",
    "tycho/detail/ODEPhase.h":             "tycho/detail/optimal_control/phase/ode_phase.h",
    "tycho/detail/ODEPhaseBase.h":         "tycho/detail/optimal_control/phase/ode_phase_base.h",
    "tycho/detail/OptimalControlProblem.h": "tycho/detail/optimal_control/phase/optimal_control_problem.h",
    "tycho/detail/PhaseIndexer.h":         "tycho/detail/optimal_control/phase/phase_indexer.h",

    # optimal_control/transcription/
    "tycho/detail/Blocked_ODE_Wrapper.h":  "tycho/detail/optimal_control/transcription/blocked_ode_wrapper.h",
    "tycho/detail/LGLCoeffs.h":            "tycho/detail/optimal_control/transcription/lgl_coeffs.h",
    "tycho/detail/LGLControlSplines.h":    "tycho/detail/optimal_control/transcription/lgl_control_splines.h",
    "tycho/detail/LGLDefects.h":           "tycho/detail/optimal_control/transcription/lgl_defects.h",
    "tycho/detail/LGLIntegrals.h":         "tycho/detail/optimal_control/transcription/lgl_integrals.h",
    "tycho/detail/LGLInterpFunctions.h":   "tycho/detail/optimal_control/transcription/lgl_interp_functions.h",
    "tycho/detail/LGLInterpTable.h":       "tycho/detail/optimal_control/transcription/lgl_interp_table.h",
    "tycho/detail/MeshSpacingConstraints.h": "tycho/detail/optimal_control/transcription/mesh_spacing_constraints.h",
    "tycho/detail/ShootingDefects.h":      "tycho/detail/optimal_control/transcription/shooting_defects.h",
    "tycho/detail/TranscriptionSizing.h":  "tycho/detail/optimal_control/transcription/transcription_sizing.h",
    "tycho/detail/TrapezoidalDefects.h":   "tycho/detail/optimal_control/transcription/trapezoidal_defects.h",
    "tycho/detail/TrapezoidalIntegrals.h": "tycho/detail/optimal_control/transcription/trapezoidal_integrals.h",

    # optimal_control/interp/
    "tycho/detail/InterpTable1D.h":        "tycho/detail/optimal_control/interp/interp_table_1d.h",
    "tycho/detail/InterpTable2D.h":        "tycho/detail/optimal_control/interp/interp_table_2d.h",
    "tycho/detail/InterpTable3D.h":        "tycho/detail/optimal_control/interp/interp_table_3d.h",
    "tycho/detail/InterpTable4D.h":        "tycho/detail/optimal_control/interp/interp_table_4d.h",

    # optimal_control/builder/
    "tycho/detail/ocp_wrapper.h":          "tycho/detail/optimal_control/builder/ocp_wrapper.h",
    "tycho/detail/ode_builder.h":          "tycho/detail/optimal_control/builder/ode_builder.h",
    "tycho/detail/phase_wrapper.h":        "tycho/detail/optimal_control/builder/phase_wrapper.h",
    "tycho/detail/runtime_ode.h":          "tycho/detail/optimal_control/builder/runtime_ode.h",
    "tycho/detail/var_registry.h":         "tycho/detail/optimal_control/builder/var_registry.h",

    # solvers/
    "tycho/detail/ConstraintFunction.h":   "tycho/detail/solvers/constraint_function.h",
    "tycho/detail/IndexingData.h":         "tycho/detail/solvers/indexing_data.h",
    "tycho/detail/IterateInfo.h":          "tycho/detail/solvers/iterate_info.h",
    "tycho/detail/Jet.h":                  "tycho/detail/solvers/jet.h",
    "tycho/detail/NonLinearProgram.h":     "tycho/detail/solvers/non_linear_program.h",
    "tycho/detail/ObjectiveFunction.h":    "tycho/detail/solvers/objective_function.h",
    "tycho/detail/OptimizationProblem.h":  "tycho/detail/solvers/optimization_problem.h",
    "tycho/detail/OptimizationProblemBase.h": "tycho/detail/solvers/optimization_problem_base.h",
    "tycho/detail/PSIOPT.h":               "tycho/detail/solvers/psiopt.h",
    "tycho/detail/SizingSpecs.h":          "tycho/detail/solvers/sizing_specs.h",
    "tycho/detail/SolverFlags.h":          "tycho/detail/solvers/solver_flags.h",
    "tycho/detail/SolverFunctionBase.h":   "tycho/detail/solvers/solver_function_base.h",
    "tycho/detail/SolverInit.h":           "tycho/detail/solvers/solver_init.h",
    "tycho/detail/SolverInterfaceSpecs.h": "tycho/detail/solvers/solver_interface_specs.h",

    # solvers/linear/
    "tycho/detail/AccelerateInterface.h":  "tycho/detail/solvers/linear/accelerate_interface.h",
    "tycho/detail/AccelerateUtils.h":      "tycho/detail/solvers/linear/accelerate_utils.h",
    "tycho/detail/PardisoInterface.h":     "tycho/detail/solvers/linear/pardiso_interface.h",

    # astro/
    "tycho/detail/CR3BPModel.h":           "tycho/detail/astro/cr3bp_model.h",
    "tycho/detail/J2.h":                   "tycho/detail/astro/j2.h",
    "tycho/detail/KeplerModel.h":          "tycho/detail/astro/kepler_model.h",
    "tycho/detail/KeplerPropagator.h":     "tycho/detail/astro/kepler_propagator.h",
    "tycho/detail/KeplerUtils.h":          "tycho/detail/astro/kepler_utils.h",
    "tycho/detail/LambertSolvers.h":       "tycho/detail/astro/lambert_solvers.h",
    "tycho/detail/MEEDynamics.h":          "tycho/detail/astro/mee_dynamics.h",
    "tycho/detail/ThrusterModels.h":       "tycho/detail/astro/thruster_models.h",

    # ---- aggregate headers renamed in src/ ----
    "Tycho_VectorFunctions.h":             "tycho_vector_functions.h",
    "Tycho_OptimalControl.h":              "tycho_optimal_control.h",
    "Tycho_Solvers.h":                     "tycho_solvers.h",
    "Tycho_Utils.h":                       "tycho_utils.h",
    "Tycho_Astro.h":                       "tycho_astro.h",
    "Tycho_Integrators.h":                 "tycho_integrators.h",
    "Tycho_TypeDefs.h":                    "tycho_typedefs.h",

    # ---- private headers renamed in src/Utils/ ----
    "Utils/ColorText.h":                   "Utils/color_text.h",
    "Utils/EigenSTL.h":                    "Utils/eigen_stl.h",
    "ColorText.h":                         "color_text.h",
    "EigenSTL.h":                          "eigen_stl.h",

    # ---- forwarding header bare names that appear in src/*.cpp ----
    # These are includes like #include "PSIOPT.h" that resolved via _TYCHO_SRC_INCLUDES
    # They must become full tycho/detail/... paths.
    # (Listed here in addition to the "tycho/detail/..." forms above to handle both cases)
}

# Files and directories to process
SEARCH_DIRS = [
    "include/tycho",
    "src",
    "tests",
    "bench",
    "extensions",
]

EXTENSIONS = {".h", ".cpp", ".cc", ".cxx"}


def process_file(path: Path, dry_run: bool) -> int:
    """Return number of replacements made."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return 0

    original = content
    count = 0

    # Match #include "..." lines and replace the path inside quotes
    def replace_include(m):
        nonlocal count
        old_path = m.group(1)
        if old_path in INCLUDE_MAP:
            count += 1
            return f'#include "{INCLUDE_MAP[old_path]}"'
        return m.group(0)

    content = re.sub(r'#include\s+"([^"]+)"', replace_include, content)

    if content != original:
        if dry_run:
            print(f"  Would update: {path} ({count} replacements)")
        else:
            path.write_text(content, encoding="utf-8")
            print(f"  Updated: {path} ({count} replacements)")

    return count


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("DRY RUN — no files will be modified\n")

    total_files = 0
    total_replacements = 0

    for search_dir in SEARCH_DIRS:
        base = REPO_ROOT / search_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in EXTENSIONS and path.is_file():
                n = process_file(path, dry_run)
                if n > 0:
                    total_files += 1
                    total_replacements += n

    print(f"\nDone: {total_replacements} replacements in {total_files} files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the migration script in dry-run mode first**

```bash
python scripts/migrate_includes.py --dry-run
```

Review the output. Every file that references old paths should appear. Verify nothing unexpected shows up.

- [ ] **Step 3: Run the migration script for real**

```bash
python scripts/migrate_includes.py
```

---

## Task 8: Manual Updates (Files That Need Human Attention)

**Files:**
- Modify: `src/pch.h`
- Modify: `src/CMakeLists.txt`
- Modify: `src/Bindings/CMakeLists.txt` (if applicable)
- Modify: aggregate headers' internal includes
- Modify: `src/Bindings/VectorFunctions/python_arg_parsing.h` (check content)

The migration script handles `#include "tycho/detail/..."` paths mechanically, but some files require additional care.

- [ ] **Step 1: Rewrite src/pch.h Tycho-specific includes block**

The `pch.h` currently includes forwarding headers by bare name like `"Utils/ThreadPool.h"`. The migration script handles any that match `INCLUDE_MAP`, but the pch.h needs a manual check.

Open `src/pch.h` and ensure the Tycho-specific section looks exactly like this:

```cpp
#include "tycho/detail/typedefs/eigen_types.h"

// Real private headers (no detail/ counterpart)
#include "Utils/eigen_stl.h"
#include "Utils/fmtlib.h"

// Utils
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

- [ ] **Step 2: Update src/CMakeLists.txt add_library() source lists**

Open `src/CMakeLists.txt` and make these changes:

**`pch` target:** Update `target_precompile_headers`:
```cmake
target_precompile_headers(pch PRIVATE pch.h
    VectorFunctions/tycho_vector_functions.h)
```

**`optimalcontrol` target:** Replace forwarding `.h` entries with renamed `.cpp` entries:
```cmake
add_library(optimalcontrol STATIC
    OptimalControl/tycho_optimal_control.h
    OptimalControl/ode_phase_base.cpp
    OptimalControl/phase_indexer.cpp
    OptimalControl/optimal_control_problem.cpp
    OptimalControl/lgl_interp_table.cpp
    OptimalControl/runtime_ode.cpp
)
```

**`psiopt` target:** Remove all forwarding `.h` entries, rename `.cpp` entries:
```cmake
add_library(psiopt STATIC
    Solvers/psiopt.cpp
    Solvers/non_linear_program.cpp
    Solvers/optimization_problem.cpp
    Solvers/optimization_problem_base.cpp
    Solvers/solver_init.cpp
)
```
Note: `Solvers/OptimizationProblemBase.cpp` — check if this `.cpp` file exists (it may be header-only). Only list `.cpp` files that actually exist.

**`utils` target:** Remove forwarding headers, update filenames:
```cmake
add_library(utils STATIC
    Utils/color_text.h Utils/color_text.cpp
    Utils/get_core_count.cpp
    Utils/memory_management.cpp
    Utils/thread_pool.cpp
)
```

**`astro` target:** Remove forwarding headers, update filenames:
```cmake
add_library(astro STATIC
    Astro/tycho_astro.h
    Astro/lambert_solvers.cpp
    Astro/accelleration_models.cpp
)
```

- [ ] **Step 3: Check src/Bindings/pch_nb.h**

Open `src/Bindings/pch_nb.h` and verify the `FunctionRegistry.h` and `JetBind.h` includes are still correct (they use bare relative names, no `detail/` paths, so the migration script won't touch them). No changes expected here.

- [ ] **Step 4: Check aggregate headers still compile**

Each aggregate header (`src/*/tycho_*.h`) internally includes both real private headers and (formerly) detail headers. After the migration script, verify they have no leftover old paths:

```bash
grep -r "tycho/detail/[A-Z]" src/
```

Expected: no output. If any appear, fix them manually by replacing with the new snake_case subdirectory path.

- [ ] **Step 5: Check for bare includes in src/*.cpp files**

```bash
grep -rn '^#include "[A-Z]' src/ --include="*.cpp"
```

These are bare `#include "FooBar.h"` that resolved via forwarding headers. The migration script only handles paths in the INCLUDE_MAP. If any remain, add them to the INCLUDE_MAP and rerun, or fix manually.

Also check for forwarding-header-style includes:
```bash
grep -rn '^#include "[A-Z][a-zA-Z]*/[A-Z]' src/ --include="*.cpp"
```

- [ ] **Step 6: Verify python_arg_parsing.h content**

Check that `src/Bindings/VectorFunctions/python_arg_parsing.h` now contains the real implementation (not a forwarding include):
```bash
wc -l src/Bindings/VectorFunctions/python_arg_parsing.h
```
Expected: more than 3 lines. If it's still just a forwarding header, the copy step in Task 4 Step 18 failed — redo it by copying from git history:
```bash
git show HEAD:include/tycho/detail/PythonArgParsing.h > src/Bindings/VectorFunctions/python_arg_parsing.h
```

Also update any `#include "PythonArgParsing.h"` references in Bindings to `#include "python_arg_parsing.h"`.

---

## Task 9: Update extensions/ Directory

**Files:**
- Modify: `extensions/Tycho_Extensions.h`
- Modify: `extensions/Tycho_Extensions.cpp` (if it exists)

- [ ] **Step 1: Check and update extensions/ includes**

```bash
grep -rn "#include" extensions/ | grep -v "^Binary"
```

Update any references to old aggregate header names (e.g., `Tycho_VectorFunctions.h` → `tycho_vector_functions.h`) and any `detail/` paths that weren't caught by the migration script.

---

## Task 10: Attempt First Build

**Files:** none (build artifacts only)

- [ ] **Step 1: Reconfigure CMake**

```bash
cd build
cmake --preset linux-clang-release
```

- [ ] **Step 2: Attempt build and capture errors**

```bash
ninja -j6 all 2>&1 | head -100
```

There will likely be errors. This step exists to surface them systematically.

- [ ] **Step 3: Fix errors iteratively**

For each error:
- If it's a "file not found" for a header: the `#include` path wasn't updated. Add it to the migration script's `INCLUDE_MAP` and rerun, or fix manually.
- If it's a compilation error (not a missing file): it's likely a legitimate issue in the file content — inspect and fix.
- If it's a CMake error: check `src/CMakeLists.txt` and `src/Bindings/CMakeLists.txt` for stale filenames.

Repeat `ninja -j6 all` until the build is clean.

- [ ] **Step 4: Commit all include path fixes and CMake updates**

```bash
git add -A
git commit -m "refactor: update all #include paths and CMakeLists for restructured headers"
```

---

## Task 11: Run Full Test Suite

**Files:** none

- [ ] **Step 1: Run C++ tests**

```bash
ctest --output-on-failure
```

Expected: all tests pass. If any fail, investigate — test failures here are likely due to missed include paths in test files.

- [ ] **Step 2: Run Python examples**

```bash
conda activate tycho
python scripts/run_examples.py
```

Expected: all 38 examples pass.

- [ ] **Step 3: Run C++ brachistochrone example**

```bash
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: output contains "Optimal Solution Found" with objective near 1.8013.

- [ ] **Step 4: Verify no stale paths remain**

```bash
# No PascalCase detail/ includes remain
grep -r "tycho/detail/[A-Z]" include/ src/ tests/ bench/ extensions/

# No flat (non-subdirectory) detail/ includes remain
grep -rP 'include "tycho/detail/[a-z_]+\.h"' include/ src/ tests/ bench/ extensions/

# No forwarding header shims remain in src/ (excluding aggregate headers)
grep -rn 'include "tycho/detail/' src/ | grep -v "tycho_"
```

All should return empty.

---

## Task 12: Delete Migration Script and Update Documentation

**Files:**
- Delete: `scripts/migrate_includes.py`
- Modify: `CLAUDE.md`
- Modify: `doc/VectorFunction.md`
- Modify: `doc/Bindings.md`

- [ ] **Step 1: Delete the temporary migration script**

```bash
git rm scripts/migrate_includes.py
```

- [ ] **Step 2: Update CLAUDE.md repository structure section**

Open `CLAUDE.md` and update:
- The repository structure section — change `detail/` to reflect new subdirectory layout
- Remove references to forwarding headers
- Update aggregate header names from `Tycho_*.h` to `tycho_*.h`
- Update the `Bindings/` section description
- Update `src/` forwarding header descriptions (remove entirely)

- [ ] **Step 3: Update doc/VectorFunction.md and doc/Bindings.md**

```bash
grep -n "detail/" doc/VectorFunction.md doc/Bindings.md
grep -n "src/VectorFunctions/" doc/VectorFunction.md doc/Bindings.md
grep -n "Tycho_" doc/VectorFunction.md doc/Bindings.md
```

Update any stale paths in these files to reflect the new structure.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor: update documentation for project restructuring"
```

---

## Task 13: Final Verification and PR

- [ ] **Step 1: Full clean build from scratch**

```bash
cd build
cmake --preset linux-clang-release
ninja -j6 all
```

- [ ] **Step 2: Full test suite**

```bash
ctest --output-on-failure
conda activate tycho && python scripts/run_examples.py
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

- [ ] **Step 3: clang-format**

```bash
cd build && ninja clang-format
git diff --stat
```

If formatting changes were made: `git add -A && git commit -m "style: apply clang-format after restructuring"`

- [ ] **Step 4: Create PR**

```bash
gh pr create --title "refactor: project reorganization (Phase 1)" --body "$(cat <<'EOF'
## Summary
- Move Python examples into `examples/python_examples/`
- Remove ~143 vestigial forwarding headers from `src/`
- Rename all `detail/` headers and `src/*.cpp` files to snake_case
- Reorganize `include/tycho/detail/` into domain subdirectories (`utils/`, `typedefs/`, `vf/`, `integrators/`, `optimal_control/`, `solvers/`, `astro/`)
- Update all `#include` paths, CMakeLists.txt, and documentation

**Note:** This is a pure structural reorganization. No code identifiers (class names, function names, variables) are changed. Phase 2 will migrate identifiers to the agreed naming conventions.

## Test plan
- [ ] `ninja -j6 all` succeeds from clean build
- [ ] `ctest --output-on-failure` — all C++ tests pass
- [ ] `python scripts/run_examples.py` — all 38 Python examples pass
- [ ] `brachistochrone_cpp` reports "Optimal Solution Found"

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
