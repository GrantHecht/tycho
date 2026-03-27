# Tycho Project Reorganization — Design Spec

**Date:** 2026-03-26
**Scope:** Phase 1 of a broader cleanup — structural reorganization only, no naming convention changes to code identifiers.
**Delivery:** Single PR covering all changes.

---

## Overview

This spec covers four workstreams delivered as one atomic PR, executed in this order:

1. Move Python examples into `examples/python_examples/`
2. Remove vestigial forwarding headers from `src/`
3. Rename all headers and `.cpp` source files to snake_case
4. Reorganize `include/tycho/detail/` into domain subdirectories

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
```

### Target layout

```
examples/
├── python_examples/
│   ├── <27 scripts>
│   ├── MeshRefinement/      (4 scripts, preserved)
│   └── UpdatedInterface/    (7 scripts, preserved)
└── cpp_examples/
    ├── static/
    └── builder/
```

### Changes required

- `git mv` the 27 root Python scripts into `examples/python_examples/`
- `git mv` `MeshRefinement/` and `UpdatedInterface/` into `examples/python_examples/`
- Delete `examples/Plots/` entirely
- Update `scripts/run_examples.py` to scan `examples/python_examples/` instead of `examples/`
- Update any other scripts or docs referencing example paths

---

## Step 2: Remove Forwarding Headers from `src/`

### Current state

Directories like `src/VectorFunctions/`, `src/OptimalControl/`, `src/Integrators/`, `src/TypeDefs/`, `src/Utils/`, and `src/Astro/` contain `.h` files that are one-line `#include` forwards to `include/tycho/detail/`:

```cpp
#pragma once
#include "tycho/detail/VectorFunction.h"
```

These serve no purpose — actual `.cpp` files bypass them and include from `detail/` directly.

### Action

- Delete all forwarding `.h` files from `src/` subdirectories
- Update `src/Bindings/` files that reference forwarding headers to use `tycho/detail/...` paths directly
- Update CMakeLists.txt `target_include_directories` if any `src/` subdirectories were listed solely for forwarding header access
- Review aggregate headers (`Tycho_VectorFunctions.h`, `Tycho_OptimalControl.h`, etc.) in `src/` — either update to include from `tycho/detail/` or delete if redundant with the public top-level headers

---

## Step 3: Rename All Files to Snake_case

### Headers (`include/tycho/detail/`)

All 150 headers renamed from PascalCase to snake_case. Mapping rule:

- `DenseFunctionBase.h` → `dense_function_base.h`
- `ODEPhaseBase.h` → `ode_phase_base.h`
- `PSIOPT.h` → `psiopt.h`
- `CR3BPModel.h` → `cr3bp_model.h`
- `LGLCoeffs.h` → `lgl_coeffs.h`
- `J2.h` → `j2.h`

Headers already in snake_case (e.g., `ode_builder.h`, `thread_pool.h`) are unchanged.

### Source files (`src/`)

All `.cpp` files renamed to snake_case. Examples:

- `src/OptimalControl/RuntimeODE.cpp` → `src/OptimalControl/runtime_ode.cpp`
- `src/Solvers/PSIOPT.cpp` → `src/Solvers/psiopt.cpp`

### Include path updates

Every `#include` referencing a renamed file must be updated. This affects:

- All `detail/` headers (internal cross-includes)
- Top-level public headers (`include/tycho/*.h`)
- `src/` source and binding files
- `pch.h` / `pch_bindings.h`
- `tests/cpp/` test files
- `bench/cpp/` benchmark files
- CMakeLists.txt files (any explicit source file lists)

---

## Step 4: Reorganize `detail/` into Subdirectories

### Target structure

```
include/tycho/detail/
├── utils/                              # Foundational utilities (13 headers)
│   ├── type_storage.h
│   ├── sizing_helpers.h
│   ├── crtp_base.h
│   ├── thread_pool.h
│   ├── get_core_count.h
│   ├── math_functions.h
│   ├── std_extensions.h
│   ├── function_return_type.h
│   ├── type_name.h
│   ├── flat_map.h
│   ├── memory_management.h
│   ├── timer.h
│   └── tuple_iterator.h
│
├── typedefs/                           # Type system fundamentals (1 header)
│   └── eigen_types.h
│
├── vf/                                 # Vector functions (~50 headers)
│   ├── core/                           # Base classes, specs, concepts
│   │   ├── dense_function_base.h
│   │   ├── dense_function.h
│   │   ├── dense_function_specs.h
│   │   ├── dense_function_operations.h
│   │   ├── dense_scalar_function_base.h
│   │   ├── computable_base.h
│   │   ├── computable.h
│   │   ├── sparse_function_base.h
│   │   ├── vector_function.h
│   │   ├── vector_function_concepts.h
│   │   ├── expression_fwd_declarations.h
│   │   ├── function_domains.h
│   │   ├── function_type_def_macros.h
│   │   ├── function_templates.h
│   │   ├── functional_flags.h
│   │   ├── input_output_size.h
│   │   ├── assignment_types.h
│   │   └── object_detectors.h
│   │
│   ├── expressions/                    # Composition & structural operations
│   │   ├── nested_function.h
│   │   ├── call_and_append.h
│   │   ├── stacked.h
│   │   ├── segment.h
│   │   ├── for.h
│   │   ├── parsed_input.h
│   │   ├── function_holder.h
│   │   ├── lambda_function.h
│   │   └── summation.h
│   │
│   ├── operators/                      # Math, products, norms
│   │   ├── binary_math.h
│   │   ├── cwise_operators.h
│   │   ├── cwise_product.h
│   │   ├── cwise_sum.h
│   │   ├── operator_overloads.h
│   │   ├── math_overloads.h
│   │   ├── arc_tan2.h
│   │   ├── sign_function.h
│   │   ├── root_finder.h
│   │   ├── cross_product.h
│   │   ├── dot_product.h
│   │   ├── vector_products.h
│   │   ├── vector_scalar_function_product.h
│   │   ├── vector_scalar_function_division.h
│   │   ├── function_vector_sums.h
│   │   ├── norms.h
│   │   ├── normalized.h
│   │   ├── matrix_function.h
│   │   ├── matrix_product.h
│   │   └── matrix_inverse.h
│   │
│   ├── derivatives/                    # Autodiff, finite differences
│   │   ├── dense_derivatives.h
│   │   ├── dense_autodiff_fwd.h
│   │   ├── dense_fdiff_cent_array.h
│   │   ├── dense_fdiff_fwd.h
│   │   ├── fd_coeffs.h
│   │   ├── fd_deriv_arbitrary.h
│   │   ├── fd_deriv_uniform.h
│   │   ├── detect_diagonal.h
│   │   └── detect_super_scalar.h
│   │
│   ├── scaling/                        # Scaling & padding
│   │   ├── scaled.h
│   │   ├── io_scaled.h
│   │   ├── padded.h
│   │   └── auto_scaling_utils.h
│   │
│   ├── type_erasure/                   # GenericFunction, conditionals, comparatives
│   │   ├── generic_function.h
│   │   ├── gf_type_erasure.h
│   │   ├── generic_conditional.h
│   │   ├── generic_comparative.h
│   │   ├── conditional.h
│   │   ├── conditional_base.h
│   │   ├── comparative.h
│   │   └── autodiff_function.h
│   │
│   └── common/                         # Leaf function types
│       ├── common_functions.h
│       ├── constant.h
│       ├── value.h
│       ├── value_lock.h
│       └── elements.h
│
├── integrators/                        # RK steppers + coefficients (3 headers)
│   ├── rk_coeffs.h
│   ├── rk_steppers.h
│   └── integrator.h
│
├── optimal_control/                    # Phase, ODE, transcription (~25 headers)
│   ├── core/                           # Flags, sizes, types
│   │   ├── optimal_control_flags.h
│   │   ├── ode_sizes.h
│   │   ├── ode_arguments.h
│   │   ├── interface_types.h
│   │   ├── state_function.h
│   │   └── link_function.h
│   │
│   ├── phase/                          # Phase & ODE definitions
│   │   ├── ode.h
│   │   ├── ode_phase_base.h
│   │   ├── ode_phase.h
│   │   ├── optimal_control_problem.h
│   │   ├── phase_indexer.h
│   │   └── mesh_iterate_info.h
│   │
│   ├── transcription/                  # Collocation methods, defects, integrals
│   │   ├── lgl_coeffs.h
│   │   ├── lgl_defects.h
│   │   ├── lgl_integrals.h
│   │   ├── lgl_control_splines.h
│   │   ├── lgl_interp_table.h
│   │   ├── lgl_interp_functions.h
│   │   ├── trapezoidal_defects.h
│   │   ├── trapezoidal_integrals.h
│   │   ├── shooting_defects.h
│   │   ├── transcription_sizing.h
│   │   ├── mesh_spacing_constraints.h
│   │   └── blocked_ode_wrapper.h
│   │
│   ├── interp/                         # Interpolation tables
│   │   ├── interp_table_1d.h
│   │   ├── interp_table_2d.h
│   │   ├── interp_table_3d.h
│   │   └── interp_table_4d.h
│   │
│   └── builder/                        # Builder API
│       ├── var_registry.h
│       ├── runtime_ode.h
│       ├── ode_builder.h
│       ├── phase_wrapper.h
│       └── ocp_wrapper.h
│
├── solvers/                            # PSIOPT + NLP (~17 headers)
│   ├── solver_flags.h
│   ├── solver_init.h
│   ├── solver_function_base.h
│   ├── solver_interface_specs.h
│   ├── sizing_specs.h
│   ├── constraint_function.h
│   ├── objective_function.h
│   ├── non_linear_program.h
│   ├── optimization_problem_base.h
│   ├── optimization_problem.h
│   ├── psiopt.h
│   ├── jet.h
│   ├── iterate_info.h
│   ├── indexing_data.h
│   └── linear/                         # Sparse linear solver backends
│       ├── pardiso_interface.h
│       ├── accelerate_interface.h
│       └── accelerate_utils.h
│
└── astro/                              # Astrodynamics (8 headers)
    ├── kepler_model.h
    ├── kepler_propagator.h
    ├── kepler_utils.h
    ├── cr3bp_model.h
    ├── j2.h
    ├── mee_dynamics.h
    ├── lambert_solvers.h
    └── thruster_models.h
```

### Design decisions

- **`vf/` mirrors the `tycho::vf` namespace** agreed upon for Phase 2.
- **LGL headers in `optimal_control/transcription/`** — these are collocation transcription machinery included by `optimal_control.h`, not `integrators.h`.
- **Interpolation tables in `optimal_control/interp/`** — used for table-driven dynamics within the OC framework.
- **`typedefs/` gets its own directory** for symmetry, even with one file.
- **`PythonArgParsing.h`** belongs in `src/Bindings/`, not `detail/`. Moved there as part of this cleanup.

### Include path updates

Every `#include` referencing a moved file must be updated (second pass — step 3 handled renames, this step handles new directory paths). This affects the same set of files as step 3.

---

## Verification Criteria

Before merging this PR:

1. `cd build && ninja -j6 all` — clean build succeeds
2. `ctest --output-on-failure` — all C++ unit tests pass
3. `python scripts/run_examples.py` — all 38 Python examples pass (from new `python_examples/` path)
4. `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` — converges with "Optimal Solution Found"
5. No stale `#include` paths remain (grep for old paths returns empty)
6. No forwarding headers remain in `src/` subdirectories

---

## Out of Scope

- **Naming convention migration** (class names, function names, variables, namespaces) — deferred to Phase 2, done module by module
- **`src/` directory renaming** (e.g., `src/VectorFunctions/` → `src/vf/`) — may be addressed in Phase 2 alongside namespace changes
- **Python package restructuring** (`tychopy/`) — not part of this effort
- **Build system modernization** — CMake changes are limited to updating file references
