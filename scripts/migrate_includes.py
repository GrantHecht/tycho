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

    # ---- PythonArgParsing renamed in Bindings ----
    "PythonArgParsing.h":                  "python_arg_parsing.h",
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
