#pragma once

/// @file vector_functions.h
/// @brief Umbrella header for the VectorFunction subsystem.
///
/// Includes the full public VectorFunction API: the CRTP base types, the
/// operator/expression DSL, the type-erased @ref tycho::vf::GenericFunction
/// family, the common-function library, and the VF concepts. Include this
/// single header to author dynamics, constraints, and objectives.

/// @defgroup vf VectorFunctions
/// @brief The core DSL for defining dynamics, constraints, and objectives.
///
/// Everything in Tycho's problem-definition layer is expressed as a
/// VectorFunction. The subsystem provides:
/// - **CRTP dispatch** of `compute_impl` for primal evaluation,
/// - **multi-mode derivatives** (analytic, finite-difference, EnzymeAD),
/// - **SuperScalar dispatch** for SIMD-batched evaluation, and
/// - an **expression DSL** (`+`, `*`, `.dot()`, `.segment()`, …) that
///   composes primitive functions into larger ones while tracking the
///   sparsity (input domain) of the resulting Jacobian.
///
/// See the @ref tycho::vf::DenseFunctionBase reference for the operations
/// available on every VectorFunction.

#include "tycho/typedefs.h"
#include "tycho/utils.h"

// Core VF types
#include "tycho/detail/vf/core/dense_function_base.h"
#include "tycho/detail/vf/core/vector_function.h"
#include "tycho/detail/vf/operators/math_overloads.h"
#include "tycho/detail/vf/operators/operator_overloads.h"

// Type erasure
#include "tycho/detail/vf/type_erasure/generic_comparative.h"
#include "tycho/detail/vf/type_erasure/generic_conditional.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"

// Common functions library
#include "tycho/detail/vf/common/common_functions.h"

// Concepts
#include "tycho/detail/vf/core/vector_function_concepts.h"

// Extern template declarations — suppress implicit instantiation of common
// VF types in consuming TUs (explicit instantiations in vf_instantiations lib)
#include "tycho/detail/vf/extern_templates.h"
