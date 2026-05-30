// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/dense_function_base.h"

namespace tycho::vf {

/// @brief Selects how a VectorFunction's Jacobian and Hessian are evaluated.
///
/// This mode is passed as a template argument to a user `VectorFunction` to choose
/// the derivative-evaluation strategy. The first template slot selects the Jacobian
/// mode; the second (where present) selects the Hessian mode. The chosen mode
/// determines which `DenseFirstDerivatives` / `DenseSecondDerivatives` specialization
/// is instantiated.
/// @ingroup vf
enum class DenseDerivativeMode {
    Analytic,       ///< Hand-written analytic derivatives supplied by the function itself.
    FDiffFwd,       ///< Forward (first-order) finite-difference Jacobian.
    FDiffCentArray, ///< Central finite-difference Jacobian via packed SuperScalar arrays.
    EnzymeAD,       ///< Enzyme automatic differentiation (compiler-plugin AD).
};

/// @brief Mixes a chosen Jacobian-evaluation strategy into a dense VectorFunction.
///
/// The primary template provides the default (analytic) behavior by inheriting the
/// base function directly; explicit specializations supply the finite-difference and
/// Enzyme Jacobian implementations selected by @p JMode.
/// @tparam Derived  The concrete VectorFunction type (CRTP self type).
/// @tparam IR       Input dimension (rows of the argument), or `Eigen::Dynamic`.
/// @tparam OR       Output dimension (rows of the result), or `Eigen::Dynamic`.
/// @tparam JMode    Jacobian-evaluation mode selecting the specialization.
/// @ingroup vf
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseFirstDerivatives : DenseFunctionBase<Derived, IR, OR> {
    /// @brief The dense function base providing the primal `compute` interface.
    using Base = DenseFunctionBase<Derived, IR, OR>;
};

/// @brief Mixes chosen Jacobian and Hessian strategies into a dense VectorFunction.
///
/// Layers a Hessian-evaluation strategy on top of @ref DenseFirstDerivatives, which
/// already supplies the Jacobian. The primary template defaults to analytic Hessians;
/// finite-difference and Enzyme modes are provided by specializations.
/// @tparam Derived  The concrete VectorFunction type (CRTP self type).
/// @tparam IR       Input dimension (rows of the argument), or `Eigen::Dynamic`.
/// @tparam OR       Output dimension (rows of the result), or `Eigen::Dynamic`.
/// @tparam Jmode    Jacobian-evaluation mode (forwarded to @ref DenseFirstDerivatives).
/// @tparam Hmode    Hessian-evaluation mode selecting the specialization.
/// @ingroup vf
template <class Derived, int IR, int OR, DenseDerivativeMode Jmode, DenseDerivativeMode Hmode>
struct DenseSecondDerivatives : DenseFirstDerivatives<Derived, IR, OR, Jmode> {
    /// @brief The first-derivative layer providing the Jacobian interface.
    using Base = DenseFirstDerivatives<Derived, IR, OR, Jmode>;
};

} // namespace tycho::vf
