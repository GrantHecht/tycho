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

namespace tycho::vf {

/// @brief Assignment tag selecting plain `target = value` semantics.
/// @ingroup vf
struct DirectAssignment {};

/// @brief Assignment tag selecting `target += value` semantics.
/// @ingroup vf
struct PlusEqualsAssignment {};

/// @brief Assignment tag selecting `target -= value` semantics.
/// @ingroup vf
struct MinusEqualsAssignment {};

/// @brief Assignment tag selecting aliased `target = value` semantics.
/// @ingroup vf
///
/// Used where the target may alias one of the operands, disabling the
/// `noalias()` fast path.
struct AliasedDirectAssignment {};

/// @brief Assignment tag selecting scaled `target = scale * value` semantics.
/// @ingroup vf
/// @tparam Scalar  Numeric type of the scale factor.
template <class Scalar> struct ScaledDirectAssignment {
    Scalar value; ///< @brief Scale factor applied to the assigned value.
    /// @brief Constructs the tag from a scale factor.
    /// @param v  Scale factor to store.
    ScaledDirectAssignment(Scalar v) : value(v) {}
};

/// @brief Assignment tag selecting scaled `target += scale * value` semantics.
/// @ingroup vf
/// @tparam Scalar  Numeric type of the scale factor.
template <class Scalar> struct ScaledPlusEqualsAssignment {
    Scalar value; ///< @brief Scale factor applied to the accumulated value.
    /// @brief Constructs the tag from a scale factor.
    /// @param v  Scale factor to store.
    ScaledPlusEqualsAssignment(Scalar v) : value(v) {}
};

} // namespace tycho::vf