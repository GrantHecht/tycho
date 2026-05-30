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

#include "tycho/detail/vf/core/vector_function_concepts.h"

namespace tycho::vf {

/// @internal
/// @brief Trait: reports whether `T` is a SuperScalar (SIMD-lane-packed) scalar type.
///
/// Wraps the `IsSuperScalar<T>` concept as a `std::bool_constant` so it can be used in
/// trait-style metaprogramming. SuperScalar scalars (packed `Eigen::Array` columns)
/// drive the vectorized derivative paths.
/// @tparam T  Candidate scalar type.
/// @endinternal
template <class T> struct Is_SuperScalar : std::bool_constant<IsSuperScalar<T>> {};

} // namespace tycho::vf
