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

namespace tycho::utils {

/// @internal
/// @brief Primary template for extracting the return type from a callable
/// type. Specializations cover free functions, member functions, and
/// cv-qualified member functions.
template <typename T> struct return_type;

/// @internal
/// @brief Specialization for plain free-function pointers.
template <typename R, typename... Args> struct return_type<R (*)(Args...)> {
    using type = R; ///< @internal The return type of the callable.
};

/// @internal
/// @brief Specialization for non-const member-function pointers.
template <typename R, typename C, typename... Args> struct return_type<R (C::*)(Args...)> {
    using type = R; ///< @internal The return type of the callable.
};

/// @internal
/// @brief Specialization for const member-function pointers.
template <typename R, typename C, typename... Args> struct return_type<R (C::*)(Args...) const> {
    using type = R; ///< @internal The return type of the callable.
};

/// @internal
/// @brief Specialization for volatile member-function pointers.
template <typename R, typename C, typename... Args> struct return_type<R (C::*)(Args...) volatile> {
    using type = R; ///< @internal The return type of the callable.
};

/// @internal
/// @brief Specialization for const-volatile member-function pointers.
template <typename R, typename C, typename... Args>
struct return_type<R (C::*)(Args...) const volatile> {
    using type = R; ///< @internal The return type of the callable.
};

/// @internal
/// @brief Convenience alias for `return_type<T>::type`.
template <typename T> using return_type_t = typename return_type<T>::type;

} // namespace tycho::utils
