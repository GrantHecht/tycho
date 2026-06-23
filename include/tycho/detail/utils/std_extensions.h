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

#include <type_traits>

namespace std {

/// @internal
/// @brief Type trait that strips both `const` and reference qualifiers from T
/// in a single step (equivalent to `std::remove_const_t<std::remove_reference_t<T>>`).
template <class T> struct remove_const_reference {
    using type = ///< @internal Resolved type with const and reference removed.
        typename std::remove_const<typename std::remove_reference<T>::type>::type;
};

} // namespace std
