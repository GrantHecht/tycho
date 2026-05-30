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

/// @brief Injects the standard scalar-templated VF matrix type aliases.
///
/// Expands to `Output`, `Input`, `Gradient`, `Jacobian`, and `Hessian`
/// member-template aliases that forward to the corresponding aliases on
/// @p Base, so derived VectorFunction classes inherit them without
/// re-declaration.
/// @param Base  Base class providing the templated `Output`/`Input`/… aliases.
#define VF_TYPE_ALIASES(Base)                                                                      \
    template <class Scalar> using Output = typename Base::template Output<Scalar>;                 \
    template <class Scalar> using Input = typename Base::template Input<Scalar>;                   \
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;             \
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;             \
    template <class Scalar> using Hessian = typename Base::template Hessian<Scalar>;

} // namespace tycho::vf
