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
//   - Extracted ConvergenceFlags enum, severity ordering operator, and forward declaration from
//   psiopt.h
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
// =============================================================================

#pragma once
#include <compare>

namespace tycho {

/// Optimizer convergence status. Lives in tycho:: (not tycho::solvers) so callers
/// outside the solvers module can reference it directly. Placed in a forward-
/// declaration header to break include cycles with dependent modules.
enum class ConvergenceFlags {
    CONVERGED = 0,
    ACCEPTABLE = 1,
    NOTCONVERGED = 2,
    DIVERGING = 3,
};

// Severity ordering: CONVERGED < ACCEPTABLE < NOTCONVERGED < DIVERGING
constexpr auto operator<=>(ConvergenceFlags a, ConvergenceFlags b) {
    return static_cast<int>(a) <=> static_cast<int>(b);
}

} // namespace tycho

namespace tycho::solvers {
class PSIOPT;
} // namespace tycho::solvers
