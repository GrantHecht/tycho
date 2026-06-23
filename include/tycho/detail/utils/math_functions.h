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

/// @brief Compute the factorial of a non-negative integer.
/// @param i  Non-negative integer.
/// @return  `i!` (returns 1 for i = 0 or i = 1).
inline int factorial(int i) { return (i == 1 || i == 0) ? 1 : factorial(i - 1) * i; }

/// @brief Compute the product of all integers strictly between two bounds.
///
/// Returns the product of the integers in the open interval formed by @p i
/// and @p j, handling the degenerate cases where either argument is zero or
/// the arguments are adjacent (returns the larger one) or equal (returns 1).
///
/// @param i  First bound (non-negative integer).
/// @param j  Second bound (non-negative integer).
/// @return  The partial-factorial product between @p i and @p j.
inline int factorial_div(int i, int j) {
    if (i == 0) {
        return factorial_div(i + 1, j);
    } else if (j == 0) {
        return factorial_div(i, j + 1);
    } else if (j == i + 1) {
        return j;
    } else if (i == j + 1) {
        return i;
    } else if (i > j) {
        return i * factorial_div(i - 1, j);
    } else if (j > i) {
        return j * factorial_div(i, j - 1);
    } else if (i == j) {
        return 1;
    } else {
        return 0;
    }
}
} // namespace tycho::utils
