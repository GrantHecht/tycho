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

/// @brief Compute the partial factorial `max(i,j)! / min(i,j)!`.
///
/// Returns the product of all integers from `min(i,j)+1` to `max(i,j)`
/// inclusive.  Degenerate cases: equal arguments return 1; adjacent arguments
/// return the larger value; zero arguments are normalised to 1 before
/// recursing.
///
/// @param i  First non-negative integer bound.
/// @param j  Second non-negative integer bound.
/// @return  Product of integers in `(min(i,j), max(i,j)]`; 1 when `i == j`.
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
