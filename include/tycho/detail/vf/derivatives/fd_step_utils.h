// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace tycho::vf::detail {

/// @brief Relative finite-difference step scale: @f$ \max(1, |x_i|) @f$.
///
/// Multiplying a nominal (absolute) step by this scale keeps the step
/// proportional to the input magnitude, avoiding both catastrophic
/// cancellation at large @f$ |x_i| @f$ (where a fixed absolute step underflows
/// the representable spacing) and an oversized step near zero.
///
/// Elementwise for SuperScalar (Eigen array) scalars so each lane gets a
/// scale matched to its own magnitude.
/// @tparam Scalar  Either an arithmetic type (`double`) or an Eigen array
///                 (SuperScalar) type.
/// @param xi  The input component to scale the step by.
/// @return The relative step scale for @p xi.
template <class Scalar> inline Scalar fd_step_scale(const Scalar &xi) {
    if constexpr (std::is_arithmetic_v<Scalar>) {
        return std::max(Scalar(1), std::abs(xi));
    } else {
        return xi.abs().max(1.0);
    }
}

} // namespace tycho::vf::detail
