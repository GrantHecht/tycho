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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <fmt/format.h>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho::vf {

/// @brief Direction of a finite-difference stencil relative to the sample point.
/// @ingroup vf
enum class FDCoeffType {
    Backwards, ///< Backward (left-sided) stencil.
    Central,   ///< Centered (symmetric) stencil.
    Forwards,  ///< Forward (right-sided) stencil.
};

/// @brief Fixed-size array of finite-difference weights.
/// @tparam SZ  Number of stencil nodes (weights).
/// @ingroup vf
template <int SZ> using arr = std::array<double, SZ>;

/// @brief Compile-time table of finite-difference stencil weights.
///
/// The primary template is undefined; each `(Order, Accuracy, Dir, Shift)` combination
/// has an explicit specialization (or, for `Backwards`, a generic specialization
/// derived from the matching `Forwards` table). Each holds the node count `N` and the
/// per-node `weights`. A derivative is formed as
/// @f$ f^{(\mathrm{Order})}(x) \approx h^{-\mathrm{Order}} \sum_k \mathrm{weights}[k]\, f(x_k) @f$.
/// @tparam Order     Derivative order to approximate.
/// @tparam Accuracy  Order of accuracy @f$ O(h^{\mathrm{Accuracy}}) @f$.
/// @tparam Dir       Stencil direction (forward, central, or backward).
/// @tparam Shift     Stencil offset for one-sided semi-shifted stencils.
/// @ingroup vf
template <int Order, int Accuracy, FDCoeffType Dir, int Shift> struct FDCoeffs;

////////////////////////////////////////////////////////////////////////////////
//// Forward Coefficients
/// No shift
// 1st derivatives
/// @brief Forward stencil: 1st derivative, O(h^2), no shift.
template <> struct FDCoeffs<1, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 3;                        ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1.5, 2, -0.5}; ///< Per-node weights.
};

/// @brief Forward stencil: 1st derivative, O(h^4), no shift.
template <> struct FDCoeffs<1, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 5; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-25 / 12.0, 4, -3, 4 / 3.0, -1 / 4.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 1st derivative, O(h^6), no shift.
template <> struct FDCoeffs<1, 6, FDCoeffType::Forwards, 0> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-49 / 20.0, 6,       -15 / 2.0, 20 / 3.0,
                                       -15 / 4.0,  6 / 5.0, -1 / 6.0}; ///< Per-node weights.
};

// 2nd derivatives
/// @brief Forward stencil: 2nd derivative, O(h^2), no shift.
template <> struct FDCoeffs<2, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 4;                       ///< Number of stencil nodes.
    static constexpr arr<N> weights = {2, -5, 4, -1}; ///< Per-node weights.
};

/// @brief Forward stencil: 2nd derivative, O(h^4), no shift.
template <> struct FDCoeffs<2, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 6; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {15 / 4.0, -77 / 6.0, 107 / 6.0,
                                       -13,      61 / 12.0, -5 / 6.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 2nd derivative, O(h^6), no shift.
template <> struct FDCoeffs<2, 6, FDCoeffType::Forwards, 0> {
    static constexpr int N = 8; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {469 / 90.0,  -223 / 10.0,  879 / 20.0, -949 / 18.0, 41,
                                       -201 / 10.0, 1019 / 180.0, -7 / 10.0}; ///< Per-node weights.
};

// 3rd derivatives
/// @brief Forward stencil: 3rd derivative, O(h^2), no shift.
template <> struct FDCoeffs<3, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 5;                                        ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-5 / 2.0, 9, -12, 7, -3 / 2.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 3rd derivative, O(h^4), no shift.
template <> struct FDCoeffs<3, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-49 / 8.0,  29, -461 / 8.0, 62,
                                       -307 / 8.0, 13, -15 / 8.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 3rd derivative, O(h^6), no shift.
template <> struct FDCoeffs<3, 6, FDCoeffType::Forwards, 0> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-801 / 80.0, 349 / 6.0,    -18353 / 120.0,
                                       2391 / 10.0, -1457 / 30.0, -561 / 8.0,
                                       527 / 30.0,  -469 / 240.0}; ///< Per-node weights.
};

// 4th derivatives
/// @brief Forward stencil: 4th derivative, O(h^2), no shift.
template <> struct FDCoeffs<4, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 6;                                  ///< Number of stencil nodes.
    static constexpr arr<N> weights = {3, -14, 26, -24, 11, -2}; ///< Per-node weights.
};

/// @brief Forward stencil: 4th derivative, O(h^4), no shift.
template <> struct FDCoeffs<4, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 8; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {28 / 3.0,   -111 / 2.0, 142,     -1219 / 6.0, 176,
                                       -185 / 2.0, 82 / 3.0,   -7 / 2.0}; ///< Per-node weights.
};

//------------------------------------------------------------------------------
/// 1 shift
// 1st derivatives
/// @brief Forward stencil: 1st derivative, O(h^4), shift 1.
template <> struct FDCoeffs<1, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 5; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1 / 4.0, -5 / 6.0, 3 / 2.0, -1 / 2.0,
                                       1 / 12.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 1st derivative, O(h^6), shift 1.
template <> struct FDCoeffs<1, 6, FDCoeffType::Forwards, 1> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1 / 6.0, -77 / 60.0, 5 / 2.0, -5 / 3.0,
                                       5 / 6.0,  -1 / 4.0,   1 / 30.0}; ///< Per-node weights.
};

// 2nd derivatives
/// @brief Forward stencil: 2nd derivative, O(h^4), shift 1.
template <> struct FDCoeffs<2, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 6; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {5 / 6.0, -5 / 4.0, -1 / 3.0,
                                       7 / 6.0, -1 / 2.0, 1 / 12.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 2nd derivative, O(h^6), shift 1.
template <> struct FDCoeffs<2, 6, FDCoeffType::Forwards, 1> {
    static constexpr int N = 8; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {7 / 10.0, -7 / 18.0, -27 / 10.0, 19 / 4.0, -67 / 18.0,
                                       9 / 5.0,  -1 / 2.0,  11 / 180.0}; ///< Per-node weights.
};

// 3rd derivatives
/// @brief Forward stencil: 3rd derivative, O(h^2), shift 1.
template <> struct FDCoeffs<3, 2, FDCoeffType::Forwards, 1> {
    static constexpr int N = 5;                                       ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-3 / 2.0, 5, -6, 3, -1 / 2.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 3rd derivative, O(h^4), shift 1.
template <> struct FDCoeffs<3, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-15 / 8.0, 7, -83 / 8.0, 8,
                                       -29 / 8.0, 1, -1 / 8.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 3rd derivative, O(h^6), shift 1.
template <> struct FDCoeffs<3, 6, FDCoeffType::Forwards, 1> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {
        -469 / 240.0, 303 / 40.0, -731 / 60.0, 269 / 24.0, -57 / 8.0,
        407 / 120.0,  -67 / 60.0, 9 / 40.0,    -1 / 48.0}; ///< Per-node weights.
};

// 4th derivatives
/// @brief Forward stencil: 4th derivative, O(h^2), shift 1.
template <> struct FDCoeffs<4, 2, FDCoeffType::Forwards, 1> {
    static constexpr int N = 6;                                ///< Number of stencil nodes.
    static constexpr arr<N> weights = {2, -9, 16, -14, 6, -1}; ///< Per-node weights.
};

/// @brief Forward stencil: 4th derivative, O(h^4), shift 1.
template <> struct FDCoeffs<4, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 8; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {7 / 2.0, -56 / 3.0, 85 / 2.0, -54, 251 / 6.0,
                                       -20,     11 / 2.0,  -2 / 3.0}; ///< Per-node weights.
};

//------------------------------------------------------------------------------
/// 2 shift
// 1st derivatives
/// @brief Forward stencil: 1st derivative, O(h^6), shift 2.
template <> struct FDCoeffs<1, 6, FDCoeffType::Forwards, 2> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {1 / 30.0, -2 / 5.0, -7 / 12.0, 4 / 3.0,
                                       -1 / 2.0, 2 / 15.0, -1 / 60.0}; ///< Per-node weights.
};

// 2nd derivatives
/// @brief Forward stencil: 2nd derivative, O(h^6), shift 2.
template <> struct FDCoeffs<2, 6, FDCoeffType::Forwards, 2> {
    static constexpr int N = 8; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-11 / 180.0, 107 / 90.0, -21 / 10.0, 13 / 18.0, 17 / 36.0,
                                       -3 / 10.0,   4 / 45.0,   -1 / 90.0}; ///< Per-node weights.
};

// 3rd derivatives
/// @brief Forward stencil: 3rd derivative, O(h^4), shift 2.
template <> struct FDCoeffs<3, 4, FDCoeffType::Forwards, 2> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1 / 8.0, -1, 35 / 8.0, -6,
                                       29 / 8.0, -1, 1 / 8.0}; ///< Per-node weights.
};

/// @brief Forward stencil: 3rd derivative, O(h^6), shift 2.
template <> struct FDCoeffs<3, 6, FDCoeffType::Forwards, 2> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {
        -1 / 48.0, -53 / 30.0,  273 / 40.0, -313 / 30.0, 103 / 12.0,
        -9 / 2.0,  197 / 120.0, -11 / 30.0, 3 / 80.0}; ///< Per-node weights.
};

// 4th derivatives
/// @brief Forward stencil: 4th derivative, O(h^4), shift 2.
template <> struct FDCoeffs<4, 4, FDCoeffType::Forwards, 2> {
    static constexpr int N = 8; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {2 / 3.0, -11 / 6.0, 0,      31 / 6.0, -22 / 3.0,
                                       9 / 2.0, -4 / 3.0,  1 / 6.0}; ///< Per-node weights.
};

////////////////////////////////////////////////////////////////////////////////
//// Backward Coefficients
/// @brief Backward stencil specialization derived by reflecting the matching forward stencil.
///
/// Reuses the `Forwards` table for the same `(Order, Accuracy, Shift)` and reverses its
/// node order, applying the parity sign @ref C so the result is the correct backward
/// (left-sided) stencil.
/// @tparam Order     Derivative order to approximate.
/// @tparam Accuracy  Order of accuracy @f$ O(h^{\mathrm{Accuracy}}) @f$.
/// @tparam Shift     Stencil offset for semi-shifted stencils.
/// @ingroup vf
template <int Order, int Accuracy, int Shift>
struct FDCoeffs<Order, Accuracy, FDCoeffType::Backwards, Shift>
    : FDCoeffs<Order, Accuracy, FDCoeffType::Forwards, Shift> {
    /// @brief The matching forward stencil this backward stencil is reflected from.
    using Base = FDCoeffs<Order, Accuracy, FDCoeffType::Forwards, Shift>;
    using Base::N;

    /// @brief Reverses the node order of @p a and applies the parity sign @ref C.
    /// @param a  The forward stencil weights to reflect.
    /// @return The reflected (backward) stencil weights.
    static constexpr arr<N> flip(const arr<N> a) {
        arr<N> out;
        for (int i = 0; i < N; i++) {
            out[i] = C * a[N - i];
        }
        return out;
    }

    static constexpr int C = ((Order / 2) == ((Order + 1) / 2))
                                 ? (1)
                                 : (-1); ///< Parity sign: +1 for even derivative order, -1 for odd.
    static constexpr arr<N> weights = flip(Base::weights); ///< Per-node backward weights.
};

////////////////////////////////////////////////////////////////////////////////
//// Centered Coefficients
// 1st derivatives
/// @brief Central stencil: 1st derivative, O(h^2).
template <> struct FDCoeffs<1, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 3;                       ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-0.5, 0, 0.5}; ///< Per-node weights.
};

/// @brief Central stencil: 1st derivative, O(h^4).
template <> struct FDCoeffs<1, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 5; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {1 / 12.0, -2 / 3.0, 0, 2 / 3.0,
                                       -1 / 12.0}; ///< Per-node weights.
};

/// @brief Central stencil: 1st derivative, O(h^6).
template <> struct FDCoeffs<1, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1 / 60.0, 3 / 20.0,  -3 / 4.0, 0,
                                       3 / 4.0,   -3 / 20.0, 1 / 60.0}; ///< Per-node weights.
};

/// @brief Central stencil: 1st derivative, O(h^8).
template <> struct FDCoeffs<1, 8, FDCoeffType::Central, 0> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {
        1 / 280.0, -4 / 105.0, 1 / 5.0,   -4 / 5.0,  0,
        4 / 5.0,   -1 / 5.0,   4 / 105.0, -1 / 280.0}; ///< Per-node weights.
};

// 2nd derivatives
/// @brief Central stencil: 2nd derivative, O(h^2).
template <> struct FDCoeffs<2, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 3;                   ///< Number of stencil nodes.
    static constexpr arr<N> weights = {1, -2, 1}; ///< Per-node weights.
};

/// @brief Central stencil: 2nd derivative, O(h^4).
template <> struct FDCoeffs<2, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 5; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1 / 12.0, 4 / 3.0, -5 / 2.0, 4 / 3.0,
                                       -1 / 12.0}; ///< Per-node weights.
};

/// @brief Central stencil: 2nd derivative, O(h^6).
template <> struct FDCoeffs<2, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {1 / 90.0, -3 / 20.0, 3 / 2.0, -49 / 18.0,
                                       3 / 2.0,  -3 / 20.0, 1 / 90.0}; ///< Per-node weights.
};

/// @brief Central stencil: 2nd derivative, O(h^8).
template <> struct FDCoeffs<2, 8, FDCoeffType::Central, 0> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {
        -1 / 560.0, 8 / 315.0, -1 / 5.0,  8 / 5.0,   -205 / 72.0,
        8 / 5.0,    -1 / 5.0,  8 / 315.0, -1 / 560.0}; ///< Per-node weights.
};

// 3rd derivatives
/// @brief Central stencil: 3rd derivative, O(h^2).
template <> struct FDCoeffs<3, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 5;                              ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-0.5, 1, 0, -1, 0.5}; ///< Per-node weights.
};

/// @brief Central stencil: 3rd derivative, O(h^4).
template <> struct FDCoeffs<3, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {1 / 8.0,   -1, 13 / 8.0, 0,
                                       -13 / 8.0, 1,  -1 / 8.0}; ///< Per-node weights.
};

/// @brief Central stencil: 3rd derivative, O(h^6).
template <> struct FDCoeffs<3, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {
        -7 / 240.0, 3 / 10.0,    -169 / 120.0, 61 / 30.0, 0,
        -61 / 30.0, 169 / 120.0, -3 / 10.0,    7 / 240.0}; ///< Per-node weights.
};

// 4th derivatives
/// @brief Central stencil: 4th derivative, O(h^2).
template <> struct FDCoeffs<4, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 5;                          ///< Number of stencil nodes.
    static constexpr arr<N> weights = {1, -4, 6, -4, 1}; ///< Per-node weights.
};

/// @brief Central stencil: 4th derivative, O(h^4).
template <> struct FDCoeffs<4, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 7; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {-1 / 6.0,  2, -13 / 2.0, 28 / 3.0,
                                       -13 / 2.0, 2, -1 / 6.0}; ///< Per-node weights.
};

/// @brief Central stencil: 4th derivative, O(h^6).
template <> struct FDCoeffs<4, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 9; ///< Number of stencil nodes.
    static constexpr arr<N> weights = {
        7 / 240.0,   -2 / 5.0,   169 / 60.0, -122 / 15.0, 91 / 8.0,
        -122 / 15.0, 169 / 60.0, -2 / 5.0,   7 / 240.0}; ///< Per-node weights.
};

} // namespace tycho::vf
