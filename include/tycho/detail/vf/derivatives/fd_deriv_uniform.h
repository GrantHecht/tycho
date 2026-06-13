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

#include "tycho/detail/vf/derivatives/fd_coeffs.h"
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

// Notes: "time axis" must be monotonic increasing

/// @brief Finite-difference differentiation of data sampled on a uniform axis.
///
/// Differentiates a series of vector samples assumed equally spaced along a monotonic
/// increasing axis, choosing forward, central, or backward stencils (from @ref FDCoeffs)
/// depending on the sample's distance from the data boundaries.
/// @tparam DType     Data type: an `Eigen::Matrix` type for each sample.
/// @tparam Order     Derivative order to compute.
/// @tparam Accuracy  Order of accuracy @f$ O(h^{\mathrm{Accuracy}}) @f$.
/// @ingroup vf
template <class DType, int Order, int Accuracy> struct FinDiffDerivUniform {
  public:
    /// @brief Stencil-coefficient table for the given direction and shift.
    template <FDCoeffType Dir, int Shift> using Coeffs = FDCoeffs<Order, Accuracy, Dir, Shift>;
    /// @brief Scalar element type of the data samples.
    using Scalar = typename DType::Scalar;

    int axis;   ///< Index of the independent (time) axis within each sample.
    int length; ///< Number of samples.
    std::vector<Eigen::MatrixBase<DType>> data; ///< The sampled data series.
    Scalar h;                                   ///< Uniform spacing along the axis.

    static constexpr int acc =
        2 * ((Accuracy + 1) / 2);     ///< Accuracy rounded up to the nearest even number.
    static constexpr int ord = Order; ///< Derivative order.

    static constexpr int cent_sten_size = ((ord / 2) == ((ord + 1) / 2))
                                              ? (ord - 1 + acc)
                                              : (ord + acc); ///< Centered-stencil node count.
    static constexpr int fb_sten_size = acc + ord; ///< Forward/backward-stencil node count.

    /// @brief Sets the index of the independent (time) axis within each sample.
    /// @param i  Axis index.
    inline void set_axis_id(int i) { this->axis = i; }

    /// @brief Loads the sampled data series and derives the uniform spacing @ref h.
    /// @param d  Samples to differentiate; must contain at least @ref fb_sten_size entries.
    inline void set_data(std::vector<Eigen::MatrixBase<DType>> d) {
        if (d.size() < this->fb_sten_size) {
            std::cout << "ERROR: Not enough data for desired derivative/accuracy" << std::endl;
            return;
        } else {
            this->data = d;
            this->length = d.size();
            this->h = d[1][axis] - d[0][axis];
        }
    }

    /// @brief Computes the finite-difference derivative at a single sample index.
    ///
    /// Selects a forward, central, or backward stencil based on @p i's distance from the
    /// data boundaries and writes the result into @p dout.
    /// @tparam DerivType  Output Eigen type.
    /// @param i     Sample index at which to evaluate the derivative.
    /// @param dout  Output buffer receiving the derivative.
    template <class DerivType>
    inline DType deriv_at(const int i, Eigen::MatrixBase<DerivType> &dout) const {
        if (i < 0 || i > length - 1) {
            std::cout << "ERROR: Index out of bounds" << std::endl;
            return;
        }

        // Calc shift
        FDCoeffType dir;
        int shift;
        std::vector<int> stencil;
        if (i < fb_sten_size / 2) { // Forward / semi-forward
            dir = FDCoeffType::Forwards;
            shift = i;
            for (int j = 0; j < fb_sten_size; j++) {
                stencil.push_back(i - j);
            }
        } else if (length - i < fb_sten_size / 2) { // Backward / semi-backward
            dir = FDCoeffType::Backwards;
            shift = (length - i);
            for (int j = length - fb_sten_size; j < length; j++) {
                stencil.push_back(i - j);
            }
        } else { // Centered
            dir = FDCoeffType::Central;
            shift = 0;
            int lb = cent_sten_size / 2;
            for (int j = 0; j < cent_sten_size; j++) {
                stencil.push_back(i + j - lb);
            }
        }
        int sz = stencil.size();

        // Get weights
        // using CF = Coeffs<dir, shift>;

        // Calc deriv
        dout.setZero();
        for (int j = 0; j < sz; j++) {
            // dout += CF::weights[j] * data[i + stencil[j]];
        }
    }
};

} // namespace tycho::vf
