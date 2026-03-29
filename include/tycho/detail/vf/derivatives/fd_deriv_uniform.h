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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho::vf {

// Notes: "time axis" must be monotonic increasing

/*!
 * @brief
 *
 * @tparam DType Data Type: An Eigen::Matrix type
 * @tparam Accuracy O(h^Accuracy)
 * @tparam Order Take the "Order-th" derivative
 */
template <class DType, int Order, int Accuracy> struct FinDiffDerivUniform {
  public:
    template <FDCoeffType Dir, int Shift> using Coeffs = FDCoeffs<Order, Accuracy, Dir, Shift>;
    using Scalar = typename DType::Scalar;

    int axis;
    int length;
    std::vector<Eigen::MatrixBase<DType>> data;
    Scalar h;

    static constexpr int acc = 2 * ((Accuracy + 1) / 2);
    static constexpr int ord = Order;

    static constexpr int cent_sten_size =
        ((ord / 2) == ((ord + 1) / 2)) ? (ord - 1 + acc) : (ord + acc);
    static constexpr int fb_sten_size = acc + ord;

    inline void set_axis_id(int i) { this->axis = i; }

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
