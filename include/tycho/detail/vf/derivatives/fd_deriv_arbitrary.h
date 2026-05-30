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

#include <Eigen/Cholesky>

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

/// @brief Finite-differences a series of vector samples along one axis.
///
/// Computes the per-sample derivative of @p BL with respect to the chosen @p axis using
/// a mix of forward, central, and backward stencils at the boundaries and interior.
/// @tparam VectorType  Eigen vector type of each sample.
/// @param BL       The sampled data series; spacing is inferred from the first two samples.
/// @param axis     Index of the independent axis within each sample.
/// @param inctime  Whether the returned samples retain the axis (time) component.
/// @return One derivative per input sample.
template <class VectorType>
std::vector<VectorType> FDiffData(const std::vector<VectorType> &BL, int axis, bool inctime) {
    std::vector<VectorType> BLDot = BL;
    double h = BL[1][axis] - BL[0][axis];
    for (int i = 0; i < BL.size(); i++) {
        if (!inctime) {
            BLDot[i].resize(axis);
        }
        if (i < 2) {
            BLDot[i].head(axis) = ((1.0 / 3.0) * BL[i + 3].head(axis) - 1.5 * BL[i + 2].head(axis) +
                                   3.0 * BL[i + 1].head(axis) - (11.0 / 6) * BL[i].head(axis)) *
                                  (1.0 / h);
        } else if (i > (BL.size() - 3)) {
            BLDot[i].head(axis) = (BL[i].head(6) - BL[i - 1].head(axis)) * (1.0 / h);
        } else {
            BLDot[i].head(axis) =
                ((-1.0 / 12) * BL[i + 2].head(axis) + (2.0 / 3.0) * BL[i + 1].head(axis) +
                 (-2.0 / 3.0) * BL[i - 1].head(axis) + (1.0 / 12.0) * BL[i - 2].head(axis)) *
                (1.0 / h);
        }
    }
    return BLDot;
}

/// @brief Finite-difference differentiation of data sampled on an arbitrary (non-uniform) axis.
///
/// Unlike @ref FinDiffDerivUniform, the samples need not be equally spaced: stencil
/// weights are solved per evaluation from the actual node offsets via a Vandermonde
/// system, choosing forward, central, or backward stencils near the boundaries.
/// @tparam DType  Data type: an `Eigen::Matrix` type for each sample.
/// @ingroup vf
template <class DType> struct FDDerivArbitrary {
  public:
    /// @brief Scalar element type of the data samples.
    using Scalar = typename DType::Scalar;

    int axis;                ///< Index of the independent axis within each sample.
    int length;              ///< Number of samples.
    std::vector<DType> data; ///< The sampled data series.

    /// @brief Sets the index of the independent axis within each sample.
    /// @param i  Axis index.
    inline void set_axis(int i) { this->axis = i; }

    /// @brief Loads the sampled data series.
    /// @param d  Samples to differentiate.
    inline void set_data(const std::vector<DType> &d) {
        this->data = d;
        this->length = d.size();
    }

    /// @brief Constructs an empty differentiator; set axis and data before use.
    FDDerivArbitrary() {};
    /// @brief Constructs a differentiator with the given axis and data.
    /// @param i  Index of the independent axis within each sample.
    /// @param d  Samples to differentiate.
    FDDerivArbitrary(int i, const std::vector<DType> &d) {
        this->set_axis(i);
        this->set_data(d);
    }

    /*!
     * @brief Finite-difference derivative at a single sample index, written
     *        into a caller-provided buffer.
     *
     * @tparam DerivType Output Eigen::Matrix type.
     * @param i Sample index at which to evaluate the derivative.
     * @param dout Output buffer receiving the derivative.
     * @param Order Order of the derivative to compute.
     * @param Accuracy Finite-difference accuracy order (rounded up to the
     *                 nearest even number).
     */
    template <class DerivType>
    inline void deriv_at(const int i, DerivType &dout, const int Order, const int Accuracy) const {
        const int acc = 2 * ((Accuracy + 1) / 2);
        const int ord = Order;

        const int cent_sten_size = ((ord / 2) == ((ord + 1) / 2)) ? (ord - 1 + acc) : (ord + acc);
        const int fb_sten_size = acc + ord;

        if (length < fb_sten_size) {
            std::cout << "ERROR: Requested accuracy too high for given data" << std::endl;
        }

        if (i < 0 || i > length - 1) {
            std::cout << "ERROR: Index out of bounds" << std::endl;
            return;
        }

        Scalar t0 = data[i][axis];

        // Calc steps and stencil
        Eigen::Matrix<Scalar, -1, 1> steps;
        std::vector<int> stencil;
        if (i < fb_sten_size / 2) { // Forward / semi-forward
            steps.resize(fb_sten_size);
            for (int j = 0; j < fb_sten_size; j++) {
                steps[j] = data[j][axis] - t0;
                stencil.push_back(j - i);
            }
        } else if (length - 1 - i < fb_sten_size / 2) { // Backward / semi-backward
            steps.resize(fb_sten_size);
            for (int j = length - fb_sten_size; j < length; j++) {
                int k = j - length + fb_sten_size;
                steps[k] = data[j][axis] - t0;
                stencil.push_back(j - i);
            }
        } else { // Centered
            steps.resize(cent_sten_size);
            int lb = cent_sten_size / 2;
            for (int j = 0; j < cent_sten_size; j++) {
                steps[j] = data[i - lb + j][axis] - t0;
                stencil.push_back(j - lb);
            }
        }

        // Calc weights
        int sz = steps.size();

        Eigen::Matrix<Scalar, -1, -1> mat(sz, sz);
        for (int j = 0; j < sz; j++) {
            mat.row(j) = steps.array().pow(j);
        }

        Eigen::Matrix<Scalar, -1, 1> vec(sz);
        vec.setZero();
        vec[ord] = tycho::utils::factorial(ord);

        Eigen::Matrix<Scalar, -1, 1> weights = mat.colPivHouseholderQr().solve(vec);

        // Calc deriv
        dout.setZero();
        for (int j = 0; j < sz; j++) {
            dout += weights[j] * data[i + stencil[j]];
        }
    }

    /*!
     * @brief Finite-difference derivative at a single sample index, returned
     *        as a new vector.
     *
     * @tparam DerivType Output Eigen::Matrix type.
     * @param i Sample index at which to evaluate the derivative.
     * @param Order Order of the derivative to compute.
     * @param Accuracy Finite-difference accuracy order (rounded up to the
     *                 nearest even number).
     * @return The derivative at index @p i.
     */
    template <class DerivType>
    inline DerivType deriv_at(const int i, const int Order, const int Accuracy) const {
        DerivType dout(this->data[0].size());
        this->deriv_at<DType>(i, dout, Order, Accuracy);
        dout[axis] = this->data[i][axis];
        return dout;
    }

    /*!
     * @brief Calculate all derivatives at once, into a caller-provided vector.
     *
     * @tparam DerivType Output Eigen::Matrix type.
     * @param bulk Output vector receiving one derivative per sample index.
     * @param Order Order of the derivative to compute.
     * @param Accuracy Finite-difference accuracy order (rounded up to the
     *                 nearest even number).
     */
    template <class DerivType>
    inline void deriv(std::vector<DerivType> &bulk, const int Order, const int Accuracy) const {
        bulk.resize(length);
        for (int i = 0; i < length; i++) {
            bulk[i] = deriv_at<DerivType>(i, Order, Accuracy);
        }
    }

    /*!
     * @brief Calculate all derivatives at once, returned as a new vector.
     *
     * @tparam DerivType Output Eigen::Matrix type.
     * @param Order Order of the derivative to compute.
     * @param Accuracy Finite-difference accuracy order (rounded up to the
     *                 nearest even number).
     * @return One derivative per sample index.
     */
    template <class DerivType>
    inline std::vector<DerivType> deriv(const int Order, const int Accuracy) const {
        std::vector<DerivType> bulk;
        deriv<DerivType>(bulk, Order, Accuracy);
        return bulk;
    }

    /// @brief Python-binding shim: returns all derivatives at once.
    /// @internal
    /// @tparam DerivType  Output Eigen type.
    /// @param Order     Derivative order to compute.
    /// @param Accuracy  Finite-difference accuracy order.
    /// @return One derivative per sample index.
    /// @endinternal
    template <class DerivType>
    inline std::vector<DerivType> allderiv_python(const int Order, const int Accuracy) const {
        return deriv<DerivType>(Order, Accuracy);
    }
    /// @brief Python-binding shim: returns the derivative at a single sample index.
    /// @internal
    /// @tparam DerivType  Output Eigen type.
    /// @param i         Sample index at which to evaluate the derivative.
    /// @param Order     Derivative order to compute.
    /// @param Accuracy  Finite-difference accuracy order.
    /// @return The derivative at index @p i.
    /// @endinternal
    template <class DerivType>
    inline DerivType ithderiv_python(const int i, const int Order, const int Accuracy) const {
        return deriv_at<DerivType>(i, Order, Accuracy);
    }
};

} // namespace tycho::vf
