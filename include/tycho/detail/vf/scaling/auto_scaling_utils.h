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

/// @file
/// @brief Utilities for automatically choosing output scale factors.
#pragma once
#include "tycho/detail/vf/scaling/io_scaled.h"
#include "tycho/vector_functions.h"
#include <algorithm>
#include <array>
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

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::vf {

/// @ingroup vf
/// @brief Computes output scale factors that normalize each Jacobian row.
///
/// For each test input the function's input-scaled Jacobian is evaluated and the
/// L2 norm of every row recorded. The returned output scale for each row is the
/// reciprocal of the mean row norm across the test inputs (rows whose mean norm
/// is non-finite or near zero default to a scale of 1). Applying these scales
/// (e.g. via @ref IOScaled) drives each scaled Jacobian row to roughly unit norm,
/// improving conditioning for the optimizer.
/// @tparam Func  VectorFunction type to analyze.
/// @param func          The function whose Jacobian rows are normalized.
/// @param input_scales  Per-component input scale factors applied during evaluation.
/// @param test_inputs   Sample inputs at which the Jacobian is evaluated.
/// @return Per-row output scale factors (length `func.output_rows()`).
template <class Func>
Eigen::VectorXd calc_jacobian_row_scales(const Func &func, const Eigen::VectorXd &input_scales,
                                         std::vector<Eigen::VectorXd> &test_inputs) {

    Eigen::MatrixXd rownorms(func.output_rows(), test_inputs.size());
    Eigen::VectorXd output_scales(func.output_rows());
    output_scales.setOnes();
    IOScaled<Func> scaled_func(func, input_scales, output_scales);

    Eigen::VectorXd fx(func.output_rows());
    Eigen::MatrixXd jx(func.output_rows(), func.input_rows());

    for (int i = 0; i < test_inputs.size(); i++) {
        fx.setZero();
        jx.setZero();
        scaled_func.compute_jacobian(test_inputs[i], fx, jx);
        for (int j = 0; j < func.output_rows(); j++) {
            rownorms(j, i) = jx.row(j).norm();
        }
    }

    for (int j = 0; j < func.output_rows(); j++) {
        double avg = 1.0;
        if (rownorms.row(j).mean() < 1.0e-12) {

        } else if (!std::isfinite(rownorms.row(j).mean())) {

        } else {
            avg = rownorms.row(j).mean();
        }

        output_scales[j] = 1.0 / avg;
    }

    return output_scales;
}

} // namespace tycho::vf
