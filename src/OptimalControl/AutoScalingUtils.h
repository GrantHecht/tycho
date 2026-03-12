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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once
#include "CommonFunctions/IOScaled.h"
#include "VectorFunctions/Tycho_VectorFunctions.h"
#include "pch.h"

namespace Tycho {

template <class Func>
Eigen::VectorXd calc_jacobian_row_scales(const Func &func, const Eigen::VectorXd &input_scales,
                                         std::vector<Eigen::VectorXd> &test_inputs) {

    Eigen::MatrixXd rownorms(func.ORows(), test_inputs.size());
    Eigen::VectorXd output_scales(func.ORows());
    output_scales.setOnes();
    IOScaled<Func> scaled_func(func, input_scales, output_scales);

    Eigen::VectorXd fx(func.ORows());
    Eigen::MatrixXd jx(func.ORows(), func.IRows());

    for (int i = 0; i < test_inputs.size(); i++) {
        fx.setZero();
        jx.setZero();
        scaled_func.compute_jacobian(test_inputs[i], fx, jx);
        for (int j = 0; j < func.ORows(); j++) {
            rownorms(j, i) = jx.row(j).norm();
        }
    }

    for (int j = 0; j < func.ORows(); j++) {
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

} // namespace Tycho
