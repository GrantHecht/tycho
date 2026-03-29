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
#include <string>
#include <variant>
#include <vector>

#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <fmt/format.h>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho::oc {

using VarIndexType = std::variant<int, Eigen::VectorXi, std::string, std::vector<std::string>>;
using ScaleType = std::variant<double, Eigen::VectorXd, ScaleModes, std::string>;
using RegionType = std::variant<PhaseRegionFlags, std::string>;

static PhaseRegionFlags get_PhaseRegion(RegionType reg_t) {
    PhaseRegionFlags reg;

    if (std::holds_alternative<PhaseRegionFlags>(reg_t)) {
        reg = std::get<PhaseRegionFlags>(reg_t);
    } else if (std::holds_alternative<std::string>(reg_t)) {
        reg = strto_PhaseRegionFlag(std::get<std::string>(reg_t));
    }
    return reg;
}

static std::tuple<ScaleModes, bool, Eigen::VectorXd> get_scale_info(int orows, ScaleType scale_t) {

    Eigen::VectorXd OutputScales(orows);
    OutputScales.setOnes();
    ScaleModes ScaleMode = ScaleModes::AUTO;
    bool ScalesSet = false;
    if (std::holds_alternative<double>(scale_t)) {
        OutputScales *= std::get<double>(scale_t);
        ScaleMode = ScaleModes::CUSTOM;
        ScalesSet = true;

    } else if (std::holds_alternative<Eigen::VectorXd>(scale_t)) {
        OutputScales = std::get<Eigen::VectorXd>(scale_t);
        ScaleMode = ScaleModes::CUSTOM;
        ScalesSet = true;

        if (OutputScales.size() != orows) {
            throw std::invalid_argument(
                "Scaling vector size does not match output size of function");
        }
    } else if (std::holds_alternative<ScaleModes>(scale_t)) {
        ScaleMode = std::get<ScaleModes>(scale_t);
        if (ScaleMode == ScaleModes::CUSTOM) {
            throw std::invalid_argument("ScaleModes::CUSTOM requires explicit scale values (pass a "
                                        "double or VectorXd instead)");
        }
        ScalesSet = (ScaleMode != ScaleModes::AUTO);
    } else if (std::holds_alternative<std::string>(scale_t)) {
        ScaleMode = strto_ScaleMode(std::get<std::string>(scale_t));
        if (ScaleMode == ScaleModes::CUSTOM) {
            throw std::invalid_argument("ScaleModes::CUSTOM requires explicit scale values (pass a "
                                        "double or VectorXd instead)");
        }
        ScalesSet = (ScaleMode != ScaleModes::AUTO);
    }

    return std::tuple{ScaleMode, ScalesSet, OutputScales};
}

} // namespace tycho::oc
