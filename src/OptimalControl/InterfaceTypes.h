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
#include <string>
#include <variant>
#include <vector>

#include "OptimalControlFlags.h"
#include "pch.h"

namespace Tycho {

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

} // namespace Tycho
