#pragma once
#include <string>
#include <variant>
#include <vector>

#include "OptimalControlFlags.h"
#include "pch.h"

namespace Tycho {

using VarIndexType = std::variant<int, Eigen::VectorXi, std::string, std::vector<std::string>>;
#ifdef TYCHO_PYTHON_BINDINGS
using ScaleType = std::variant<double, Eigen::VectorXd, std::string, py::none>;
#else
using ScaleType = std::variant<double, Eigen::VectorXd, std::string>;
#endif
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

static std::tuple<std::string, bool, Eigen::VectorXd> get_scale_info(int orows, ScaleType scale_t) {

    Eigen::VectorXd OutputScales(orows);
    OutputScales.setOnes();
    std::string ScaleMode = "auto";
    bool ScalesSet = false;
    if (std::holds_alternative<double>(scale_t)) {
        OutputScales *= std::get<double>(scale_t);
        ScaleMode = "custom";
        ScalesSet = true;

    } else if (std::holds_alternative<Eigen::VectorXd>(scale_t)) {
        OutputScales = std::get<Eigen::VectorXd>(scale_t);
        ScaleMode = "custom";
        ScalesSet = true;

        if (OutputScales.size() != orows) {
            throw std::invalid_argument(
                "Scaling vector size does not match output size of function");
        }
    } else if (std::holds_alternative<std::string>(scale_t)) {
        ScaleMode = std::get<std::string>(scale_t);

        if (ScaleMode == "auto") {
            ScalesSet = false;
        } else if (ScaleMode == "none") {
            ScalesSet = true;
        } else {
            throw std::invalid_argument(fmt::format("Unrecognized Scale Mode:{0:}", ScaleMode));
        }
#ifdef TYCHO_PYTHON_BINDINGS
    } else if (std::holds_alternative<py::none>(scale_t)) {
        ScaleMode = "none";
        ScalesSet = true;
#endif
    }

    return std::tuple{ScaleMode, ScalesSet, OutputScales};
}

} // namespace Tycho