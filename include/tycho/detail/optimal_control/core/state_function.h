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
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
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

namespace tycho::oc {

template <class FuncType> struct StateFunction {
    FuncType Func;
    PhaseRegionFlags RegionFlag = PhaseRegionFlags::NotSet;
    Eigen::VectorXi XtUVars;
    Eigen::VectorXi OPVars;
    Eigen::VectorXi SPVars;

    Eigen::VectorXi EXTVars; // dirty i know

    ScaleModes ScaleMode = ScaleModes::AUTO;
    bool ScalesSet = false;
    Eigen::VectorXd OutputScales;

    int StorageIndex = 0;
    int PhaseLocalIndex = 0;
    int GlobalIndex = 0;

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv,
                  Eigen::VectorXi spv) {
        this->RegionFlag = Reg;
        this->Func = f;
        this->XtUVars = xtuv;
        this->OPVars = opv;
        this->SPVars = spv;
        this->OutputScales = Eigen::VectorXd::Ones(this->Func.output_rows());
    }

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv,
                  Eigen::VectorXi spv, ScaleType scale_t)
        : StateFunction(f, Reg, xtuv, opv, spv) {

        auto [ScaleMode, ScalesSet, OutputScales] = get_scale_info(this->Func.output_rows(), scale_t);
        this->OutputScales = OutputScales;
        this->ScaleMode = ScaleMode;
        this->ScalesSet = ScalesSet;
    }

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv) {
        this->Func = f;
        this->OutputScales = Eigen::VectorXd::Ones(this->Func.output_rows());

        switch (Reg) {
        case PhaseRegionFlags::ODEParams: {
            this->RegionFlag = PhaseRegionFlags::Params;
            this->XtUVars.resize(0);
            this->OPVars = xtuv;
            this->SPVars.resize(0);
            break;
        }
        case PhaseRegionFlags::StaticParams: {
            this->RegionFlag = PhaseRegionFlags::Params;
            this->XtUVars.resize(0);
            this->OPVars.resize(0);
            this->SPVars = xtuv;
            break;
        }
        default: {
            this->RegionFlag = Reg;
            this->XtUVars = xtuv;
            this->OPVars.resize(0);
            this->SPVars.resize(0);
            break;
        }
        }
    }
    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, ScaleType scale_t)
        : StateFunction(f, Reg, xtuv) {
        auto [ScaleMode, ScalesSet, OutputScales] = get_scale_info(this->Func.output_rows(), scale_t);
        this->OutputScales = OutputScales;
        this->ScaleMode = ScaleMode;
        this->ScalesSet = ScalesSet;
    }

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, PhaseRegionFlags ParReg,
                  Eigen::VectorXi pv) {
        this->Func = f;
        this->RegionFlag = Reg;
        this->XtUVars = xtuv;
        this->OutputScales = Eigen::VectorXd::Ones(this->Func.output_rows());

        switch (ParReg) {
        case PhaseRegionFlags::ODEParams: {
            this->OPVars = pv;
            this->SPVars.resize(0);
            break;
        }
        case PhaseRegionFlags::StaticParams: {
            this->RegionFlag = PhaseRegionFlags::Params;
            this->OPVars.resize(0);
            this->SPVars = pv;
            break;
        }
        default: {
            throw std::invalid_argument(
                "Param region flag must be either StaticParams or ODEParams");
        }
        }
    }
    StateFunction() {}
};

} // namespace tycho::oc
