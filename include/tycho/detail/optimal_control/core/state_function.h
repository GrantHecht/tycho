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
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/type_name.h"

namespace tycho::oc {

template <class FuncType> struct StateFunction {
    FuncType func_;
    PhaseRegionFlags region_flag_ = PhaseRegionFlags::NotSet;
    Eigen::VectorXi xtu_vars_;
    Eigen::VectorXi op_vars_;
    Eigen::VectorXi sp_vars_;

    Eigen::VectorXi ext_vars_; // dirty i know

    ScaleModes scale_mode_ = ScaleModes::AUTO;
    bool scales_set_ = false;
    Eigen::VectorXd output_scales_;

    int storage_index_ = 0;
    int phase_local_index_ = 0;
    int global_index_ = 0;

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv,
                  Eigen::VectorXi spv) {
        this->region_flag_ = Reg;
        this->func_ = f;
        this->xtu_vars_ = xtuv;
        this->op_vars_ = opv;
        this->sp_vars_ = spv;
        this->output_scales_ = Eigen::VectorXd::Ones(this->func_.output_rows());
    }

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv,
                  Eigen::VectorXi spv, ScaleType scale_t)
        : StateFunction(f, Reg, xtuv, opv, spv) {

        auto [scale_mode, scales_set, output_scales] =
            get_scale_info(this->func_.output_rows(), scale_t);
        this->output_scales_ = output_scales;
        this->scale_mode_ = scale_mode;
        this->scales_set_ = scales_set;
    }

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv) {
        this->func_ = f;
        this->output_scales_ = Eigen::VectorXd::Ones(this->func_.output_rows());

        switch (Reg) {
        case PhaseRegionFlags::ODEParams: {
            this->region_flag_ = PhaseRegionFlags::Params;
            this->xtu_vars_.resize(0);
            this->op_vars_ = xtuv;
            this->sp_vars_.resize(0);
            break;
        }
        case PhaseRegionFlags::StaticParams: {
            this->region_flag_ = PhaseRegionFlags::Params;
            this->xtu_vars_.resize(0);
            this->op_vars_.resize(0);
            this->sp_vars_ = xtuv;
            break;
        }
        default: {
            this->region_flag_ = Reg;
            this->xtu_vars_ = xtuv;
            this->op_vars_.resize(0);
            this->sp_vars_.resize(0);
            break;
        }
        }
    }
    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, ScaleType scale_t)
        : StateFunction(f, Reg, xtuv) {
        auto [scale_mode, scales_set, output_scales] =
            get_scale_info(this->func_.output_rows(), scale_t);
        this->output_scales_ = output_scales;
        this->scale_mode_ = scale_mode;
        this->scales_set_ = scales_set;
    }

    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, PhaseRegionFlags ParReg,
                  Eigen::VectorXi pv) {
        this->func_ = f;
        this->region_flag_ = Reg;
        this->xtu_vars_ = xtuv;
        this->output_scales_ = Eigen::VectorXd::Ones(this->func_.output_rows());

        switch (ParReg) {
        case PhaseRegionFlags::ODEParams: {
            this->op_vars_ = pv;
            this->sp_vars_.resize(0);
            break;
        }
        case PhaseRegionFlags::StaticParams: {
            this->region_flag_ = PhaseRegionFlags::Params;
            this->op_vars_.resize(0);
            this->sp_vars_ = pv;
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
