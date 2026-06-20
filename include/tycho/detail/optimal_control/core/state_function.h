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

/// @ingroup optimal_control
/// @brief A VectorFunction attached to a phase region with its variable bindings.
///
/// Bundles a user function with the phase region it acts on and the specific
/// state/time/control (@c xtu), ODE-parameter (@c op), and static-parameter
/// (@c sp) variable indices it reads, plus its output scaling. This is the
/// internal record a phase keeps for each constraint or objective the user adds.
/// @tparam FuncType  The wrapped VectorFunction type.
template <class FuncType> struct StateFunction {
    FuncType func_; ///< The wrapped VectorFunction.
    PhaseRegionFlags region_flag_ =
        PhaseRegionFlags::NotSet; ///< Phase region the function acts on.
    Eigen::VectorXi xtu_vars_;    ///< State/time/control variable indices the function reads.
    Eigen::VectorXi op_vars_;     ///< ODE-parameter variable indices the function reads.
    Eigen::VectorXi sp_vars_;     ///< Static-parameter variable indices the function reads.

    Eigen::VectorXi ext_vars_; ///< Extra (externally-supplied) variable indices.

    ScaleModes scale_mode_ = ScaleModes::AUTO; ///< How the output scales were determined.
    bool scales_set_ = false;                  ///< Whether explicit output scales were set.
    Eigen::VectorXd output_scales_;            ///< Per-output scale factors.

    int storage_index_ = 0;     ///< Index of this function within the phase's storage.
    int phase_local_index_ = 0; ///< Index of this function within its phase.
    int global_index_ = 0;      ///< Index of this function within the whole problem.

    /// @brief Construct with explicit state, ODE-param, and static-param bindings.
    /// @param f     The wrapped VectorFunction.
    /// @param Reg   Phase region the function acts on.
    /// @param xtuv  State/time/control variable indices.
    /// @param opv   ODE-parameter variable indices.
    /// @param spv   Static-parameter variable indices.
    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv,
                  Eigen::VectorXi spv) {
        this->region_flag_ = Reg;
        this->func_ = f;
        this->xtu_vars_ = xtuv;
        this->op_vars_ = opv;
        this->sp_vars_ = spv;
        this->output_scales_ = Eigen::VectorXd::Ones(this->func_.output_rows());
    }

    /// @brief Construct with explicit bindings and an output-scale specification.
    /// @param f        The wrapped VectorFunction.
    /// @param Reg      Phase region the function acts on.
    /// @param xtuv     State/time/control variable indices.
    /// @param opv      ODE-parameter variable indices.
    /// @param spv      Static-parameter variable indices.
    /// @param scale_t  Output-scale specifier (see @ref ScaleType).
    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv,
                  Eigen::VectorXi spv, ScaleType scale_t)
        : StateFunction(f, Reg, xtuv, opv, spv) {

        auto [scale_mode, scales_set, output_scales] =
            get_scale_info(this->func_.output_rows(), scale_t);
        this->output_scales_ = output_scales;
        this->scale_mode_ = scale_mode;
        this->scales_set_ = scales_set;
    }

    /// @brief Construct from a single variable group, dispatching on the region.
    ///
    /// When @p Reg is @c ODEParams or @c StaticParams, @p xtuv is interpreted as
    /// the corresponding parameter indices; otherwise it is the state/time/control
    /// indices.
    /// @param f     The wrapped VectorFunction.
    /// @param Reg   Phase region (or parameter region) the function acts on.
    /// @param xtuv  Variable indices, interpreted per @p Reg.
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
    /// @brief Construct from a single variable group plus an output-scale specification.
    /// @param f        The wrapped VectorFunction.
    /// @param Reg      Phase region (or parameter region) the function acts on.
    /// @param xtuv     Variable indices, interpreted per @p Reg.
    /// @param scale_t  Output-scale specifier (see @ref ScaleType).
    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, ScaleType scale_t)
        : StateFunction(f, Reg, xtuv) {
        auto [scale_mode, scales_set, output_scales] =
            get_scale_info(this->func_.output_rows(), scale_t);
        this->output_scales_ = output_scales;
        this->scale_mode_ = scale_mode;
        this->scales_set_ = scales_set;
    }

    /// @brief Construct binding both a state region and a parameter region.
    /// @param f       The wrapped VectorFunction.
    /// @param Reg     Phase region for the state/time/control variables.
    /// @param xtuv    State/time/control variable indices.
    /// @param ParReg  Parameter region — must be @c ODEParams or @c StaticParams.
    /// @param pv      Parameter variable indices.
    /// @throws std::invalid_argument if @p ParReg is neither @c ODEParams nor @c StaticParams.
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
    /// @brief Default constructor; leaves all bindings empty.
    StateFunction() {}
};

} // namespace tycho::oc
