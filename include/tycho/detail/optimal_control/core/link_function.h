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
/// @brief A VectorFunction that couples states across phases in a multi-phase problem.
///
/// Bundles a user function with the set of phases it links, the region of each
/// linked phase it reads, the per-phase variable bindings, and any shared
/// link-parameter indices. This is the internal record an
/// @ref OptimalControlProblem keeps for each link constraint or objective.
/// @tparam FuncType  The wrapped VectorFunction type.
template <class FuncType> struct LinkFunction {
    FuncType func_;                                ///< The wrapped VectorFunction.
    LinkFlags link_flag_ = LinkFlags::ReadRegions; ///< The link pairing this function uses.

    Eigen::Matrix<PhaseRegionFlags, -1, 1> phase_reg_flags_; ///< Region of each linked phase.
    std::vector<Eigen::VectorXi> phases_to_link_; ///< Phase indices for each link application.
    std::vector<Eigen::VectorXi> xtu_vars_;    ///< Per-phase state/time/control variable indices.
    std::vector<Eigen::VectorXi> op_vars_;     ///< Per-phase ODE-parameter variable indices.
    std::vector<Eigen::VectorXi> sp_vars_;     ///< Per-phase static-parameter variable indices.
    std::vector<Eigen::VectorXi> link_params_; ///< Shared link-parameter variable indices.

    ScaleModes scale_mode_ = ScaleModes::AUTO; ///< How the output scales were determined.
    bool scales_set_ = false;                  ///< Whether explicit output scales were set.
    Eigen::VectorXd output_scales_;            ///< Per-output scale factors.

    int storage_index_ = 0; ///< Index of this link within the problem's storage.
    int global_index_ = 0;  ///< Index of this link within the whole problem.

    /// @internal
    /// @brief Populate all members, broadcasting empty variable groups as needed.
    /// @param f        The wrapped VectorFunction.
    /// @param RegFlags Region of each linked phase.
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      Per-phase state/time/control variable indices.
    /// @param opv      Per-phase ODE-parameter variable indices.
    /// @param spv      Per-phase static-parameter variable indices.
    /// @param lv       Shared link-parameter variable indices.
    /// @throws std::invalid_argument if both @p PTL and @p lv are empty, or if a
    ///         non-empty @p PTL and @p RegFlags have mismatched sizes.
    /// @endinternal
    void init(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
              std::vector<Eigen::VectorXi> PTL, std::vector<Eigen::VectorXi> xtv,
              std::vector<Eigen::VectorXi> opv, std::vector<Eigen::VectorXi> spv,
              std::vector<Eigen::VectorXi> lv) {

        Eigen::VectorXi empty;
        empty.resize(0);

        int nappl = std::max(PTL.size(), lv.size());
        int nxs = xtv.size();
        int nos = opv.size();
        int nss = spv.size();

        int nmax = std::max({nxs, nos, nss});

        if (nappl == 0) {
            throw std::invalid_argument("PTL vector and link param vector cannot both have 0 size");
        }

        if (PTL.size() == 0) {
            if (RegFlags.size() != 0) {
                throw std::invalid_argument(
                    "PTL vector element and RegFlags vector must be same size");
            }
            for (int i = 0; i < nappl; i++) {
                PTL.push_back(empty);
            }
        }
        if (lv.size() == 0) {
            for (int i = 0; i < nappl; i++) {
                lv.push_back(empty);
            }
        }

        if (nmax > 0 && PTL.size() > 0) {
            if (nxs == 0) {
                for (int i = 0; i < nmax; i++) {
                    xtv.push_back(empty);
                }
            }

            if (nos == 0) {
                for (int i = 0; i < nmax; i++) {
                    opv.push_back(empty);
                }
            }

            if (nss == 0) {
                for (int i = 0; i < nmax; i++) {
                    spv.push_back(empty);
                }
            }
        }

        this->phases_to_link_ = PTL;
        this->phase_reg_flags_ = RegFlags;
        this->func_ = f;

        this->xtu_vars_ = xtv;
        this->op_vars_ = opv;
        this->sp_vars_ = spv;
        this->link_params_ = lv;
        this->output_scales_ = Eigen::VectorXd::Ones(this->func_.output_rows());
    }

    /// @internal
    /// @brief Translate a @ref LinkFlags pairing into the per-phase region vector.
    /// @param Flag  The link pairing to expand.
    /// @return Two-element (or empty, for @c LinkParams) region vector.
    /// @endinternal
    Eigen::Matrix<PhaseRegionFlags, -1, 1> make_phase_reg_flags(LinkFlags Flag) {
        Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags;
        this->link_flag_ = Flag;

        switch (Flag) {
        case LinkFlags::BackToFront: {
            RegFlags.resize(2);
            RegFlags << PhaseRegionFlags::Back, PhaseRegionFlags::Front;
            break;
        }
        case LinkFlags::BackToBack: {
            RegFlags.resize(2);
            RegFlags << PhaseRegionFlags::Back, PhaseRegionFlags::Back;

            break;
        }
        case LinkFlags::FrontToBack: {
            RegFlags.resize(2);
            RegFlags << PhaseRegionFlags::Front, PhaseRegionFlags::Back;

            break;
        }
        case LinkFlags::FrontToFront: {
            RegFlags.resize(2);
            RegFlags << PhaseRegionFlags::Front, PhaseRegionFlags::Front;
            break;
        }
        case LinkFlags::ParamsToParams: {
            RegFlags.resize(2);
            RegFlags << PhaseRegionFlags::Params, PhaseRegionFlags::Params;
            break;
        }
        case LinkFlags::PathToPath: {
            RegFlags.resize(2);
            RegFlags << PhaseRegionFlags::Path, PhaseRegionFlags::Path;
            break;
        }
        case LinkFlags::LinkParams: {
            RegFlags.resize(0);
            break;
        }
        default:
            break;
        }
        return RegFlags;
    }

    /// @brief Construct from an explicit per-phase region vector and full bindings.
    /// @param f        The wrapped VectorFunction.
    /// @param RegFlags Region of each linked phase.
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      Per-phase state/time/control variable indices.
    /// @param opv      Per-phase ODE-parameter variable indices.
    /// @param spv      Per-phase static-parameter variable indices.
    /// @param lv       Shared link-parameter variable indices.
    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv, std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv) {
        this->init(f, RegFlags, PTL, xtv, opv, spv, lv);
    }

    /// @brief Construct from an explicit region vector, full bindings, and a scale spec.
    /// @param f        The wrapped VectorFunction.
    /// @param RegFlags Region of each linked phase.
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      Per-phase state/time/control variable indices.
    /// @param opv      Per-phase ODE-parameter variable indices.
    /// @param spv      Per-phase static-parameter variable indices.
    /// @param lv       Shared link-parameter variable indices.
    /// @param scale_t  Output-scale specifier (see @ref ScaleType).
    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv, std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv, ScaleType scale_t) {
        this->init(f, RegFlags, PTL, xtv, opv, spv, lv);

        auto [scale_mode, scales_set, output_scales] =
            get_scale_info(this->func_.output_rows(), scale_t);
        this->output_scales_ = output_scales;
        this->scale_mode_ = scale_mode;
        this->scales_set_ = scales_set;
    }

    /// @brief Construct from a @ref LinkFlags pairing and full bindings.
    /// @param f     The wrapped VectorFunction.
    /// @param Flag  Link pairing (regions are derived from it).
    /// @param PTL   Phases-to-link index groups.
    /// @param xtv   Per-phase state/time/control variable indices.
    /// @param opv   Per-phase ODE-parameter variable indices.
    /// @param spv   Per-phase static-parameter variable indices.
    /// @param lv    Shared link-parameter variable indices.
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv) {
        this->init(f, make_phase_reg_flags(Flag), PTL, xtv, opv, spv, lv);
    }
    /// @brief Construct from a @ref LinkFlags pairing, full bindings, and a scale spec.
    /// @param f        The wrapped VectorFunction.
    /// @param Flag     Link pairing (regions are derived from it).
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      Per-phase state/time/control variable indices.
    /// @param opv      Per-phase ODE-parameter variable indices.
    /// @param spv      Per-phase static-parameter variable indices.
    /// @param lv       Shared link-parameter variable indices.
    /// @param scale_t  Output-scale specifier (see @ref ScaleType).
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv,
                 ScaleType scale_t) {
        this->init(f, make_phase_reg_flags(Flag), PTL, xtv, opv, spv, lv);
        auto [scale_mode, scales_set, output_scales] =
            get_scale_info(this->func_.output_rows(), scale_t);
        this->output_scales_ = output_scales;
        this->scale_mode_ = scale_mode;
        this->scales_set_ = scales_set;
    }
    /// @brief Construct from per-phase region names and full bindings.
    /// @param f        The wrapped VectorFunction.
    /// @param RegFlags Region name of each linked phase.
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      Per-phase state/time/control variable indices.
    /// @param opv      Per-phase ODE-parameter variable indices.
    /// @param spv      Per-phase static-parameter variable indices.
    /// @param lv       Shared link-parameter variable indices.
    LinkFunction(FuncType f, std::vector<std::string> RegFlags, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv) {
        this->init(f, strto_PhaseRegionFlag(RegFlags), PTL, xtv, opv, spv, lv);
    }
    /// @brief Construct from a link-flag name and full bindings.
    /// @param f     The wrapped VectorFunction.
    /// @param Flag  Link-pairing name (regions are derived from it).
    /// @param PTL   Phases-to-link index groups.
    /// @param xtv   Per-phase state/time/control variable indices.
    /// @param opv   Per-phase ODE-parameter variable indices.
    /// @param spv   Per-phase static-parameter variable indices.
    /// @param lv    Shared link-parameter variable indices.
    LinkFunction(FuncType f, std::string Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv) {
        this->init(f, make_phase_reg_flags(strto_LinkFlag(Flag)), PTL, xtv, opv, spv, lv);
    }

    /// @brief Construct from an explicit region vector and state bindings only.
    /// @param f        The wrapped VectorFunction.
    /// @param RegFlags Region of each linked phase.
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      Per-phase state/time/control variable indices (params left empty).
    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, std::vector<Eigen::VectorXi> xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, RegFlags, PTL, xtv, emptyvec, emptyvec, empty_vec_lv);
    }
    /// @brief Construct from a @ref LinkFlags pairing and state bindings only.
    /// @param f     The wrapped VectorFunction.
    /// @param Flag  Link pairing (regions are derived from it).
    /// @param PTL   Phases-to-link index groups.
    /// @param xtv   Per-phase state/time/control variable indices (params left empty).
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, make_phase_reg_flags(Flag), PTL, xtv, emptyvec, emptyvec, empty_vec_lv);
    }
    /// @brief Construct from a region vector with one shared state-binding for all phases.
    /// @param f        The wrapped VectorFunction.
    /// @param RegFlags Region of each linked phase.
    /// @param PTL      Phases-to-link index groups.
    /// @param xtv      State/time/control indices applied to every linked phase.
    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, Eigen::VectorXi xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, RegFlags, PTL, xtvvec, emptyvec, emptyvec, empty_vec_lv);
    }
    /// @brief Construct from a @ref LinkFlags pairing with one shared state-binding.
    /// @param f     The wrapped VectorFunction.
    /// @param Flag  Link pairing (regions are derived from it).
    /// @param PTL   Phases-to-link index groups.
    /// @param xtv   State/time/control indices applied to every linked phase.
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 Eigen::VectorXi xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, make_phase_reg_flags(Flag), PTL, xtvvec, emptyvec, emptyvec, empty_vec_lv);
    }

    /// @brief Construct from a link-flag name with one shared state-binding.
    /// @param f     The wrapped VectorFunction.
    /// @param Flag  Link-pairing name (regions are derived from it).
    /// @param PTL   Phases-to-link index groups.
    /// @param xtv   State/time/control indices applied to every linked phase.
    LinkFunction(FuncType f, std::string Flag, std::vector<Eigen::VectorXi> PTL,
                 Eigen::VectorXi xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, make_phase_reg_flags(strto_LinkFlag(Flag)), PTL, xtvvec, emptyvec, emptyvec,
                   empty_vec_lv);
    }

    /// @brief Default constructor; leaves all bindings empty.
    LinkFunction() {}
};

} // namespace tycho::oc
