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

template <class FuncType> struct LinkFunction {
    FuncType func_;
    LinkFlags link_flag_ = LinkFlags::ReadRegions;

    Eigen::Matrix<PhaseRegionFlags, -1, 1> phase_reg_flags_;
    std::vector<Eigen::VectorXi> phases_to_link_;
    std::vector<Eigen::VectorXi> xtu_vars_;
    std::vector<Eigen::VectorXi> op_vars_;
    std::vector<Eigen::VectorXi> sp_vars_;
    std::vector<Eigen::VectorXi> link_params_;

    ScaleModes scale_mode_ = ScaleModes::AUTO;
    bool scales_set_ = false;
    Eigen::VectorXd output_scales_;

    int storage_index_ = 0;
    int global_index_ = 0;

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

    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv, std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv) {
        this->init(f, RegFlags, PTL, xtv, opv, spv, lv);
    }

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

    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv) {
        this->init(f, make_phase_reg_flags(Flag), PTL, xtv, opv, spv, lv);
    }
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
    LinkFunction(FuncType f, std::vector<std::string> RegFlags, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv) {
        this->init(f, strto_PhaseRegionFlag(RegFlags), PTL, xtv, opv, spv, lv);
    }
    LinkFunction(FuncType f, std::string Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv, std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv, std::vector<Eigen::VectorXi> lv) {
        this->init(f, make_phase_reg_flags(strto_LinkFlag(Flag)), PTL, xtv, opv, spv, lv);
    }

    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, std::vector<Eigen::VectorXi> xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, RegFlags, PTL, xtv, emptyvec, emptyvec, empty_vec_lv);
    }
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, make_phase_reg_flags(Flag), PTL, xtv, emptyvec, emptyvec, empty_vec_lv);
    }
    LinkFunction(FuncType f, Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL, Eigen::VectorXi xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, RegFlags, PTL, xtvvec, emptyvec, emptyvec, empty_vec_lv);
    }
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL,
                 Eigen::VectorXi xtv) {
        Eigen::VectorXi empty;
        empty.resize(0);
        std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
        std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
        std::vector<Eigen::VectorXi> empty_vec_lv(PTL.size(), empty);

        this->init(f, make_phase_reg_flags(Flag), PTL, xtvvec, emptyvec, emptyvec, empty_vec_lv);
    }

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

    LinkFunction() {}
};

} // namespace tycho::oc
