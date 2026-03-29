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
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/sizing_helpers.h"
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

// Import commonly-used types from vf and utils namespaces.
using utils::FlatMap;
using utils::SZ_SUM;

template <int _XV, int _UV, int _PV> struct ODEConstSizes {
    static const int XV = _XV;
    static const int UV = _UV;
    static const int PV = _PV;
    static const int XtV = SZ_SUM<_XV, 1>::value;
    static const int XtUV = SZ_SUM<_XV, 1, _UV>::value;
    static const int XtUPV = SZ_SUM<_XV, 1, _UV, _PV>::value;
};

template <int _XV, int _UV, int _PV> struct ODEXVSizes : ODEConstSizes<_XV, _UV, _PV> {
    inline int t_var() const { return this->XV; }
    inline int x_vars() const { return this->XV; }
    inline int xt_vars() const { return this->XtV; }
    void set_xvars(int xv) {}
};

template <int _UV, int _PV> struct ODEXVSizes<-1, _UV, _PV> : ODEConstSizes<-1, _UV, _PV> {
    inline int t_var() const { return this->XVdynamic; }
    inline int x_vars() const { return this->XVdynamic; }
    inline int xt_vars() const { return this->XtVdynamic; }
    void set_xvars(int xv) {
        this->XVdynamic = xv;
        this->XtVdynamic = xv + 1;
    }

  protected:
    int XVdynamic = 0;
    int XtVdynamic = 0;
};

template <int _XV, int _UV, int _PV> struct ODEXUVSizes : ODEXVSizes<_XV, _UV, _PV> {
    inline int u_vars() const { return this->UV; }
    inline int xtu_vars() const { return this->UV + this->xt_vars(); }
    void set_uvars(int uv) {}
};
template <int _XV, int _PV> struct ODEXUVSizes<_XV, -1, _PV> : ODEXVSizes<_XV, -1, _PV> {
    inline int u_vars() const { return this->UVdynamic; }
    inline int xtu_vars() const { return this->UVdynamic + this->xt_vars(); }
    void set_uvars(int uv) { this->UVdynamic = uv; }

  protected:
    int UVdynamic = 0;
};

template <int _XV, int _UV, int _PV> struct ODEXUPVSizes : ODEXUVSizes<_XV, _UV, _PV> {
    inline int p_vars() const { return this->PV; }
    inline int xtu_p_vars() const { return this->PV + this->xtu_vars(); }
    void set_pvars(int pv) {}
    void set_xt_up_vars(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
    }
};

template <int _XV, int _UV> struct ODEXUPVSizes<_XV, _UV, -1> : ODEXUVSizes<_XV, _UV, -1> {
    inline int p_vars() const { return this->PVdynamic; }
    inline int xtu_p_vars() const { return this->PVdynamic + this->xtu_vars(); }
    void set_pvars(int pv) { this->PVdynamic = pv; }
    void set_xt_up_vars(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
    }

  protected:
    int PVdynamic = 0;
};

template <int _XV, int _UV, int _PV> struct ODESize : ODEXUPVSizes<_XV, _UV, _PV> {

    FlatMap<std::string, Eigen::VectorXi> xtu_p_idxs_;

    void add_idx(const std::string &name, const Eigen::VectorXi &idx) {
        if (idx.size() == 0) {
            throw std::invalid_argument(
                fmt::format("Variable index group with name: {0:} has no elements.", name));
        }
        if (xtu_p_idxs_.contains(name)) {
            throw std::invalid_argument(
                fmt::format("Variable index group with name: {0:} already exists.", name));
        }
        xtu_p_idxs_.insert(name, idx);
    }

    void add_idx(const std::string &name, int indx) {
        Eigen::VectorXi idxv(1);
        idxv[0] = indx;
        this->add_idx(name, idxv);
    }

    Eigen::VectorXi idx(const std::string &name) const {
        if (!xtu_p_idxs_.contains(name)) {
            throw std::invalid_argument(
                fmt::format("No variable index group with name: {0:} exists.", name));
        }
        return xtu_p_idxs_.at(name);
    }

    void set_idxs(const FlatMap<std::string, Eigen::VectorXi> &idxs) {
        for (const auto &[name, idx] : idxs) {
            if (idx.size() == 0) {
                throw std::invalid_argument(
                    fmt::format("Variable index group with name: {0:} has no elements.", name));
            }
        }
        this->xtu_p_idxs_ = idxs;
    }
    FlatMap<std::string, Eigen::VectorXi> get_idxs() const { return this->xtu_p_idxs_; }

    Eigen::VectorXi x_idxs() const {
        Eigen::VectorXi idxs(this->x_vars());
        std::iota(idxs.begin(), idxs.end(), 0);
        return idxs;
    }
    Eigen::VectorXi xt_idxs() const {
        Eigen::VectorXi idxs(this->xt_vars());
        std::iota(idxs.begin(), idxs.end(), 0);
        return idxs;
    }
    Eigen::VectorXi xtu_idxs() const {
        Eigen::VectorXi idxs(this->xtu_vars());
        std::iota(idxs.begin(), idxs.end(), 0);
        return idxs;
    }
    Eigen::VectorXi u_idxs() const {
        Eigen::VectorXi idxs(this->u_vars());
        std::iota(idxs.begin(), idxs.end(), this->xt_vars());
        return idxs;
    }
    Eigen::VectorXi p_idxs() const {
        Eigen::VectorXi idxs(this->p_vars());
        std::iota(idxs.begin(), idxs.end(), this->xtu_vars());
        return idxs;
    }

    Eigen::VectorXi idxs_impl(const Eigen::VectorXi &zidxs, const Eigen::VectorXi &idxs) const {

        auto minelem = *std::min_element(zidxs.begin(), zidxs.end());
        auto maxelem = *std::max_element(zidxs.begin(), zidxs.end());

        if (minelem < 0 || maxelem >= idxs.size()) {
            throw std::invalid_argument("Indexing error in ODESizes idxs");
        }

        Eigen::VectorXi nidxs(zidxs.size());

        for (int i = 0; i < zidxs.size(); i++) {
            nidxs[i] = idxs[zidxs[i]];
        }

        return nidxs;
    }

    Eigen::VectorXi idxs_impl(int zidx, const Eigen::VectorXi &idxs) const {
        Eigen::VectorXi zidxs(1);
        zidxs[0] = zidx;
        return this->idxs_impl(zidxs, idxs);
    }

    Eigen::VectorXi x_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->x_idxs());
    }
    Eigen::VectorXi xt_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->xt_idxs());
    }
    Eigen::VectorXi xtu_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->xtu_idxs());
    }
    Eigen::VectorXi u_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->u_idxs());
    }
    Eigen::VectorXi p_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->p_idxs());
    }

    Eigen::VectorXi x_idxs(int zidxs) const { return idxs_impl(zidxs, this->x_idxs()); }
    Eigen::VectorXi xt_idxs(int zidxs) const { return idxs_impl(zidxs, this->xt_idxs()); }
    Eigen::VectorXi xtu_idxs(int zidxs) const { return idxs_impl(zidxs, this->xtu_idxs()); }
    Eigen::VectorXi u_idxs(int zidxs) const { return idxs_impl(zidxs, this->u_idxs()); }
    Eigen::VectorXi p_idxs(int zidxs) const { return idxs_impl(zidxs, this->p_idxs()); }
};

} // namespace tycho::oc
