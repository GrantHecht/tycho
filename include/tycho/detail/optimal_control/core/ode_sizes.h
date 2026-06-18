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

/// @internal
/// @brief Compile-time ODE dimension constants for state, control, and parameters.
///
/// Holds the four derived sizes (state, state+time, state+time+control,
/// state+time+control+params) computed from the three template dimensions.
/// @tparam _XV  State-vector dimension (`Eigen::Dynamic` for runtime size).
/// @tparam _UV  Control-vector dimension (`Eigen::Dynamic` for runtime size).
/// @tparam _PV  Parameter-vector dimension (`Eigen::Dynamic` for runtime size).
/// @endinternal
template <int _XV, int _UV, int _PV> struct ODEConstSizes {
    static constexpr int XV = _XV;                          ///< State-vector dimension.
    static constexpr int UV = _UV;                          ///< Control-vector dimension.
    static constexpr int PV = _PV;                          ///< Parameter-vector dimension.
    static constexpr int XtV = SZ_SUM<_XV, 1>::value;       ///< State + time dimension.
    static constexpr int XtUV = SZ_SUM<_XV, 1, _UV>::value; ///< State + time + control dimension.
    static constexpr int XtUPV =
        SZ_SUM<_XV, 1, _UV, _PV>::value; ///< State + time + control + params.
};

/// @internal
/// @brief State-and-time size accessors layer of the ODE size trait stack.
///
/// Static-size specialization: state and time dimensions are known at compile
/// time, so the setters are no-ops.
/// @tparam _XV  State-vector dimension.
/// @tparam _UV  Control-vector dimension.
/// @tparam _PV  Parameter-vector dimension.
/// @endinternal
template <int _XV, int _UV, int _PV> struct ODEXVSizes : ODEConstSizes<_XV, _UV, _PV> {
    inline int t_var() const { return this->XV; }    ///< @brief Index of the time variable.
    inline int x_vars() const { return this->XV; }   ///< @brief Number of state variables.
    inline int xt_vars() const { return this->XtV; } ///< @brief Number of state + time variables.
    /// @brief No-op setter (state size is fixed at compile time).
    /// @param xv  Ignored.
    void set_xvars(int xv) {}
};

/// @internal
/// @brief Runtime-state specialization of @ref ODEXVSizes (`_XV == Eigen::Dynamic`).
/// @tparam _UV  Control-vector dimension.
/// @tparam _PV  Parameter-vector dimension.
/// @endinternal
template <int _UV, int _PV> struct ODEXVSizes<-1, _UV, _PV> : ODEConstSizes<-1, _UV, _PV> {
    inline int t_var() const { return this->xv_dynamic_; }  ///< @brief Index of the time variable.
    inline int x_vars() const { return this->xv_dynamic_; } ///< @brief Number of state variables.
    inline int xt_vars() const {
        return this->xtv_dynamic_;
    } ///< @brief Number of state + time variables.
    /// @brief Set the runtime state dimension.
    /// @param xv  Number of state variables.
    void set_xvars(int xv) {
        this->xv_dynamic_ = xv;
        this->xtv_dynamic_ = xv + 1;
    }

  protected:
    int xv_dynamic_ = 0;  ///< Runtime state dimension.
    int xtv_dynamic_ = 0; ///< Runtime state + time dimension.
};

/// @internal
/// @brief Control-size accessors layer of the ODE size trait stack.
/// @tparam _XV  State-vector dimension.
/// @tparam _UV  Control-vector dimension.
/// @tparam _PV  Parameter-vector dimension.
/// @endinternal
template <int _XV, int _UV, int _PV> struct ODEXUVSizes : ODEXVSizes<_XV, _UV, _PV> {
    inline int u_vars() const { return this->UV; } ///< @brief Number of control variables.
    inline int xtu_vars() const {
        return this->UV + this->xt_vars();
    } ///< @brief Number of state + time + control variables.
    /// @brief No-op setter (control size is fixed at compile time).
    /// @param uv  Ignored.
    void set_uvars(int uv) {}
};
/// @internal
/// @brief Runtime-control specialization of @ref ODEXUVSizes (`_UV == Eigen::Dynamic`).
/// @tparam _XV  State-vector dimension.
/// @tparam _PV  Parameter-vector dimension.
/// @endinternal
template <int _XV, int _PV> struct ODEXUVSizes<_XV, -1, _PV> : ODEXVSizes<_XV, -1, _PV> {
    inline int u_vars() const { return this->uv_dynamic_; } ///< @brief Number of control variables.
    inline int xtu_vars() const {
        return this->uv_dynamic_ + this->xt_vars();
    } ///< @brief Number of state + time + control variables.
    /// @brief Set the runtime control dimension.
    /// @param uv  Number of control variables.
    void set_uvars(int uv) { this->uv_dynamic_ = uv; }

  protected:
    int uv_dynamic_ = 0; ///< Runtime control dimension.
};

/// @internal
/// @brief Parameter-size accessors layer of the ODE size trait stack.
/// @tparam _XV  State-vector dimension.
/// @tparam _UV  Control-vector dimension.
/// @tparam _PV  Parameter-vector dimension.
/// @endinternal
template <int _XV, int _UV, int _PV> struct ODEXUPVSizes : ODEXUVSizes<_XV, _UV, _PV> {
    inline int p_vars() const { return this->PV; } ///< @brief Number of parameter variables.
    inline int xtu_p_vars() const {
        return this->PV + this->xtu_vars();
    } ///< @brief Total state + time + control + parameter variables.
    /// @brief No-op setter (parameter size is fixed at compile time).
    /// @param pv  Ignored.
    void set_pvars(int pv) {}
    /// @brief Set all runtime variable dimensions at once.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    /// @param pv  Number of parameter variables.
    void set_xt_up_vars(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
    }
};

/// @internal
/// @brief Runtime-parameter specialization of @ref ODEXUPVSizes (`_PV == Eigen::Dynamic`).
/// @tparam _XV  State-vector dimension.
/// @tparam _UV  Control-vector dimension.
/// @endinternal
template <int _XV, int _UV> struct ODEXUPVSizes<_XV, _UV, -1> : ODEXUVSizes<_XV, _UV, -1> {
    inline int p_vars() const {
        return this->pv_dynamic_;
    } ///< @brief Number of parameter variables.
    inline int xtu_p_vars() const {
        return this->pv_dynamic_ + this->xtu_vars();
    } ///< @brief Total state + time + control + parameter variables.
    /// @brief Set the runtime parameter dimension.
    /// @param pv  Number of parameter variables.
    void set_pvars(int pv) { this->pv_dynamic_ = pv; }
    /// @brief Set all runtime variable dimensions at once.
    /// @param xv  Number of state variables.
    /// @param uv  Number of control variables.
    /// @param pv  Number of parameter variables.
    void set_xt_up_vars(int xv, int uv, int pv) {
        this->set_xvars(xv);
        this->set_uvars(uv);
        this->set_pvars(pv);
    }

  protected:
    int pv_dynamic_ = 0; ///< Runtime parameter dimension.
};

/// @internal
/// @brief Full ODE size descriptor: dimensions plus named variable index groups.
///
/// Top of the size trait stack. Adds a registry of named index groups (e.g.
/// state subsets a user wants to refer to by name) and convenience accessors
/// that return the absolute indices of each variable block within the packed
/// `[state, time, control, params]` argument vector.
/// @tparam _XV  State-vector dimension.
/// @tparam _UV  Control-vector dimension.
/// @tparam _PV  Parameter-vector dimension.
/// @endinternal
template <int _XV, int _UV, int _PV> struct ODESize : ODEXUPVSizes<_XV, _UV, _PV> {

    FlatMap<std::string, Eigen::VectorXi> xtu_p_idxs_; ///< Registry of named variable index groups.

    /// @brief Register a named group of variable indices.
    /// @param name  Group name (must be unique).
    /// @param idx   Indices belonging to the group (must be non-empty).
    /// @throws std::invalid_argument if @p idx is empty or @p name already exists.
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

    /// @brief Register a named group consisting of a single variable index.
    /// @param name  Group name (must be unique).
    /// @param indx  The single index belonging to the group.
    /// @throws std::invalid_argument if @p name already exists.
    void add_idx(const std::string &name, int indx) {
        Eigen::VectorXi idxv(1);
        idxv[0] = indx;
        this->add_idx(name, idxv);
    }

    /// @brief Look up a named variable index group.
    /// @param name  Group name.
    /// @return The indices registered under @p name.
    /// @throws std::invalid_argument if no group named @p name exists.
    Eigen::VectorXi idx(const std::string &name) const {
        if (!xtu_p_idxs_.contains(name)) {
            throw std::invalid_argument(
                fmt::format("No variable index group with name: {0:} exists.", name));
        }
        return xtu_p_idxs_.at(name);
    }

    /// @brief Replace the entire named-index-group registry.
    /// @param idxs  Map of group name to indices; each group must be non-empty.
    /// @throws std::invalid_argument if any group is empty.
    void set_idxs(const FlatMap<std::string, Eigen::VectorXi> &idxs) {
        for (const auto &[name, idx] : idxs) {
            if (idx.size() == 0) {
                throw std::invalid_argument(
                    fmt::format("Variable index group with name: {0:} has no elements.", name));
            }
        }
        this->xtu_p_idxs_ = idxs;
    }
    /// @brief Return the full named-index-group registry.
    /// @return Map of group name to indices.
    FlatMap<std::string, Eigen::VectorXi> get_idxs() const { return this->xtu_p_idxs_; }

    /// @brief Absolute indices of the state block `[0, x_vars)`.
    /// @return Index vector of the state variables.
    Eigen::VectorXi x_idxs() const {
        Eigen::VectorXi idxs(this->x_vars());
        std::iota(idxs.begin(), idxs.end(), 0);
        return idxs;
    }
    /// @brief Absolute indices of the state + time block `[0, xt_vars)`.
    /// @return Index vector of the state and time variables.
    Eigen::VectorXi xt_idxs() const {
        Eigen::VectorXi idxs(this->xt_vars());
        std::iota(idxs.begin(), idxs.end(), 0);
        return idxs;
    }
    /// @brief Absolute indices of the state + time + control block `[0, xtu_vars)`.
    /// @return Index vector of the state, time, and control variables.
    Eigen::VectorXi xtu_idxs() const {
        Eigen::VectorXi idxs(this->xtu_vars());
        std::iota(idxs.begin(), idxs.end(), 0);
        return idxs;
    }
    /// @brief Absolute indices of the control block within the packed argument vector.
    /// @return Index vector of the control variables.
    Eigen::VectorXi u_idxs() const {
        Eigen::VectorXi idxs(this->u_vars());
        std::iota(idxs.begin(), idxs.end(), this->xt_vars());
        return idxs;
    }
    /// @brief Absolute indices of the parameter block within the packed argument vector.
    /// @return Index vector of the parameter variables.
    Eigen::VectorXi p_idxs() const {
        Eigen::VectorXi idxs(this->p_vars());
        std::iota(idxs.begin(), idxs.end(), this->xtu_vars());
        return idxs;
    }

    /// @internal
    /// @brief Map relative (zero-based) indices into a block's absolute index vector.
    /// @param zidxs  Relative indices into @p idxs (each in `[0, idxs.size())`).
    /// @param idxs   The block's absolute index vector to gather from.
    /// @return Absolute indices `idxs[zidxs[i]]`.
    /// @throws std::invalid_argument if any relative index is out of range.
    /// @endinternal
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

    /// @internal
    /// @brief Single-index overload of @ref idxs_impl.
    /// @param zidx  Relative index into @p idxs.
    /// @param idxs  The block's absolute index vector to gather from.
    /// @return One-element vector with the absolute index `idxs[zidx]`.
    /// @endinternal
    Eigen::VectorXi idxs_impl(int zidx, const Eigen::VectorXi &idxs) const {
        Eigen::VectorXi zidxs(1);
        zidxs[0] = zidx;
        return this->idxs_impl(zidxs, idxs);
    }

    /// @brief Absolute indices of a state-block subset selected by relative indices.
    /// @param zidxs  Relative indices into the state block.
    /// @return Absolute indices of the selected state variables.
    Eigen::VectorXi x_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->x_idxs());
    }
    /// @brief Absolute indices of a state+time-block subset selected by relative indices.
    /// @param zidxs  Relative indices into the state + time block.
    /// @return Absolute indices of the selected state/time variables.
    Eigen::VectorXi xt_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->xt_idxs());
    }
    /// @brief Absolute indices of a state+time+control-block subset by relative indices.
    /// @param zidxs  Relative indices into the state + time + control block.
    /// @return Absolute indices of the selected state/time/control variables.
    Eigen::VectorXi xtu_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->xtu_idxs());
    }
    /// @brief Absolute indices of a control-block subset selected by relative indices.
    /// @param zidxs  Relative indices into the control block.
    /// @return Absolute indices of the selected control variables.
    Eigen::VectorXi u_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->u_idxs());
    }
    /// @brief Absolute indices of a parameter-block subset selected by relative indices.
    /// @param zidxs  Relative indices into the parameter block.
    /// @return Absolute indices of the selected parameter variables.
    Eigen::VectorXi p_idxs(const Eigen::VectorXi &zidxs) const {
        return idxs_impl(zidxs, this->p_idxs());
    }

    /// @brief Absolute index of a single state variable.
    /// @param zidxs  Relative state index.
    /// @return One-element vector with the absolute state index.
    Eigen::VectorXi x_idxs(int zidxs) const { return idxs_impl(zidxs, this->x_idxs()); }
    /// @brief Absolute index of a single state-or-time variable.
    /// @param zidxs  Relative state + time index.
    /// @return One-element vector with the absolute index.
    Eigen::VectorXi xt_idxs(int zidxs) const { return idxs_impl(zidxs, this->xt_idxs()); }
    /// @brief Absolute index of a single state/time/control variable.
    /// @param zidxs  Relative state + time + control index.
    /// @return One-element vector with the absolute index.
    Eigen::VectorXi xtu_idxs(int zidxs) const { return idxs_impl(zidxs, this->xtu_idxs()); }
    /// @brief Absolute index of a single control variable.
    /// @param zidxs  Relative control index.
    /// @return One-element vector with the absolute control index.
    Eigen::VectorXi u_idxs(int zidxs) const { return idxs_impl(zidxs, this->u_idxs()); }
    /// @brief Absolute index of a single parameter variable.
    /// @param zidxs  Relative parameter index.
    /// @return One-element vector with the absolute parameter index.
    Eigen::VectorXi p_idxs(int zidxs) const { return idxs_impl(zidxs, this->p_idxs()); }
};

} // namespace tycho::oc
