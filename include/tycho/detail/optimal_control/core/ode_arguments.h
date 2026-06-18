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
#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/vector_functions.h"

namespace tycho::oc {

using vf::Arguments;

/// @ingroup optimal_control
/// @brief Identity VectorFunction over an ODE argument vector with semantic indexing.
///
/// Extends @c tycho::vf::Arguments with ODE-aware accessors: the packed
/// argument vector is laid out as @c [x0..x(XV-1), t, u0..u(UV-1), p0..p(PV-1)]
/// and may be indexed by semantic tags (@c XVar, @c UVar, @c PVar, @c TVar, and
/// their vector/segment variants) instead of raw offsets. This is the entry
/// point a user builds ODE dynamics expressions from.
/// @tparam _XV  State-vector dimension (`Eigen::Dynamic` for runtime size).
/// @tparam _UV  Control-vector dimension (`Eigen::Dynamic` for runtime size).
/// @tparam _PV  Parameter-vector dimension (`Eigen::Dynamic` for runtime size).
template <int _XV = -1, int _UV = -1, int _PV = -1>
struct ODEArguments : Arguments<ODESize<_XV, _UV, _PV>::XtUPV>, ODESize<_XV, _UV, _PV> {

    /// @brief Convenience alias for the underlying Arguments base class.
    using Base = Arguments<ODESize<_XV, _UV, _PV>::XtUPV>;

    /// @brief Construct with runtime state, control, and parameter dimensions.
    /// @param Xv  Number of state variables.
    /// @param Uv  Number of control variables.
    /// @param Pv  Number of parameter variables.
    /// @note Only available when at least one of @p _XV, @p _UV, @p _PV is dynamic.
    ODEArguments(int Xv, int Uv, int Pv)
        requires(_XV < 0 || _UV < 0 || _PV < 0)
        : Base(Xv + Uv + Pv + 1) {
        this->set_xt_up_vars(Xv, Uv, Pv);
    }
    /// @brief Construct with runtime state and control dimensions (zero parameters).
    /// @param Xv  Number of state variables.
    /// @param Uv  Number of control variables.
    ODEArguments(int Xv, int Uv)
        requires(_XV < 0 || _UV < 0 || _PV < 0)
        : ODEArguments(Xv, Uv, 0) {}
    /// @brief Construct with a runtime state dimension (zero controls and parameters).
    /// @param Xv  Number of state variables.
    ODEArguments(int Xv)
        requires(_XV < 0 || _UV < 0 || _PV < 0)
        : ODEArguments(Xv, 0) {}

    /// @brief Default constructor for fully compile-time-sized arguments.
    /// @note Only available when @p _XV, @p _UV, and @p _PV are all non-negative.
    ODEArguments()
        requires(_XV >= 0 && _UV >= 0 && _PV >= 0)
        : Base(_XV + _UV + _PV + 1) {
        this->set_xt_up_vars(_XV, _UV, _PV);
    }

    // Offset-aware access via semantic variable tags.
    // ODE vector layout: [x0..x(XV-1), t, u0..u(UV-1), p0..p(PV-1)]

    // ---- Scalar access ----

    /// @brief Access the @p I-th state variable by @c tycho::vf::XVarTag.
    /// @tparam I  Compile-time state index.
    /// @return Scalar element expression for state variable @p I.
    template <int I> decltype(auto) operator[](vf::XVarTag<I>) const {
        static_assert(_XV >= 0, "XVar requires compile-time-known state dimension (XV >= 0)");
        static_assert(I < _XV, "XVar index out of range for this ODE sizing");
        return this->template coeff<I>();
    }

    /// @brief Access the @p I-th control variable by @c tycho::vf::UVarTag.
    /// @tparam I  Compile-time control index.
    /// @return Scalar element expression for control variable @p I.
    template <int I> decltype(auto) operator[](vf::UVarTag<I>) const {
        static_assert(_UV >= 0, "UVar requires compile-time-known control dimension (UV >= 0)");
        static_assert(I < _UV, "UVar index out of range for this ODE sizing");
        return this->template coeff<_XV + 1 + I>();
    }

    /// @brief Access the @p I-th parameter variable by @c tycho::vf::PVarTag.
    /// @tparam I  Compile-time parameter index.
    /// @return Scalar element expression for parameter variable @p I.
    template <int I> decltype(auto) operator[](vf::PVarTag<I>) const {
        static_assert(_PV >= 0, "PVar requires compile-time-known parameter dimension (PV >= 0)");
        static_assert(I < _PV, "PVar index out of range for this ODE sizing");
        return this->template coeff<_XV + 1 + _UV + I>();
    }

    /// @brief Access the time variable by @c tycho::vf::TVarTag.
    /// @return Scalar element expression for the time variable.
    decltype(auto) operator[](vf::TVarTag) const {
        static_assert(_XV >= 0, "TVar requires compile-time-known XV");
        return this->template coeff<_XV>();
    }

    // ---- Full-vector access ----

    /// @brief Access the full state vector by @c tycho::vf::XVecTag.
    /// @return Segment expression spanning all state variables.
    decltype(auto) operator[](vf::XVecTag) const {
        static_assert(_XV >= 0, "XVec requires compile-time-known XV");
        static_assert(_XV > 0, "XVec requires at least one state variable (XV > 0)");
        return this->template segment<_XV, 0>();
    }

    /// @brief Access the full control vector by @c tycho::vf::UVecTag.
    /// @return Segment expression spanning all control variables.
    decltype(auto) operator[](vf::UVecTag) const {
        static_assert(_UV >= 0, "UVec requires compile-time-known UV");
        static_assert(_UV > 0, "UVec requires at least one control variable (UV > 0)");
        return this->template segment<_UV, _XV + 1>();
    }

    /// @brief Access the full parameter vector by @c tycho::vf::PVecTag.
    /// @return Segment expression spanning all parameter variables.
    decltype(auto) operator[](vf::PVecTag) const {
        static_assert(_PV >= 0, "PVec requires compile-time-known PV");
        static_assert(_PV > 0, "PVec requires at least one parameter variable (PV > 0)");
        return this->template segment<_PV, _XV + 1 + _UV>();
    }

    // ---- Sub-vector access ----

    /// @brief Access a contiguous slice of the state vector by @c tycho::vf::XSegTag.
    /// @tparam Start  Compile-time start index within the state block.
    /// @tparam Size   Compile-time length of the slice.
    /// @return Segment expression for the selected state slice.
    template <int Start, int Size> decltype(auto) operator[](vf::XSegTag<Start, Size>) const {
        static_assert(_XV >= 0, "XSeg requires compile-time-known state dimension (XV >= 0)");
        static_assert(Start + Size <= _XV, "XSeg out of range for this ODE sizing");
        return this->template segment<Size, Start>();
    }

    /// @brief Access a contiguous slice of the control vector by @c tycho::vf::USegTag.
    /// @tparam Start  Compile-time start index within the control block.
    /// @tparam Size   Compile-time length of the slice.
    /// @return Segment expression for the selected control slice.
    template <int Start, int Size> decltype(auto) operator[](vf::USegTag<Start, Size>) const {
        static_assert(_UV >= 0, "USeg requires compile-time-known control dimension (UV >= 0)");
        static_assert(Start + Size <= _UV, "USeg out of range for this ODE sizing");
        return this->template segment<Size, _XV + 1 + Start>();
    }

    /// @brief Access a contiguous slice of the parameter vector by @c tycho::vf::PSegTag.
    /// @tparam Start  Compile-time start index within the parameter block.
    /// @tparam Size   Compile-time length of the slice.
    /// @return Segment expression for the selected parameter slice.
    template <int Start, int Size> decltype(auto) operator[](vf::PSegTag<Start, Size>) const {
        static_assert(_PV >= 0, "PSeg requires compile-time-known parameter dimension (PV >= 0)");
        static_assert(Start + Size <= _PV, "PSeg out of range for this ODE sizing");
        return this->template segment<Size, _XV + 1 + _UV + Start>();
    }
};

} // namespace tycho::oc
