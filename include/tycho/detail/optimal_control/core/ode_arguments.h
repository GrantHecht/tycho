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

template <int _XV = -1, int _UV = -1, int _PV = -1>
struct ODEArguments : Arguments<ODESize<_XV, _UV, _PV>::XtUPV>, ODESize<_XV, _UV, _PV> {

    using Base = Arguments<ODESize<_XV, _UV, _PV>::XtUPV>;

    ODEArguments(int Xv, int Uv, int Pv) : Base(Xv + Uv + Pv + 1) {
        this->set_xt_up_vars(Xv, Uv, Pv);
    }
    ODEArguments(int Xv, int Uv) : ODEArguments(Xv, Uv, 0) {}
    ODEArguments(int Xv) : ODEArguments(Xv, 0) {}

    // Default constructor for compile-time-sized ODEArguments (XV, UV, PV all >= 0).
    ODEArguments()
        requires(_XV >= 0 && _UV >= 0 && _PV >= 0)
        : ODEArguments(_XV, _UV, _PV) {}

    // Offset-aware access via semantic variable tags.
    // ODE vector layout: [x0..x(XV-1), t, u0..u(UV-1), p0..p(PV-1)]

    // ---- Scalar access ----

    template <int I> decltype(auto) operator[](vf::XVarTag<I>) const {
        static_assert(_XV >= 0, "XVar requires compile-time-known state dimension (XV >= 0)");
        static_assert(I < _XV, "XVar index out of range for this ODE sizing");
        return this->template coeff<I>();
    }

    template <int I> decltype(auto) operator[](vf::UVarTag<I>) const {
        static_assert(_UV >= 0, "UVar requires compile-time-known control dimension (UV >= 0)");
        static_assert(I < _UV, "UVar index out of range for this ODE sizing");
        return this->template coeff<_XV + 1 + I>();
    }

    template <int I> decltype(auto) operator[](vf::PVarTag<I>) const {
        static_assert(_PV >= 0, "PVar requires compile-time-known parameter dimension (PV >= 0)");
        static_assert(I < _PV, "PVar index out of range for this ODE sizing");
        return this->template coeff<_XV + 1 + _UV + I>();
    }

    decltype(auto) operator[](vf::TVarTag) const {
        static_assert(_XV >= 0, "TVar requires compile-time-known XV");
        return this->template coeff<_XV>();
    }

    // ---- Full-vector access ----

    decltype(auto) operator[](vf::XVecTag) const {
        static_assert(_XV >= 0, "XVec requires compile-time-known XV");
        static_assert(_XV > 0, "XVec requires at least one state variable (XV > 0)");
        return this->template segment<_XV, 0>();
    }

    decltype(auto) operator[](vf::UVecTag) const {
        static_assert(_UV >= 0, "UVec requires compile-time-known UV");
        static_assert(_UV > 0, "UVec requires at least one control variable (UV > 0)");
        return this->template segment<_UV, _XV + 1>();
    }

    decltype(auto) operator[](vf::PVecTag) const {
        static_assert(_PV >= 0, "PVec requires compile-time-known PV");
        static_assert(_PV > 0, "PVec requires at least one parameter variable (PV > 0)");
        return this->template segment<_PV, _XV + 1 + _UV>();
    }

    // ---- Sub-vector access ----

    template <int Start, int Size> decltype(auto) operator[](vf::XSegTag<Start, Size>) const {
        static_assert(_XV >= 0, "XSeg requires compile-time-known state dimension (XV >= 0)");
        static_assert(Start + Size <= _XV, "XSeg out of range for this ODE sizing");
        return this->template segment<Size, Start>();
    }

    template <int Start, int Size> decltype(auto) operator[](vf::USegTag<Start, Size>) const {
        static_assert(_UV >= 0, "USeg requires compile-time-known control dimension (UV >= 0)");
        static_assert(Start + Size <= _UV, "USeg out of range for this ODE sizing");
        return this->template segment<Size, _XV + 1 + Start>();
    }

    template <int Start, int Size> decltype(auto) operator[](vf::PSegTag<Start, Size>) const {
        static_assert(_PV >= 0, "PSeg requires compile-time-known parameter dimension (PV >= 0)");
        static_assert(Start + Size <= _PV, "PSeg out of range for this ODE sizing");
        return this->template segment<Size, _XV + 1 + _UV + Start>();
    }
};

} // namespace tycho::oc
