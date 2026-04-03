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
    // XVar<I> → index I          (states start at 0)
    // UVar<I> → index XV + 1 + I (controls after states + time)
    // PVar<I> → index XV + 1 + UV + I (params after controls)

    template <int I> decltype(auto) operator[](vf::XVarTag<I>) const {
        static_assert(_XV >= 0 && I < _XV, "XVar index out of range for this ODE sizing");
        return this->template coeff<I>();
    }

    template <int I> decltype(auto) operator[](vf::UVarTag<I>) const {
        static_assert(_UV >= 0 && I < _UV, "UVar index out of range for this ODE sizing");
        return this->template coeff<_XV + 1 + I>();
    }

    template <int I> decltype(auto) operator[](vf::PVarTag<I>) const {
        static_assert(_PV >= 0 && I < _PV, "PVar index out of range for this ODE sizing");
        return this->template coeff<_XV + 1 + _UV + I>();
    }
};

} // namespace tycho::oc
