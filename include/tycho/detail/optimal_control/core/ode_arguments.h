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
#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/vector_functions.h"

namespace tycho::oc {

template <int _XV = -1, int _UV = -1, int _PV = -1>
struct ODEArguments : Arguments<ODESize<_XV, _UV, _PV>::XtUPV>, ODESize<_XV, _UV, _PV> {

    using Base = Arguments<ODESize<_XV, _UV, _PV>::XtUPV>;

    ODEArguments(int Xv, int Uv, int Pv) : Base(Xv + Uv + Pv + 1) { this->set_xt_up_vars(Xv, Uv, Pv); }
    ODEArguments(int Xv, int Uv) : ODEArguments(Xv, Uv, 0) {}
    ODEArguments(int Xv) : ODEArguments(Xv, 0) {}
};

} // namespace tycho::oc
