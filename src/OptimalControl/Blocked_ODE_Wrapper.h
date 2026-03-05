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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once
#include "Utils/SizingHelpers.h"
#include "pch.h"

namespace Tycho {

template <class DODE> struct Blocked_ODE_Wrapper : DODE {
    static const int UV = 0;
    static const int PV = SZ_SUM<DODE::PV, DODE::UV>::value;
    static const int XtUV = DODE::XtV;
    using Base = DODE;

    inline int XtUVars() const { return Base::XtVars(); }
    inline int UVars() const { return 0; }
    inline int PVars() const { return Base::UVars() + Base::PVars(); }

    Blocked_ODE_Wrapper() {};
    Blocked_ODE_Wrapper(const DODE &ode) : Base(ode) {}
};

} // namespace Tycho