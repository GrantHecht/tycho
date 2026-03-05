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

namespace Tycho {

template <int _CS, int ODEXV, int ODEUV, int ODEPV> struct DefectConstSizes {
    static const int CS = _CS;
    static const int IS = SZ_NEG<CS - 1>::value;
    static const int XTU = SZ_SUM<ODEXV, ODEUV, 1>::value;
    static const int DefIRC = SZ_SUM<SZ_PROD<CS, XTU>::value, ODEPV>::value;
    static const int DefORC = SZ_PROD<IS, ODEXV>::value;
};

} // namespace Tycho
