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

namespace Tycho {

enum class ParsedIOFlags {
    HiddenInput = -2,
    IngoreOutput = -1,
    NotContiguous,
    Contiguous,
};

enum class ThreadingFlags : int {
    RoundRobin = -4,
    MainThread = -3,
    NeedsPool = -2,
    ByApplication = -1,
    Thread0 = 0,
    Thread1 = 1,
    Thread2 = 2,
    Thread3 = 3,
    Thread4 = 4,
    Thread5 = 5,
    Thread6 = 6,
    Thread7 = 7,
    Thread8 = 8,
    Thread9 = 9,
    Thread10 = 10,
    Thread11 = 11,
    Thread12 = 12,
    Thread13 = 13,
    Thread14 = 14,
    Thread15 = 15,
    Thread16 = 16,
    Thread17 = 17,
    Thread18 = 18,
    Thread19 = 19,
    Thread20 = 20,
    Thread21 = 21,
    Thread22 = 22,
    Thread23 = 23,
    Thread24 = 24,
    Thread25 = 25,
    Thread26 = 26,
    Thread27 = 27,
    Thread28 = 28,
};

enum class VarTypes {
    NonLinear = 0,
    Linear = 1,
    Quadratic = 2,
    Inactive = 3,
    BiLinear = 4,
};
} // namespace Tycho
