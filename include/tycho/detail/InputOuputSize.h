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

template <int IR, int OR> struct InputOutputSize {
    static const int InputRows = IR;
    static const int OutputRows = OR;
};

template <> struct InputOutputSize<-1, -1> {
    int InputRows = 0;
    int OutputRows = 0;
};

template <int OR> struct InputOutputSize<-1, OR> {
    int InputRows = 0;
    static const int OutputRows = OR;
};

template <int IR> struct InputOutputSize<IR, -1> {
    static const int InputRows = IR;
    int OutputRows = 0;
};

} // namespace Tycho
