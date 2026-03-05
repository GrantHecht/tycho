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

#include "DenseFunction.h"

namespace Tycho {

enum DenseDerivativeModes {
    Analytic,
    FDiffFwd,
    FDiffCentArray,
    AutodiffFwd,
};

template <class Derived, int IR, int OR, int Jmode>
struct DenseFirstDerivatives : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
};

template <class Derived, int IR, int OR, int Jmode, int Hmode>
struct DenseSecondDerivatives : DenseFirstDerivatives<Derived, IR, OR, Jmode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, Jmode>;
};

template <class Derived, int IR, int OR, int Jmode, int Hmode>
struct DenseDerivatives : DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode>;
    DENSE_FUNCTION_BASE_TYPES(Base)
};

} // namespace Tycho
