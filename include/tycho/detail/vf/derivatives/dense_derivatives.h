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

#include "tycho/detail/vf/core/dense_function.h"

namespace tycho::vf {

enum class DenseDerivativeMode {
    Analytic,
    FDiffFwd,
    FDiffCentArray,
    AutodiffFwd,
};

template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseFirstDerivatives : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
};

template <class Derived, int IR, int OR, DenseDerivativeMode Jmode, DenseDerivativeMode Hmode>
struct DenseSecondDerivatives : DenseFirstDerivatives<Derived, IR, OR, Jmode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, Jmode>;
};

template <class Derived, int IR, int OR, DenseDerivativeMode Jmode, DenseDerivativeMode Hmode>
struct DenseDerivatives : DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode>;
    DENSE_FUNCTION_BASE_TYPES(Base)
};

} // namespace tycho::vf
