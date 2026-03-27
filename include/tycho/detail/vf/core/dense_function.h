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

#include "tycho/detail/vf/core/dense_function_base.h"
#include "tycho/detail/vf/core/dense_scalar_function_base.h"

namespace Tycho {

template <class Derived, int IR, int OR> struct DenseFunction : DenseFunctionBase<Derived, IR, OR> {
    using Base = DenseFunctionBase<Derived, IR, OR>;
};

template <class Derived, int IR>
struct DenseFunction<Derived, IR, 1> : DenseScalarFunctionBase<Derived, IR> {
    using Base = DenseScalarFunctionBase<Derived, IR>;
};
} // namespace Tycho
