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

#include "GenericConditional.h"
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"
#include "pch.h"

namespace Tycho {

template <int IR> struct GenericComparative : rubber_types::TypeErasure<ConditionalSpec<IR>> {
    using Base = rubber_types::TypeErasure<ConditionalSpec<IR>>;

    GenericComparative() {}
    template <class T> GenericComparative(const T &t) : Base(t) {}
    GenericComparative(const GenericComparative<IR> &obj) {
        this->reset_container(obj.get_container());
    }
};

} // namespace Tycho
