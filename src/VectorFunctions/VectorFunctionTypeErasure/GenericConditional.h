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
//   - PR 9: Replaced rubber_types with TypeStorage<ConditionalBase<IR>>
// =============================================================================

#pragma once
#include "ConditionalTypeErasure.h"
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"
#include "pch.h"

namespace Tycho {

template <int IR> struct GenericConditional {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    static const bool IsConditional = true;
    static const int IRC = IR;

    TypeStorage<ConditionalBase<IR>> storage;

    GenericConditional() = default;

    template <class T> GenericConditional(const T &t) {
        storage.template emplace<ConditionalModel<IR, std::decay_t<T>>>(t);
    }

    GenericConditional(const GenericConditional &) = default;
    GenericConditional(GenericConditional &&) noexcept = default;
    GenericConditional &operator=(const GenericConditional &) = default;
    GenericConditional &operator=(GenericConditional &&) noexcept = default;

    std::string name() const { return storage.get().name(); }
    int IRows() const { return storage.get().IRows(); }

    template <class InTypeT> bool compute(const Eigen::MatrixBase<InTypeT> &x) const {
        InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

} // namespace Tycho
