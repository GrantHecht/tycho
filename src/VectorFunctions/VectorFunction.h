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

#include "DenseDerivatives.h"
#include "DenseDifferentiation/DenseAutodiffFwd.h"
#include "DenseDifferentiation/DenseFDiffCentArray.h"
#include "DenseDifferentiation/DenseFDiffFwd.h"
#include "Utils/FunctionReturnType.h"

namespace Tycho {

template <class Derived, int IR, int OR, DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct VectorFunction : DenseDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseDerivatives<Derived, IR, OR, Jm, Hm>;
    DENSE_FUNCTION_BASE_TYPES(Base)
};

template <class Derived, class ExprImpl, class... Ts>
struct VectorExpression
    : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    using Base =
        typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    VectorExpression(Ts... ts) : Base(ExprImpl::Definition(ts...)) {}
    VectorExpression() {};

    /////////////////////////////////////
    void InitExpression(Ts... ts) { *this = Base(ExprImpl::Definition(ts...)); }
};

template <class Derived, class ExprImpl>
struct VectorExpression<Derived, ExprImpl>
    : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    using Base =
        typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    VectorExpression() : Base(ExprImpl::Definition()) {}

    /////////////////////////////////////
    void InitExpression() { *this = Base(ExprImpl::Definition()); }
};

#define BUILD_FROM_EXPRESSION(NAME, IMPL, ...)                                                     \
    struct NAME : VectorExpression<NAME, IMPL, __VA_ARGS__> {                                      \
        using Base = VectorExpression<NAME, IMPL, __VA_ARGS__>;                                    \
        using Base::Base;                                                                          \
    };

} // namespace Tycho
