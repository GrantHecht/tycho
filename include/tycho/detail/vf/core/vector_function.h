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

#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/vf/derivatives/dense_autodiff_fwd.h"
#include "tycho/detail/vf/derivatives/dense_derivatives.h"
#include "tycho/detail/vf/derivatives/dense_fdiff_cent_array.h"
#include "tycho/detail/vf/derivatives/dense_fdiff_fwd.h"

namespace tycho::vf {

template <class Derived, int IR, int OR, DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct VectorFunction : DenseSecondDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
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
    void init_expression(Ts... ts) { *this = Base(ExprImpl::Definition(ts...)); }
};

template <class Derived, class ExprImpl>
struct VectorExpression<Derived, ExprImpl>
    : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    using Base =
        typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    VectorExpression() : Base(ExprImpl::Definition()) {}

    /////////////////////////////////////
    void init_expression() { *this = Base(ExprImpl::Definition()); }
};

#define BUILD_FROM_EXPRESSION(NAME, IMPL, ...)                                                     \
    struct NAME : VectorExpression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__> {                        \
        using Base = VectorExpression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__>;                      \
        using Base::Base;                                                                          \
    };

} // namespace tycho::vf
