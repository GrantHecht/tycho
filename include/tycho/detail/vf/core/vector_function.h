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
#include "tycho/detail/vf/derivatives/dense_derivatives.h"
#include "tycho/detail/vf/derivatives/dense_enzyme.h"
#include "tycho/detail/vf/derivatives/dense_fdiff_cent_array.h"
#include "tycho/detail/vf/derivatives/dense_fdiff_fwd.h"

namespace tycho::vf {

/// @brief CRTP base for user-defined VectorFunctions with selectable derivative modes.
///
/// This is the primary entry point for authoring a custom VectorFunction. A
/// user struct inherits from `VectorFunction<MyFunc, IR, OR, ...>` and provides
/// a `compute_impl` (and, for analytic modes, `compute_jacobian_impl` etc.).
/// The chosen Jacobian and Hessian modes determine whether derivatives come
/// from user-supplied analytic code, finite differences, or Enzyme AD.
///
/// Layered atop @ref DenseSecondDerivatives, which supplies the derivative
/// dispatch machinery, and ultimately @ref DenseFunctionBase / @ref
/// ComputableBase for the evaluation, expression-builder, and solver-interface
/// methods.
///
/// @tparam Derived  CRTP derived (user) function type.
/// @tparam IR       Input rows at compile time (`-1` for dynamic).
/// @tparam OR       Output rows at compile time (`-1` for dynamic).
/// @tparam Jm       Jacobian derivative mode (defaults to Analytic).
/// @tparam Hm       Hessian derivative mode (defaults to Analytic).
/// @ingroup vf
template <class Derived, int IR, int OR, DenseDerivativeMode Jm = DenseDerivativeMode::Analytic,
          DenseDerivativeMode Hm = DenseDerivativeMode::Analytic>
struct VectorFunction : DenseSecondDerivatives<Derived, IR, OR, Jm, Hm> {
    /// @brief Immediate base class supplying the first/second derivative dispatch.
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jm, Hm>;
    VF_TYPE_ALIASES(Base)
};

/// @brief CRTP wrapper that materializes a named function from an expression definition.
///
/// `VectorExpression` lets a named struct be defined in terms of an
/// expression-template `Definition(...)` factory: the wrapper inherits from the
/// concrete function type produced by `ExprImpl::Definition` (rebased onto
/// @p Derived) and forwards constructor arguments to that factory. This is the
/// machinery behind the `BUILD_FROM_EXPRESSION` macro.
///
/// @tparam Derived  CRTP derived (named expression) type.
/// @tparam ExprImpl Implementation type exposing a static `Definition(...)` factory.
/// @tparam Ts       Constructor argument types forwarded to `ExprImpl::Definition`.
/// @ingroup vf
template <class Derived, class ExprImpl, class... Ts>
struct VectorExpression
    : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    /// @brief Concrete function type produced by `ExprImpl::Definition`, rebased onto @p Derived.
    using Base =
        typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    /// @brief Construct by invoking the expression factory with @p ts.
    /// @param ts  Arguments forwarded to `ExprImpl::Definition`.
    VectorExpression(Ts... ts) : Base(ExprImpl::Definition(ts...)) {}
    /// @brief Default-construct an empty (uninitialized) expression.
    VectorExpression() {};

    /// @brief (Re)initialize this expression from the factory arguments @p ts.
    /// @param ts  Arguments forwarded to `ExprImpl::Definition`.
    void init_expression(Ts... ts) { *this = Base(ExprImpl::Definition(ts...)); }
};

/// @brief Nullary specialization of @ref VectorExpression (no factory arguments).
///
/// @tparam Derived  CRTP derived (named expression) type.
/// @tparam ExprImpl Implementation type exposing a static nullary `Definition()` factory.
/// @ingroup vf
template <class Derived, class ExprImpl>
struct VectorExpression<Derived, ExprImpl>
    : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    /// @brief Concrete function type produced by `ExprImpl::Definition`, rebased onto @p Derived.
    using Base =
        typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    /// @brief Construct by invoking the nullary expression factory.
    VectorExpression() : Base(ExprImpl::Definition()) {}

    /// @brief (Re)initialize this expression from the nullary factory.
    void init_expression() { *this = Base(ExprImpl::Definition()); }
};

/// @brief Define a named VectorFunction struct from an expression implementation.
///
/// Expands to a struct @p NAME deriving from
/// `VectorExpression<NAME, IMPL, ...>`, inheriting its constructors. Use this to
/// give a reusable name to an expression-template composition. See
/// @ref tycho::vf::VectorExpression.
///
/// @param NAME  Name of the generated struct.
/// @param IMPL  Implementation type with a static `Definition(...)` factory.
/// @param ...   Optional constructor argument types forwarded to the factory.
#define BUILD_FROM_EXPRESSION(NAME, IMPL, ...)                                                     \
    struct NAME : VectorExpression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__> {                        \
        using Base = VectorExpression<NAME, IMPL __VA_OPT__(, ) __VA_ARGS__>;                      \
        using Base::Base;                                                                          \
    };

} // namespace tycho::vf
