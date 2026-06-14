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
//   - PR 9: Replaced rubber_types with tycho::utils::TypeStorage<ConditionalBase<IR>>
// =============================================================================

#pragma once
#include "tycho/detail/vf/common/common_functions.h"
#include "tycho/detail/vf/type_erasure/conditional_base.h"
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::vf {

/// @brief Type-erased boolean predicate over an input vector.
///
/// GenericConditional holds any value-typed conditional (a comparison such as
/// `a < b`, a logical combination via AND/OR, or a constant) behind a single
/// uniform interface. It is the runtime-polymorphic predicate the library uses
/// to drive IfElseFunction branching, and is exposed to Python as the
/// `Conditional` type. Copies are deep (the wrapped predicate is cloned).
/// @tparam IR  Compile-time input-row count (-1 for dynamic).
/// @ingroup vf
template <int IR> struct GenericConditional {
    /// @brief Const reference to the fixed-size input vector accepted by compute().
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    static constexpr bool is_conditional = true; ///< @brief Trait flag: this is a conditional.
    static constexpr int IRC = IR;               ///< @brief Compile-time input-row count.

    /// @brief Type-erased storage holding the wrapped predicate.
    tycho::utils::TypeStorage<ConditionalBase<IR>> storage;

    /// @brief Construct an empty predicate (no wrapped function).
    GenericConditional() = default;

    /// @brief Construct by type-erasing an arbitrary predicate @p t.
    /// @tparam T  Any type satisfying the conditional interface (input_rows, compute).
    /// @param t   Predicate to wrap and store.
    template <class T> GenericConditional(const T &t) {
        storage.template emplace<ConditionalModel<IR, std::decay_t<T>>>(t);
    }

    /// @brief Deep-copy constructor (clones the wrapped predicate).
    GenericConditional(const GenericConditional &) = default;
    /// @brief Move constructor.
    GenericConditional(GenericConditional &&) noexcept = default;
    /// @brief Deep-copy assignment (clones the wrapped predicate).
    /// @return Reference to this object.
    GenericConditional &operator=(const GenericConditional &) = default;
    /// @brief Move assignment.
    /// @return Reference to this object.
    GenericConditional &operator=(GenericConditional &&) noexcept = default;

    /// @brief Number of input rows the wrapped predicate accepts.
    /// @return Input-row count.
    int input_rows() const { return storage.get().input_rows(); }

    /// @brief Evaluate the wrapped predicate at @p x.
    /// @tparam InTypeT  Eigen expression type of the input vector.
    /// @param x  Input vector.
    /// @return Boolean result of the predicate.
    template <class InTypeT> bool compute(const Eigen::MatrixBase<InTypeT> &x) const {
        InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

} // namespace tycho::vf
