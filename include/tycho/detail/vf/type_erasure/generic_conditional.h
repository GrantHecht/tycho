// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
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

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/interior/typedefs/eigen_types.h>
#include <hven/detail/interior/utils/flat_map.h>
#include <hven/detail/interior/utils/function_return_type.h>
#include <hven/detail/interior/utils/get_core_count.h>
#include <hven/detail/interior/utils/math_functions.h>
#include <hven/detail/interior/utils/sizing_helpers.h>
#include <hven/detail/interior/utils/std_extensions.h>
#include <hven/detail/interior/utils/thread_pool.h>
#include <hven/detail/interior/utils/type_name.h>
#include <hven/detail/interior/utils/type_storage.h>

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
    int input_rows() const {
        if (storage.empty()) {
            throw std::runtime_error(
                "GenericConditional is empty (default-constructed); assign a predicate before use");
        }
        return storage.get().input_rows();
    }

    /// @brief Runtime read-set: conservatively the full input range.
    ///
    /// The wrapped predicate's structure (and therefore its true read-set) is
    /// erased behind @ref ConditionalBase, which has no `input_domain()` in its
    /// virtual interface — so this reports the full `[0, input_rows())` range,
    /// which is always a safe (if not maximally sparse) over-approximation for
    /// any caller unioning domains. GenericConditional is the TestFunc type
    /// behind the Python-facing `ifelse()` DSL and behind AND/OR combinations
    /// (`ConditionalStatement<GenericConditional<IR>, GenericConditional<IR>>`),
    /// so both IfElseFunction's runtime domain union and ConditionalStatement's
    /// own `input_domain()` depend on this (VF_REVIEW 1.11).
    /// @return The full input-row range as a single-column domain matrix.
    [[nodiscard]] DomainMatrix input_domain() const {
        DomainMatrix d(2, 1);
        d(0, 0) = 0;
        d(1, 0) = this->input_rows();
        return d;
    }

    /// @brief Evaluate the wrapped predicate at @p x.
    /// @tparam InTypeT  Eigen expression type of the input vector.
    /// @param x  Input vector.
    /// @return Boolean result of the predicate.
    template <class InTypeT> bool compute(const Eigen::MatrixBase<InTypeT> &x) const {
        if (storage.empty()) {
            throw std::runtime_error(
                "GenericConditional is empty (default-constructed); assign a predicate before use");
        }
        InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

} // namespace tycho::vf
