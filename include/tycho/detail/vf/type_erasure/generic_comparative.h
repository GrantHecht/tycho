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
#include "tycho/detail/vf/type_erasure/generic_conditional.h"
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

/// @brief Type-erased boolean predicate, distinct from GenericConditional.
///
/// GenericComparative has the same interface and behaviour as GenericConditional
/// but is a separate type so the Python bindings can register "Comparative" and
/// "Conditional" as distinct nanobind classes. It wraps any value-typed predicate
/// behind a uniform interface; copies are deep.
/// @tparam IR  Compile-time input-row count (-1 for dynamic).
/// @ingroup vf
template <int IR> struct GenericComparative {
    /// @brief Const reference to the fixed-size input vector accepted by compute().
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    static constexpr bool is_conditional = true; ///< @brief Trait flag: this is a conditional.
    static constexpr int IRC = IR;               ///< @brief Compile-time input-row count.

    /// @brief Type-erased storage holding the wrapped predicate.
    tycho::utils::TypeStorage<ConditionalBase<IR>> storage;

    /// @brief Construct an empty predicate (no wrapped function).
    GenericComparative() = default;

    /// @brief Construct by type-erasing an arbitrary predicate @p t.
    /// @tparam T  Any type satisfying the conditional interface (input_rows, compute).
    /// @param t   Predicate to wrap and store.
    template <class T> GenericComparative(const T &t) {
        storage.template emplace<ConditionalModel<IR, std::decay_t<T>>>(t);
    }

    /// @brief Deep-copy constructor (clones the wrapped predicate).
    GenericComparative(const GenericComparative &) = default;
    /// @brief Move constructor.
    GenericComparative(GenericComparative &&) noexcept = default;
    /// @brief Deep-copy assignment (clones the wrapped predicate).
    /// @return Reference to this object.
    GenericComparative &operator=(const GenericComparative &) = default;
    /// @brief Move assignment.
    /// @return Reference to this object.
    GenericComparative &operator=(GenericComparative &&) noexcept = default;

    /// @brief Number of input rows the wrapped predicate accepts.
    /// @return Input-row count.
    /// @throws std::runtime_error if this GenericComparative is empty (default-constructed
    ///   and never assigned a predicate).
    int input_rows() const {
        if (storage.empty()) {
            throw std::runtime_error(
                "GenericComparative is empty (default-constructed); assign a predicate before "
                "use");
        }
        return storage.get().input_rows();
    }

    /// @brief Runtime read-set: conservatively the full input range.
    ///
    /// The wrapped predicate's structure (and therefore its true read-set) is
    /// erased behind @ref ConditionalBase, which has no `input_domain()` in its
    /// virtual interface — so this reports the full `[0, input_rows())` range,
    /// which is always a safe (if not maximally sparse) over-approximation for
    /// any caller unioning domains (e.g. IfElseFunction, VF_REVIEW 1.11).
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
    /// @throws std::runtime_error if this GenericComparative is empty (default-constructed
    ///   and never assigned a predicate).
    template <class InTypeT> bool compute(const Eigen::MatrixBase<InTypeT> &x) const {
        if (storage.empty()) {
            throw std::runtime_error(
                "GenericComparative is empty (default-constructed); assign a predicate before "
                "use");
        }
        InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

} // namespace tycho::vf
