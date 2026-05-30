// =============================================================================
// New file in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt)
//
// Defines ConditionalBase<IR>, ConditionalModel<IR,T>, and the
// ConditionalStorable concept for GenericConditional / GenericComparative.
// Replaces the rubber_types ConditionalSpec Model/ExternalInterface.
// =============================================================================

#pragma once

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

/// @internal
/// @brief Abstract type-erasure interface for conditional predicates.
///
/// Defines the virtual contract that backs GenericConditional / GenericComparative.
/// A concrete predicate is wrapped by ConditionalModel and stored via
/// tycho::utils::TypeStorage. Replaces the legacy rubber_types ConditionalSpec
/// Concept/ExternalInterface.
/// @tparam IR  Compile-time input-row count (-1 for dynamic).
/// @endinternal
template <int IR> struct ConditionalBase {
    /// @internal
    /// @brief Const reference to the predicate's fixed-size input vector.
    /// @endinternal
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    /// @internal
    /// @brief Virtual destructor for safe deletion through the base pointer.
    /// @endinternal
    virtual ~ConditionalBase() = default;
    /// @internal
    /// @brief Number of input rows the wrapped predicate accepts.
    /// @return Input-row count.
    /// @endinternal
    virtual int input_rows() const = 0;
    /// @internal
    /// @brief Evaluate the wrapped predicate at @p x.
    /// @param x  Input vector.
    /// @return Boolean result of the predicate.
    /// @endinternal
    virtual bool compute(const Eigen::MatrixBase<InType> &x) const = 0;
    /// @internal
    /// @brief Deep-copy this predicate into another TypeStorage slot.
    /// @param s  Destination storage to emplace the clone into.
    /// @endinternal
    virtual void clone_into(tycho::utils::TypeStorage<ConditionalBase<IR>> &s) const = 0;
};

/// @internal
/// @brief Concrete adapter holding a value-typed predicate @p T behind ConditionalBase.
/// @tparam IR  Compile-time input-row count (-1 for dynamic).
/// @tparam T   Wrapped predicate type (e.g. ConditionalStatement, ConstantConditional).
/// @endinternal
template <int IR, typename T> struct ConditionalModel final : ConditionalBase<IR> {
    /// @internal
    /// @brief Const reference to the predicate's fixed-size input vector.
    /// @endinternal
    using InType = typename ConditionalBase<IR>::InType;

    T data_; ///< @internal Owned predicate instance. @endinternal
    /// @internal
    /// @brief Construct by taking ownership of a predicate value.
    /// @param t  Predicate to move into the model.
    /// @endinternal
    explicit ConditionalModel(T t) : data_(std::move(t)) {}

    /// @internal
    /// @brief Forward to the wrapped predicate's input_rows().
    /// @return Input-row count.
    /// @endinternal
    int input_rows() const override { return data_.input_rows(); }
    /// @internal
    /// @brief Forward to the wrapped predicate's compute().
    /// @param x  Input vector.
    /// @return Boolean result of the predicate.
    /// @endinternal
    bool compute(const Eigen::MatrixBase<InType> &x) const override { return data_.compute(x); }
    /// @internal
    /// @brief Emplace a copy of this model into @p s.
    /// @param s  Destination storage.
    /// @endinternal
    void clone_into(tycho::utils::TypeStorage<ConditionalBase<IR>> &s) const override {
        s.template emplace<ConditionalModel<IR, T>>(data_);
    }
};

} // namespace tycho::vf
