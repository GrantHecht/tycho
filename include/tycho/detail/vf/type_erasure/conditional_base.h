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

#include "tycho/detail/eigen_types.h"
#include "tycho/detail/std_extensions.h"
#include "tycho/detail/math_functions.h"
#include "tycho/detail/type_name.h"
#include "tycho/detail/type_storage.h"
#include "tycho/detail/sizing_helpers.h"
#include "tycho/detail/thread_pool.h"
#include "tycho/detail/flat_map.h"
#include "tycho/detail/function_return_type.h"
#include "tycho/detail/get_core_count.h"
#include "tycho/detail/crtp_base.h"

namespace Tycho {

template <int IR> struct ConditionalBase {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    virtual ~ConditionalBase() = default;
    virtual int IRows() const = 0;
    virtual bool compute(const Eigen::MatrixBase<InType> &x) const = 0;
    virtual void clone_into(TypeStorage<ConditionalBase<IR>> &s) const = 0;
};

template <int IR, typename T> struct ConditionalModel final : ConditionalBase<IR> {
    using InType = typename ConditionalBase<IR>::InType;

    T data_;
    explicit ConditionalModel(T t) : data_(std::move(t)) {}

    int IRows() const override { return data_.IRows(); }
    bool compute(const Eigen::MatrixBase<InType> &x) const override { return data_.compute(x); }
    void clone_into(TypeStorage<ConditionalBase<IR>> &s) const override {
        s.template emplace<ConditionalModel<IR, T>>(data_);
    }
};

} // namespace Tycho
