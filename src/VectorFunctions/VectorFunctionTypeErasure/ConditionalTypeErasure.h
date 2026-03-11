// =============================================================================
// New file in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt)
//
// Defines ConditionalBase<IR>, ConditionalModel<IR,T>, and the
// ConditionalStorable concept for GenericConditional / GenericComparative.
// Replaces the rubber_types ConditionalSpec Model/ExternalInterface.
// =============================================================================

#pragma once

#include "pch.h"

namespace Tycho {

template <int IR> struct ConditionalBase {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    virtual ~ConditionalBase() = default;
    virtual std::string name() const = 0;
    virtual int IRows() const = 0;
    virtual bool compute(const Eigen::MatrixBase<InType> &x) const = 0;
    virtual void clone_into(TypeStorage<ConditionalBase<IR>> &s) const = 0;
};

template <int IR, typename T> struct ConditionalModel final : ConditionalBase<IR> {
    using InType = typename ConditionalBase<IR>::InType;

    T data_;
    explicit ConditionalModel(T t) : data_(std::move(t)) {}

    std::string name() const override { return data_.name(); }
    int IRows() const override { return data_.IRows(); }
    bool compute(const Eigen::MatrixBase<InType> &x) const override { return data_.compute(x); }
    void clone_into(TypeStorage<ConditionalBase<IR>> &s) const override {
        s.template emplace<ConditionalModel<IR, T>>(data_);
    }
};

} // namespace Tycho
