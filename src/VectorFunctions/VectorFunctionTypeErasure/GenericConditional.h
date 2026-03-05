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
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"
#include "pch.h"

namespace Tycho {

template <int IR> struct ConditionalSpec {

    template <class Scalar> using Input = Eigen::Matrix<Scalar, IR, 1>;
    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;
    using InType = Eigen::Ref<const Input<double>>;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Concept { // abstract base class for model.
        virtual ~Concept() = default;
        // Your (internal) interface goes here.
        virtual std::string name() const = 0;
        virtual int IRows() const = 0;
        virtual bool compute(ConstVectorBaseRef<InType> x) const = 0;
    };
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Holder> struct Model : public Holder, public virtual Concept {
        using Holder::Holder;
        // Pass through to encapsulated value.
        virtual std::string name() const override { return rubber_types::model_get(this).name(); }
        virtual int IRows() const override { return rubber_types::model_get(*this).IRows(); }

        virtual bool compute(ConstVectorBaseRef<InType> x) const override {
            return rubber_types::model_get(this).compute(x);
        }
    };
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Container> struct ExternalInterface : public Container {
        using Container::Container;
        static const bool IsConditional = true;
        static const bool IRC = IR;

        // Define the external interface. Should match encapsulated type.
        std::string name() const { return rubber_types::interface_get(this).name(); }
        int IRows() const { return rubber_types::interface_get(*this).IRows(); }

        template <class InTypeT> bool compute(ConstVectorBaseRef<InTypeT> x) const {
            InType xt(x.derived());
            return rubber_types::interface_get(this).compute(xt);
        }
    };
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

template <int IR> struct GenericConditional : rubber_types::TypeErasure<ConditionalSpec<IR>> {
    using Base = rubber_types::TypeErasure<ConditionalSpec<IR>>;

    GenericConditional() {}
    template <class T> GenericConditional(const T &t) : Base(t) {}
    GenericConditional(const GenericConditional<IR> &obj) {
        this->reset_container(obj.get_container());
    }
};

} // namespace Tycho