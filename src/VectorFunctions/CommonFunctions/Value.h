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

#include "VectorFunction.h"

namespace Tycho {

template <int OSZ> struct Value : VectorFunction<Value<OSZ>, OSZ, OSZ> {
    using Base = VectorFunction<Value<OSZ>, OSZ, OSZ>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    template <class Func> using REARGUMENT = Value<OSZ>;

    typename Base::template Output<double> value;

    Value() { this->value.setOnes(); }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value;
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        fx = this->value;
    }

    template <class Func, int FuncIRC, int ii>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, ii> &f) const {
        return Value<OSZ>();
    }
};

} // namespace Tycho
