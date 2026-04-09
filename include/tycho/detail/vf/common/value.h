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

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <int OSZ> struct Value : VectorFunction<Value<OSZ>, OSZ, OSZ> {
    using Base = VectorFunction<Value<OSZ>, OSZ, OSZ>;
    VF_TYPE_ALIASES(Base)

    template <class Func> using REARGUMENT = Value<OSZ>;

    typename Base::template Output<double> value_;

    Value() { this->value_.setOnes(); }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value_;
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        fx = this->value_;
    }

    template <class Func, int FuncIRC, int ii>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, ii> &f) const {
        return Value<OSZ>();
    }
};

} // namespace tycho::vf
