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
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace Tycho {

template <int IR, int OR> struct Constant : VectorFunction<Constant<IR, OR>, IR, OR> {
    using Base = VectorFunction<Constant<IR, OR>, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    static const bool IsLinearFunction = true;
    static const bool IsVectorizable = true;

    Output<double> value;

    Constant(int ir, Output<double> val) {
        this->setIORows(ir, val.size());
        value = val;

        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = 0;
        this->set_input_domain(this->IRows(), {dmn});
    }

    Constant() {}

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value.template cast<Scalar>();
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value.template cast<Scalar>();
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        fx = this->value.template cast<Scalar>();
    }
};

} // namespace Tycho
