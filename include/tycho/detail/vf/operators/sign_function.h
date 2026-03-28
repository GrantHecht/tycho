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

namespace tycho::vf {

template <class Func>
struct SignFunction : VectorFunction<SignFunction<Func>, Func::IRC, Func::ORC> {
    using Base = VectorFunction<SignFunction<Func>, Func::IRC, Func::ORC>;

    Func func;
    DENSE_FUNCTION_BASE_TYPES(Base);

    // using INPUT_DOMAIN = SingleDomain<Func::IRC, 0, 0>;

    SignFunction() {}
    SignFunction(Func func) : func(std::move(func)) {
        this->set_io_rows(this->func.input_rows(), this->func.output_rows());
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = 0;
        this->set_input_domain(this->input_rows(), {dmn});
    }

    template <class OutType> void sign_impl(OutType &fx) const {
        typedef typename OutType::Scalar Scalar;
        if constexpr (Is_SuperScalar<Scalar>::value) {
            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] = fx[i].sign();
            }
        } else {
            fx = fx.array().sign();
        }
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        this->func.compute(x, fx);
        this->sign_impl(fx);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        this->func.compute(x, fx);
        this->sign_impl(fx);
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        this->func.compute(x, fx);
        this->sign_impl(fx);
    }
};

} // namespace tycho::vf