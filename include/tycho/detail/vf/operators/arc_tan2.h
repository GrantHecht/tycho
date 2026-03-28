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

struct ArcTan2Op : VectorFunction<ArcTan2Op, 2, 1, DenseDerivativeMode::Analytic,
                                  DenseDerivativeMode::Analytic> {

    using Base = VectorFunction<ArcTan2Op, 2, 1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool is_vectorizable = true;

    template <class Scalar> static Scalar calc_arc_tan2(Scalar yy, Scalar xx) {
        Scalar fx;
        if constexpr (Is_SuperScalar<Scalar>::value) {
            for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
                fx[i] = atan2(yy[i], xx[i]);
            }
        } else {
            fx = atan2(yy, xx);
        }
        return fx;
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);

        Scalar denom = xx * xx + yy * yy;

        jx(0, 0) = xx / denom;
        jx(0, 1) = -yy / denom;
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

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);

        Scalar denom = xx * xx + yy * yy;

        jx(0, 0) = xx / denom;
        jx(0, 1) = -yy / denom;

        adjgrad[0] = adjvars[0] * xx / denom;
        adjgrad[1] = -adjvars[0] * yy / denom;

        adjhess(0, 0) = -2 * xx * yy / (denom * denom);
        adjhess(1, 1) = 2 * xx * yy / (denom * denom);

        adjhess(0, 1) = 1 / denom - 2 * xx * xx / (denom * denom);
        adjhess(1, 0) = -1 / denom + 2 * yy * yy / (denom * denom);

        adjhess *= adjvars[0];
    }
};

template <class YFunc, class XFunc> struct ArcTan2Impl {
    static auto Definition(YFunc yf, XFunc xf) { return ArcTan2Op().eval(StackedOutputs{yf, xf}); }
};

} // namespace tycho::vf