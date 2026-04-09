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

struct ArcTan2Op : VectorFunction<ArcTan2Op, 2, 1, DenseDerivativeMode::Analytic,
                                  DenseDerivativeMode::Analytic> {

    using Base = VectorFunction<ArcTan2Op, 2, 1, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);
    static constexpr bool is_vectorizable = true;

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
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        Scalar yy = x[0];
        Scalar xx = x[1];
        fx[0] = calc_arc_tan2(yy, xx);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

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
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

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