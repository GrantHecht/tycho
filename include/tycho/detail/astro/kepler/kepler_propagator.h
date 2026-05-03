// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Reimplemented as a hand-written VF backed by kepler_lcd_iterate +
//     codegen'd primal/residual + IFT composition layer (no DSL).
// =============================================================================
#pragma once
#include "tycho/detail/astro/kepler/kepler_propagator_ift.h"
#include "tycho/vector_functions.h"

namespace tycho::astro {

using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::MatRef;
using vf::VecRef;
using vf::VectorFunction;

struct KeplerPropagator
    : VectorFunction<KeplerPropagator, 7, 6,
                     DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<KeplerPropagator, 7, 6,
                                DenseDerivativeMode::Analytic, DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    double mu_ = 1.0;
    static constexpr bool is_vectorizable = true;

    KeplerPropagator() { this->set_io_rows(7, 6); }
    explicit KeplerPropagator(double mu) : mu_(mu) { this->set_io_rows(7, 6); }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        detail::kepler_propagate<Scalar>(R0, V0, dt, mu_, out);
        fx = out;
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x,
                                      CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        Eigen::Matrix<Scalar, 6, 7> jac;
        detail::kepler_propagate_jacobian<Scalar>(R0, V0, dt, mu_, out, jac);
        fx = out;
        jx = jac;
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        using Scalar = typename InType::Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> ag = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> ah = adjhess_.const_cast_derived();
        Vector3<Scalar> R0 = x.template head<3>();
        Vector3<Scalar> V0 = x.template segment<3>(3);
        Scalar dt = x[6];
        Vector6<Scalar> out;
        Eigen::Matrix<Scalar, 6, 7> jac;
        Vector7<Scalar> grad;
        Eigen::Matrix<Scalar, 7, 7> hess;
        Vector6<Scalar> lm = adjvars;
        detail::kepler_propagate_adjoint_hessian<Scalar>(R0, V0, dt, mu_, lm, out, jac, grad, hess);
        fx = out;
        jx = jac;
        ag = grad;
        ah = hess;
    }
};

} // namespace tycho::astro
