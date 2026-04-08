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

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include "tycho/detail/vf/derivatives/dense_derivatives.h"

namespace tycho::vf {

template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::AutodiffFwd>
    : DenseFunctionBase<Derived, IR, OR> {
    using Base = DenseFunctionBase<Derived, IR, OR>;
    VF_TYPE_ALIASES(Base);

    template <class Scalar> using dual = autodiff::detail::HigherOrderDual<1U, Scalar>;

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        Input<dual<Scalar>> xdual = x.template cast<dual<Scalar>>();
        Output<dual<Scalar>> fdual(this->output_rows());
        fdual.setZero();

        for (int i = 0; i < this->input_rows(); i++) {
            xdual[i].grad = Scalar(1.0);
            this->derived().compute(xdual, fdual);

            for (int j = 0; j < this->output_rows(); j++) {
                jx(j, i) = fdual[j].grad;
            }

            if (i == 0) {
                for (int j = 0; j < this->output_rows(); j++) {
                    fx[j] = fdual[j].val;
                }
            }

            fdual.setZero();
            xdual[i].grad = Scalar(0.0);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::AutodiffFwd>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
    VF_TYPE_ALIASES(Base);
    // using Base::adjointhessian;

    template <class Scalar> using dual = autodiff::detail::HigherOrderDual<2U, Scalar>;

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {

        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> gx = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> hx = adjhess_.const_cast_derived();

        Input<dual<Scalar>> xdual = x.template cast<dual<Scalar>>();
        Output<dual<Scalar>> fdual(this->output_rows());

        for (int i = 0; i < this->input_rows(); i++) {

            xdual[i].grad.val = 1.0;
            dual<Scalar> vv;

            for (int j = i; j < this->input_rows(); j++) {

                fdual.setZero();

                xdual[j].val.grad = 1.0;

                this->derived().compute(xdual, fdual);

                xdual[j].val.grad = 0.0;

                vv = 0.0;
                for (int k = 0; k < this->output_rows(); k++) {

                    vv += fdual[k] * adjvars[k];
                }

                hx(i, j) = vv.grad.grad;
                hx(j, i) = hx(i, j);
            }
            gx[i] = vv.grad.val;
            for (int k = 0; k < this->output_rows(); k++) {
                jx(k, i) = fdual[k].grad.val;
            }
            if (i == 0) {
                for (int j = 0; j < this->output_rows(); j++) {
                    fx[j] = fdual[j].val.val;
                }
            }

            xdual[i].grad.val = 0.0;
        }
    }
};

} // namespace tycho::vf
