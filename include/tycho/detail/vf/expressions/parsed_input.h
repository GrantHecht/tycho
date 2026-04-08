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

template <class Func, int IRC, int ORC>
struct ParsedInput
    : VectorFunction<ParsedInput<Func, IRC, ORC>, IRC, ORC, DenseDerivativeMode::Analytic> {
    using Base =
        VectorFunction<ParsedInput<Func, IRC, ORC>, IRC, ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);

    Func func_;
    typename Func::template Input<int> varlocs_;
    bool contiguous_ = false;

    ParsedInput() {}
    ParsedInput(Func f, const typename Func::template Input<int> &vlocs, int irr)
        : func_(std::move(f)), varlocs_(vlocs) {
        this->set_io_rows(irr, this->func_.output_rows());

        this->contiguous_ = true;
        for (int i = 0; i < varlocs_.size() - 1; i++) {
            int delta = varlocs_[i + 1] - varlocs_[i];
            if (delta != 1)
                this->contiguous_ = false;
        }
        // this->set_input_domain(irr, func_.input_domain());
    }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;

        if (this->contiguous_) {
            this->func_.compute(x.segment(varlocs_[0], this->func_.input_rows()), fx_);

        } else {
            typename Func::template Input<Scalar> xin(this->func_.input_rows());
            for (int i = 0; i < this->func_.input_rows(); i++) {
                xin[i] = x[this->varlocs_[i]];
            }
            this->func_.compute(xin, fx_);
        }
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();

        using Scalar = typename InType::Scalar;

        if (this->contiguous_) {
            this->func_.compute_jacobian(x.segment(varlocs_[0], this->func_.input_rows()), fx_,
                                         jx.middleCols(varlocs_[0], this->func_.input_rows()));
        } else {
            typename Func::template Input<Scalar> xin(this->func_.input_rows());
            typename Func::template Jacobian<Scalar> jxin(this->func_.output_rows(), this->func_.input_rows());
            jxin.setZero();
            for (int i = 0; i < this->func_.input_rows(); i++) {
                xin[i] = x[this->varlocs_[i]];
            }
            this->func_.compute_jacobian(xin, fx_, jxin);
            for (int i = 0; i < this->func_.input_rows(); i++) {
                jx.col(this->varlocs_[i]) = jxin.col(i);
            }
        }
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        using Scalar = typename InType::Scalar;

        if (this->contiguous_) {
            this->func_.compute_jacobian_adjointgradient_adjointhessian(
                x.segment(varlocs_[0], this->func_.input_rows()), fx_,
                jx.middleCols(varlocs_[0], this->func_.input_rows()),
                adjgrad.segment(varlocs_[0], this->func_.input_rows()),
                adjhess.block(varlocs_[0], varlocs_[0], this->func_.input_rows(),
                              this->func_.input_rows()),
                adjvars);

        } else {
            typename Func::template Input<Scalar> xin(this->func_.input_rows());
            typename Func::template Jacobian<Scalar> jxin(this->func_.output_rows(), this->func_.input_rows());
            typename Func::template Gradient<Scalar> gxin(this->func_.input_rows());
            typename Func::template Hessian<Scalar> hxin(this->func_.input_rows(), this->func_.input_rows());
            jxin.setZero();
            hxin.setZero();
            gxin.setZero();

            for (int i = 0; i < this->func_.input_rows(); i++) {
                xin[i] = x[this->varlocs_[i]];
            }

            this->func_.compute_jacobian_adjointgradient_adjointhessian(xin, fx_, jxin, gxin, hxin,
                                                                        adjvars);

            for (int i = 0; i < this->func_.input_rows(); i++) {
                jx.col(this->varlocs_[i]) = jxin.col(i);
                adjgrad[this->varlocs_[i]] = gxin[i];
            }
            for (int i = 0; i < this->func_.input_rows(); i++) {
                for (int j = 0; j < this->func_.input_rows(); j++) {
                    adjhess(this->varlocs_[j], this->varlocs_[i]) = hxin(j, i);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
