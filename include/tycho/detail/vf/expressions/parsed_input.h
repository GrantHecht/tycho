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
    DENSE_FUNCTION_BASE_TYPES(Base);

    SUB_FUNCTION_IO_TYPES(Func);
    Func func;
    Func_Input<int> varlocs;
    bool contiguous = false;

    ParsedInput() {}
    ParsedInput(Func f, const Func_Input<int> &varlocs, int irr)
        : func(std::move(f)), varlocs(varlocs) {
        this->set_io_rows(irr, this->func.output_rows());

        this->contiguous = true;
        for (int i = 0; i < varlocs.size() - 1; i++) {
            int delta = varlocs[i + 1] - varlocs[i];
            if (delta != 1)
                this->contiguous = false;
        }
        // this->set_input_domain(irr, func.input_domain());
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;

        if (this->contiguous) {
            this->func.compute(x.segment(varlocs[0], this->func.input_rows()), fx_);

        } else {
            Func_Input<Scalar> xin(this->func.input_rows());
            for (int i = 0; i < this->func.input_rows(); i++) {
                xin[i] = x[this->varlocs[i]];
            }
            this->func.compute(xin, fx_);
        }
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();

        using Scalar = typename InType::Scalar;

        if (this->contiguous) {
            this->func.compute_jacobian(x.segment(varlocs[0], this->func.input_rows()), fx_,
                                        jx.middleCols(varlocs[0], this->func.input_rows()));
        } else {
            Func_Input<Scalar> xin(this->func.input_rows());
            Func_jacobian<Scalar> jxin(this->func.output_rows(), this->func.input_rows());
            jxin.setZero();
            for (int i = 0; i < this->func.input_rows(); i++) {
                xin[i] = x[this->varlocs[i]];
            }
            this->func.compute_jacobian(xin, fx_, jxin);
            for (int i = 0; i < this->func.input_rows(); i++) {
                jx.col(this->varlocs[i]) = jxin.col(i);
            }
        }
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        using Scalar = typename InType::Scalar;

        if (this->contiguous) {
            this->func.compute_jacobian_adjointgradient_adjointhessian(
                x.segment(varlocs[0], this->func.input_rows()), fx_,
                jx.middleCols(varlocs[0], this->func.input_rows()),
                adjgrad.segment(varlocs[0], this->func.input_rows()),
                adjhess.block(varlocs[0], varlocs[0], this->func.input_rows(), this->func.input_rows()),
                adjvars);

        } else {
            Func_Input<Scalar> xin(this->func.input_rows());
            Func_jacobian<Scalar> jxin(this->func.output_rows(), this->func.input_rows());
            Func_gradient<Scalar> gxin(this->func.input_rows());
            Func_hessian<Scalar> hxin(this->func.input_rows(), this->func.input_rows());
            jxin.setZero();
            hxin.setZero();
            gxin.setZero();

            for (int i = 0; i < this->func.input_rows(); i++) {
                xin[i] = x[this->varlocs[i]];
            }

            this->func.compute_jacobian_adjointgradient_adjointhessian(xin, fx_, jxin, gxin, hxin,
                                                                       adjvars);

            for (int i = 0; i < this->func.input_rows(); i++) {
                jx.col(this->varlocs[i]) = jxin.col(i);
                adjgrad[this->varlocs[i]] = gxin[i];
            }
            for (int i = 0; i < this->func.input_rows(); i++) {
                for (int j = 0; j < this->func.input_rows(); j++) {
                    adjhess(this->varlocs[j], this->varlocs[i]) = hxin(j, i);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
