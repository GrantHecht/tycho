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
struct IOScaled
    : VectorFunction<IOScaled<Func>, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    using Base =
        VectorFunction<IOScaled<Func>, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static const bool is_linear_function = Func::is_linear_function;
    static const bool is_vectorizable = Func::is_vectorizable;

    Input<double> input_scales;
    Output<double> output_scales;

    IOScaled() {}

    IOScaled(Func f, const Input<double> &input_scales, const Output<double> &output_scales)
        : func(std::move(f)) {
        this->set_io_rows(this->func.input_rows(), this->func.output_rows());
        this->set_input_domain(this->input_rows(), {this->func.input_domain()});

        this->input_scales = input_scales;
        this->output_scales = output_scales;

        this->EnableVectorization = this->func.EnableVectorization;
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &x_scaled) {
            for (int i = 0; i < this->input_rows(); i++) {
                x_scaled[i] = this->input_scales[i] * x[i];
            }

            this->func.compute(x_scaled, fx);

            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] *= this->output_scales[i];
            }
        };

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Input<Scalar>>(this->input_rows(), 1));
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {

        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &x_scaled) {
            for (int i = 0; i < this->input_rows(); i++) {
                x_scaled[i] = this->input_scales[i] * x[i];
            }

            this->func.compute_jacobian(x_scaled, fx, jx);

            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] *= this->output_scales[i];
                jx.row(i) *= Scalar(this->output_scales[i]);
            }
            for (int i = 0; i < this->input_rows(); i++) {
                jx.col(i) *= Scalar(this->input_scales[i]);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Input<Scalar>>(this->input_rows(), 1));
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

        auto Impl = [&](auto &x_scaled, auto &l_scaled) {
            for (int i = 0; i < this->input_rows(); i++) {
                x_scaled[i] = this->input_scales[i] * x[i];
            }
            for (int i = 0; i < this->output_rows(); i++) {
                l_scaled[i] = this->output_scales[i] * adjvars[i];
            }
            this->func.compute_jacobian_adjointgradient_adjointhessian(x_scaled, fx, jx, adjgrad,
                                                                       adjhess, l_scaled);

            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] *= this->output_scales[i];
                jx.row(i) *= Scalar(this->output_scales[i]);
            }
            for (int i = 0; i < this->input_rows(); i++) {
                jx.col(i) *= Scalar(this->input_scales[i]);
                adjhess.col(i) *= Scalar(this->input_scales[i]);
                adjgrad[i] *= this->input_scales[i];
            }

            for (int i = 0; i < this->input_rows(); i++) {
                adjhess.row(i) *= Scalar(this->input_scales[i]);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<Input<Scalar>>(this->input_rows(), 1),
                                    tycho::utils::TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
};

} // namespace tycho::vf