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

/// @file
/// @brief VectorFunction wrapper applying per-component input and output scales.
#pragma once

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @ingroup vf
/// @brief Scales both the input and output of a wrapped function elementwise.
///
/// On evaluation each input component is multiplied by the matching
/// @ref input_scales factor before being passed to the wrapped function, and
/// each result component is multiplied by the matching @ref output_scales
/// factor. Derivatives are transformed consistently: Jacobian rows by the
/// output scales and columns by the input scales, and the adjoint gradient and
/// Hessian by the corresponding factors.
/// @tparam Func  Wrapped VectorFunction type.
template <class Func>
struct IOScaled
    : VectorFunction<IOScaled<Func>, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base =
        VectorFunction<IOScaled<Func>, Func::IRC, Func::ORC, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);
    /// @brief The wrapped function whose input and output are scaled.
    Func func_;
    /// @brief Input-domain sparsity inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    /// @brief Linearity inherited from the wrapped function (scaling is linear).
    static constexpr bool is_linear_function = Func::is_linear_function;
    /// @brief Vectorizability inherited from the wrapped function.
    static constexpr bool is_vectorizable = Func::is_vectorizable;

    /// @brief Per-component input scale factors applied before evaluation.
    Input<double> input_scales;
    /// @brief Per-component output scale factors applied after evaluation.
    Output<double> output_scales;

    /// @brief Constructs an empty IO-scaled wrapper (configured later).
    IOScaled() {}

    /// @brief Constructs an IO-scaled wrapper around a function.
    /// @param f              Function to wrap.
    /// @param input_scales   Per-component input scale factors.
    /// @param output_scales  Per-component output scale factors.
    IOScaled(Func f, const Input<double> &input_scales, const Output<double> &output_scales)
        : func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {this->func_.input_domain()});

        this->input_scales = input_scales;
        this->output_scales = output_scales;

        this->enable_vectorization_ = this->func_.enable_vectorization_;
    }

    /// @internal
    /// @brief Evaluates the wrapped function on scaled input and scales its output.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param x    Input vector (scaled internally before evaluation).
    /// @param fx_  Output vector receiving the scaled result.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto &x_scaled) {
            for (int i = 0; i < this->input_rows(); i++) {
                x_scaled[i] = this->input_scales[i] * x[i];
            }

            this->func_.compute(x_scaled, fx);

            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] *= this->output_scales[i];
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<Scalar>>(this->input_rows(), 1));
    }
    /// @internal
    /// @brief Evaluates the scaled value and the correspondingly scaled Jacobian.
    ///
    /// Jacobian rows are scaled by the output scales and columns by the input
    /// scales.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param x    Input vector (scaled internally before evaluation).
    /// @param fx_  Output vector receiving the scaled result.
    /// @param jx_  Jacobian buffer receiving the row/column-scaled Jacobian.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {

        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &x_scaled) {
            for (int i = 0; i < this->input_rows(); i++) {
                x_scaled[i] = this->input_scales[i] * x[i];
            }

            this->func_.compute_jacobian(x_scaled, fx, jx);

            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] *= this->output_scales[i];
                jx.row(i) *= Scalar(this->output_scales[i]);
            }
            for (int i = 0; i < this->input_rows(); i++) {
                jx.col(i) *= Scalar(this->input_scales[i]);
            }
        };

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<Scalar>>(this->input_rows(), 1));
    }

    /// @internal
    /// @brief Evaluates the scaled value, Jacobian, adjoint gradient, and Hessian.
    ///
    /// The adjoint variables are pre-scaled by the output scales; the resulting
    /// adjoint gradient and Hessian are then scaled by the input scales.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param x        Input vector (scaled internally before evaluation).
    /// @param fx_      Output vector receiving the scaled result.
    /// @param jx_      Jacobian buffer receiving the row/column-scaled Jacobian.
    /// @param adjgrad_ Adjoint-gradient buffer receiving the input-scaled gradient.
    /// @param adjhess_ Adjoint-Hessian buffer receiving the input-scaled Hessian.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
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

        auto Impl = [&](auto &x_scaled, auto &l_scaled) {
            for (int i = 0; i < this->input_rows(); i++) {
                x_scaled[i] = this->input_scales[i] * x[i];
            }
            for (int i = 0; i < this->output_rows(); i++) {
                l_scaled[i] = this->output_scales[i] * adjvars[i];
            }
            this->func_.compute_jacobian_adjointgradient_adjointhessian(x_scaled, fx, jx, adjgrad,
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

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<Input<Scalar>>(this->input_rows(), 1),
            tycho::utils::TempSpec<Output<Scalar>>(this->output_rows(), 1));
    }
};

} // namespace tycho::vf