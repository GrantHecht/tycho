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

#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/utils/lambda_jump_table.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::LambdaJumpTable;
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

template <int OR>
struct InterpFunction : VectorFunction<InterpFunction<OR>, 1, OR, DenseDerivativeMode::Analytic,
                                       DenseDerivativeMode::Analytic> {
    using Base = VectorFunction<InterpFunction<OR>, 1, OR, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    static constexpr int TempSize = SZ_SUM<OR, 1>::value;
    std::shared_ptr<LGLInterpTable> table;
    Eigen::VectorXi vars;
    InterpFunction() {}

    InterpFunction(std::shared_ptr<LGLInterpTable> tab, Eigen::VectorXi v) {
        this->table = tab;
        this->vars = v;

        if (v.maxCoeff() + 1 > tab->xtu_vars_) {
            throw std::invalid_argument("Interpolation table has incorrect dimensions");
        }

        this->set_io_rows(1, this->vars.size());
        // this->setHessFDSteps(tab->delta_t_/10.0);
    }
    InterpFunction(std::shared_ptr<LGLInterpTable> tab) {
        this->table = tab;
        if (this->table->x_vars_ != OR || this->table->u_vars_ > 0) {
            throw std::invalid_argument("Interpolation table has incorrect dimensions");
        }
        this->vars.resize(tab->x_vars_);
        for (int i = 0; i < this->vars.size(); i++)
            this->vars[i] = i;
        this->set_io_rows(1, this->vars.size());
        // this->setHessFDSteps(tab->delta_t_ / 10.0);
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        auto Impl = [&](auto maxsize) {
            Scalar t = x[0];
            Eigen::Matrix<Scalar, TempSize, 1, 0, maxsize.value, 1> state;
            state.resize(this->table->xtu_vars_);
            this->table->interpolate_ref(t, state);
            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] = state[this->vars[i]];
            }
        };

        if constexpr (OR > 0)
            Impl(std::integral_constant<int, TempSize>());
        else
            LambdaJumpTable<4, 8, 16>::run(Impl, this->table->xtu_vars_);
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto maxsize) {
            Scalar t = x[0];
            Eigen::Matrix<Scalar, TempSize, 2, 0, maxsize.value, 2> state;
            state.resize(this->table->xtu_vars_, 2);
            this->table->interpolate_deriv_ref(t, state);
            for (int i = 0; i < this->output_rows(); i++) {
                fx[i] = state(this->vars[i], 0);
                jx(i, 0) = state(this->vars[i], 1);
            }
        };

        if constexpr (OR > 0)
            Impl(std::integral_constant<int, TempSize>());
        else
            LambdaJumpTable<4, 8, 16>::run(Impl, this->table->xtu_vars_);
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;

        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        MatrixBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> hx = adjhess_.const_cast_derived();

        auto Impl = [&](auto maxsize) {
            if (this->table->method_ == TranscriptionModes::LGL3) {
                Scalar t = x[0];
                Eigen::Matrix<Scalar, TempSize, 3, 0, maxsize.value, 3> state;
                state.resize(this->table->xtu_vars_, 3);
                this->table->interpolate_2nd_deriv_ref(t, state);
                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i] = state(this->vars[i], 0);
                    jx(i, 0) = state(this->vars[i], 1);
                    adjgrad[0] += jx(i, 0) * adjvars[i];
                    hx(0, 0) += (state(this->vars[i], 2)) * adjvars[i];
                }
            } else {
                Scalar t = x[0];
                Eigen::Matrix<Scalar, TempSize, 2, 0, maxsize.value, 2> state;
                state.resize(this->table->xtu_vars_, 2);
                this->table->interpolate_deriv_ref(t, state);
                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i] = state(this->vars[i], 0);
                    jx(i, 0) = state(this->vars[i], 1);
                    adjgrad[0] += jx(i, 0) * adjvars[i];
                }
                state.setZero();
                Scalar h = Scalar(this->table->delta_t_ / 10.0);
                this->table->interpolate_deriv_ref(t + h, state);
                for (int i = 0; i < this->output_rows(); i++) {
                    hx(0, 0) += (state(this->vars[i], 1) - jx(i, 0)) * adjvars[i] / h;
                }
            }
        };

        if constexpr (OR > 0)
            Impl(std::integral_constant<int, TempSize>());
        else
            LambdaJumpTable<4, 8, 16>::run(Impl, this->table->xtu_vars_);
    }
};

} // namespace tycho::oc
