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
using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::MatRef;
using vf::ThreadingFlags;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

/// @ingroup optimal_control
/// @brief VectorFunction that interpolates selected variables of a trajectory table.
///
/// Maps a scalar time input to the interpolated values of selected
/// state/time/control variables read from an @ref LGLInterpTable, with analytic
/// first and second derivatives. Used to inject a trajectory's control or state
/// history into expressions (e.g. control interpolants in shooting).
/// @tparam OR  Compile-time output dimension (number of variables interpolated).
template <int OR>
struct InterpFunction : VectorFunction<InterpFunction<OR>, 1, OR, DenseDerivativeMode::Analytic,
                                       DenseDerivativeMode::Analytic> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<InterpFunction<OR>, 1, OR, DenseDerivativeMode::Analytic,
                                DenseDerivativeMode::Analytic>;
    VF_TYPE_ALIASES(Base);

    static constexpr int TempSize = SZ_SUM<OR, 1>::value; ///< Scratch-buffer size hint.
    std::shared_ptr<LGLInterpTable> table;                ///< The trajectory table to interpolate.
    Eigen::VectorXi vars; ///< Indices of the table variables to output.
    /// @brief Default constructor; leaves the table and variables unset.
    InterpFunction() {}

    /// @brief Construct interpolating an explicit set of table variables.
    /// @param tab  The trajectory table to interpolate.
    /// @param v    Indices of the table variables to output.
    /// @throws std::invalid_argument if any index exceeds the table's dimensions.
    InterpFunction(std::shared_ptr<LGLInterpTable> tab, Eigen::VectorXi v) {
        this->table = tab;
        this->vars = v;

        if (v.maxCoeff() + 1 > tab->xtu_vars_) {
            throw std::invalid_argument("Interpolation table has incorrect dimensions");
        }

        this->set_io_rows(1, this->vars.size());
        // this->setHessFDSteps(tab->delta_t_/10.0);
    }
    /// @brief Construct interpolating all state variables of the table.
    /// @param tab  The trajectory table to interpolate.
    /// @throws std::invalid_argument if the table's state size != @p OR or it has controls.
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

    /// @internal
    /// @brief Interpolate the selected variables at the input time.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Input time (1-vector).
    /// @param fx_  Output interpolated-values vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

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

    /// @internal
    /// @brief Interpolate the selected variables and their time derivative (Jacobian).
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Input time (1-vector).
    /// @param fx_  Output interpolated-values vector to write.
    /// @param jx_  Output Jacobian (d/dt) to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

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

    /// @internal
    /// @brief Interpolate values, time derivative, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Input time (1-vector).
    /// @param fx_      Output interpolated-values vector to write.
    /// @param jx_      Output Jacobian (d/dt) to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;

        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        MatRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> hx = adjhess_.const_cast_derived();

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
