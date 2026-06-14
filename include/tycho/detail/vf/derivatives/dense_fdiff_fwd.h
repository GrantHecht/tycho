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

#include "tycho/detail/vf/derivatives/dense_derivatives.h"

namespace tycho::vf {

/// @brief Forward finite-difference Jacobian specialization of @ref DenseFirstDerivatives.
///
/// Approximates the Jacobian columnwise with a first-order forward difference
/// @f$ \partial f / \partial x_i \approx (f(x + h\,e_i) - f(x)) / h @f$, requiring
/// @f$ \mathrm{IR}+1 @f$ primal evaluations per Jacobian.
/// @tparam Derived  The concrete VectorFunction type (CRTP self type).
/// @tparam IR       Input dimension (rows), or `Eigen::Dynamic`.
/// @tparam OR       Output dimension (rows), or `Eigen::Dynamic`.
/// @ingroup vf
template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::FDiffFwd>
    : DenseFunctionBase<Derived, IR, OR> {
    /// @brief The dense function base providing the primal `compute` interface.
    using Base = DenseFunctionBase<Derived, IR, OR>;
    VF_TYPE_ALIASES(Base)

    /// @brief Constructs the mode with a default Jacobian step size of 1e-7.
    DenseFirstDerivatives() { this->set_jac_fd_steps(1.0e-7); }

    /// @brief Sets a per-input-dimension Jacobian finite-difference step size.
    /// @param steps  Step size for each input dimension.
    void set_jac_fd_steps(const Input<double> &steps) { this->jac_fd_steps = steps; }
    /// @brief Sets a single Jacobian finite-difference step size for all input dimensions.
    /// @param step  Step size applied uniformly to every input dimension.
    void set_jac_fd_steps(double step) {
        this->jac_fd_steps.resize(this->input_rows());
        this->jac_fd_steps.setConstant(step);
    }

    /// @brief Computes the function value and forward finite-difference Jacobian.
    /// @internal
    /// Stores the primal in @p fx_ and the Jacobian in @p jx_; requires
    /// @f$ \mathrm{IR}+1 @f$ calls to the underlying function.
    /// @tparam InType   Eigen type of the input vector @p x.
    /// @tparam OutType  Eigen type of the output vector @p fx_.
    /// @tparam JacType  Eigen type of the Jacobian matrix @p jx_.
    /// @param x    Input vector at which to evaluate.
    /// @param fx_  Output function value, written in place.
    /// @param jx_  Output Jacobian, written in place.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        this->derived().compute(x, fx_);
        Input<Scalar> xi = x;
        Output<Scalar> fi(this->output_rows());
        fi.setZero();
        for (int i = 0; i < this->input_rows(); i++) {
            xi[i] += this->jac_fd_steps[i];
            this->derived().compute(xi, fi);

            jx.col(i) = (fi - fx) / Scalar(this->jac_fd_steps[i]);

            fi.setZero();
            xi[i] = x[i];
        }
    }

  protected:
    /// @brief Per-input-dimension Jacobian finite-difference step sizes.
    Eigen::VectorXd jac_fd_steps;
};

////////////////////////////////////////////////////////////////////////////////

/// @brief Forward finite-difference Hessian specialization of @ref DenseSecondDerivatives.
///
/// Forms the adjoint Hessian by forward-differencing the adjoint gradient
/// @f$ g(x) = J(x)^\top \lambda @f$ columnwise, then symmetrizing. The Jacobian is
/// supplied by the chosen @p JMode first-derivative layer.
/// @tparam Derived  The concrete VectorFunction type (CRTP self type).
/// @tparam IR       Input dimension (rows), or `Eigen::Dynamic`.
/// @tparam OR       Output dimension (rows), or `Eigen::Dynamic`.
/// @tparam JMode    Jacobian-evaluation mode used for the first-derivative layer.
/// @ingroup vf
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::FDiffFwd>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    /// @brief The first-derivative layer providing the Jacobian interface.
    using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
    VF_TYPE_ALIASES(Base)

    /// @brief Constructs the mode with a default Hessian step size of 1e-7.
    DenseSecondDerivatives() { this->set_hess_fd_steps(1.0e-7); }
    using Base::adjointhessian;
    /// @brief Sets a per-input-dimension Hessian finite-difference step size.
    /// @param steps  Step size for each input dimension.
    void set_hess_fd_steps(const Input<double> &steps) { this->hess_fd_steps = steps; }
    /// @brief Sets a single Hessian finite-difference step size for all input dimensions.
    /// @param step  Step size applied uniformly to every input dimension.
    void set_hess_fd_steps(double step) {
        this->hess_fd_steps = Input<double>::Constant(this->input_rows(), step);
    }

    /// @brief Computes the adjoint Hessian by finite-differencing the adjoint gradient.
    ///
    /// Differentiates the adjoint gradient @f$ J^\top \lambda @f$ columnwise and
    /// symmetrizes the result.
    /// @tparam InType       Eigen type of the input vector @p x.
    /// @tparam AdjHessType  Eigen type of the output adjoint Hessian @p adjhess_.
    /// @tparam AdjVarType   Eigen type of the adjoint coefficient vector @p adjvars.
    /// @param x        Input vector at which to evaluate.
    /// @param adjhess_ Output adjoint Hessian, written in place.
    /// @param adjvars  Adjoint (Lagrange) coefficients weighting each output row.
    template <class InType, class AdjHessType, class AdjVarType>
    inline void adjointhessian(CVecRef<InType> x, CMatRef<AdjHessType> adjhess_,
                               CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Gradient<Scalar> ag(this->input_rows());
        Gradient<Scalar> agi(this->input_rows());
        ag.setZero();
        agi.setZero();
        this->adjointgradient(x, ag, adjvars);

        Input<Scalar> xi = x;
        for (int i = 0; i < this->input_rows(); i++) {
            // if (this->VariableTypes[i] == VarTypes::Linear) {
            if (false) {
                for (int j = 0; j < this->input_rows(); j++) {
                    adjhess(j, i) = Scalar(0.0);
                }
            } else {
                xi[i] += this->hess_fd_steps[i];
                this->adjointgradient(xi, agi, adjvars);
                for (int j = 0; j < this->input_rows(); j++) {
                    adjhess(j, i) = (agi[j] - ag[j]) / Scalar(this->hess_fd_steps[i]);
                }
                agi.setZero();
                xi[i] = x[i];
            }
        }
        adjhess = (adjhess + adjhess.transpose()).eval() * Scalar(0.5);
    }

    /// @brief Computes value, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @internal
    /// Delegates value/Jacobian/adjoint-gradient to the base function, then forms the
    /// adjoint Hessian via @ref adjointhessian.
    /// @tparam InType       Eigen type of the input vector @p x.
    /// @tparam OutType      Eigen type of the output value @p fx_.
    /// @tparam JacType      Eigen type of the Jacobian @p jx_.
    /// @tparam AdjGradType  Eigen type of the adjoint gradient @p adjgrad_.
    /// @tparam AdjHessType  Eigen type of the adjoint Hessian @p adjhess_.
    /// @tparam AdjVarType   Eigen type of the adjoint coefficients @p adjvars.
    /// @param x        Input vector at which to evaluate.
    /// @param fx_      Output value, written in place.
    /// @param jx_      Output Jacobian, written in place.
    /// @param adjgrad_ Output adjoint gradient, written in place.
    /// @param adjhess_ Output adjoint Hessian, written in place.
    /// @param adjvars  Adjoint (Lagrange) coefficients weighting each output row.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        this->derived().compute_jacobian_adjointgradient(x, fx_, jx_, adjgrad_, adjvars);
        adjointhessian(x, adjhess_, adjvars);
    }

  protected:
    /// @brief Per-input-dimension Hessian finite-difference step sizes.
    Input<double> hess_fd_steps;
};

} // namespace tycho::vf
