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
#include "tycho/detail/vf/derivatives/fd_step_utils.h"

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

    /// @brief Default forward-difference Jacobian step (relative; see compute_jacobian_impl).
    static constexpr double kDefaultJacStep = 1.0e-7;

    /// @brief Constructs the mode with the default Jacobian step size.
    DenseFirstDerivatives() { this->set_jac_fd_steps(kDefaultJacStep); }

    /// @brief Set I/O rows and keep the FD step vector sized to the input count.
    ///
    /// The mode constructor runs before a dynamic-size function knows its
    /// sizes, so the step vector starts empty; re-applying the default here
    /// removes the old requirement that every dynamic-size wrapper manually
    /// re-set the steps after set_io_rows.
    /// @param inputrows   Number of input rows.
    /// @param outputrows  Number of output rows.
    void set_io_rows(int inputrows, int outputrows) {
        Base::set_io_rows(inputrows, outputrows);
        if (this->jac_fd_steps.size() != inputrows) {
            this->set_jac_fd_steps(kDefaultJacStep);
        }
    }

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
            const Scalar h = Scalar(this->jac_fd_steps[i]) * detail::fd_step_scale(x[i]);
            xi[i] += h;
            // Realized step (x+h)-x absorbs the rounding of the perturbed point.
            const Scalar hr = xi[i] - x[i];
            this->derived().compute(xi, fi);

            jx.col(i) = (fi - fx) / hr;

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

    /// @brief Default Hessian FD step.
    ///
    /// When the gradient being differenced is itself finite-differenced
    /// (nested FD, i.e. @p JMode is also an FD mode — forward or central), an
    /// eps^(1/3)-scale step (5e-6) is the standard choice — differencing an
    /// already-noisy FD gradient with too fine a step amplifies that noise.
    /// Otherwise (an analytic or Enzyme Jacobian feeding the Hessian FD), the
    /// finer sqrt-eps-scale step (1e-7) is appropriate.
    static constexpr double kDefaultHessStep =
        (JMode == DenseDerivativeMode::FDiffFwd || JMode == DenseDerivativeMode::FDiffCentArray)
            ? 5.0e-6
            : 1.0e-7;

    /// @brief Constructs the mode with the default Hessian step size.
    DenseSecondDerivatives() { this->set_hess_fd_steps(kDefaultHessStep); }

    /// @brief Set I/O rows and keep the Hessian FD step vector sized to the input count.
    ///
    /// Chains to @ref Base::set_io_rows (the first-derivative layer), which
    /// re-applies its own Jacobian step default; this layer additionally
    /// re-applies the Hessian step default so dynamic-size functions never
    /// read a stale (or empty) Hessian step vector.
    /// @param inputrows   Number of input rows.
    /// @param outputrows  Number of output rows.
    void set_io_rows(int inputrows, int outputrows) {
        Base::set_io_rows(inputrows, outputrows);
        if (this->hess_fd_steps.size() != inputrows) {
            this->set_hess_fd_steps(kDefaultHessStep);
        }
    }

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
        Gradient<Scalar> ag(this->input_rows());
        ag.setZero();
        this->adjointgradient(x, ag, adjvars);
        adjointhessian_from_gradient(x, adjhess_, adjvars, ag);
    }

    /// @brief Computes value, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @internal
    /// Delegates value/Jacobian/adjoint-gradient to the base function, then forms the
    /// adjoint Hessian via `adjointhessian_from_gradient()`, reusing the base-point
    /// adjoint gradient just computed by `compute_jacobian_adjointgradient` instead of
    /// recomputing it (which would cost another IR+1 primal evaluations under FD Jacobian
    /// modes — see `adjointhessian_from_gradient()`).
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
        adjointhessian_from_gradient(x, adjhess_, adjvars, adjgrad_);
    }

  private:
    /// @brief Forms the adjoint Hessian by finite-differencing the adjoint gradient,
    /// given the already-computed base-point adjoint gradient @p ag0.
    ///
    /// Shared by @ref adjointhessian (which computes @p ag0 itself) and
    /// @ref compute_jacobian_adjointgradient_adjointhessian_impl (which passes through
    /// the gradient its caller already computed), eliminating one redundant base-point
    /// adjoint-gradient evaluation (IR+1 primal evals when the Jacobian mode is FD) per
    /// combined Jacobian/adjoint-gradient/adjoint-Hessian call.
    /// @tparam InType       Eigen type of the input vector @p x.
    /// @tparam AdjHessType  Eigen type of the output adjoint Hessian @p adjhess_.
    /// @tparam AdjVarType   Eigen type of the adjoint coefficient vector @p adjvars.
    /// @tparam AdjGradType  Eigen type of the base-point adjoint gradient @p ag0.
    /// @param x        Input vector at which to evaluate.
    /// @param adjhess_ Output adjoint Hessian, written in place.
    /// @param adjvars  Adjoint (Lagrange) coefficients weighting each output row.
    /// @param ag0      Base-point adjoint gradient @f$ J(x)^\top \lambda @f$, already
    ///                 evaluated at @p x with the same @p adjvars.
    /// @pre @p ag0 must not alias @p adjhess_ (the Hessian is written while @p ag0 is
    ///      still being read column-by-column).
    template <class InType, class AdjHessType, class AdjVarType, class AdjGradType>
    inline void adjointhessian_from_gradient(CVecRef<InType> x, CMatRef<AdjHessType> adjhess_,
                                             CVecRef<AdjVarType> adjvars,
                                             CVecRef<AdjGradType> ag0) const {
        typedef typename InType::Scalar Scalar;
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        Gradient<Scalar> agi(this->input_rows());
        agi.setZero();

        Input<Scalar> xi = x;
        for (int i = 0; i < this->input_rows(); i++) {
            const Scalar h = Scalar(this->hess_fd_steps[i]) * detail::fd_step_scale(x[i]);
            xi[i] += h;
            const Scalar hr = xi[i] - x[i];
            this->adjointgradient(xi, agi, adjvars);
            for (int j = 0; j < this->input_rows(); j++) {
                adjhess(j, i) = (agi[j] - ag0[j]) / hr;
            }
            agi.setZero();
            xi[i] = x[i];
        }
        adjhess = (adjhess + adjhess.transpose()).eval() * Scalar(0.5);
    }

  protected:
    /// @brief Per-input-dimension Hessian finite-difference step sizes.
    Input<double> hess_fd_steps;
};

} // namespace tycho::vf
