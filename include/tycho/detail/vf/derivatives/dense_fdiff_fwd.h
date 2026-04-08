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

//! First derivatives using forward finite difference
/*!
  \tparam IR Input Rows
  \tparam OR Output Rows
*/
template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::FDiffFwd>
    : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
    VF_TYPE_ALIASES(Base)

    DenseFirstDerivatives() { this->set_jac_fd_steps(1.0e-7); }

    //! Set step size for each input dimension
    void set_jac_fd_steps(const Input<double> &steps) { this->jac_fd_steps = steps; }
    //! Set step size for all input dimensions
    void set_jac_fd_steps(double step) {
        this->jac_fd_steps.resize(this->input_rows());
        this->jac_fd_steps.setConstant(step);
    }

    //! Jacobian implementation
    /*!
      \tparam InType Eigen type of input x
      \tparam OutType Eigen type of output fx
      \tparam JacType Eigen type of jacobian jx
      \param x const reference to input vector
      \param fx_ const reference to output vector
      \param jx_ const reference to jacobian matrix
      Calculates function and jacobian and stores them in fx_ and jx_. Requires
      (IR+1) function calls.
    */
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
    Eigen::VectorXd jac_fd_steps;
};

////////////////////////////////////////////////////////////////////////////////

//! Second derivatives using forward finite difference
/*!
  \tparam IR Input Rows
  \tparam OR Output Rows
  \tparam JMode Jacobian Mode (enumerator)
*/
template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::FDiffFwd>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
    VF_TYPE_ALIASES(Base)

    DenseSecondDerivatives() { this->set_hess_fd_steps(1.0e-7); }
    using Base::adjointhessian;
    //! Set step size for each input dimension
    void set_hess_fd_steps(const Input<double> &steps) { this->hess_fd_steps = steps; }
    //! Set step size for all input dimensions
    void set_hess_fd_steps(double step) {
        this->hess_fd_steps = Input<double>::Constant(this->input_rows(), step);
    }

    //! Adjoint hessian implementation
    /*!
      \tparam InType Eigen type of input x
      \tparam AdjHessType Eigen type of output adjoint hessian matrix
      \tparam AdjVarType Eigen type of adjoint coefficient vector
      \param x const reference to input vector
      \param adjhess_ const reference to adjoint hessian matrix
      \param adjvars const reference to adjoint coefficient vector
      Calculates adjoint hessian matrix by taking the derivative of the adjoint
      gradient vector.
    */
    template <class InType, class AdjHessType, class AdjVarType>
    inline void adjointhessian(CVecRef<InType> x,
                               CMatRef<AdjHessType> adjhess_,
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

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
        this->derived().compute_jacobian_adjointgradient(x, fx_, jx_, adjgrad_, adjvars);
        adjointhessian(x, adjhess_, adjvars);
    }

  protected:
    Input<double> hess_fd_steps;
};

} // namespace tycho::vf
