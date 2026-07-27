// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/vf/derivatives/dense_derivatives.h"
#include "tycho/detail/vf/derivatives/fd_step_utils.h"

namespace tycho::vf {

/// @brief Central finite-difference Jacobian specialization of @ref DenseFirstDerivatives.
///
/// Approximates the Jacobian columnwise with a second-order central difference
/// @f$ \partial f / \partial x_i \approx (f(x + h\,e_i) - f(x - h\,e_i)) / 2h @f$. The
/// two perturbed evaluations per column are packed into a 2-lane SuperScalar array so
/// both stencil nodes are computed in one function call.
/// @tparam Derived  The concrete VectorFunction type (CRTP self type).
/// @tparam IR       Input dimension (rows), or `Eigen::Dynamic`.
/// @tparam OR       Output dimension (rows), or `Eigen::Dynamic`.
/// @ingroup vf
template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::FDiffCentArray>
    : DenseFunctionBase<Derived, IR, OR> {
    /// @brief The dense function base providing the primal `compute` interface.
    using Base = DenseFunctionBase<Derived, IR, OR>;
    VF_TYPE_ALIASES(Base)

    /// @brief Default central-difference Jacobian step (relative; see compute_jacobian_impl).
    static constexpr double kDefaultJacStep = 1.0e-5;

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

    /// @brief Computes the function value and central finite-difference Jacobian.
    /// @internal
    /// Stores the primal in @p fx_ and the Jacobian in @p jx_, using 2-lane SuperScalar
    /// arrays to evaluate the symmetric stencil nodes per column simultaneously.
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
        Input<Eigen::Array<Scalar, 2, 1>> xi = x.template cast<Eigen::Array<Scalar, 2, 1>>();
        Output<Eigen::Array<Scalar, 2, 1>> fi(this->output_rows());
        fi.setZero();
        for (int i = 0; i < this->input_rows(); i++) {
            const Scalar h = Scalar(this->jac_fd_steps[i]) * detail::fd_step_scale(x[i]);
            xi[i][0] += h;
            xi[i][1] -= h;
            this->derived().compute(xi, fi);
            for (int j = 0; j < this->output_rows(); j++) {
                jx(j, i) = (fi[j][0] - fi[j][1]) / (xi[i][0] - xi[i][1]);
            }
            fi.setZero();
            xi[i] = x[i];
        }
    }

  protected:
    /// @brief Per-input-dimension Jacobian finite-difference step sizes.
    Eigen::VectorXd jac_fd_steps;
};
} // namespace tycho::vf
