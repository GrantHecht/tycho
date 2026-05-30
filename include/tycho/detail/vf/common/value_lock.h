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
/// @brief Zero-residual placeholder for substituted (locked) phase variables.
#pragma once
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

/// @ingroup vf
/// @brief Zero-residual placeholder paired with ODEPhaseBase::sub_variables().
///
/// `sub_variables()` replaces the corresponding primal variables with fixed
/// substituted values *before* the constraint is evaluated — the variables
/// never reach the optimizer, so the constraint residual is identically zero at
/// every iteration. The Jacobian is the identity so PSIOPT receives a correct
/// sparsity pattern.
///
/// `compute_impl` explicitly writes `fx = 0` rather than relying on the caller
/// to pre-zero the RHS buffer. An earlier implementation wrote `fx = x` and
/// corrupted multi-phase solves; the current implementation writes the correct
/// zero residual unconditionally so any future caller (unit test, new
/// transcription, direct `.compute()`) sees the same behavior.
/// @tparam USZ  Compile-time number of substituted variables (input/output rows).
template <int USZ>
struct SubstitutedVarPlaceholder : VectorFunction<SubstitutedVarPlaceholder<USZ>, USZ, USZ> {
    /// @brief Convenience alias for the CRTP VectorFunction base.
    using Base = VectorFunction<SubstitutedVarPlaceholder<USZ>, USZ, USZ>;

    /// @brief Marks the placeholder as a linear function (identity Jacobian).
    static constexpr bool is_linear_function = true;

    /// @brief Constructs a placeholder over a given number of variables.
    /// @param usz  Number of substituted variables.
    SubstitutedVarPlaceholder(int usz) { this->set_io_rows(usz, usz); }
    /// @brief Constructs an empty placeholder (size set later).
    SubstitutedVarPlaceholder() {}

    /// @internal
    /// @brief Writes the identically zero residual, ignoring the input.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @param fx_  Output residual vector, set to zero.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> & /*x*/,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        fx.setZero();
    }
    /// @internal
    /// @brief Writes the zero residual and the identity Jacobian.
    /// @tparam InType   Eigen input vector expression type.
    /// @tparam OutType  Eigen output vector expression type.
    /// @tparam JacType  Eigen Jacobian matrix expression type.
    /// @param fx_  Output residual vector, set to zero.
    /// @param jx_  Jacobian buffer; its diagonal is set to one.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> & /*x*/,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        fx.setZero();
        jx.diagonal().setOnes();
    }
    /// @internal
    /// @brief Writes the zero residual, identity Jacobian, and adjoint gradient.
    ///
    /// With an identity Jacobian the adjoint gradient equals the adjoint
    /// variables and the adjoint Hessian is zero.
    /// @tparam InType       Eigen input vector expression type.
    /// @tparam OutType      Eigen output vector expression type.
    /// @tparam JacType      Eigen Jacobian matrix expression type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector expression type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix expression type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector expression type.
    /// @param fx_      Output residual vector, set to zero.
    /// @param jx_      Jacobian buffer; its diagonal is set to one.
    /// @param adjgrad_ Adjoint-gradient buffer; set equal to @p adjvars.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> & /*x*/, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const & /*adjhess_*/,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        fx.setZero();
        jx.diagonal().setOnes();
        adjgrad = adjvars; // identity Jacobian => adjgrad = J^T * adjvars = adjvars
    }
};

} // namespace tycho::vf
