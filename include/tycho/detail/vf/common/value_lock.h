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

// Zero-residual placeholder paired with ODEPhaseBase::sub_variables().
// sub_variables() replaces the corresponding primal variables with fixed
// substituted values *before* the constraint is evaluated — the variables
// never reach the optimizer, so the constraint residual is identically
// zero at every iteration. The Jacobian is identity so PSIOPT receives a
// correct sparsity pattern.
//
// compute_impl explicitly writes fx = 0 rather than relying on the caller
// to pre-zero the RHS buffer. An earlier implementation in this file wrote
// fx = x and corrupted multi-phase solves; the current implementation
// writes the correct zero residual unconditionally so any future caller
// (unit test, new transcription, direct .compute()) sees the same
// behavior.
template <int USZ>
struct SubstitutedVarPlaceholder : VectorFunction<SubstitutedVarPlaceholder<USZ>, USZ, USZ> {
    using Base = VectorFunction<SubstitutedVarPlaceholder<USZ>, USZ, USZ>;

    static constexpr bool is_linear_function = true;

    SubstitutedVarPlaceholder(int usz) { this->set_io_rows(usz, usz); }
    SubstitutedVarPlaceholder() {}

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> & /*x*/,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        fx.setZero();
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> & /*x*/,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        fx.setZero();
        jx.diagonal().setOnes();
    }
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
