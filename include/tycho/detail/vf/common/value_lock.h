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

template <int USZ> struct LockArgs : VectorFunction<LockArgs<USZ>, USZ, USZ> {
    using Base = VectorFunction<LockArgs<USZ>, USZ, USZ>;

    static constexpr bool is_linear_function = true;

    LockArgs(int usz) { this->set_io_rows(usz, usz); }
    LockArgs() {}

    // Note: LockArgs is a "value-lock" constraint. It pairs with
    // ODEPhaseBase::sub_variables() which replaces the corresponding primal
    // variables with fixed substituted values *before* the constraint is
    // evaluated. Because the variables are never seen by the optimizer,
    // the only useful residual at eval time is 0 — the constraint is
    // structurally "satisfied" as long as the substitution has happened.
    //
    // compute_impl / compute_jacobian_impl therefore intentionally leave
    // `fx` untouched (PSIOPT zeroes the RHS between iterations, so `fx`
    // enters as 0). Writing `fx = x` would leak the substituted values
    // back into the residual and cause a spurious non-zero constraint
    // error at every iteration.
    //
    // This matches asset_asrl's historical behavior — an earlier tycho
    // refactor accidentally added `fx = x` assignments here, which caused
    // e.g. multi_spacecraft_opt to diverge because the value-lock residual
    // was reported as the locked state itself (|[rx=1, vy=1, ...]|_inf = 1).
    // Fixed in commit 8ffc2a9; regression test in
    // tests/cpp/vector_functions/test_vf_value_lock.cpp.

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> & /*x*/,
                             Eigen::MatrixBase<OutType> const & /*fx_*/) const {
        // intentionally empty — see note above
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> & /*x*/,
                                      Eigen::MatrixBase<OutType> const & /*fx_*/,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        jx.diagonal().setOnes();
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> & /*x*/, Eigen::MatrixBase<OutType> const & /*fx_*/,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const & /*adjhess_*/,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        jx.diagonal().setOnes();
        adjgrad = adjvars; // identity Jacobian => adjgrad = J^T * adjvars = adjvars
    }
};

} // namespace tycho::vf
