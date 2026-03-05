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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once
#include "VectorFunctions/VectorFunction.h"

namespace Tycho {

template <int USZ> struct LockArgs : VectorFunction<LockArgs<USZ>, USZ, USZ> {
    using Base = VectorFunction<LockArgs<USZ>, USZ, USZ>;

    static const bool IsLinearFunction = true;

    LockArgs(int usz) { this->setIORows(usz, usz); }
    LockArgs() {}
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        // Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        // Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
        for (int i = 0; i < this->IRows(); i++)
            jx(i, i) = Scalar(1.0);
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        // Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        for (int i = 0; i < this->IRows(); i++)
            jx(i, i) = Scalar(1.0);
        adjgrad = jx * adjvars;
    }
};

} // namespace Tycho