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

#include "tycho/detail/vf/core/computable_base.h"

namespace tycho::vf {

template <class Derived, int IR, int OR>
struct SparseFunctionBase : ComputableBase<Derived, IR, OR> {
    using Base = ComputableBase<Derived, IR, OR>;
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;

    template <class Scalar> using Jacobian = Eigen::SparseMatrix<Scalar>;
    template <class Scalar> using Hessian = Eigen::SparseMatrix<Scalar>;

    template <class Scalar> using ConstMatrixBaseRef = const Eigen::SparseMatrixBase<Scalar> &;
    template <class Scalar> using MatrixBaseRef = Eigen::SparseMatrixBase<Scalar> &;

  protected:
    mutable Jacobian<double> JacobianTemplate;
    mutable Hessian<double> HessianTemplate;
};

} // namespace tycho::vf
