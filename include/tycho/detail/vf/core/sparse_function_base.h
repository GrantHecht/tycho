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

/// @brief CRTP base for VectorFunctions whose Jacobian/Hessian are sparse.
/// @ingroup vf
///
/// The sparse counterpart to `DenseFunctionBase`: it extends `ComputableBase`
/// with `Jacobian` and `Hessian` aliases backed by `Eigen::SparseMatrix`
/// (rather than the dense matrices `DenseFunctionBase` supplies) and provides
/// cached sparsity templates.
/// @tparam Derived  The concrete sparse VectorFunction (CRTP).
/// @tparam IR  Input row count (`-1` if dynamic).
/// @tparam OR  Output row count (`-1` if dynamic).
template <class Derived, int IR, int OR>
struct SparseFunctionBase : ComputableBase<Derived, IR, OR> {
    using Base = ComputableBase<Derived, IR, OR>; ///< @brief The CRTP compute base.
    /// @brief Output vector type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    /// @brief Input vector type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    /// @brief Gradient vector type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;

    /// @brief Sparse Jacobian matrix type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Jacobian = Eigen::SparseMatrix<Scalar>;
    /// @brief Sparse Hessian matrix type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Hessian = Eigen::SparseMatrix<Scalar>;

    /// @brief Const reference to a sparse matrix expression.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using ConstMatrixBaseRef = const Eigen::SparseMatrixBase<Scalar> &;
    /// @brief Mutable reference to a sparse matrix expression.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using MatrixBaseRef = Eigen::SparseMatrixBase<Scalar> &;

  protected:
    mutable Jacobian<double> JacobianTemplate; ///< @brief Cached Jacobian sparsity pattern.
    mutable Hessian<double> HessianTemplate;   ///< @brief Cached Hessian sparsity pattern.
};

} // namespace tycho::vf
