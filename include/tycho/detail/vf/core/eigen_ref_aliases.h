// =============================================================================
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
//
// Namespace-scope Eigen ref-type aliases shared by all VF classes.
// These replace the 8 per-class aliases formerly injected by the
// DENSE_FUNCTION_BASE_TYPES macro.
// =============================================================================

#pragma once
#include <Eigen/Core>

namespace tycho::vf {

// CVecRef and CMatRef (likewise VecRef/MatRef) are intentionally identical —
// Eigen::MatrixBase covers both vectors and matrices. The separate aliases
// document usage-site semantics (vector vs matrix context).

/// @brief Const reference to an Eigen `MatrixBase` used in a vector context.
/// @tparam T  Concrete Eigen expression type.
template <class T> using CVecRef = const Eigen::MatrixBase<T> &;
/// @brief Mutable reference to an Eigen `MatrixBase` used in a vector context.
/// @tparam T  Concrete Eigen expression type.
template <class T> using VecRef = Eigen::MatrixBase<T> &;
/// @brief Const reference to an Eigen `MatrixBase` used in a matrix context.
/// @tparam T  Concrete Eigen expression type.
template <class T> using CMatRef = const Eigen::MatrixBase<T> &;
/// @brief Mutable reference to an Eigen `MatrixBase` used in a matrix context.
/// @tparam T  Concrete Eigen expression type.
template <class T> using MatRef = Eigen::MatrixBase<T> &;
/// @brief Const reference to an Eigen `EigenBase` (any Eigen expression).
/// @tparam T  Concrete Eigen expression type.
template <class T> using CEigRef = const Eigen::EigenBase<T> &;
/// @brief Mutable reference to an Eigen `EigenBase` (any Eigen expression).
/// @tparam T  Concrete Eigen expression type.
template <class T> using EigRef = Eigen::EigenBase<T> &;
/// @brief Const reference to an Eigen `DiagonalBase` expression.
/// @tparam T  Concrete Eigen diagonal expression type.
template <class T> using CDiagRef = const Eigen::DiagonalBase<T> &;
/// @brief Mutable reference to an Eigen `DiagonalBase` expression.
/// @tparam T  Concrete Eigen diagonal expression type.
template <class T> using DiagRef = Eigen::DiagonalBase<T> &;

} // namespace tycho::vf
