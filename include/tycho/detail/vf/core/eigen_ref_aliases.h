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
template <class T> using CVecRef = const Eigen::MatrixBase<T> &;
template <class T> using VecRef = Eigen::MatrixBase<T> &;
template <class T> using CMatRef = const Eigen::MatrixBase<T> &;
template <class T> using MatRef = Eigen::MatrixBase<T> &;
template <class T> using CEigRef = const Eigen::EigenBase<T> &;
template <class T> using EigRef = Eigen::EigenBase<T> &;
template <class T> using CDiagRef = const Eigen::DiagonalBase<T> &;
template <class T> using DiagRef = Eigen::DiagonalBase<T> &;

} // namespace tycho::vf
