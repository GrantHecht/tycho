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
//   - pybind11 header references removed
// =============================================================================

#pragma once

namespace tycho::vf {

#define DENSE_FUNCTION_BASE_TYPES(Base)                                                            \
    template <class Scalar> using Output = typename Base::template Output<Scalar>;                 \
    template <class Scalar> using Input = typename Base::template Input<Scalar>;                   \
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;             \
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;             \
    template <class Scalar> using Hessian = typename Base::template Hessian<Scalar>;               \
    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;          \
    template <class Scalar> using VectorBaseRef = Eigen::MatrixBase<Scalar> &;                     \
    template <class Scalar> using ConstMatrixBaseRef = const Eigen::MatrixBase<Scalar> &;          \
    template <class Scalar> using MatrixBaseRef = Eigen::MatrixBase<Scalar> &;                     \
    template <class Scalar> using ConstEigenBaseRef = const Eigen::EigenBase<Scalar> &;            \
    template <class Scalar> using EigenBaseRef = Eigen::EigenBase<Scalar> &;                       \
    template <class Scalar> using ConstDiagonalBaseRef = const Eigen::DiagonalBase<Scalar> &;      \
    template <class Scalar> using DiagonalBaseRef = Eigen::DiagonalBase<Scalar> &;

#define SUB_FUNCTION_IO_TYPES(Func)                                                                \
    template <class Scalar> using Func##_Output = typename Func::template Output<Scalar>;          \
    template <class Scalar> using Func##_Input = typename Func::template Input<Scalar>;            \
    template <class Scalar> using Func##_gradient = typename Func::template Gradient<Scalar>;      \
    template <class Scalar> using Func##_jacobian = typename Func::template Jacobian<Scalar>;      \
    template <class Scalar> using Func##_hessian = typename Func::template Hessian<Scalar>;

} // namespace tycho::vf
