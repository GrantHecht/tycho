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

#include <Eigen/Core>

#include "tycho/detail/typedefs/super_scalar_traits.h"

namespace tycho {

template <class Scalar, int rows> using Vector = Eigen::Matrix<Scalar, rows, 1>;

template <class Scalar> using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <class Scalar> using Vector1 = Eigen::Matrix<Scalar, 1, 1>;

template <class Scalar> using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

template <class Scalar> using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

template <class Scalar> using Vector4 = Eigen::Matrix<Scalar, 4, 1>;

template <class Scalar> using Vector5 = Eigen::Matrix<Scalar, 5, 1>;

template <class Scalar> using Vector6 = Eigen::Matrix<Scalar, 6, 1>;

template <class Scalar> using Vector7 = Eigen::Matrix<Scalar, 7, 1>;

template <class Scalar> using Vector8 = Eigen::Matrix<Scalar, 8, 1>;

template <class Scalar> using Vector9 = Eigen::Matrix<Scalar, 9, 1>;

template <class Scalar> using Vector10 = Eigen::Matrix<Scalar, 10, 1>;

template <class Scalar> using Vector11 = Eigen::Matrix<Scalar, 11, 1>;

template <class Scalar> using Vector12 = Eigen::Matrix<Scalar, 12, 1>;

template <class Scalar> using Vector13 = Eigen::Matrix<Scalar, 13, 1>;

template <class Scalar> using Vector14 = Eigen::Matrix<Scalar, 14, 1>;

template <class Type> using EigenRef = Eigen::Ref<Type>;

template <class Type> using ConstEigenRef = const Eigen::Ref<const Type> &;

template <class Scalar, int MaxSize> using MaxVector = Eigen::Matrix<Scalar, -1, 1, 0, MaxSize, 1>;

template <class Scalar, int MaxSize>
using MaxMatrix = Eigen::Matrix<Scalar, -1, -1, 0, MaxSize, MaxSize>;

using IOint = int;

using DomainMatrix = Eigen::Matrix<IOint, 2, -1>;

template <class Scalar, int Sz> using SuperScalarType = Eigen::Array<Scalar, Sz, 1>;

#if defined(__AVX512F__)
using DefaultSuperScalar = SuperScalarType<double, 8>; // 512-bit = 8 doubles
#elif defined(__AVX__) || defined(__AVX2__)
using DefaultSuperScalar = SuperScalarType<double, 4>; // 256-bit = 4 doubles
#else
using DefaultSuperScalar = SuperScalarType<double, 2>; // 128-bit = 2 doubles (SSE2/NEON)
#endif

} // namespace tycho
