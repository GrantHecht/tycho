// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/interior/typedefs/eigen_types.h>
#include <hven/detail/interior/utils/flat_map.h>
#include <hven/detail/interior/utils/function_return_type.h>
#include <hven/detail/interior/utils/get_core_count.h>
#include <hven/detail/interior/utils/math_functions.h>
#include <hven/detail/interior/utils/sizing_helpers.h>
#include <hven/detail/interior/utils/std_extensions.h>
#include <hven/detail/interior/utils/thread_pool.h>
#include <hven/detail/interior/utils/type_name.h>
#include <hven/detail/interior/utils/type_storage.h>

namespace tycho::vf {

/// @internal
/// @brief Trait: detects whether `T` is an Eigen diagonal-matrix type.
///
/// The primary template inherits `std::false_type`; the specializations report
/// `std::true_type` for `Eigen::DiagonalMatrix` and `Eigen::DiagonalWrapper` (and its
/// const form). Used by the derivative machinery to take a cheaper code path when a
/// Jacobian is structurally diagonal.
/// @tparam T  Candidate matrix type.
/// @endinternal
template <class T> struct Is_EigenDiagonalMatrix : std::false_type {};

/// @internal
/// @brief Specialization marking `Eigen::DiagonalMatrix` as diagonal.
/// @tparam Scalar     Element type of the diagonal matrix.
/// @tparam Rows_Cols  Compile-time dimension of the square matrix.
/// @endinternal
template <class Scalar, int Rows_Cols>
struct Is_EigenDiagonalMatrix<Eigen::DiagonalMatrix<Scalar, Rows_Cols>> : std::true_type {};

/// @internal
/// @brief Specialization marking `Eigen::DiagonalWrapper` as diagonal.
/// @tparam T  Wrapped expression type.
/// @endinternal
template <class T> struct Is_EigenDiagonalMatrix<Eigen::DiagonalWrapper<T>> : std::true_type {};
/// @internal
/// @brief Specialization marking `const Eigen::DiagonalWrapper` as diagonal.
/// @tparam T  Wrapped expression type.
/// @endinternal
template <class T>
struct Is_EigenDiagonalMatrix<const Eigen::DiagonalWrapper<T>> : std::true_type {};

} // namespace tycho::vf
