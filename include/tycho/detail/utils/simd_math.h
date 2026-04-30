// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// SuperScalar math overloads filling gaps in Eigen's free-function math API.
//
// Eigen 3.4 ships free overloads for the unary cwise operations (cos, sin,
// sqrt, ...) via EIGEN_ARRAY_DECLARE_GLOBAL_UNARY, but the binary cwise
// operations (notably atan2) only exist as ArrayBase member functions. This
// header adds the missing free overloads in the Eigen namespace so ADL on a
// SuperScalar Eigen::Array Scalar (the W-lane primal dispatch path) finds
// them - matching how cos/sin already resolve. Per-lane scalar fallback;
// no SIMD intrinsic implementation. Sufficient for VFs whose body uses
// 2-arg atan2 under is_vectorizable=true (e.g. CartesianToModified).
//
// RETIRE WHEN: the Eigen submodule is bumped to Eigen 5, which adds a free
// Eigen::atan2(ArrayBase, ArrayBase) overload backed by scalar_atan2_op
// (true SIMD via the packet backend) - obsoletes this scalar shim.
// =============================================================================
#pragma once

#include <Eigen/Core>
#include <cmath>

namespace Eigen {

// Template on ArrayBase so the overload matches both concrete Array<>
// values and Eigen's lazy CwiseBinaryOp / CwiseUnaryOp expression templates
// that the codegen output passes in. Each operand is evaluated into a plain
// Array first, then atan2 is applied per lane via std::atan2.
template <typename DerivedY, typename DerivedX>
inline auto atan2(const ArrayBase<DerivedY> &y,
                  const ArrayBase<DerivedX> &x) {
    using Scalar = typename DerivedY::Scalar;
    constexpr int Rows = DerivedY::RowsAtCompileTime;
    constexpr int Cols = DerivedY::ColsAtCompileTime;
    using Result = Array<Scalar, Rows, Cols>;
    Result y_eval = y.derived();
    Result x_eval = x.derived();
    Result result(y_eval.rows(), y_eval.cols());
    for (Index i = 0; i < y_eval.size(); ++i)
        result(i) = std::atan2(y_eval(i), x_eval(i));
    return result;
}

} // namespace Eigen
