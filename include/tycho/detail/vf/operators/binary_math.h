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
#include "tycho/detail/vf/core/expression_fwd_declarations.h"

namespace tycho::vf {

/// @brief Form the vector cross product @f$ f_1 \times f_2 @f$ of two VectorFunctions.
/// @tparam Func1  Derived type of the first (left) operand function.
/// @tparam IR1    Compile-time input rows of the first operand.
/// @tparam OR1    Compile-time output rows of the first operand.
/// @tparam Func2  Derived type of the second (right) operand function.
/// @tparam IR2    Compile-time input rows of the second operand.
/// @tparam OR2    Compile-time output rows of the second operand.
/// @param f1  Left operand (3-vector valued function).
/// @param f2  Right operand (3-vector valued function).
/// @return A FunctionCrossProduct expression evaluating the 3D vector cross product.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto cross_product(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                   const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return FunctionCrossProduct<Func1, Func2>(f1.derived(), f2.derived());
}

/// @brief Form the quaternion product @f$ f_1 \otimes f_2 @f$ of two VectorFunctions.
/// @tparam Func1  Derived type of the first (left) operand function.
/// @tparam IR1    Compile-time input rows of the first operand.
/// @tparam OR1    Compile-time output rows of the first operand.
/// @tparam Func2  Derived type of the second (right) operand function.
/// @tparam IR2    Compile-time input rows of the second operand.
/// @tparam OR2    Compile-time output rows of the second operand.
/// @param f1  Left operand (quaternion-valued function).
/// @param f2  Right operand (quaternion-valued function).
/// @return A FunctionQuatProduct expression evaluating the quaternion product.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto quat_product(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                  const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return FunctionQuatProduct<Func1, Func2>(f1.derived(), f2.derived());
}

/// @brief Form the scalar dot product @f$ f_1 \cdot f_2 @f$ of two VectorFunctions.
/// @tparam Func1  Derived type of the first (left) operand function.
/// @tparam IR1    Compile-time input rows of the first operand.
/// @tparam OR1    Compile-time output rows of the first operand.
/// @tparam Func2  Derived type of the second (right) operand function.
/// @tparam IR2    Compile-time input rows of the second operand.
/// @tparam OR2    Compile-time output rows of the second operand.
/// @param f1  Left operand function.
/// @param f2  Right operand function.
/// @return A FunctionDotProduct expression evaluating the scalar dot product.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto dot_product(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                 const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return FunctionDotProduct<Func1, Func2>(f1.derived(), f2.derived());
}

/// @brief Form the element-wise (Hadamard) product @f$ f_1 \odot f_2 @f$ of two VectorFunctions.
/// @tparam Func1  Derived type of the first (left) operand function.
/// @tparam IR1    Compile-time input rows of the first operand.
/// @tparam OR1    Compile-time output rows of the first operand.
/// @tparam Func2  Derived type of the second (right) operand function.
/// @tparam IR2    Compile-time input rows of the second operand.
/// @tparam OR2    Compile-time output rows of the second operand.
/// @param f1  Left operand function.
/// @param f2  Right operand function.
/// @return A CwiseFunctionProduct expression evaluating the element-wise product.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto cwise_product(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                   const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return CwiseFunctionProduct<Func1, Func2>(f1.derived(), f2.derived());
}
/// @brief Form the element-wise quotient @f$ f_1 \oslash f_2 @f$ of two VectorFunctions.
///
/// Implemented as the element-wise product of @p f1 with the element-wise inverse of @p f2.
/// @tparam Func1  Derived type of the numerator function.
/// @tparam IR1    Compile-time input rows of the numerator.
/// @tparam OR1    Compile-time output rows of the numerator.
/// @tparam Func2  Derived type of the denominator function.
/// @tparam IR2    Compile-time input rows of the denominator.
/// @tparam OR2    Compile-time output rows of the denominator.
/// @param f1  Numerator function.
/// @param f2  Denominator function.
/// @return A CwiseFunctionProduct expression evaluating the element-wise quotient.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto cwise_quotient(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                    const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return CwiseFunctionProduct<Func1, CwiseInverse<Func2>>(f1.derived(),
                                                            CwiseInverse<Func2>(f2.derived()));
}

} // namespace tycho::vf