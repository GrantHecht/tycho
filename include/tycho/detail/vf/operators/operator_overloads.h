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

#include "tycho/detail/vf/common/common_functions.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"

namespace tycho::vf {

/////////////////////// Scalar Multiplication and
/// Division//////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Scale a VectorFunction by a scalar on the right, @f$ f \cdot s @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @param func  Operand function.
/// @param s     Scalar multiplier.
/// @return A Scaled expression scaling @p func by @p s.
template <class Derived, int IR, int OR>
decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR> &func, double s) {
    return Scaled<Derived>(func.derived(), s);
}

/// @brief Scale a type-erased GenericFunction by a scalar on the right.
/// @tparam IR  Compile-time input rows.
/// @tparam OR  Compile-time output rows.
/// @param func  Type-erased operand function.
/// @param s     Scalar multiplier.
/// @return A Scaled expression scaling @p func by @p s.
template <int IR, int OR> auto operator*(const GenericFunction<IR, OR> &func, double s) {
    return Scaled<GenericFunction<IR, OR>>(func, s);
}

/// @brief Scale a VectorFunction by a compile-time constant factor.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam VALUE    Static-scale wrapper carrying the compile-time factor.
/// @param func  Operand function.
/// @param s     Static-scale tag selecting the compile-time factor.
/// @return A StaticScaled expression applying the compile-time factor to @p func.
template <class Derived, int IR, int OR, class VALUE>
decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR> &func, StaticScaleBase<VALUE> s) {
    return StaticScaled<Derived, VALUE>(func.derived());
}

/// @brief Scale a VectorFunction by a scalar on the left, @f$ s \cdot f @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @param s     Scalar multiplier.
/// @param func  Operand function.
/// @return A Scaled expression scaling @p func by @p s.
template <class Derived, int IR, int OR>
decltype(auto) operator*(double s, const DenseFunctionBase<Derived, IR, OR> &func) {
    return Scaled<Derived>(func.derived(), s);
}

/// @brief Scale a type-erased GenericFunction by a scalar on the left.
/// @tparam IR  Compile-time input rows.
/// @tparam OR  Compile-time output rows.
/// @param s     Scalar multiplier.
/// @param func  Type-erased operand function.
/// @return A Scaled expression scaling @p func by @p s.
template <int IR, int OR> auto operator*(double s, const GenericFunction<IR, OR> &func) {
    return Scaled<GenericFunction<IR, OR>>(func, s);
}

/// @brief Element-wise scale a VectorFunction by a constant Eigen vector (right).
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the per-row scale vector.
/// @param func  Operand function.
/// @param s     Per-row scale factors.
/// @return A RowScaled expression scaling each output row of @p func.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator*(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return RowScaled<Derived>(func.derived(), s);
}
/// @brief Element-wise scale a VectorFunction by a constant Eigen vector (left).
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the per-row scale vector.
/// @param s     Per-row scale factors.
/// @param func  Operand function.
/// @return A RowScaled expression scaling each output row of @p func.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator*(const Eigen::MatrixBase<OutType> &s,
                         const DenseFunctionBase<Derived, IR, OR> &func) {
    return RowScaled<Derived>(func.derived(), s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Fold a scalar factor into an existing Scaled expression (right).
/// @tparam Derived  Wrapped function type of the Scaled expression.
/// @param func  Existing scaled expression.
/// @param s     Additional scalar multiplier.
/// @return A Scaled expression with the combined scale factor.
template <class Derived> decltype(auto) operator*(const Scaled<Derived> &func, double s) {
    return Scaled<Derived>(func.func_, func.scale_value_ * s);
}
/// @brief Fold a scalar factor into an existing Scaled expression (left).
/// @tparam Derived  Wrapped function type of the Scaled expression.
/// @param s     Additional scalar multiplier.
/// @param func  Existing scaled expression.
/// @return A Scaled expression with the combined scale factor.
template <class Derived> decltype(auto) operator*(double s, const Scaled<Derived> &func) {
    return Scaled<Derived>(func.func_, func.scale_value_ * s);
}
/// @brief Fold a scalar factor into an existing RowScaled expression (right).
/// @tparam Derived  Wrapped function type of the RowScaled expression.
/// @param func  Existing row-scaled expression.
/// @param s     Additional scalar multiplier.
/// @return A RowScaled expression with the combined per-row scale.
template <class Derived> decltype(auto) operator*(const RowScaled<Derived> &func, double s) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_ * s);
}
/// @brief Fold a scalar factor into an existing RowScaled expression (left).
/// @tparam Derived  Wrapped function type of the RowScaled expression.
/// @param s     Additional scalar multiplier.
/// @param func  Existing row-scaled expression.
/// @return A RowScaled expression with the combined per-row scale.
template <class Derived> decltype(auto) operator*(double s, const RowScaled<Derived> &func) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_ * s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Fold a per-row Eigen vector into an existing RowScaled expression (right).
/// @tparam Derived  Wrapped function type of the RowScaled expression.
/// @tparam OutType  Eigen type of the per-row scale vector.
/// @param func  Existing row-scaled expression.
/// @param s     Per-row scale factors to fold in.
/// @return A RowScaled expression with the element-wise combined per-row scale.
template <class Derived, class OutType>
decltype(auto) operator*(const RowScaled<Derived> &func, const Eigen::MatrixBase<OutType> &s) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_.cwiseProduct(s));
}
/// @brief Fold a per-row Eigen vector into an existing RowScaled expression (left).
/// @tparam Derived  Wrapped function type of the RowScaled expression.
/// @tparam OutType  Eigen type of the per-row scale vector.
/// @param s     Per-row scale factors to fold in.
/// @param func  Existing row-scaled expression.
/// @return A RowScaled expression with the element-wise combined per-row scale.
template <class Derived, class OutType>
decltype(auto) operator*(const Eigen::MatrixBase<OutType> &s, const RowScaled<Derived> &func) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_.cwiseProduct(s));
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Divide a VectorFunction by a scalar, @f$ f / s @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @param func  Operand function.
/// @param s     Scalar divisor.
/// @return A Scaled expression scaling @p func by @f$ 1/s @f$.
template <class Derived, int IR, int OR>
decltype(auto) operator/(const DenseFunctionBase<Derived, IR, OR> &func, double s) {
    return Scaled<Derived>(func.derived(), 1.0 / s);
}
/// @brief Divide a VectorFunction element-wise by a constant Eigen vector.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the per-row divisor vector.
/// @param func  Operand function.
/// @param s     Per-row divisors.
/// @return A RowScaled expression scaling each output row of @p func by @f$ 1/s_i @f$.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator/(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return RowScaled<Derived>(func.derived(), s.cwiseInverse());
}
/// @brief Fold a scalar divisor into an existing Scaled expression.
/// @tparam Derived  Wrapped function type of the Scaled expression.
/// @param func  Existing scaled expression.
/// @param s     Scalar divisor.
/// @return A Scaled expression with the combined scale factor.
template <class Derived> decltype(auto) operator/(const Scaled<Derived> &func, double s) {
    return Scaled<Derived>(func.func_, func.scale_value_ / s);
}
/// @brief Fold a scalar divisor into an existing RowScaled expression.
/// @tparam Derived  Wrapped function type of the RowScaled expression.
/// @param func  Existing row-scaled expression.
/// @param s     Scalar divisor.
/// @return A RowScaled expression with the combined per-row scale.
template <class Derived> decltype(auto) operator/(const RowScaled<Derived> &func, double s) {
    return RowScaled<Derived>(func.func_, func.row_scale_values_ / s);
}

/// @brief Divide a scalar by a scalar-valued VectorFunction, @f$ s / f @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param s     Scalar numerator.
/// @param func  Scalar-valued denominator function.
/// @return A Scaled expression scaling the element-wise inverse of @p func by @p s.
template <class Derived, int IR>
decltype(auto) operator/(double s, const DenseFunctionBase<Derived, IR, 1> &func) {
    return Scaled<CwiseInverse<Derived>>(CwiseInverse<Derived>(func.derived()), s);
}

/////////////////////// Scalar Addition
/// Subtraction///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Add a constant Eigen vector to a VectorFunction (right), @f$ f + v @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the addend vector.
/// @param func  Operand function.
/// @param s     Constant vector to add.
/// @return A FunctionPlusVector expression adding @p s to @p func.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator+(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<Derived>(func.derived(), s);
}

/// @brief Add a constant Eigen vector to a type-erased GenericFunction (right).
/// @tparam IR       Compile-time input rows.
/// @tparam OR       Compile-time output rows.
/// @tparam OutType  Eigen type of the addend vector.
/// @param func  Type-erased operand function.
/// @param s     Constant vector to add.
/// @return A FunctionPlusVector expression adding @p s to @p func.
template <int IR, int OR, class OutType>
auto operator+(const GenericFunction<IR, OR> &func, const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, s);
}

/// @brief Add a constant Eigen vector to a VectorFunction (left), @f$ v + f @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the addend vector.
/// @param s     Constant vector to add.
/// @param func  Operand function.
/// @return A FunctionPlusVector expression adding @p s to @p func.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator+(const Eigen::MatrixBase<OutType> &s,
                         const DenseFunctionBase<Derived, IR, OR> &func) {
    return FunctionPlusVector<Derived>(func.derived(), s);
}

/// @brief Add a constant Eigen vector to a type-erased GenericFunction (left).
/// @tparam IR       Compile-time input rows.
/// @tparam OR       Compile-time output rows.
/// @tparam OutType  Eigen type of the addend vector.
/// @param s     Constant vector to add.
/// @param func  Type-erased operand function.
/// @return A FunctionPlusVector expression adding @p s to @p func.
template <int IR, int OR, class OutType>
auto operator+(const Eigen::MatrixBase<OutType> &s, const GenericFunction<IR, OR> &func) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, s);
}

/// @brief Subtract a constant Eigen vector from a VectorFunction, @f$ f - v @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the subtrahend vector.
/// @param func  Operand function.
/// @param s     Constant vector to subtract.
/// @return A FunctionPlusVector expression adding @f$ -s @f$ to @p func.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator-(const DenseFunctionBase<Derived, IR, OR> &func,
                         const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<Derived>(func.derived(), (-s));
}

/// @brief Subtract a constant Eigen vector from a type-erased GenericFunction.
/// @tparam IR       Compile-time input rows.
/// @tparam OR       Compile-time output rows.
/// @tparam OutType  Eigen type of the subtrahend vector.
/// @param func  Type-erased operand function.
/// @param s     Constant vector to subtract.
/// @return A FunctionPlusVector expression adding @f$ -s @f$ to @p func.
template <int IR, int OR, class OutType>
auto operator-(const GenericFunction<IR, OR> &func, const Eigen::MatrixBase<OutType> &s) {
    return FunctionPlusVector<GenericFunction<IR, OR>>(func, (-s));
}

/// @brief Subtract a VectorFunction from a constant Eigen vector, @f$ v - f @f$.
/// @tparam Derived  Derived type of the operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @tparam OR       Compile-time output rows of the operand.
/// @tparam OutType  Eigen type of the minuend vector.
/// @param s     Constant vector minuend.
/// @param func  Operand function.
/// @return A FunctionPlusVector expression evaluating @f$ -f + v @f$.
template <class Derived, int IR, int OR, class OutType>
decltype(auto) operator-(const Eigen::MatrixBase<OutType> &s,
                         const DenseFunctionBase<Derived, IR, OR> &func) {
    return FunctionPlusVector<Scaled<Derived>>(Scaled<Derived>(func.derived(), -1.0), s);
}

/// @brief Add a scalar to a scalar-valued VectorFunction (right), @f$ f + s @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @param s     Scalar addend.
/// @return A FunctionPlusVector expression adding @p s to @p func.
template <class Derived, int IR>
decltype(auto) operator+(const DenseFunctionBase<Derived, IR, 1> &func, double s) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), st);
}
/// @brief Add a scalar to a scalar-valued VectorFunction (left), @f$ s + f @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param s     Scalar addend.
/// @param func  Scalar-valued operand function.
/// @return A FunctionPlusVector expression adding @p s to @p func.
template <class Derived, int IR>
decltype(auto) operator+(double s, const DenseFunctionBase<Derived, IR, 1> &func) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), st);
}
/// @brief Subtract a scalar from a scalar-valued VectorFunction, @f$ f - s @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @param s     Scalar to subtract.
/// @return A FunctionPlusVector expression adding @f$ -s @f$ to @p func.
template <class Derived, int IR>
decltype(auto) operator-(const DenseFunctionBase<Derived, IR, 1> &func, double s) {
    Vector1<double> st;
    st[0] = s;
    return FunctionPlusVector<Derived>(func.derived(), (-st));
}
/// @brief Subtract a scalar-valued VectorFunction from a scalar, @f$ s - f @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param s     Scalar minuend.
/// @param func  Scalar-valued operand function.
/// @return A FunctionPlusVector expression evaluating @f$ -f + s @f$.
template <class Derived, int IR>
decltype(auto) operator-(double s, const DenseFunctionBase<Derived, IR, 1> &func) {
    Vector1<double> st;
    st[0] = s;

    return FunctionPlusVector<Scaled<Derived>>(Scaled<Derived>(func.derived(), -1.0), st);
}

/////////////////////// Function Scalar Multiplication And Division
///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Multiply a vector-valued function by a scalar-valued function (vector on left).
/// @tparam VFunc  Derived type of the vector-valued function.
/// @tparam IR1    Compile-time input rows of the vector function.
/// @tparam OR     Compile-time output rows of the vector function.
/// @tparam SFunc  Derived type of the scalar-valued function.
/// @tparam IR2    Compile-time input rows of the scalar function.
/// @param vf  Vector-valued function.
/// @param sf  Scalar-valued multiplier function.
/// @return A VectorScalarFunctionProduct expression scaling @p vf by @p sf.
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<VFunc, IR1, OR> &vf,
                         const DenseFunctionBase<SFunc, IR2, 1> &sf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between vector and scalar functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
/// @brief Multiply a scalar-valued function by a vector-valued function (scalar on left).
/// @tparam VFunc  Derived type of the vector-valued function.
/// @tparam IR1    Compile-time input rows of the vector function.
/// @tparam OR     Compile-time output rows of the vector function.
/// @tparam SFunc  Derived type of the scalar-valued function.
/// @tparam IR2    Compile-time input rows of the scalar function.
/// @param sf  Scalar-valued multiplier function.
/// @param vf  Vector-valued function.
/// @return A VectorScalarFunctionProduct expression scaling @p vf by @p sf.
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR2, 1> &sf,
                         const DenseFunctionBase<VFunc, IR1, OR> &vf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between scalar and vector functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
/// @brief Multiply two scalar-valued functions, @f$ s_1 \cdot s_2 @f$.
/// @tparam VFunc  Derived type of the second scalar function.
/// @tparam IR1    Compile-time input rows of the second function.
/// @tparam SFunc  Derived type of the first scalar function.
/// @tparam IR2    Compile-time input rows of the first function.
/// @param sf  First scalar-valued function (the @p SFunc operand).
/// @param vf  Second scalar-valued function (the @p VFunc operand).
/// @return A VectorScalarFunctionProduct expression of the two scalar functions.
template <class VFunc, int IR1, class SFunc, int IR2>
decltype(auto) operator*(const DenseFunctionBase<SFunc, IR2, 1> &sf,
                         const DenseFunctionBase<VFunc, IR1, 1> &vf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator*: input size mismatch between scalar functions");
    return VectorScalarFunctionProduct<VFunc, SFunc>(vf.derived(), sf.derived());
}
/// @brief Divide a vector-valued function by a scalar-valued function, @f$ vf / sf @f$.
/// @tparam VFunc  Derived type of the vector-valued function.
/// @tparam IR1    Compile-time input rows of the vector function.
/// @tparam OR     Compile-time output rows of the vector function.
/// @tparam SFunc  Derived type of the scalar-valued function.
/// @tparam IR2    Compile-time input rows of the scalar function.
/// @param vf  Vector-valued numerator function.
/// @param sf  Scalar-valued denominator function.
/// @return A VectorScalarFunctionDivision expression dividing @p vf by @p sf.
template <class VFunc, int IR1, int OR, class SFunc, int IR2>
decltype(auto) operator/(const DenseFunctionBase<VFunc, IR1, OR> &vf,
                         const DenseFunctionBase<SFunc, IR2, 1> &sf) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator/: input size mismatch between vector and scalar functions");
    return VectorScalarFunctionDivision<VFunc, SFunc>(vf.derived(), sf.derived());
}

/////////////////////// Function Addition
/// Subtraction/////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Add two VectorFunctions, @f$ f_1 + f_2 @f$.
/// @tparam Func1  Derived type of the left operand function.
/// @tparam IR1    Compile-time input rows of the left operand.
/// @tparam OR1    Compile-time output rows of the left operand.
/// @tparam Func2  Derived type of the right operand function.
/// @tparam IR2    Compile-time input rows of the right operand.
/// @tparam OR2    Compile-time output rows of the right operand.
/// @param f1  Left operand function.
/// @param f2  Right operand function.
/// @return A TwoFunctionSum expression evaluating the sum of the two functions.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
decltype(auto) operator+(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                         const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator+: input size mismatch — both functions have "
                  "static sizes that don't match");
    static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
                  "VF operator+: output size mismatch — both functions have "
                  "static sizes that don't match");
    return TwoFunctionSum<Func1, Func2>(f1.derived(), f2.derived());
}

/// @brief Append a third function to a two-function sum, flattening into a MultiFunctionSum.
/// @tparam Func1  First function type of the existing sum.
/// @tparam IR     Compile-time input rows of the appended operand.
/// @tparam OR     Compile-time output rows of the appended operand.
/// @tparam Func2  Second function type of the existing sum.
/// @tparam Func3  Derived type of the appended operand function.
/// @param f2  Operand function to append.
/// @param f1  Existing two-function sum.
/// @return A MultiFunctionSum aggregating all three operands.
template <class Func1, int IR, int OR, class Func2, class Func3>
decltype(auto) operator+(const DenseFunctionBase<Func3, IR, OR> &f2,
                         const TwoFunctionSum<Func1, Func2> &f1) {
    static_assert(IR == TwoFunctionSum<Func1, Func2>::IRC || IR < 0 ||
                      TwoFunctionSum<Func1, Func2>::IRC < 0,
                  "VF operator+: input size mismatch with sum operand");
    static_assert(OR == TwoFunctionSum<Func1, Func2>::ORC || OR < 0 ||
                      TwoFunctionSum<Func1, Func2>::ORC < 0,
                  "VF operator+: output size mismatch with sum operand");
    return MultiFunctionSum<Func1, Func2, Func3>(f1.func1, f1.func2, f2.derived());
}
/// @brief Append a fourth function to a three-function sum.
/// @tparam Func1  First function type of the existing sum.
/// @tparam IR     Compile-time input rows of the appended operand.
/// @tparam OR     Compile-time output rows of the appended operand.
/// @tparam Func2  Second function type of the existing sum.
/// @tparam Func3  Third function type of the existing sum.
/// @tparam Func4  Derived type of the appended operand function.
/// @param f2  Operand function to append.
/// @param f1  Existing three-function sum.
/// @return A MultiFunctionSum aggregating all four operands.
template <class Func1, int IR, int OR, class Func2, class Func3, class Func4>
decltype(auto) operator+(const DenseFunctionBase<Func4, IR, OR> &f2,
                         const MultiFunctionSum<Func1, Func2, Func3> &f1) {
    static_assert(IR == MultiFunctionSum<Func1, Func2, Func3>::IRC || IR < 0 ||
                      MultiFunctionSum<Func1, Func2, Func3>::IRC < 0,
                  "VF operator+: input size mismatch with sum operand");
    static_assert(OR == MultiFunctionSum<Func1, Func2, Func3>::ORC || OR < 0 ||
                      MultiFunctionSum<Func1, Func2, Func3>::ORC < 0,
                  "VF operator+: output size mismatch with sum operand");
    return MultiFunctionSum<Func1, Func2, Func3, Func4>(f1.func1, f1.func2, std::get<0>(f1.funcs),
                                                        f2.derived());
}

/// @brief Subtract two VectorFunctions, @f$ f_1 - f_2 @f$.
/// @tparam Func1  Derived type of the left operand function.
/// @tparam IR1    Compile-time input rows of the left operand.
/// @tparam OR1    Compile-time output rows of the left operand.
/// @tparam Func2  Derived type of the right operand function.
/// @tparam IR2    Compile-time input rows of the right operand.
/// @tparam OR2    Compile-time output rows of the right operand.
/// @param f1  Left operand function.
/// @param f2  Right operand function.
/// @return A FunctionDifference expression evaluating the difference of the two functions.
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
decltype(auto) operator-(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                         const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF operator-: input size mismatch — both functions have "
                  "static sizes that don't match");
    static_assert(OR1 == OR2 || OR1 < 0 || OR2 < 0,
                  "VF operator-: output size mismatch — both functions have "
                  "static sizes that don't match");
    return FunctionDifference<Func1, Func2>(f1.derived(), f2.derived());
}

////////////Comparisons///////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Less-than comparison of two scalar-valued functions, @f$ \text{lhs} < \text{rhs} @f$.
/// @tparam Func1  Derived type of the left-hand function.
/// @tparam IR1    Compile-time input rows of the left-hand function.
/// @tparam Func2  Derived type of the right-hand function.
/// @tparam IR2    Compile-time input rows of the right-hand function.
/// @param lhs  Left-hand scalar function.
/// @param rhs  Right-hand scalar function.
/// @return A ConditionalStatement encoding the less-than test.
template <class Func1, int IR1, class Func2, int IR2>
auto operator<(const DenseFunctionBase<Func1, IR1, 1> &lhs,
               const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::LessThanFlag,
                                              rhs.derived());
}
/// @brief Less-than-or-equal comparison of two scalar functions, @f$ \text{lhs} \le \text{rhs} @f$.
/// @tparam Func1  Derived type of the left-hand function.
/// @tparam IR1    Compile-time input rows of the left-hand function.
/// @tparam Func2  Derived type of the right-hand function.
/// @tparam IR2    Compile-time input rows of the right-hand function.
/// @param lhs  Left-hand scalar function.
/// @param rhs  Right-hand scalar function.
/// @return A ConditionalStatement encoding the less-than-or-equal test.
template <class Func1, int IR1, class Func2, int IR2>
auto operator<=(const DenseFunctionBase<Func1, IR1, 1> &lhs,
                const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::LessThanEqualToFlag,
                                              rhs.derived());
}
/// @brief Greater-than comparison of two scalar functions, @f$ \text{lhs} > \text{rhs} @f$.
/// @tparam Func1  Derived type of the left-hand function.
/// @tparam IR1    Compile-time input rows of the left-hand function.
/// @tparam Func2  Derived type of the right-hand function.
/// @tparam IR2    Compile-time input rows of the right-hand function.
/// @param lhs  Left-hand scalar function.
/// @param rhs  Right-hand scalar function.
/// @return A ConditionalStatement encoding the greater-than test.
template <class Func1, int IR1, class Func2, int IR2>
auto operator>(const DenseFunctionBase<Func1, IR1, 1> &lhs,
               const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::GreaterThanFlag,
                                              rhs.derived());
}
/// @brief Greater-than-or-equal comparison of two scalar functions, @f$ \text{lhs} \ge \text{rhs}
/// @f$.
/// @tparam Func1  Derived type of the left-hand function.
/// @tparam IR1    Compile-time input rows of the left-hand function.
/// @tparam Func2  Derived type of the right-hand function.
/// @tparam IR2    Compile-time input rows of the right-hand function.
/// @param lhs  Left-hand scalar function.
/// @param rhs  Right-hand scalar function.
/// @return A ConditionalStatement encoding the greater-than-or-equal test.
template <class Func1, int IR1, class Func2, int IR2>
auto operator>=(const DenseFunctionBase<Func1, IR1, 1> &lhs,
                const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(
        lhs.derived(), ConditionalFlags::GreaterThanEqualToFlag, rhs.derived());
}
/// @brief Equality comparison of two scalar functions, @f$ \text{lhs} = \text{rhs} @f$.
/// @tparam Func1  Derived type of the left-hand function.
/// @tparam IR1    Compile-time input rows of the left-hand function.
/// @tparam Func2  Derived type of the right-hand function.
/// @tparam IR2    Compile-time input rows of the right-hand function.
/// @param lhs  Left-hand scalar function.
/// @param rhs  Right-hand scalar function.
/// @return A ConditionalStatement encoding the equality test.
template <class Func1, int IR1, class Func2, int IR2>
auto operator==(const DenseFunctionBase<Func1, IR1, 1> &lhs,
                const DenseFunctionBase<Func2, IR2, 1> &rhs) {
    static_assert(IR1 == IR2 || IR1 < 0 || IR2 < 0,
                  "VF comparison: input size mismatch between scalar functions");
    return ConditionalStatement<Func1, Func2>(lhs.derived(), ConditionalFlags::EqualToFlag,
                                              rhs.derived());
}

/// @brief Less-than comparison of a scalar function against a constant, @f$ \text{lhs} < c @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs   Scalar function.
/// @param rhsv  Constant threshold.
/// @return A ConditionalStatement comparing @p lhs against a constant.
template <class Func1, int IR>
auto operator<(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::LessThanFlag, rhs.derived());
}
/// @brief Less-than-or-equal comparison of a scalar function against a constant, @f$ \text{lhs} \le
/// c @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs   Scalar function.
/// @param rhsv  Constant threshold.
/// @return A ConditionalStatement comparing @p lhs against a constant.
template <class Func1, int IR>
auto operator<=(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::LessThanEqualToFlag, rhs.derived());
}
/// @brief Greater-than comparison of a scalar function against a constant, @f$ \text{lhs} > c @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs   Scalar function.
/// @param rhsv  Constant threshold.
/// @return A ConditionalStatement comparing @p lhs against a constant.
template <class Func1, int IR>
auto operator>(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::GreaterThanFlag, rhs.derived());
}
/// @brief Greater-than-or-equal comparison of a scalar function against a constant, @f$ \text{lhs}
/// \ge c @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs   Scalar function.
/// @param rhsv  Constant threshold.
/// @return A ConditionalStatement comparing @p lhs against a constant.
template <class Func1, int IR>
auto operator>=(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::GreaterThanEqualToFlag, rhs.derived());
}
/// @brief Equality comparison of a scalar function against a constant, @f$ \text{lhs} = c @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs   Scalar function.
/// @param rhsv  Constant threshold.
/// @return A ConditionalStatement comparing @p lhs against a constant.
template <class Func1, int IR>
auto operator==(const DenseFunctionBase<Func1, IR, 1> &lhs, double rhsv) {
    Vector1<double> tmp;
    tmp[0] = rhsv;
    Constant<IR, 1> rhs(lhs.input_rows(), tmp);
    return ConditionalStatement<Func1, Constant<IR, 1>>(
        lhs.derived(), ConditionalFlags::EqualToFlag, rhs.derived());
}

/// @brief Less-than comparison of a constant against a scalar function, @f$ c < \text{rhs} @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs  Constant threshold.
/// @param rhs  Scalar function.
/// @return A ConditionalStatement equivalent to @f$ \text{rhs} > c @f$.
template <class Func1, int IR>
auto operator<(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs > lhs);
}
/// @brief Less-than-or-equal comparison of a constant against a scalar function, @f$ c \le
/// \text{rhs} @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs  Constant threshold.
/// @param rhs  Scalar function.
/// @return A ConditionalStatement equivalent to @f$ \text{rhs} \ge c @f$.
template <class Func1, int IR>
auto operator<=(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs >= lhs);
}
/// @brief Greater-than comparison of a constant against a scalar function, @f$ c > \text{rhs} @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs  Constant threshold.
/// @param rhs  Scalar function.
/// @return A ConditionalStatement equivalent to @f$ \text{rhs} < c @f$.
template <class Func1, int IR>
auto operator>(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs < lhs);
}
/// @brief Greater-than-or-equal comparison of a constant against a scalar function, @f$ c \ge
/// \text{rhs} @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs  Constant threshold.
/// @param rhs  Scalar function.
/// @return A ConditionalStatement equivalent to @f$ \text{rhs} \le c @f$.
template <class Func1, int IR>
auto operator>=(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs <= lhs);
}
/// @brief Equality comparison of a constant against a scalar function, @f$ c = \text{rhs} @f$.
/// @tparam Func1  Derived type of the scalar function.
/// @tparam IR     Compile-time input rows of the function.
/// @param lhs  Constant threshold.
/// @param rhs  Scalar function.
/// @return A ConditionalStatement equivalent to @f$ \text{rhs} = c @f$.
template <class Func1, int IR>
auto operator==(double lhs, const DenseFunctionBase<Func1, IR, 1> &rhs) {
    return (rhs == lhs);
}

////////////////////// Matrix Products /////////////////////////////////

/// @brief Matrix-matrix product of two matrix-valued function views, @f$ M_1 M_2 @f$.
/// @tparam M1             Wrapped function type of the left matrix view.
/// @tparam M2             Wrapped function type of the right matrix view.
/// @tparam M1Rows         Compile-time row count of the left matrix.
/// @tparam M1Cols_M2Rows  Shared inner dimension (left cols / right rows).
/// @tparam M2Cols         Compile-time column count of the right matrix.
/// @tparam M1Major        Storage order of the left matrix view.
/// @tparam M2Major        Storage order of the right matrix view.
/// @param m1  Left matrix-valued function view.
/// @param m2  Right matrix-valued function view.
/// @return A MatrixFunctionProduct expression evaluating the matrix product.
template <class M1, class M2, int M1Rows, int M1Cols_M2Rows, int M2Cols, int M1Major, int M2Major>
decltype(auto) operator*(const MatrixFunctionView<M1, M1Rows, M1Cols_M2Rows, M1Major> &m1,
                         const MatrixFunctionView<M2, M1Cols_M2Rows, M2Cols, M2Major> &m2) {

    using MatFunc1 = MatrixFunctionView<M1, M1Rows, M1Cols_M2Rows, M1Major>;
    using MatFunc2 = MatrixFunctionView<M2, M1Cols_M2Rows, M2Cols, M2Major>;

    return MatrixFunctionProduct<MatFunc1, MatFunc2>(m1, m2);
}

/// @brief Matrix-vector product of a matrix view and a VF expression, @f$ M v @f$.
///
/// Wraps the VF in a column-vector MatrixFunctionView, then delegates to the existing
/// MatrixFunctionProduct.
/// @tparam M1       Wrapped function type of the matrix view.
/// @tparam M1Rows   Compile-time row count of the matrix.
/// @tparam M1Cols   Compile-time column count of the matrix.
/// @tparam M1Major  Storage order of the matrix view.
/// @tparam Func     Derived type of the vector function.
/// @tparam IR       Compile-time input rows of the vector function.
/// @tparam OR       Compile-time output rows of the vector function.
/// @param m  Matrix-valued function view.
/// @param v  Vector-valued function.
/// @return A MatrixFunctionProduct expression evaluating the matrix-vector product.
template <class M1, int M1Rows, int M1Cols, int M1Major, class Func, int IR, int OR>
decltype(auto) operator*(const MatrixFunctionView<M1, M1Rows, M1Cols, M1Major> &m,
                         const DenseFunctionBase<Func, IR, OR> &v) {
    auto col_vec = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>(v.derived(), m.matrix_cols_, 1);

    using MatFunc1 = MatrixFunctionView<M1, M1Rows, M1Cols, M1Major>;
    using MatFunc2 = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>;

    return MatrixFunctionProduct<MatFunc1, MatFunc2>(m, col_vec);
}
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////// pack — type-erase to GenericFunction ///////////////////

/// @brief Type-erase this function into a GenericFunction of matching I/O sizes.
///
/// Out-of-line definition of DenseFunctionBase::pack(); wraps the derived function in a
/// runtime-polymorphic GenericFunction so it can be stored and passed by value.
/// @tparam Derived  CRTP-derived function type.
/// @tparam IR       Compile-time input rows.
/// @tparam OR       Compile-time output rows.
/// @return A GenericFunction<IR, OR> wrapping the derived function.
template <class Derived, int IR, int OR>
GenericFunction<IR, OR> DenseFunctionBase<Derived, IR, OR>::pack() const {
    return GenericFunction<IR, OR>(this->derived());
}

/////////////////////// MatrixFunctionView::inverse() /////////////////////

/// @brief Compute the matrix inverse of a square matrix-valued function view.
///
/// Out-of-line definition of MatrixFunctionView::inverse(); selects a fixed-size
/// MatrixInverse for 2x2/3x3 and a dynamic one otherwise, composing it with the
/// underlying function. Throws std::invalid_argument if the matrix is not square.
/// @tparam Func    Wrapped function type of the matrix view.
/// @tparam MRows   Compile-time row count of the matrix.
/// @tparam MCols   Compile-time column count of the matrix.
/// @tparam MMajor  Storage order of the matrix view.
/// @return A MatrixFunctionView wrapping the inverse expression.
template <class Func, int MRows, int MCols, int MMajor>
auto MatrixFunctionView<Func, MRows, MCols, MMajor>::inverse() const {
    if (this->matrix_rows_ != this->matrix_cols_) {
        throw std::invalid_argument("MatrixFunctionView::inverse: matrix must be square");
    }
    int size = this->matrix_rows_;
    GenericFunction<-1, -1> inv_func;
    if (size == 2) {
        inv_func = MatrixInverse<2, MMajor>(size);
    } else if (size == 3) {
        inv_func = MatrixInverse<3, MMajor>(size);
    } else {
        inv_func = MatrixInverse<-1, MMajor>(size);
    }
    GenericFunction<-1, -1> composed(inv_func.eval(static_cast<const Func &>(*this)));
    return MatrixFunctionView<GenericFunction<-1, -1>, -1, -1, MMajor>(composed, size, size);
}

/////////////////////// quat_rotate ////////////////////////////////////////

/// @brief Rotate a 3-element vector function by a 4-element quaternion function (scalar-last).
///
/// Computes @f$ q \otimes (v, 0) \otimes q^{-1} @f$ and extracts the vector part, assuming a
/// unit quaternion.
/// @tparam QFunc  Derived type of the quaternion function.
/// @tparam QIR    Compile-time input rows of the quaternion function.
/// @tparam QOR    Compile-time output rows of the quaternion function.
/// @tparam VFunc  Derived type of the vector function.
/// @tparam VIR    Compile-time input rows of the vector function.
/// @tparam VOR    Compile-time output rows of the vector function.
/// @param q  Quaternion-valued function (scalar-last).
/// @param v  3-vector-valued function to rotate.
/// @return An expression evaluating the rotated 3-vector.
template <class QFunc, int QIR, int QOR, class VFunc, int VIR, int VOR>
auto quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR> &q,
                 const DenseFunctionBase<VFunc, VIR, VOR> &v) {
    auto qinv =
        StackedOutputs{q.derived().template head<3>() * (-1.0), q.derived().template coeff<3>()};
    auto vq = v.derived().template padded_lower<1>();
    auto qvq = FunctionQuatProduct<QFunc, decltype(vq)>{q.derived(), vq};
    return FunctionQuatProduct<decltype(qvq), decltype(qinv)>{qvq, qinv}.template head<3>();
}

/// @brief Rotate a constant 3-vector by a quaternion-valued function (scalar-last).
/// @tparam QFunc  Derived type of the quaternion function.
/// @tparam QIR    Compile-time input rows of the quaternion function.
/// @tparam QOR    Compile-time output rows of the quaternion function.
/// @param q  Quaternion-valued function (scalar-last).
/// @param v  Constant 3-vector to rotate.
/// @return An expression evaluating the rotated 3-vector.
template <class QFunc, int QIR, int QOR>
auto quat_rotate(const DenseFunctionBase<QFunc, QIR, QOR> &q, const Eigen::Vector3d &v) {
    auto v_const = Constant<-1, 3>(q.derived().input_rows(), v);
    return quat_rotate(q, v_const);
}

/////////////////////// cwise_product Eigen overloads //////////////////////

/// @brief Element-wise product of a VectorFunction and a constant Eigen vector (right).
///
/// Uses RowScaled internally (same as the Python binding).
/// @tparam Func  Derived type of the operand function.
/// @tparam IR    Compile-time input rows of the operand.
/// @tparam OR    Compile-time output rows of the operand.
/// @param f  Operand function.
/// @param v  Constant per-row factors.
/// @return A RowScaled expression scaling each output row of @p f by @p v.
template <class Func, int IR, int OR>
auto cwise_product(const DenseFunctionBase<Func, IR, OR> &f, const Eigen::VectorXd &v) {
    return RowScaled<Func>(f.derived(), v);
}

/// @brief Element-wise product of a constant Eigen vector and a VectorFunction (left).
/// @tparam Func  Derived type of the operand function.
/// @tparam IR    Compile-time input rows of the operand.
/// @tparam OR    Compile-time output rows of the operand.
/// @param v  Constant per-row factors.
/// @param f  Operand function.
/// @return A RowScaled expression scaling each output row of @p f by @p v.
template <class Func, int IR, int OR>
auto cwise_product(const Eigen::VectorXd &v, const DenseFunctionBase<Func, IR, OR> &f) {
    return RowScaled<Func>(f.derived(), v);
}

} // namespace tycho::vf
