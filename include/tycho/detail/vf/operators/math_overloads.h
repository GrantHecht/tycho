// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Overloads the common mathematical functions contained in cmath for Tycho scalar
// functions.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once
#include "tycho/detail/vf/common/common_functions.h"

////////////////////////////////////////////////////////////////////////////
/////////////////////// CMath Overloads ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/// @brief Sine of a scalar-valued VectorFunction, @f$ \sin(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseSin expression applying @f$ \sin @f$ to @p func.
template <class Derived, int IR>
auto sin(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseSin<Derived>(func.derived());
}

/// @brief Cosine of a scalar-valued VectorFunction, @f$ \cos(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseCos expression applying @f$ \cos @f$ to @p func.
template <class Derived, int IR>
auto cos(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseCos<Derived>(func.derived());
}

/// @brief Tangent of a scalar-valued VectorFunction, @f$ \tan(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseTan expression applying @f$ \tan @f$ to @p func.
template <class Derived, int IR>
auto tan(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseTan<Derived>(func.derived());
}

/// @brief Arcsine of a scalar-valued VectorFunction, @f$ \arcsin(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseArcSin expression applying @f$ \arcsin @f$ to @p func.
template <class Derived, int IR>
auto asin(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcSin<Derived>(func.derived());
}

/// @brief Arccosine of a scalar-valued VectorFunction, @f$ \arccos(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseArcCos expression applying @f$ \arccos @f$ to @p func.
template <class Derived, int IR>
auto acos(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcCos<Derived>(func.derived());
}

/// @brief Arctangent of a scalar-valued VectorFunction, @f$ \arctan(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseArcTan expression applying @f$ \arctan @f$ to @p func.
template <class Derived, int IR>
auto atan(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcTan<Derived>(func.derived());
}

/// @brief Two-argument arctangent of scalar-valued VectorFunctions, @f$ \operatorname{atan2}(y, x)
/// @f$.
/// @tparam Derived1  Derived type of the numerator (y) function.
/// @tparam IR1       Compile-time input rows of the numerator.
/// @tparam Derived2  Derived type of the denominator (x) function.
/// @tparam IR2       Compile-time input rows of the denominator.
/// @param yf  Numerator (y) scalar function.
/// @param xf  Denominator (x) scalar function.
/// @return An expression evaluating the quadrant-aware arctangent of @p yf over @p xf.
template <class Derived1, int IR1, class Derived2, int IR2>
auto atan2(const tycho::vf::DenseFunctionBase<Derived1, IR1, 1> &yf,
           const tycho::vf::DenseFunctionBase<Derived2, IR2, 1> &xf) {
    return tycho::vf::ArcTan2Op().eval(
        tycho::vf::StackedOutputs<Derived1, Derived2>(yf.derived(), xf.derived()));
}

/// @brief Square root of a scalar-valued VectorFunction, @f$ \sqrt{f} @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseSqrt expression applying @f$ \sqrt{\cdot} @f$ to @p func.
template <class Derived, int IR>
auto sqrt(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseSqrt<Derived>(func.derived());
}

/// @brief Exponential of a scalar-valued VectorFunction, @f$ e^{f} @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseExp expression applying @f$ e^{\cdot} @f$ to @p func.
template <class Derived, int IR>
auto exp(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseExp<Derived>(func.derived());
}

/// @brief Natural logarithm of a scalar-valued VectorFunction, @f$ \ln(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseLog expression applying @f$ \ln @f$ to @p func.
template <class Derived, int IR>
auto log(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseLog<Derived>(func.derived());
}

/// @brief Hyperbolic sine of a scalar-valued VectorFunction, @f$ \sinh(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseSinH expression applying @f$ \sinh @f$ to @p func.
template <class Derived, int IR>
auto sinh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseSinH<Derived>(func.derived());
}

/// @brief Hyperbolic cosine of a scalar-valued VectorFunction, @f$ \cosh(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseCosH expression applying @f$ \cosh @f$ to @p func.
template <class Derived, int IR>
auto cosh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseCosH<Derived>(func.derived());
}

/// @brief Hyperbolic tangent of a scalar-valued VectorFunction, @f$ \tanh(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseTanH expression applying @f$ \tanh @f$ to @p func.
template <class Derived, int IR>
auto tanh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseTanH<Derived>(func.derived());
}

/// @brief Inverse hyperbolic sine of a scalar-valued VectorFunction, @f$ \operatorname{arcsinh}(f)
/// @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseArcSinH expression applying @f$ \operatorname{arcsinh} @f$ to @p func.
template <class Derived, int IR>
auto asinh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcSinH<Derived>(func.derived());
}

/// @brief Inverse hyperbolic cosine of a scalar-valued VectorFunction, @f$
/// \operatorname{arccosh}(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseArcCosH expression applying @f$ \operatorname{arccosh} @f$ to @p func.
template <class Derived, int IR>
auto acosh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcCosH<Derived>(func.derived());
}

/// @brief Inverse hyperbolic tangent of a scalar-valued VectorFunction, @f$
/// \operatorname{arctanh}(f) @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseArcTanH expression applying @f$ \operatorname{arctanh} @f$ to @p func.
template <class Derived, int IR>
auto atanh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcTanH<Derived>(func.derived());
}
/// @brief Absolute value of a scalar-valued VectorFunction, @f$ |f| @f$.
/// @tparam Derived  Derived type of the scalar operand function.
/// @tparam IR       Compile-time input rows of the operand.
/// @param func  Scalar-valued operand function.
/// @return A CwiseAbs expression applying @f$ |\cdot| @f$ to @p func.
template <class Derived, int IR>
auto abs(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseAbs<Derived>(func.derived());
}
