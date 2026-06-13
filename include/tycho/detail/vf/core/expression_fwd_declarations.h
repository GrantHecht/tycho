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

#include "tycho/detail/vf/core/vector_function_concepts.h"
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

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::vf {

template <class Derived, int IR, int OR> struct DenseFunctionBase;

template <int IR> struct Arguments;

template <int IR, int OR> struct GenericFunction;

template <int IR> struct GenericConditional;

template <class LHS, class RHS> struct ConditionalStatement;

template <class TestFunc, class TrueFunc, class FalseFunc> struct IfElseFunction;

template <int IR> struct GenericComparative;

template <class... Funcs> struct ComparativeFunction;

template <int IR, int OR, int ST> struct Segment;

template <int EL> struct Element;

template <int IR, int EL1, int... ELS> struct Elements;

template <int IR, int OR> struct Constant;
///////////////////////////////////////////////////////////////////////////////////////

template <int OR> struct Value;

template <class T> struct Is_Segment;
template <class T> struct Is_Arguments;
template <class T> struct Is_ScaledSegment;

template <class Derived, class Func, int IR, int OR> struct FunctionHolder;

///////////////////////////////////////////////////////////////////////////////////////
template <class OuterFunc, class InnerFunc> struct NestedFunction;

/// @brief Selects the result type and factory for composing two functions.
///
/// The primary template produces a `NestedFunction<OuterFunc, InnerFunc>`;
/// specializations collapse trivial or fusible compositions (e.g. nesting on
/// `Arguments`, or fusing two `Segment`s) into a simpler equivalent function.
/// @internal
/// @tparam OuterFunc  Outer (second-applied) function type.
/// @tparam InnerFunc  Inner (first-applied) function type.
/// @endinternal
template <class OuterFunc, class InnerFunc> struct NestedFunctionSelector {
    using type = NestedFunction<OuterFunc, InnerFunc>; ///< @brief Resulting nested function type.
    /// @brief Builds the nested function from outer and inner functions.
    /// @param ofunc  Outer function.
    /// @param ifunc  Inner function.
    /// @return The composed function.
    static decltype(auto) make_nested(OuterFunc ofunc, InnerFunc ifunc) {
        return type(ofunc, ifunc);
    }
};
/// @brief Specialization: nesting on identity `Arguments` returns the outer function.
/// @internal
/// @tparam OuterFunc  Outer function type.
/// @tparam IR  Input row count of the inner `Arguments`.
/// @endinternal
template <class OuterFunc, int IR> struct NestedFunctionSelector<OuterFunc, Arguments<IR>> {
    using type = OuterFunc; ///< @brief Resulting (unwrapped) function type.
    /// @brief Returns the outer function unchanged.
    /// @param ofunc  Outer function.
    /// @param ifunc  Inner identity `Arguments` (ignored).
    /// @return The outer function.
    static decltype(auto) make_nested(OuterFunc ofunc, Arguments<IR> ifunc) { return ofunc; }
};
/// @brief Specialization: an outer identity `Arguments` returns the inner function.
/// @internal
/// @tparam InnerFunc  Inner function type.
/// @tparam IR  Input row count of the outer `Arguments`.
/// @endinternal
template <class InnerFunc, int IR> struct NestedFunctionSelector<Arguments<IR>, InnerFunc> {
    using type = InnerFunc; ///< @brief Resulting (unwrapped) function type.
    /// @brief Returns the inner function unchanged.
    /// @param ofunc  Outer identity `Arguments` (ignored).
    /// @param ifunc  Inner function.
    /// @return The inner function.
    static decltype(auto) make_nested(Arguments<IR> ofunc, InnerFunc ifunc) { return ifunc; }
};
/// @brief Specialization: fuses two nested `Segment`s into a single `Segment`.
/// @internal
/// @tparam IR  Inner segment input rows.
/// @tparam OR  Shared intermediate size (inner output / outer input).
/// @tparam ST  Inner segment start offset.
/// @tparam OR2  Outer segment output rows.
/// @tparam ST2  Outer segment start offset.
/// @endinternal
template <int IR, int OR, int ST, int OR2, int ST2>
struct NestedFunctionSelector<Segment<OR, OR2, ST2>, Segment<IR, OR, ST>> {
    /// @brief Builds the fused segment with combined start offsets.
    /// @param ofunc  Outer segment.
    /// @param ifunc  Inner segment.
    /// @return The single equivalent `Segment`.
    static decltype(auto) make_nested(Segment<OR, OR2, ST2> ofunc, Segment<IR, OR, ST> ifunc) {
        return Segment<IR, OR2, SZ_SUM<ST, ST2>::value>(ifunc.input_rows(), ofunc.output_rows(),
                                                        ifunc.seg_start_ + ofunc.seg_start_);
    }
};

/// @brief Specialization: fuses `Elements` selected from a `Segment` into shifted `Elements`.
/// @internal
/// @tparam IR  Inner segment input rows.
/// @tparam OR  Shared intermediate size.
/// @tparam ST  Inner segment start offset.
/// @tparam EL1  First selected element index.
/// @tparam ELS  Remaining selected element indices.
/// @endinternal
template <int IR, int OR, int ST, int EL1, int... ELS>
struct NestedFunctionSelector<Elements<OR, EL1, ELS...>, Segment<IR, OR, ST>> {
    /// @brief Builds the fused/shifted elements selection.
    /// @param ofunc  Outer element selection.
    /// @param ifunc  Inner segment.
    /// @return The fused function (shifted `Elements`, or a `NestedFunction` fallback).
    static decltype(auto) make_nested(Elements<OR, EL1, ELS...> ofunc, Segment<IR, OR, ST> ifunc) {
        if constexpr (IR >= 0 && OR >= 0 && ST >= 0 && EL1 >= 0) {
            return Elements<IR, EL1 + ST, ELS + ST...>(ifunc.input_rows());
        } else {
            return NestedFunction<decltype(ofunc), decltype(ifunc)>(ofunc, ifunc);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////

template <class Func1, class Func2, class... Funcs> struct StackedOutputsSelector;
template <class Func1, class Func2, class... Funcs>
    requires MutuallyStackable<Func1, Func2, Funcs...>
struct StackedOutputs;

template <class Func> struct DynamicStackedOutputs;

///////////////////////////////////////////////////////////////////////////////////////
template <class Func, int UP, int LP> struct PaddedOutput;
template <class Func, int IRC, int ORC> struct ParsedInput;

///////////////////////////////////////////////////////////////////////////////////////

template <class Func1, class Func2> struct TwoFunctionSum;

template <class Func1, class Func2, class... Funcs> struct MultiFunctionSum;

template <class Func1, class Func2> struct FunctionDifference;

///////////////////////////////////////////////////////////////////////////////////////
template <class Func, int Rows, int Cols> struct MatrixFunction;

template <class Func, int MRows, int MCols, int MMajor> struct MatrixFunctionView;
template <class MatFunc1, class MatFunc2> struct MatrixFunctionProduct;

template <int Size, int Major> struct MatrixInverse;

///////////////////////////////////////////////////////////////////////////////////////
template <class Func> struct Scaled;

template <class Func> struct RowScaled;

template <class Func, int MRows> struct MatrixScaled;

template <class Func, class Value> struct StaticScaled;

/// @brief CRTP marker base tagging a function as statically (compile-time) scaled.
/// @ingroup vf
/// @tparam Derived  The concrete statically-scaled function type (CRTP).
template <class Derived> struct StaticScaleBase {};
///////////////////////////////////////////////////////////////////////////////////////

template <class Func> struct CwiseSin;

template <class Func> struct CwiseCos;

template <class Func> struct CwiseTan;

template <class Func> struct CwiseArcSin;

template <class Func> struct CwiseArcCos;

template <class Func> struct CwiseArcTan;

template <class Func> struct CwiseSquare;

template <class Func> struct CwiseInverse;

template <class Func> struct CwiseSqrt;

template <class Func> struct CwiseExp;

template <class Func> struct CwisePow;

template <class Func> struct CwiseLog;

template <class Func> struct CwiseSinH;

template <class Func> struct CwiseCosH;

template <class Func> struct CwiseTanH;

template <class Func> struct CwiseArcSinH;

template <class Func> struct CwiseArcCosH;

template <class Func> struct CwiseArcTanH;

template <class Func> struct CwiseAbs;

struct ArcTan2Op;

template <class YFunc, class XFunc> struct ArcTan2;

template <class Func> struct SignFunction;

///////////////////////////////////////////////////////////////////////////////////////

template <class Func> struct FunctionPlusVector;

///////////////////////////////////////////////////////////////////////////////////////

template <class Func> struct CwiseScaled;

template <class Func> struct CwiseSum;

///////////////////////////////////////////////////////////////////////////////////////

template <int IR> struct Normalized;

template <int IR, int PW> struct NormalizedPower;

template <int IR> struct Norm;
template <int IR> struct SquaredNorm;
template <int IR> struct InverseNorm;
template <int IR> struct InverseSquaredNorm;
template <int IR, int PW> struct NormPower;
template <int IR, int PW> struct InverseNormPower;
///////////////////////////////////////////////////////////////////////////////////////

template <int IR> struct DotProduct;

struct CrossProduct;

template <class Func1, class Func2> struct FunctionCrossProduct;

template <class Func1, class Func2> struct FunctionQuatProduct;

template <class Func1, class Func2> struct FunctionDotProduct;

template <class VecFunc, class ScalFunc> struct VectorScalarFunctionProduct;

template <class VecFunc, class ScalFunc> struct VectorScalarFunctionDivision;

template <class Func1, class Func2> struct CwiseFunctionProduct;
///////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template <class Func> struct CallAndAppend;

} // namespace tycho::vf
