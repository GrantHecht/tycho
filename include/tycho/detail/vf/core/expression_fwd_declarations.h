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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

namespace Tycho {

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

template <class OuterFunc, class InnerFunc> struct NestedFunctionSelector {
    using type = NestedFunction<OuterFunc, InnerFunc>;
    static decltype(auto) make_nested(OuterFunc ofunc, InnerFunc ifunc) {
        return type(ofunc, ifunc);
    }
};
template <class OuterFunc, int IR> struct NestedFunctionSelector<OuterFunc, Arguments<IR>> {
    using type = OuterFunc;
    static decltype(auto) make_nested(OuterFunc ofunc, Arguments<IR> ifunc) { return ofunc; }
};
template <class InnerFunc, int IR> struct NestedFunctionSelector<Arguments<IR>, InnerFunc> {
    using type = InnerFunc;
    static decltype(auto) make_nested(Arguments<IR> ofunc, InnerFunc ifunc) { return ifunc; }
};
template <int IR, int OR, int ST, int OR2, int ST2>
struct NestedFunctionSelector<Segment<OR, OR2, ST2>, Segment<IR, OR, ST>> {
    static decltype(auto) make_nested(Segment<OR, OR2, ST2> ofunc, Segment<IR, OR, ST> ifunc) {
        return Segment<IR, OR2, SZ_SUM<ST, ST2>::value>(ifunc.IRows(), ofunc.ORows(),
                                                        ifunc.SegStart + ofunc.SegStart);
    }
};

template <int IR, int OR, int ST, int EL1, int... ELS>
struct NestedFunctionSelector<Elements<OR, EL1, ELS...>, Segment<IR, OR, ST>> {
    static decltype(auto) make_nested(Elements<OR, EL1, ELS...> ofunc, Segment<IR, OR, ST> ifunc) {
        if constexpr (IR >= 0 && OR >= 0 && ST >= 0 && EL1 >= 0) {
            return Elements<IR, EL1 + ST, ELS + ST...>(ifunc.IRows());
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

template <class Func> struct VectorMinusFunction;

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

} // namespace Tycho
