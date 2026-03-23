// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Overloads the common mathematical functions contained in cmath for ASSET scalar
// functions.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once
#include "tycho/detail/CommonFunctions.h"

////////////////////////////////////////////////////////////////////////////
/////////////////////// CMath Overloads ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR> auto sin(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseSin<Derived>(func.derived());
}

template <class Derived, int IR> auto cos(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseCos<Derived>(func.derived());
}

template <class Derived, int IR> auto tan(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseTan<Derived>(func.derived());
}

template <class Derived, int IR> auto asin(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseArcSin<Derived>(func.derived());
}

template <class Derived, int IR> auto acos(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseArcCos<Derived>(func.derived());
}

template <class Derived, int IR> auto atan(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseArcTan<Derived>(func.derived());
}

template <class Derived1, int IR1, class Derived2, int IR2>
auto atan2(const Tycho::DenseFunctionBase<Derived1, IR1, 1> &yf,
           const Tycho::DenseFunctionBase<Derived2, IR2, 1> &xf) {
    return Tycho::ArcTan2Op().eval(
        Tycho::StackedOutputs<Derived1, Derived2>(yf.derived(), xf.derived()));
}

template <class Derived, int IR> auto sqrt(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseSqrt<Derived>(func.derived());
}

template <class Derived, int IR> auto exp(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseExp<Derived>(func.derived());
}

template <class Derived, int IR> auto log(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseLog<Derived>(func.derived());
}

template <class Derived, int IR> auto sinh(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseSinH<Derived>(func.derived());
}

template <class Derived, int IR> auto cosh(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseCosH<Derived>(func.derived());
}

template <class Derived, int IR> auto tanh(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseTanH<Derived>(func.derived());
}

template <class Derived, int IR> auto asinh(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseArcSinH<Derived>(func.derived());
}

template <class Derived, int IR> auto acosh(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseArcCosH<Derived>(func.derived());
}

template <class Derived, int IR> auto atanh(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseArcTanH<Derived>(func.derived());
}
template <class Derived, int IR> auto abs(const Tycho::DenseFunctionBase<Derived, IR, 1> &func) {
    return Tycho::CwiseAbs<Derived>(func.derived());
}
