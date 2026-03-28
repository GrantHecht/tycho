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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once
#include "tycho/detail/vf/common/common_functions.h"

////////////////////////////////////////////////////////////////////////////
/////////////////////// CMath Overloads ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR> auto sin(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseSin<Derived>(func.derived());
}

template <class Derived, int IR> auto cos(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseCos<Derived>(func.derived());
}

template <class Derived, int IR> auto tan(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseTan<Derived>(func.derived());
}

template <class Derived, int IR> auto asin(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcSin<Derived>(func.derived());
}

template <class Derived, int IR> auto acos(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcCos<Derived>(func.derived());
}

template <class Derived, int IR> auto atan(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcTan<Derived>(func.derived());
}

template <class Derived1, int IR1, class Derived2, int IR2>
auto atan2(const tycho::vf::DenseFunctionBase<Derived1, IR1, 1> &yf,
           const tycho::vf::DenseFunctionBase<Derived2, IR2, 1> &xf) {
    return tycho::vf::ArcTan2Op().eval(
        tycho::vf::StackedOutputs<Derived1, Derived2>(yf.derived(), xf.derived()));
}

template <class Derived, int IR> auto sqrt(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseSqrt<Derived>(func.derived());
}

template <class Derived, int IR> auto exp(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseExp<Derived>(func.derived());
}

template <class Derived, int IR> auto log(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseLog<Derived>(func.derived());
}

template <class Derived, int IR> auto sinh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseSinH<Derived>(func.derived());
}

template <class Derived, int IR> auto cosh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseCosH<Derived>(func.derived());
}

template <class Derived, int IR> auto tanh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseTanH<Derived>(func.derived());
}

template <class Derived, int IR> auto asinh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcSinH<Derived>(func.derived());
}

template <class Derived, int IR> auto acosh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcCosH<Derived>(func.derived());
}

template <class Derived, int IR> auto atanh(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseArcTanH<Derived>(func.derived());
}
template <class Derived, int IR> auto abs(const tycho::vf::DenseFunctionBase<Derived, IR, 1> &func) {
    return tycho::vf::CwiseAbs<Derived>(func.derived());
}
