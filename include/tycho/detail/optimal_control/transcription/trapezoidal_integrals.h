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

#include "tycho/detail/optimal_control/transcription/transcription_sizing.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

template <class Integrand, int XV, int PV> struct TrapInteg_Impl {
    static auto Definition(const Integrand &integ, int xv, int pv) {
        constexpr int IRC = SZ_SUM<SZ_PROD<2, XV>::value, 2, PV>::value;
        constexpr int XTV = SZ_SUM<XV, 1>::value;
        int irows = 2 * xv + 2 + pv;
        int xtv = xv + 1;

        auto Args = Arguments<IRC>(irows);

        auto xt1 = Args.template head<XTV>(xtv);
        auto x1 = xt1.template head<XV>(xv);
        auto t1 = xt1.template tail<1>(1);

        auto xt2 = Args.template segment<XTV, XTV>(xtv, xtv);
        auto x2 = xt2.template head<XV>(xv);
        auto t2 = xt2.template tail<1>(1);

        auto h = t2 - t1;

        auto p12 = Args.template tail<PV>(pv);

        auto X1 = StackedOutputs{x1, p12};
        auto X2 = StackedOutputs{x2, p12};

        auto V1 = integ.eval(X1);
        auto V2 = integ.eval(X2);

        auto trapint = (h / 2.0) * (V1 + V2);

        return trapint;
    }
};

template <class Integrand, int XV, int PV>
struct TrapezoidalIntegral
    : VectorExpression<TrapezoidalIntegral<Integrand, XV, PV>, TrapInteg_Impl<Integrand, XV, PV>,
                       const Integrand &, int, int> {
    using Base = VectorExpression<TrapezoidalIntegral<Integrand, XV, PV>,
                                  TrapInteg_Impl<Integrand, XV, PV>, const Integrand &, int, int>;

    TrapezoidalIntegral() {}
    TrapezoidalIntegral(const Integrand &integ, int xv, int pv) : Base(integ, xv, pv) {}
};

} // namespace tycho::oc

