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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/LGLCoeffs.h"
#include "tycho/detail/TranscriptionSizing.h"
#include "tycho/vector_functions.h"

namespace Tycho {

template <class Integrand, int CS, int XV, int PV> struct LGLReducedInteg_Impl {
    template <int V> using int_const = std::integral_constant<int, V>;

    template <int Elem> struct Weight : StaticScaleBase<Weight<Elem>> {
        static constexpr double value = LGLCoeffs<CS>::Reduced_Integral_Weights[Elem];
    };

    static auto Definition(const Integrand &integ, int xv, int pv) {
        constexpr int IRC = SZ_SUM<SZ_PROD<CS, XV>::value, CS, PV>::value;
        constexpr int XTV = SZ_SUM<XV, 1>::value;
        int irows = CS * xv + CS + pv;
        int xtv = xv + 1;
        auto Args = Arguments<IRC>(irows);

        auto xt1 = Args.template head<XTV>(xtv);
        auto t1 = xt1.template tail<1>(1);
        auto xtf = Args.template segment<XTV, SZ_PROD<CS - 1, XTV>::value>((CS - 1) * xtv, xtv);
        auto tf = xtf.template tail<1>(1);
        auto h = tf - t1;
        auto p = Args.template tail<PV>(pv);

        auto integral = constexpr_forwarding_loop(
            int_const<0>(), int_const<CS>(),
            [&](auto i, auto tup) {
                auto xti =
                    Args.template segment<XTV, SZ_PROD<i.value, XTV>::value>(i.value * xtv, xtv);
                auto xi = [&]() {
                    if constexpr (PV == 0)
                        return xti.template head<XV>(xv);
                    else
                        return StackedOutputs{xti.template head<XV>(xv), p};
                }();
                auto fi = Weight<i.value>::value * (integ.eval(xi));
                auto newtup = std::tuple_cat(tup, std::make_tuple(fi));
                if constexpr (i.value == (CS - 1))
                    return make_sum_tuple(newtup).scale(h);
                else
                    return newtup;
            },
            std::tuple{});

        return integral;
    }
};

template <class Integrand, int CS, int XV, int PV>
struct LGLIntegral
    : VectorExpression<LGLIntegral<Integrand, CS, XV, PV>,
                       LGLReducedInteg_Impl<Integrand, CS, XV, PV>, const Integrand &, int, int> {
    using Base =
        VectorExpression<LGLIntegral<Integrand, CS, XV, PV>,
                         LGLReducedInteg_Impl<Integrand, CS, XV, PV>, const Integrand &, int, int>;
    using Base::EnableVectorization;
    LGLIntegral() {}
    LGLIntegral(const Integrand &integ, int xv, int pv) : Base(integ, xv, pv) {}
};

} // namespace Tycho

