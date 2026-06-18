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

#include "tycho/detail/optimal_control/transcription/transcription_sizing.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::Arguments;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::StackedOutputs;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @internal
/// @brief Expression builder for the trapezoidal-rule integral of an integrand.
///
/// Builds the VectorFunction expression approximating @f$\int g\,dt@f$ over one
/// interval as @f$\tfrac{h}{2}(g_0+g_1)@f$.
/// @tparam Integrand  The scalar integrand expression type.
/// @tparam XV         ODE state-vector dimension.
/// @tparam PV         ODE parameter-vector dimension.
/// @endinternal
template <class Integrand, int XV, int PV> struct TrapInteg_Impl {
    /// @internal
    /// @brief Build the trapezoidal integral expression over one interval.
    /// @param integ  The scalar integrand.
    /// @param xv     Runtime state dimension.
    /// @param pv     Runtime parameter dimension.
    /// @return The VectorFunction expression for the interval integral.
    /// @endinternal
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

/// @ingroup optimal_control
/// @brief VectorFunction computing the trapezoidal-rule integral of an integrand.
///
/// Wraps @c TrapInteg_Impl as a usable expression evaluating the per-interval
/// integral @f$\tfrac{h}{2}(g_0+g_1)@f$ and its derivatives.
/// @tparam Integrand  The scalar integrand expression type.
/// @tparam XV         ODE state-vector dimension.
/// @tparam PV         ODE parameter-vector dimension.
template <class Integrand, int XV, int PV>
struct TrapezoidalIntegral
    : VectorExpression<TrapezoidalIntegral<Integrand, XV, PV>, TrapInteg_Impl<Integrand, XV, PV>,
                       const Integrand &, int, int> {
    /// @brief Convenience alias for the VectorExpression base class.
    using Base = VectorExpression<TrapezoidalIntegral<Integrand, XV, PV>,
                                  TrapInteg_Impl<Integrand, XV, PV>, const Integrand &, int, int>;

    /// @brief Default constructor; leaves the integrand unset.
    TrapezoidalIntegral() {}
    /// @brief Construct from an integrand and runtime dimensions.
    /// @param integ  The scalar integrand.
    /// @param xv     Runtime state dimension.
    /// @param pv     Runtime parameter dimension.
    TrapezoidalIntegral(const Integrand &integ, int xv, int pv) : Base(integ, xv, pv) {}
};

} // namespace tycho::oc
