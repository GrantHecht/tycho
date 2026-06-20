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

#include "tycho/detail/optimal_control/transcription/lgl_coeffs.h"
#include "tycho/detail/optimal_control/transcription/transcription_sizing.h"
#include "tycho/vector_functions.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::constexpr_forwarding_loop;
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::Arguments;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::StackedOutputs;
using vf::StaticScaleBase;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @internal
/// @brief Expression builder for the LGL reduced-quadrature integral of an integrand.
///
/// Builds the VectorFunction expression that approximates @f$\int g\,dt@f$ over
/// one collocation interval using the LGL reduced integral weights.
/// @tparam Integrand  The scalar integrand expression type.
/// @tparam CS         Number of cardinal states per interval.
/// @tparam XV         ODE state-vector dimension.
/// @tparam PV         ODE parameter-vector dimension.
/// @endinternal
template <class Integrand, int CS, int XV, int PV> struct LGLReducedInteg_Impl {
    /// @brief Convenience alias for a compile-time integer constant.
    /// @tparam V  The integer value.
    template <int V> using int_const = std::integral_constant<int, V>;

    /// @internal
    /// @brief The LGL reduced-integral weight of cardinal state @p Elem as a static scale.
    /// @tparam Elem  The cardinal-state index.
    /// @endinternal
    template <int Elem> struct Weight : StaticScaleBase<Weight<Elem>> {
        static constexpr double value =
            LGLCoeffs<CS>::Reduced_Integral_Weights[Elem]; ///< The weight value.
    };

    /// @internal
    /// @brief Build the integral expression over one collocation interval.
    /// @param integ  The scalar integrand.
    /// @param xv     Runtime state dimension.
    /// @param pv     Runtime parameter dimension.
    /// @return The VectorFunction expression for the interval integral.
    /// @endinternal
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

/// @ingroup optimal_control
/// @brief VectorFunction computing the LGL reduced-quadrature integral of an integrand.
///
/// Wraps @c LGLReducedInteg_Impl as a usable expression that evaluates the
/// per-interval integral @f$\int g\,dt@f$ and its derivatives.
/// @tparam Integrand  The scalar integrand expression type.
/// @tparam CS         Number of cardinal states per interval.
/// @tparam XV         ODE state-vector dimension.
/// @tparam PV         ODE parameter-vector dimension.
template <class Integrand, int CS, int XV, int PV>
struct LGLIntegral
    : VectorExpression<LGLIntegral<Integrand, CS, XV, PV>,
                       LGLReducedInteg_Impl<Integrand, CS, XV, PV>, const Integrand &, int, int> {
    /// @brief Convenience alias for the VectorExpression base class.
    using Base =
        VectorExpression<LGLIntegral<Integrand, CS, XV, PV>,
                         LGLReducedInteg_Impl<Integrand, CS, XV, PV>, const Integrand &, int, int>;
    using Base::enable_vectorization_;
    /// @brief Default constructor; leaves the integrand unset.
    LGLIntegral() {}
    /// @brief Construct from an integrand and runtime dimensions.
    /// @param integ  The scalar integrand.
    /// @param xv     Runtime state dimension.
    /// @param pv     Runtime parameter dimension.
    LGLIntegral(const Integrand &integ, int xv, int pv) : Base(integ, xv, pv) {}
};

} // namespace tycho::oc
