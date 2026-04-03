// =============================================================================
// Compile-time named variable index tags for the static VF DSL.
//
// XVar<I> — state variable index I. Usable on any DenseFunctionBase via
//           operator[], mapping directly to coeff<I>().
//
// UVar<I> — control variable index I. Only usable on ODEArguments<XV,UV,PV>,
//           which computes the correct offset (XV + 1 + I).
//
// PVar<I> — parameter variable index I. Only usable on ODEArguments<XV,UV,PV>,
//           which computes the correct offset (XV + 1 + UV + I).
//
// Usage:
//   auto args = Arguments<5>();
//   auto v = args[XVar<2>];       // coeff<2>() — raw index, works anywhere
//
//   auto ode_args = ODEArguments<3, 1, 0>();
//   auto v     = ode_args[XVar<2>];   // coeff<2>()
//   auto theta = ode_args[UVar<0>];   // coeff<4>() — auto offset XV+1
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

namespace tycho::vf {

// Compile-time variable index tag (base type for semantic tags).
template <int I> struct VarTag {
    static_assert(I >= 0, "Variable index must be non-negative");
    static constexpr int index = I;
};

// Semantic tag types — carry intent (state / control / parameter).
// XVarTag: usable on any DenseFunctionBase (raw index).
// UVarTag, PVarTag: require ODEArguments for offset-aware dispatch.
template <int I> struct XVarTag : VarTag<I> {};
template <int I> struct UVarTag : VarTag<I> {};
template <int I> struct PVarTag : VarTag<I> {};

// Convenience constexpr variable templates
template <int I> inline constexpr XVarTag<I> XVar{};
template <int I> inline constexpr UVarTag<I> UVar{};
template <int I> inline constexpr PVarTag<I> PVar{};

} // namespace tycho::vf
