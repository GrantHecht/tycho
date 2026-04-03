// =============================================================================
// Compile-time named variable index tags for the static VF DSL.
//
// Provides XVar<I>, UVar<I>, PVar<I> tag types for named access to
// state, control, and parameter variables within Arguments<N>.
//
// Usage:
//   auto args = Arguments<5>();
//   auto v     = args[XVar<2>];   // same as args.coeff<2>()
//   auto theta = args[UVar<0>];   // same as args.coeff<XV + 1 + 0>() in ODE context
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

namespace tycho::vf {

// Raw index tag — maps directly to coeff<I>() on any Arguments<N>.
template <int I> struct VarTag {
    static constexpr int index = I;
};

// Semantic tag types — carry intent (state / control / parameter).
// When used with plain Arguments<N>, they map to a raw index.
// When used with a future ODEArguments<XV, UV, PV>, the offset is computed
// automatically from the ODE sizing.
template <int I> struct XVarTag : VarTag<I> {};
template <int I> struct UVarTag : VarTag<I> {};
template <int I> struct PVarTag : VarTag<I> {};

// Convenience constexpr variable templates
template <int I> inline constexpr XVarTag<I> XVar{};
template <int I> inline constexpr UVarTag<I> UVar{};
template <int I> inline constexpr PVarTag<I> PVar{};

} // namespace tycho::vf
