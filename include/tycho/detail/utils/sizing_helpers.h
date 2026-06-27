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

namespace tycho::utils {

/// @internal
/// @brief Compile-time size helper: treats negative values as dynamic (-1).
template <int Arg> struct SZ_NEG {
    enum { value = ((Arg < 0) ? -1 : Arg) };
};

/// @internal
/// @brief Compile-time sum of two static sizes; propagates dynamic (-1) if
/// either operand is negative.
template <int Arg1, int Arg2> struct SZ_SUMOP {
    enum { value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 + Arg2), Identity = 0 };
};

/// @internal
/// @brief Compile-time difference of two static sizes; propagates dynamic
/// (-1) if either operand is negative.
template <int Arg1, int Arg2> struct SZ_DIFF {
    enum { value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 - Arg2), Identity = 0 };
};

/// @internal
/// @brief Compile-time maximum of two static sizes; propagates dynamic (-1)
/// if either operand is negative.
template <int Arg1, int Arg2> struct SZ_MAXOP {
    enum { value = ((Arg1 < 0 || Arg2 < 0) ? -1 : ((Arg1 > Arg2) ? Arg1 : Arg2)), Identity = 0 };
};

/// @internal
/// @brief Compile-time minimum of two static sizes; propagates dynamic (-1)
/// if either operand is negative.
template <int Arg1, int Arg2> struct SZ_MINOP {
    enum {
        value = ((Arg1 < 0 || Arg2 < 0) ? -1 : ((Arg1 > Arg2) ? Arg2 : Arg1)),
        Identity = 10000
    };
};

/// @internal
/// @brief Compile-time maximum of two sizes without dynamic-propagation
/// (treats negative values as real integers).
template <int Arg1, int Arg2> struct SZ_MAXREALOP {
    enum { value = ((Arg1 > Arg2) ? Arg1 : Arg2), Identity = 0 };
};

/// @internal
/// @brief Compile-time minimum of two sizes without dynamic-propagation
/// (intended: treats negative values as real integers and returns the smaller
/// one; compare SZ_MAXREALOP).
/// @note The current implementation is known-incorrect: it propagates -1 when
/// either argument is negative (wrong for the "real integer" contract) and
/// returns the LARGER argument when `Arg1 > Arg2` (a max, not a min). This
/// struct has no call sites and is dead code — fix the body before first use.
template <int Arg1, int Arg2> struct SZ_MINREALOP {
    enum { value = ((Arg1 < 0 || Arg2 < 0) ? -1 : ((Arg1 > Arg2) ? Arg1 : Arg2)), Identity = 0 };
};

/// @internal
/// @brief Compile-time left-biased sum: propagates dynamic (-1) only when
/// the left (first) operand is negative.
template <int Arg1, int Arg2> struct SZ_LSUMOP {
    enum { value = ((Arg1 < 0) ? -1 : Arg1 + Arg2), Identity = 0 };
};

/// @internal
/// @brief Compile-time product of two static sizes; propagates dynamic (-1)
/// if either operand is negative.
template <int Arg1, int Arg2> struct SZ_PRODOP {
    enum { value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 * Arg2), Identity = 1 };
};

/// @internal
/// @brief Compile-time integer division of two static sizes; propagates
/// dynamic (-1) if either operand is negative.
template <int Arg1, int Arg2> struct SZ_DIVOP {
    enum { value = ((Arg1 < 0 || Arg2 < 0) ? -1 : Arg1 / Arg2), Identity = 1 };
};

/// @internal
/// @brief Variadic compile-time binary fold over a size-operation @p SZ_OP.
/// Base cases handle zero, one, and two arguments via the operation's
/// `Identity` value.
template <template <int, int> class SZ_OP, int... Args> struct SZ_BINOP {};

/// @internal
/// @brief Recursive case: fold Arg1 and Arg2 then recurse over the rest.
template <template <int, int> class SZ_OP, int Arg1, int Arg2, int... Args>
struct SZ_BINOP<SZ_OP, Arg1, Arg2, Args...> {
    enum { value = SZ_OP<SZ_OP<Arg1, Arg2>::value, SZ_BINOP<SZ_OP, Args...>::value>::value };
};

/// @internal
/// @brief Single-argument base case: fold Arg1 with the operation's Identity.
template <template <int, int> class SZ_OP, int Arg1> struct SZ_BINOP<SZ_OP, Arg1> {
    enum { value = SZ_OP<Arg1, SZ_OP<1, 1>::Identity>::value };
};

/// @internal
/// @brief Zero-argument base case: returns the operation's Identity value.
template <template <int, int> class SZ_OP> struct SZ_BINOP<SZ_OP> {
    enum { value = SZ_OP<0, 0>::Identity };
};

/// @internal
/// @brief Variadic compile-time sum alias for SZ_BINOP<SZ_SUMOP, Args...>.
template <int... Args> using SZ_SUM = SZ_BINOP<SZ_SUMOP, Args...>;

/// @internal
/// @brief Variadic compile-time left-biased sum alias for SZ_BINOP<SZ_LSUMOP, Args...>.
template <int... Args> using SZ_LSUM = SZ_BINOP<SZ_LSUMOP, Args...>;

/// @internal
/// @brief Variadic compile-time product alias for SZ_BINOP<SZ_PRODOP, Args...>.
template <int... Args> using SZ_PROD = SZ_BINOP<SZ_PRODOP, Args...>;

/// @internal
/// @brief Variadic compile-time division alias for SZ_BINOP<SZ_DIVOP, Args...>.
template <int... Args> using SZ_DIV = SZ_BINOP<SZ_DIVOP, Args...>;

/// @internal
/// @brief Variadic compile-time maximum alias for SZ_BINOP<SZ_MAXOP, Args...>.
template <int... Args> using SZ_MAX = SZ_BINOP<SZ_MAXOP, Args...>;

/// @internal
/// @brief Variadic compile-time minimum alias for SZ_BINOP<SZ_MINOP, Args...>.
template <int... Args> using SZ_MIN = SZ_BINOP<SZ_MINOP, Args...>;

} // namespace tycho::utils
