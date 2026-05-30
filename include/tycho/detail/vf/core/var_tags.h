// =============================================================================
// Compile-time named variable tags for the static ODE DSL.
//
// All tags are usable only on ODEArguments<XV, UV, PV>, which computes the
// correct offset into the ODE vector layout:
//   [x0..x(XV-1), t, u0..u(UV-1), p0..p(PV-1)]
//
// Scalar tags:
//   XVar<I>  — state variable at index I
//   UVar<I>  — control variable at index XV + 1 + I
//   PVar<I>  — parameter variable at index XV + 1 + UV + I
//   TVar     — time variable at index XV
//
// Full-vector tags:
//   XVec     — full state vector [0, XV)
//   UVec     — full control vector [XV+1, XV+1+UV)
//   PVec     — full parameter vector [XV+1+UV, XV+1+UV+PV)
//
// Sub-vector tags:
//   XSeg<Start, Size>  — state sub-vector starting at Start with Size elements
//   USeg<Start, Size>  — control sub-vector (auto-offset by XV+1)
//   PSeg<Start, Size>  — parameter sub-vector (auto-offset by XV+1+UV)
//
// Usage:
//   auto args = ODEArguments<6, 2, 1>();
//   auto x0    = args[XVar<0>];      // state scalar
//   auto u1    = args[UVar<1>];      // control scalar (auto-offset)
//   auto t     = args[TVar];         // time scalar
//   auto pos   = args[XSeg<0, 3>];   // state sub-vector (positions)
//   auto vel   = args[XSeg<3, 3>];   // state sub-vector (velocities)
//   auto x_all = args[XVec];         // full state vector
//   auto u_all = args[UVec];         // full control vector
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

namespace tycho::vf {

// ---- Scalar tags ----

/// @brief Compile-time variable index tag (base type for semantic tags).
/// @ingroup vf
/// @tparam I  Non-negative variable index.
template <int I> struct VarTag {
    static_assert(I >= 0, "Variable index must be non-negative");
    static constexpr int index = I; ///< @brief The stored variable index.
};

/// @brief State-variable tag at logical state index @p I.
/// @ingroup vf
/// @tparam I  State-variable index.
template <int I> struct XVarTag : VarTag<I> {};
/// @brief Control-variable tag at logical control index @p I.
/// @ingroup vf
/// @tparam I  Control-variable index.
template <int I> struct UVarTag : VarTag<I> {};
/// @brief Parameter-variable tag at logical parameter index @p I.
/// @ingroup vf
/// @tparam I  Parameter-variable index.
template <int I> struct PVarTag : VarTag<I> {};

/// @brief Tag denoting the scalar time variable.
/// @ingroup vf
struct TVarTag {};

/// @brief Inline variable: the state-variable tag at index @p I.
/// @tparam I  State-variable index.
template <int I> inline constexpr XVarTag<I> XVar{};
/// @brief Inline variable: the control-variable tag at index @p I.
/// @tparam I  Control-variable index.
template <int I> inline constexpr UVarTag<I> UVar{};
/// @brief Inline variable: the parameter-variable tag at index @p I.
/// @tparam I  Parameter-variable index.
template <int I> inline constexpr PVarTag<I> PVar{};
/// @brief Inline variable: the time-variable tag.
inline constexpr TVarTag TVar{};

// ---- Full-vector tags ----

/// @brief Tag denoting the full state vector.
/// @ingroup vf
struct XVecTag {};
/// @brief Tag denoting the full control vector.
/// @ingroup vf
struct UVecTag {};
/// @brief Tag denoting the full parameter vector.
/// @ingroup vf
struct PVecTag {};

/// @brief Inline variable: the full-state-vector tag.
inline constexpr XVecTag XVec{};
/// @brief Inline variable: the full-control-vector tag.
inline constexpr UVecTag UVec{};
/// @brief Inline variable: the full-parameter-vector tag.
inline constexpr PVecTag PVec{};

// ---- Sub-vector tags ----

/// @brief Tag denoting a contiguous state sub-vector.
/// @ingroup vf
/// @tparam Start  Offset of the first element within the state vector.
/// @tparam Size  Number of elements in the sub-vector.
template <int Start, int Size> struct XSegTag {
    static_assert(Start >= 0, "XSeg start must be non-negative");
    static_assert(Size > 0, "XSeg size must be positive");
};
/// @brief Tag denoting a contiguous control sub-vector.
/// @ingroup vf
/// @tparam Start  Offset of the first element within the control vector.
/// @tparam Size  Number of elements in the sub-vector.
template <int Start, int Size> struct USegTag {
    static_assert(Start >= 0, "USeg start must be non-negative");
    static_assert(Size > 0, "USeg size must be positive");
};
/// @brief Tag denoting a contiguous parameter sub-vector.
/// @ingroup vf
/// @tparam Start  Offset of the first element within the parameter vector.
/// @tparam Size  Number of elements in the sub-vector.
template <int Start, int Size> struct PSegTag {
    static_assert(Start >= 0, "PSeg start must be non-negative");
    static_assert(Size > 0, "PSeg size must be positive");
};

/// @brief Inline variable: the state sub-vector tag.
/// @tparam Start  Offset within the state vector.
/// @tparam Size  Number of elements.
template <int Start, int Size> inline constexpr XSegTag<Start, Size> XSeg{};
/// @brief Inline variable: the control sub-vector tag.
/// @tparam Start  Offset within the control vector.
/// @tparam Size  Number of elements.
template <int Start, int Size> inline constexpr USegTag<Start, Size> USeg{};
/// @brief Inline variable: the parameter sub-vector tag.
/// @tparam Start  Offset within the parameter vector.
/// @tparam Size  Number of elements.
template <int Start, int Size> inline constexpr PSegTag<Start, Size> PSeg{};

} // namespace tycho::vf
