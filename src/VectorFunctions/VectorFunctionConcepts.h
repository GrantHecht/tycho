// =============================================================================
// Copyright 2026-present Grant R. Hecht
// Licensed under the Apache License, Version 2.0 — see LICENSE.txt.
//
// Tycho-original file:
//   - Adds C++20 concepts for VectorFunction CRTP traits and composition
// =============================================================================

#pragma once

#include <concepts>

#include "pch.h"

namespace Tycho {

namespace detail {

template <class T> struct IsSuperScalarImpl : std::false_type {};

template <class Scalar, int Sz>
struct IsSuperScalarImpl<Eigen::Array<Scalar, Sz, 1>> : std::bool_constant<(Sz > 1)> {};

template <class T> using VectorFunctionType = std::remove_cvref_t<T>;

} // namespace detail

template <typename T>
concept IsSuperScalar = detail::IsSuperScalarImpl<detail::VectorFunctionType<T>>::value;

template <typename T>
concept IsDoubleScalar = std::same_as<detail::VectorFunctionType<T>, double>;

template <typename T>
concept Vectorizable =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::IsVectorizable); };

template <typename T>
concept LinearVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::IsLinearFunction); };

template <typename T>
concept HasDiagonalJac =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::HasDiagonalJacobian); };

template <typename T>
concept IsGenericVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::IsGenericFunction); };

template <typename T>
concept StaticallySized = requires {
    requires(detail::VectorFunctionType<T>::IRC > 0);
    requires(detail::VectorFunctionType<T>::ORC > 0);
};

template <typename T>
concept DynamicallySized = requires {
    requires(detail::VectorFunctionType<T>::IRC < 0 || detail::VectorFunctionType<T>::ORC < 0);
};

template <typename T>
concept VFSized = requires(const detail::VectorFunctionType<T> &t) {
    { t.IRows() } -> std::same_as<int>;
    { t.ORows() } -> std::same_as<int>;
};

template <typename Inner, typename Outer>
concept Composable = VFSized<Inner> && VFSized<Outer> && requires {
    requires(detail::VectorFunctionType<Inner>::ORC < 0 ||
             detail::VectorFunctionType<Outer>::IRC < 0 ||
             detail::VectorFunctionType<Inner>::ORC == detail::VectorFunctionType<Outer>::IRC);
};

template <typename F1, typename F2>
concept Stackable = VFSized<F1> && VFSized<F2> && requires {
    requires(detail::VectorFunctionType<F1>::IRC < 0 || detail::VectorFunctionType<F2>::IRC < 0 ||
             detail::VectorFunctionType<F1>::IRC == detail::VectorFunctionType<F2>::IRC);
};

namespace detail {

// Checks that Head is Stackable with every type in Tail.
template <class Head, class... Tail>
concept StackableWithAll = (Stackable<Head, Tail> && ...);

// Recursive helper: checks all N*(N-1)/2 upper-triangle pairs.
template <class...>
struct AllPairsStackableHelper : std::true_type {};

template <class Head, class... Tail>
struct AllPairsStackableHelper<Head, Tail...>
    : std::bool_constant<StackableWithAll<Head, Tail...> &&
                         AllPairsStackableHelper<Tail...>::value> {};

} // namespace detail

// Satisfied iff every pair of types in Fs... satisfies Stackable.
// Uses the transitivity of IRC equality for the static-size case, but
// explicitly checks all pairs to close the dynamic-IRC edge case where
// F1::IRC==-1 could mask an incompatible static F2::IRC vs F3::IRC.
template <class... Fs>
concept MutuallyStackable = detail::AllPairsStackableHelper<Fs...>::value;

} // namespace Tycho
