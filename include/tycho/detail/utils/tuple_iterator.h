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

#include <functional> // std::invoke
#include <tuple>      // std::tuple
#include <type_traits>

namespace tycho::utils {

//////////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Apply a function to every element of a tuple in forward order.
/// @tparam TupleType  The tuple type.
/// @tparam FunctionType  A callable invocable with each tuple element type.
template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {}

/// @internal
/// @brief Recursive worker for tuple_for_each — visits element I, then I+1.
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
void tuple_for_each(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I>(std::forward<TupleType>(t)));
    tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

/// @internal
/// @brief Entry point: apply @p f to every element of tuple @p t in forward order.
template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
    tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

/// @internal
/// @brief Base case for reverse_tuple_for_each — no-op at index 0.
template <typename TupleType, typename FunctionType>
void reverse_tuple_for_each(TupleType &&, FunctionType, std::integral_constant<size_t, 0>) {}

/// @internal
/// @brief Recursive worker for reverse_tuple_for_each — visits element I-1,
/// then I-2 down to 0.
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<I != 0>::type>
void reverse_tuple_for_each(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I - 1>(std::forward<TupleType>(t)));
    reverse_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I - 1>());
}

/// @internal
/// @brief Apply @p f to every element of tuple @p t in reverse order.
template <typename TupleType, typename FunctionType>
void reverse_tuple_for_each(TupleType &&t, FunctionType f) {
    reverse_tuple_for_each(
        std::forward<TupleType>(t), f,
        std::integral_constant<
            size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>());
}

/// @internal
/// @brief Base case for const_tuple_for_each — no-op at the end.
template <typename TupleType, typename FunctionType>
void constexpr const_tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {}

/// @internal
/// @brief Recursive worker for const_tuple_for_each.
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
void constexpr const_tuple_for_each(TupleType &&t, FunctionType f,
                                    std::integral_constant<size_t, I>) {
    f(std::get<I>(std::forward<TupleType>(t)));
    const_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

/// @internal
/// @brief Apply @p f to every element of constant tuple @p t in forward order.
template <typename TupleType, typename FunctionType>
void constexpr const_tuple_for_each(TupleType &&t, FunctionType f) {
    const_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

////////////////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Tag type carrying a compile-time loop index value.
/// @tparam I  The compile-time index.
template <int I> struct tuple_loop_index {
    static constexpr int value = I; ///< @internal The compile-time index value.
};

/// @internal
/// @brief Base case for tuple_for_loop — no-op at the end.
template <typename TupleType, typename FunctionType>
void tuple_for_loop(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {}

/// @internal
/// @brief Recursive worker for tuple_for_loop — invokes @p f with both the
/// element and its index tag.
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
void tuple_for_loop(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I>(std::forward<TupleType>(t)), std::integral_constant<size_t, I>());
    tuple_for_loop(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

/// @internal
/// @brief Apply @p f(element, index_constant) to every element of tuple @p t
/// in forward order, passing a compile-time index tag as the second argument.
template <typename TupleType, typename FunctionType>
void tuple_for_loop(TupleType &&t, FunctionType f) {
    tuple_for_loop(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

/// @internal
/// @brief Base case for reverse_tuple_for_loop — no-op at index 0.
template <typename TupleType, typename FunctionType>
void reverse_tuple_for_loop(TupleType &&, FunctionType, std::integral_constant<size_t, 0>) {}

/// @internal
/// @brief Recursive worker for reverse_tuple_for_loop — visits element I-1
/// with its index tag, then continues downward.
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<I != 0>::type>
void reverse_tuple_for_loop(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I - 1>(std::forward<TupleType>(t)), std::integral_constant<size_t, I - 1>());
    reverse_tuple_for_loop(std::forward<TupleType>(t), f, std::integral_constant<size_t, I - 1>());
}

/// @internal
/// @brief Apply @p f(element, index_constant) to every element of tuple @p t
/// in reverse order.
template <typename TupleType, typename FunctionType>
void reverse_tuple_for_loop(TupleType &&t, FunctionType f) {
    reverse_tuple_for_loop(
        std::forward<TupleType>(t), f,
        std::integral_constant<
            size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>());
}

/// @internal
/// @brief Recursive implementation for constexpr_for_loop.
template <int I, int F, typename FunctionType>
void constexpr constexpr_for_loop_impl(FunctionType f) {
    if constexpr (I < F) {
        f(std::integral_constant<int, I>());
        constexpr_for_loop_impl<I + 1, F, FunctionType>(f);
    } else
        return;
}

/// @internal
/// @brief Compile-time integer loop: invoke @p f with
/// `std::integral_constant<int, I>` for each I in [I, F).
template <int I, int F, typename FunctionType>
void constexpr constexpr_for_loop(std::integral_constant<int, I>, std::integral_constant<int, F>,
                                  FunctionType f) {
    constexpr_for_loop_impl<I, F, FunctionType>(f);
}

/// @internal
/// @brief Recursive implementation for constexpr_forwarding_loop.
template <int I, int F, typename FunctionType, typename RetType>
auto constexpr_forwarding_loop_impl(FunctionType f, RetType input) {
    if constexpr (I < F) {
        auto output = f(std::integral_constant<int, I>(), input);
        return constexpr_forwarding_loop_impl<I + 1, F, FunctionType, decltype(output)>(f, output);
    } else
        return input;
}

/// @internal
/// @brief Compile-time integer loop that threads a value through each
/// iteration: `output_i = f(integral_constant<I>, input_{i-1})`.
template <int I, int F, typename FunctionType, typename RetType>
auto constexpr_forwarding_loop(std::integral_constant<int, I>, std::integral_constant<int, F>,
                               FunctionType f, RetType input) {
    return constexpr_forwarding_loop_impl<I, F, FunctionType, RetType>(f, input);
}

/////////////////////////////////////////////////////////////////////////////////////

/// @internal
/// @brief Helper for make_array — expands an index_sequence to fill the array.
template <class Function, std::size_t... Indices>
constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>)
    -> std::array<typename std::invoke_result<Function, std::size_t>::type, sizeof...(Indices)> {
    return {{f(Indices)...}};
}

/// @internal
/// @brief Build a `std::array<R, N>` by calling `f(i)` for i in [0, N).
/// @tparam N  The array length (compile-time constant).
/// @tparam Function  A callable with signature `R(std::size_t)`.
template <int N, class Function>
constexpr auto make_array(Function f)
    -> std::array<typename std::invoke_result<Function, std::size_t>::type, N> {
    return make_array_helper(f, std::make_index_sequence<N>{});
}

} // namespace tycho::utils
