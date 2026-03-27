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

#include <functional> // std::invoke
#include <tuple>      // std::tuple
#include <type_traits>

namespace tycho::utils {

//////////////////////////////////////////////////////////////////////////////////
template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
void tuple_for_each(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I>(std::forward<TupleType>(t)));
    tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
    tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

template <typename TupleType, typename FunctionType>
void reverse_tuple_for_each(TupleType &&, FunctionType, std::integral_constant<size_t, 0>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<I != 0>::type>
void reverse_tuple_for_each(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I - 1>(std::forward<TupleType>(t)));
    reverse_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I - 1>());
}
template <typename TupleType, typename FunctionType>
void reverse_tuple_for_each(TupleType &&t, FunctionType f) {
    reverse_tuple_for_each(
        std::forward<TupleType>(t), f,
        std::integral_constant<
            size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>());
}

template <typename TupleType, typename FunctionType>
void constexpr const_tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
void constexpr const_tuple_for_each(TupleType &&t, FunctionType f,
                                    std::integral_constant<size_t, I>) {
    f(std::get<I>(std::forward<TupleType>(t)));
    const_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template <typename TupleType, typename FunctionType>
void constexpr const_tuple_for_each(TupleType &&t, FunctionType f) {
    const_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

////////////////////////////////////////////////////////////////////////////////////////

template <int I> struct tuple_loop_index {
    static const int value = I;
};

template <typename TupleType, typename FunctionType>
void tuple_for_loop(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
void tuple_for_loop(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I>(std::forward<TupleType>(t)), std::integral_constant<size_t, I>());
    tuple_for_loop(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}
template <typename TupleType, typename FunctionType>
void tuple_for_loop(TupleType &&t, FunctionType f) {
    tuple_for_loop(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

template <typename TupleType, typename FunctionType>
void reverse_tuple_for_loop(TupleType &&, FunctionType, std::integral_constant<size_t, 0>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<I != 0>::type>
void reverse_tuple_for_loop(TupleType &&t, FunctionType f, std::integral_constant<size_t, I>) {
    f(std::get<I - 1>(std::forward<TupleType>(t)), std::integral_constant<size_t, I - 1>());
    reverse_tuple_for_loop(std::forward<TupleType>(t), f, std::integral_constant<size_t, I - 1>());
}
template <typename TupleType, typename FunctionType>
void reverse_tuple_for_loop(TupleType &&t, FunctionType f) {
    reverse_tuple_for_loop(
        std::forward<TupleType>(t), f,
        std::integral_constant<
            size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>());
}

template <int I, int F, typename FunctionType>
void constexpr constexpr_for_loop_impl(FunctionType f) {
    if constexpr (I < F) {
        f(std::integral_constant<int, I>());
        constexpr_for_loop_impl<I + 1, F, FunctionType>(f);
    } else
        return;
}

template <int I, int F, typename FunctionType>
void constexpr constexpr_for_loop(std::integral_constant<int, I>, std::integral_constant<int, F>,
                                  FunctionType f) {
    constexpr_for_loop_impl<I, F, FunctionType>(f);
}

template <int I, int F, typename FunctionType, typename RetType>
auto constexpr_forwarding_loop_impl(FunctionType f, RetType input) {
    if constexpr (I < F) {
        auto output = f(std::integral_constant<int, I>(), input);
        return constexpr_forwarding_loop_impl<I + 1, F, FunctionType, decltype(output)>(f, output);
    } else
        return input;
}

template <int I, int F, typename FunctionType, typename RetType>
auto constexpr_forwarding_loop(std::integral_constant<int, I>, std::integral_constant<int, F>,
                               FunctionType f, RetType input) {
    return constexpr_forwarding_loop_impl<I, F, FunctionType, RetType>(f, input);
}

/////////////////////////////////////////////////////////////////////////////////////

template <class Function, std::size_t... Indices>
constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>)
    -> std::array<typename std::invoke_result<Function, std::size_t>::type, sizeof...(Indices)> {
    return {{f(Indices)...}};
}

template <int N, class Function>
constexpr auto make_array(Function f)
    -> std::array<typename std::invoke_result<Function, std::size_t>::type, N> {
    return make_array_helper(f, std::make_index_sequence<N>{});
}

} // namespace tycho::utils
