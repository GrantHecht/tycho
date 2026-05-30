// =============================================================================
// Copyright 2026-present Grant R. Hecht
// Licensed under the Apache License, Version 2.0 — see LICENSE.txt.
//
// Tycho-original file:
//   - Adds C++20 concepts for VectorFunction CRTP traits and composition
// =============================================================================

#pragma once

#include <concepts>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::vf {

// Import commonly-used utils helpers so that downstream vf headers can reference
// them without the utils:: prefix (they lived in the same flat namespace before
// the migration).
using utils::return_type_t;
using utils::SZ_BINOP;
using utils::SZ_MAX;
using utils::SZ_MAXOP;
using utils::SZ_MIN;
using utils::SZ_MINOP;
using utils::SZ_PROD;
using utils::SZ_PRODOP;
using utils::SZ_SUM;
using utils::SZ_SUMOP;

namespace detail {

/// @internal
/// @brief Trait: detects whether `T` is an `Eigen::Array<Scalar, Sz, 1>` with `Sz > 1`.
/// @tparam T  Candidate type.
/// @endinternal
template <class T> struct IsSuperScalarImpl : std::false_type {};

/// @internal
/// @brief Specialization recognizing a column `Eigen::Array` as SuperScalar.
/// @tparam Scalar  Array element type.
/// @tparam Sz  Compile-time lane count.
/// @endinternal
template <class Scalar, int Sz>
struct IsSuperScalarImpl<Eigen::Array<Scalar, Sz, 1>> : std::bool_constant<(Sz > 1)> {};

/// @internal
/// @brief Strips cv- and reference-qualifiers from a candidate VF type.
/// @tparam T  Type to normalize.
/// @endinternal
template <class T> using VectorFunctionType = std::remove_cvref_t<T>;

} // namespace detail

/// @brief Satisfied when @p T is a SuperScalar (multi-lane `Eigen::Array`) type.
/// @ingroup vf
/// @tparam T  Candidate scalar type.
template <typename T>
concept IsSuperScalar = detail::IsSuperScalarImpl<detail::VectorFunctionType<T>>::value;

/// @brief Satisfied when @p T is plain `double`.
/// @ingroup vf
/// @tparam T  Candidate scalar type.
template <typename T>
concept IsDoubleScalar = std::same_as<detail::VectorFunctionType<T>, double>;

/// @brief Satisfied when @p T opts into vectorized (SuperScalar) evaluation.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept Vectorizable =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_vectorizable); };

/// @brief Satisfied when @p T declares itself a linear function.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept LinearVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_linear_function); };

/// @brief Satisfied when @p T declares a diagonal Jacobian.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept HasDiagonalJac =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::has_diagonal_jacobian); };

/// @brief Satisfied when @p T is a type-erased generic VectorFunction.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept IsGenericVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_generic_function); };

/// @brief Satisfied when @p T has positive compile-time input and output sizes.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept StaticallySized = requires {
    requires(detail::VectorFunctionType<T>::IRC > 0);
    requires(detail::VectorFunctionType<T>::ORC > 0);
};

/// @brief Satisfied when @p T has a dynamic (`-1`) input or output size.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept DynamicallySized = requires {
    requires(detail::VectorFunctionType<T>::IRC < 0 || detail::VectorFunctionType<T>::ORC < 0);
};

/// @brief Satisfied when @p T exposes `input_rows()` and `output_rows()`.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept VFSized = requires(const detail::VectorFunctionType<T> &t) {
    { t.input_rows() } -> std::same_as<int>;
    { t.output_rows() } -> std::same_as<int>;
};

/// @brief Satisfied when @p Outer can be composed with @p Inner (sizes agree).
/// @ingroup vf
///
/// Requires `Inner::ORC == Outer::IRC` unless either side is dynamic.
/// @tparam Inner  Inner (first-applied) VectorFunction type.
/// @tparam Outer  Outer (second-applied) VectorFunction type.
template <typename Inner, typename Outer>
concept Composable = VFSized<Inner> && VFSized<Outer> && requires {
    requires(detail::VectorFunctionType<Inner>::ORC < 0 ||
             detail::VectorFunctionType<Outer>::IRC < 0 ||
             detail::VectorFunctionType<Inner>::ORC == detail::VectorFunctionType<Outer>::IRC);
};

/// @brief Satisfied when @p F1 and @p F2 can be stacked (matching input sizes).
/// @ingroup vf
///
/// Requires `F1::IRC == F2::IRC` unless either side is dynamic.
/// @tparam F1  First VectorFunction type.
/// @tparam F2  Second VectorFunction type.
template <typename F1, typename F2>
concept Stackable = VFSized<F1> && VFSized<F2> && requires {
    requires(detail::VectorFunctionType<F1>::IRC < 0 || detail::VectorFunctionType<F2>::IRC < 0 ||
             detail::VectorFunctionType<F1>::IRC == detail::VectorFunctionType<F2>::IRC);
};

namespace detail {

/// @internal
/// @brief Concept: `Head` is Stackable with every type in `Tail`.
/// @tparam Head  Reference type.
/// @tparam Tail  Types to check against @p Head.
/// @endinternal
template <class Head, class... Tail>
concept StackableWithAll = (Stackable<Head, Tail> && ...);

/// @internal
/// @brief Recursive helper checking all upper-triangle Stackable pairs.
/// @endinternal
template <class...> struct AllPairsStackableHelper : std::true_type {};

/// @internal
/// @brief Recursive step: `Head` vs all of `Tail`, then recurse on `Tail`.
/// @tparam Head  Current head type.
/// @tparam Tail  Remaining types.
/// @endinternal
template <class Head, class... Tail>
struct AllPairsStackableHelper<Head, Tail...>
    : std::bool_constant<StackableWithAll<Head, Tail...> &&
                         AllPairsStackableHelper<Tail...>::value> {};

} // namespace detail

/// @brief Satisfied iff every pair of types in @p Fs... satisfies `Stackable`.
/// @ingroup vf
///
/// Explicitly checks all pairs to close the dynamic-IRC edge case where
/// one `IRC == -1` could mask an incompatible static pair elsewhere.
/// @tparam Fs  VectorFunction types to be mutually stacked.
template <class... Fs>
concept MutuallyStackable = detail::AllPairsStackableHelper<Fs...>::value;

/// @brief Satisfied when @p T is a conditional VectorFunction.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept ConditionalVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_conditional); };

/// @brief Satisfied when @p T is a component-wise operator VectorFunction.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept CwiseVF =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::is_cwise_operator); };

/// @brief Satisfied when @p T declares a diagonal Hessian.
/// @ingroup vf
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept HasDiagonalHess =
    requires { requires static_cast<bool>(detail::VectorFunctionType<T>::has_diagonal_hessian); };

/// @brief Satisfied when @p T models the full dense VectorFunction interface.
/// @ingroup vf
///
/// Requires the size constants, sizing accessors, vectorization/linearity
/// flags, the scalar-templated matrix aliases, and an `INPUT_DOMAIN` member.
/// @tparam T  Candidate VectorFunction type.
template <typename T>
concept DenseVectorFunction = requires(const T &t) {
    { T::IRC } -> std::convertible_to<int>;
    { T::ORC } -> std::convertible_to<int>;
    { t.input_rows() } -> std::same_as<int>;
    { t.output_rows() } -> std::same_as<int>;
    { T::is_vectorizable } -> std::convertible_to<bool>;
    { T::is_linear_function } -> std::convertible_to<bool>;
    typename T::template Output<double>;
    typename T::template Input<double>;
    typename T::template Jacobian<double>;
    typename T::INPUT_DOMAIN;
};

} // namespace tycho::vf
