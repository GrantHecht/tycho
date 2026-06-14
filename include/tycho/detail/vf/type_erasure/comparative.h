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

#include "tycho/detail/vf/type_erasure/conditional.h"

namespace tycho::vf {

/// @brief Selects whether a ComparativeFunction computes a minimum or maximum.
/// @ingroup vf
enum class ComparativeFlags {
    MinFlag, ///< @brief Take the scalar minimum across the operand functions.
    MaxFlag, ///< @brief Take the scalar maximum across the operand functions.
};

////////////////////////////////////////////////////////////////////////////////
// Class Definition
/// @brief VectorFunction returning the min or max of two or more functions.
///
/// Implemented recursively as a chain of IfElseFunction branches built from
/// pairwise ConditionalStatement comparisons. This primary template handles three
/// or more operands; the two-operand case is a separate specialisation. The
/// result is differentiable wherever a single operand is strictly selected.
/// @tparam First  First operand function.
/// @tparam Rest   Remaining operand functions (one or more).
/// @ingroup vf
template <class First, class... Rest>
struct ComparativeFunction<First, Rest...>
    : IfElseFunction<ConditionalStatement<First, ComparativeFunction<Rest...>>, First,
                     ComparativeFunction<Rest...>> {
    /// @brief ComparativeFunction over the remaining operands (the recursive tail).
    using Second = ComparativeFunction<Rest...>;
    /// @brief Conditional comparing @p First against the tail result.
    using BaseCond = ConditionalStatement<First, Second>;
    /// @brief IfElseFunction base implementing the selection.
    using Base = IfElseFunction<BaseCond, First, Second>;

    // Static Parameters
    static constexpr bool IsComparative =
        true; ///< @brief Trait flag: this is a comparative function.

    // ---------------------------------------------------------------------------
    // Constructors
    /// @brief Construct an empty (default) comparative function.
    ComparativeFunction() {}
    /// @brief Construct a min/max over @p first and @p rest.
    /// @param type   Whether to compute the minimum or maximum.
    /// @param first  First operand function.
    /// @param rest   Remaining operand functions.
    ComparativeFunction(ComparativeFlags type, First first, Rest... rest)
        : Base(BaseCond(first,
                        type == ComparativeFlags::MinFlag ? ConditionalFlags::LessThanFlag
                                                          : ConditionalFlags::GreaterThanFlag,
                        Second(type, rest...)),
               first, Second(type, rest...)) {}
};

// =============================================================================

/// @brief Two-operand specialisation of ComparativeFunction (min/max of two functions).
/// @tparam First   First operand function.
/// @tparam Second  Second operand function.
/// @ingroup vf
template <class First, class Second>
struct ComparativeFunction<First, Second>
    : IfElseFunction<ConditionalStatement<First, Second>, First, Second> {
    /// @brief Conditional comparing the two operands.
    using BaseCond = ConditionalStatement<First, Second>;
    /// @brief IfElseFunction base implementing the selection.
    using Base = IfElseFunction<BaseCond, First, Second>;

    // Static Parameters
    static constexpr bool IsComparative =
        true; ///< @brief Trait flag: this is a comparative function.

    // ---------------------------------------------------------------------------
    // Constructors
    /// @brief Construct an empty (default) comparative function.
    ComparativeFunction() {}
    /// @brief Construct a min/max over @p first and @p second.
    /// @param type    Whether to compute the minimum or maximum.
    /// @param first   First operand function.
    /// @param second  Second operand function.
    ComparativeFunction(ComparativeFlags type, First first, Second second)
        : Base(BaseCond(first,
                        type == ComparativeFlags::MinFlag ? ConditionalFlags::LessThanFlag
                                                          : ConditionalFlags::GreaterThanFlag,
                        second),
               first, second) {}
};

} // namespace tycho::vf
