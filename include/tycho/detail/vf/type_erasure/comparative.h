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

// Enum
enum class ComparativeFlags {
    MinFlag,
    MaxFlag,
};

////////////////////////////////////////////////////////////////////////////////
// Class Definition
template <class First, class... Rest>
struct ComparativeFunction<First, Rest...>
    : IfElseFunction<ConditionalStatement<First, ComparativeFunction<Rest...>>, First,
                     ComparativeFunction<Rest...>> {
    using Second = ComparativeFunction<Rest...>;
    using BaseCond = ConditionalStatement<First, Second>;
    using Base = IfElseFunction<BaseCond, First, Second>;

    // Static Parameters
    static constexpr bool IsComparative = true;

    // ---------------------------------------------------------------------------
    // Constructors
    ComparativeFunction() {}
    ComparativeFunction(ComparativeFlags type, First first, Rest... rest)
        : Base(BaseCond(first,
                        type == ComparativeFlags::MinFlag ? ConditionalFlags::LessThanFlag
                                                          : ConditionalFlags::GreaterThanFlag,
                        Second(type, rest...)),
               first, Second(type, rest...)) {}
};

// =============================================================================

template <class First, class Second>
struct ComparativeFunction<First, Second>
    : IfElseFunction<ConditionalStatement<First, Second>, First, Second> {
    using BaseCond = ConditionalStatement<First, Second>;
    using Base = IfElseFunction<BaseCond, First, Second>;

    // Static Parameters
    static constexpr bool IsComparative = true;

    // ---------------------------------------------------------------------------
    // Constructors
    ComparativeFunction() {}
    ComparativeFunction(ComparativeFlags type, First first, Second second)
        : Base(BaseCond(first,
                        type == ComparativeFlags::MinFlag ? ConditionalFlags::LessThanFlag
                                                          : ConditionalFlags::GreaterThanFlag,
                        second),
               first, second) {}
};

} // namespace tycho::vf
