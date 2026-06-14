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

namespace tycho::vf {

/// @internal
/// @brief Defines a SFINAE trait `Detect_X` testing for a member named `X`.
///
/// The generated `Detect_##X<T>::value` is `true` iff `T` declares a member
/// named `X`. Implemented via the inherited-ambiguity idiom.
/// @param X  Member name to probe for.
/// @endinternal
#define CREATE_MEMBER_DETECTOR(X)                                                                  \
    template <typename T> class Detect_##X {                                                       \
        struct Fallback {                                                                          \
            int X;                                                                                 \
        };                                                                                         \
        struct Derived : T, Fallback {};                                                           \
                                                                                                   \
        template <typename U, U> struct Check;                                                     \
                                                                                                   \
        typedef char ArrayOfOne[1];                                                                \
        typedef char ArrayOfTwo[2];                                                                \
                                                                                                   \
        template <typename U> static ArrayOfOne &func(Check<int Fallback::*, &U::X> *);            \
        template <typename U> static ArrayOfTwo &func(...);                                        \
                                                                                                   \
      public:                                                                                      \
        typedef Detect_##X type;                                                                   \
        enum { value = sizeof(func<Derived>(0)) == 2 };                                            \
    };

} // namespace tycho::vf