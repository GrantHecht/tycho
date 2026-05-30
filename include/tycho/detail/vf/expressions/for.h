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

#include "tycho/detail/vf/expressions/nested_function.h"

namespace tycho::vf {

//! Declaration of For_Impl
template <class Derived, int N, class StartFunc, class BodyFunc> struct For_Impl;

//! User-facing For class
/*!
  \tparam N Max iteration counter. Iterates from 0 to N, inclusive.
  \tparam StartFunc The function that is affected by the loop.
  \tparam BodyFunc The function that is iteratively called on StartFunc.

  Format as follows:
  for(int i=0; i<=N; i++){
    start = body(i, start);
  }
*/
template <int N, class StartFunc, class BodyFunc>
struct For : For_Impl<For<N, StartFunc, BodyFunc>, N, StartFunc, BodyFunc> {
    /// @brief CRTP implementation base generating the unrolled loop.
    using Base = For_Impl<For<N, StartFunc, BodyFunc>, N, StartFunc, BodyFunc>;
    /// @brief Construct the loop from its starting function.
    /// @param f  Starting function forwarded to the implementation base.
    For(StartFunc f) : Base::For_Impl(f) {};
};

//! Implementation of For loop
/*!
  Uses template recursion to generate a function with the same effect as a for
  loop. The i-th level in the recursion inherits from the NestedFunction of
  `BodyFunc::Definition<i>` with `For_Impl<i-1, StartFunc, BodyFunc>`.
*/
template <class Derived, int N, class StartFunc, class BodyFunc>
struct For_Impl : NestedFunction_Impl<Derived, decltype(BodyFunc::template Definition<N>()),
                                      For_Impl<Derived, N - 1, StartFunc, BodyFunc>> {
    /// @internal
    /// @brief Outer function for this level: the body's definition at iteration N.
    /// @endinternal
    using OFuncType = decltype(BodyFunc::template Definition<N>());
    /// @internal
    /// @brief Inner function: the recursively-nested loop of one fewer iteration.
    /// @endinternal
    using IFuncType = For_Impl<Derived, N - 1, StartFunc, BodyFunc>;
    /// @internal
    /// @brief NestedFunction base composing the body over the inner loop.
    /// @endinternal
    using Base = NestedFunction_Impl<Derived, OFuncType, IFuncType>;

    For_Impl() {};
    /// @internal
    /// @brief Construct from the starting function, seeding the inner recursion.
    /// @param f  Starting function affected by the loop.
    /// @endinternal
    For_Impl(StartFunc f) : Base::NestedFunction_Impl(OFuncType(), IFuncType(f)) {};
};

//! Base Specialization of For_Impl
/*!

*/
template <class Derived, class StartFunc, class BodyFunc>
struct For_Impl<Derived, 0, StartFunc, BodyFunc>
    : NestedFunction_Impl<Derived, decltype(BodyFunc::template Definition<0>()), StartFunc> {
    /// @internal
    /// @brief Outer function at the base case: the body's definition at iteration 0.
    /// @endinternal
    using OFuncType = decltype(BodyFunc::template Definition<0>());
    /// @internal
    /// @brief NestedFunction base composing the body over the starting function.
    /// @endinternal
    using Base = NestedFunction_Impl<Derived, OFuncType, StartFunc>;

    For_Impl() {};
    /// @internal
    /// @brief Construct the base case from the starting function.
    /// @param inner  Starting function affected by the loop.
    /// @endinternal
    For_Impl(StartFunc inner) : Base::NestedFunction_Impl(OFuncType(), inner) {};
};

} // namespace tycho::vf
