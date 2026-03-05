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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

namespace Tycho {

template <typename T> struct return_type;
template <typename R, typename... Args> struct return_type<R (*)(Args...)> {
    using type = R;
};
template <typename R, typename C, typename... Args> struct return_type<R (C::*)(Args...)> {
    using type = R;
};
template <typename R, typename C, typename... Args> struct return_type<R (C::*)(Args...) const> {
    using type = R;
};
template <typename R, typename C, typename... Args> struct return_type<R (C::*)(Args...) volatile> {
    using type = R;
};
template <typename R, typename C, typename... Args>
struct return_type<R (C::*)(Args...) const volatile> {
    using type = R;
};
template <typename T> using return_type_t = typename return_type<T>::type;

} // namespace Tycho
