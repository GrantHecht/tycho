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
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "function_registry.h"
#include "tycho/detail/solvers/jet.h"

// Partial specialisation: when Arg = nb::args, dereference via args_proxy so
// the genfunc receives nb::detail::args_proxy (the unpacked argument sequence).
namespace tycho::solvers::detail {
template <class T>
struct JetInvoker<T, std::function<std::shared_ptr<T>(nb::detail::args_proxy)>, nb::args> {
    static inline std::shared_ptr<T>
    invoke(const std::function<std::shared_ptr<T>(nb::detail::args_proxy)> &f, const nb::args &a) {
        return f(*a);
    }
};
} // namespace tycho::solvers::detail

namespace tycho {

using namespace tycho::solvers;

template <> struct TychoBind<Jet> {
    static void Build(nb::module_ &m);
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
