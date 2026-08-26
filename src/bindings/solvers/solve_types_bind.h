// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Python bindings for the engine-neutral solve-call value types: Mode, the
// warm-start currency (WarmStartData/WarmExtension) and its declaration-
// identity stamp (DeclarationKey), and the result types a solve() call hands
// back (StageResult, PhaseResult, SolveResult).

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "function_registry.h"
#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/solve_types.h"

namespace tycho {

using namespace tycho::solvers;

template <> struct TychoBind<SolveResult> {
    static void build(nb::module_ &m);
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
