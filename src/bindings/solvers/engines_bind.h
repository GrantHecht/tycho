// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Python bindings for the engine handle layer: SqpSolver (with a kwargs
// constructor mapped onto SqpOptions' plain-value fields) and IpoptSolver.
// InteriorPointSolver is bound separately (interior_point_solver_bind.h) --
// it is already a tycho-visible hven type with its own settings/result
// surface.

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "function_registry.h"
#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/engines.h"

namespace tycho {

using namespace tycho::solvers;

template <> struct TychoBind<SqpSolver> {
    static void build(nb::module_ &m);
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
