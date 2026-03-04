#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "FunctionRegistry.h"
#include "Solvers/OptimizationProblem.h"
#include "Solvers/OptimizationProblemBase.h"

namespace Tycho {

template <> struct TychoBind<OptimizationProblemBase> {
    static void Build(nb::module_ &m);
};

template <> struct TychoBind<OptimizationProblem> {
    static void Build(nb::module_ &m);
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
