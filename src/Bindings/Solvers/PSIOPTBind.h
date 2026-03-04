#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "FunctionRegistry.h"
#include "Solvers/PSIOPT.h"

namespace Tycho {

template <> struct TychoBind<PSIOPT> {
    static void Build(nb::module_ &m);
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
