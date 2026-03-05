#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "FunctionRegistry.h"
#include "Utils/MemoryManagement.h"

namespace Tycho {

template <> struct TychoBind<MemoryManager> {
    static void Build(nb::module_ &m);
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
