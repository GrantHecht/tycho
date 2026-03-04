#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "FunctionRegistry.h"
#include "OptimalControl/MeshIterateInfo.h"

namespace Tycho {

template <> struct TychoBind<MeshIterateInfo> {
    static void Build(nb::module_ &m);
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
