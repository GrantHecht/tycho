#pragma once
#include "BenchUtils.h"
#include "CRTPBase.h"
#include "FunctionReturnType.h"
#include "GetCoreCount.h"
#include "LambdaJumpTable.h"
#include "MathFunctions.h"
#include "STDExtensions.h"
#include "SizingHelpers.h"
#include "ThreadPool.h"
#include "TupleIterator.h"
#include "TypeErasure.h"
#include "TypeName.h"

namespace Tycho {

#ifdef TYCHO_PYTHON_BINDINGS
void UtilsBuild(nb::module_ &m) {
    auto um = m.def_submodule("Utils", "Contains miscilanaeous utilities");
    um.def("get_core_count", &Tycho::get_core_count);
}
#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho
