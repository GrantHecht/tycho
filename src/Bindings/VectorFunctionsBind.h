#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Binding-layer aggregate for VectorFunctions.
// Included in the pch_bindings PCH after Tycho_VectorFunctions.h so that
// GenericComparative, GenericConditional, and GenericFunction are fully
// defined before these headers instantiate nb::class_<T>.

#include "VectorFunctions/CommonFunctionsBind.h"
#include "VectorFunctions/DenseFunctionBaseBind.h"
#include "VectorFunctions/GenericFunctionBind.h"
#include "VectorFunctions/PythonArgParsing.h"
#include "VectorFunctions/PythonFunctions.h"

#endif // TYCHO_PYTHON_BINDINGS
