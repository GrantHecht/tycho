#pragma once

#include "FDDerivArbitrary.h"
#include "FDDerivUniform.h"
#include "LGLInterpFunctions.h"
#include "ODE.h"
#include "ODEPhase.h"
#include "OptimalControlProblem.h"
#include "pch.h"

namespace Tycho {

#ifdef TYCHO_PYTHON_BINDINGS
void GenericODESBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart4(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart5(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart6(FunctionRegistry &reg, nb::module_ &m);

void OptimalControlBuild(FunctionRegistry &reg, nb::module_ &m);
#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho
