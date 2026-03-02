#pragma once
#include "J2.h"
#include "KeplerModel.h"
#include "KeplerPropagator.h"
#include "KeplerUtils.h"
#include "LambertSolvers.h"
#include "MEEDynamics.h"
#include "ThrusterModels.h"
#include "VectorFunctions/Tycho_VectorFunctions.h"
#include "pch.h"

namespace Tycho {

#ifdef TYCHO_PYTHON_BINDINGS
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
void KeplerUtilsBuild(FunctionRegistry &reg, nb::module_ &m);
void LambertSolversBuild(FunctionRegistry &reg, nb::module_ &m);
void AstroBuild(FunctionRegistry &reg, nb::module_ &m);
#endif // TYCHO_PYTHON_BINDINGS

}