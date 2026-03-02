#pragma once
#include "CommonFunctions/CommonFunctions.h"
#include "MathOverloads.h"
#include "OperatorOverloads.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericComparative.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"
#ifdef TYCHO_PYTHON_BINDINGS
#include "Bindings/FunctionRegistry.h"
#include "Bindings/DenseFunctionBaseBind.h"
#include "Bindings/CommonFunctionsBind.h"
#include "Bindings/PythonArgParsing.h"
#include "Bindings/PythonFunctions.h"
#include "Bindings/GenericFunctionBind.h"
#endif

namespace Tycho {

GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>> &elems);
GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>> &elems);
GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>> &elems);
GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>> &elems);

#ifdef TYCHO_PYTHON_BINDINGS
/// <summary>
/// Builds All of vector Functions module, Calls all of other build functions
/// </summary>
/// <param name="reg"></param>
/// <param name="m"></param>
void VectorFunctionBuild(FunctionRegistry &reg, nb::module_ &m);

void VectorFunctionBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void VectorFunctionBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart3(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart4(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart5(FunctionRegistry &reg, nb::module_ &m);

void BulkOperationsBuild(FunctionRegistry &reg, nb::module_ &m);

// std::vector<GenericFunction<-1, -1>> ParsePythonArgs(nb::args x,int irows=0);
// std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(nb::args x,int irows=0);

void FreeFunctionsBuild(FunctionRegistry &reg, nb::module_ &m);
void MatrixFunctionBuild(nb::module_ &m);
#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho
