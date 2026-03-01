#pragma once
#include "CommonFunctions/CommonFunctions.h"
#include "MathOverloads.h"
#include "OperatorOverloads.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericComparative.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"
#ifdef TYCHO_PYTHON_BINDINGS
#include "PythonArgParsing.h"
#include "Bindings/FunctionRegistry.h"
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
void VectorFunctionBuild(FunctionRegistry &reg, py::module &m);

void VectorFunctionBuildPart1(FunctionRegistry &reg, py::module &m);
void VectorFunctionBuildPart2(FunctionRegistry &reg, py::module &m);
void ArgsSegBuildPart1(FunctionRegistry &reg, py::module &m);
void ArgsSegBuildPart2(FunctionRegistry &reg, py::module &m);
void ArgsSegBuildPart3(FunctionRegistry &reg, py::module &m);
void ArgsSegBuildPart4(FunctionRegistry &reg, py::module &m);
void ArgsSegBuildPart5(FunctionRegistry &reg, py::module &m);

void BulkOperationsBuild(FunctionRegistry &reg, py::module &m);

// std::vector<GenericFunction<-1, -1>> ParsePythonArgs(py::args x,int irows=0);
// std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(py::args x,int irows=0);

void FreeFunctionsBuild(FunctionRegistry &reg, py::module &m);
void MatrixFunctionBuild(py::module &m);
#endif // TYCHO_PYTHON_BINDINGS

} // namespace Tycho
