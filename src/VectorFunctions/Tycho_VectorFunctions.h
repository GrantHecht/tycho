#pragma once
#include "Bindings/CommonFunctionsBind.h"
#include "Bindings/DenseFunctionBaseBind.h"
#include "Bindings/FunctionRegistry.h"
#include "Bindings/GenericFunctionBind.h"
#include "Bindings/PythonArgParsing.h"
#include "Bindings/PythonFunctions.h"
#include "CommonFunctions/CommonFunctions.h"
#include "MathOverloads.h"
#include "OperatorOverloads.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericComparative.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"

namespace Tycho {

GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>> &elems);
GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>> &elems);
GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>> &elems);
GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>> &elems);

} // namespace Tycho
