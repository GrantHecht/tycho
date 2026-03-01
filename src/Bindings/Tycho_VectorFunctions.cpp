#include "Tycho_VectorFunctions.h"

#include "CommonFunctions/IOScaled.h"

// Out-of-class definition for IOScaled<Func>::Build — must come after IOScaled.h
// and cannot be in CommonFunctionsBind.h since IOScaled.h is not included by CommonFunctions.h.
namespace Tycho {
template <class Func>
void IOScaled<Func>::Build(nb::module_ &m, const char *name) {
    auto obj = nb::class_<IOScaled<Func>>(m, name);
    obj.def(nb::init<Func, const Input<double> &, const Output<double> &>());
    Base::DenseBaseBuild(obj);
}
} // namespace Tycho

#include "CommonFunctions/InterpTable1D.h"
#include "CommonFunctions/InterpTable2D.h"
#include "CommonFunctions/InterpTable3D.h"
#include "CommonFunctions/InterpTable4D.h"
#include "Utils/fmtlib.h"

// Out-of-class definitions for InterpTable binding functions.
// Included here after all InterpTable headers so all types are complete.
#ifdef TYCHO_PYTHON_BINDINGS
#include "InterpTableBind.h"
#endif

namespace Tycho {} // namespace Tycho

void Tycho::VectorFunctionBuild(FunctionRegistry &reg, nb::module_ &m) {
    auto &mod = reg.getVectorFunctionsModule();

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using ELEM = Segment<-1, 1, -1>;

    //////////////////////////////////
    Gen::GenericBuild(reg.vfuncx);
    GenS::GenericBuild(reg.sfuncx);
    nb::implicitly_convertible<GenS, Gen>();
    VectorFunctionBuildPart1(reg, mod);
    VectorFunctionBuildPart2(reg, mod);
    ////////////////////////////////////

    ArgsSegBuildPart1(reg, mod);
    ArgsSegBuildPart2(reg, mod);
    ArgsSegBuildPart3(reg, mod);
    ArgsSegBuildPart4(reg, mod);
    ArgsSegBuildPart5(reg, mod);

    BulkOperationsBuild(reg, mod);
    FreeFunctionsBuild(reg, mod);
    MatrixFunctionBuild(mod);
    GenericConditional<-1>::ConditionalBuild(mod);
    GenericComparative<-1>::ComparativeBuild(mod);

    reg.Build_Register<PyVectorFunction<-1, -1>>(mod, "PyVectorFunction");
    reg.Build_Register<PyVectorFunction<-1, 1>>(mod, "PyScalarFunction");

    reg.Build_Register<Constant<-1, -1>>(mod, "ConstantVector");
    reg.Build_Register<Constant<-1, 1>>(mod, "ConstantScalar");

    reg.Build_Register<IOScaled<Gen>>(mod, "IOScaled");

    InterpTable1DBuild(mod);
    InterpTable2DBuild(mod);
    InterpTable3DBuild(mod);
    InterpTable4DBuild(mod);

    mod.def("ScalarDynamicStackTest", [](const std::vector<GenericFunction<-1, 1>> &funcs) {
        return GenericFunction<-1, -1>(DynamicStackedOutputs<GenericFunction<-1, 1>>{funcs});
    });

    mod.def("DynamicStackTest", [](const std::vector<GenericFunction<-1, -1>> &funcs) {
        return GenericFunction<-1, -1>(DynamicStackedOutputs<GenericFunction<-1, -1>>{funcs});
    });
}
