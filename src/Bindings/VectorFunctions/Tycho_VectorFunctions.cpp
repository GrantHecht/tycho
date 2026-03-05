// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Namespace: Tycho
// =============================================================================

#include "Tycho_VectorFunctions.h"

#include "CommonFunctions/IOScaled.h"

// TychoBind<IOScaled<Func>> specialization — must come after IOScaled.h
// and cannot be in CommonFunctionsBind.h since IOScaled.h is not included by CommonFunctions.h.
namespace Tycho {
template <class Func> struct TychoBind<IOScaled<Func>> {
    static void Build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<IOScaled<Func>>(m, name);
        obj.def(nb::init<Func, const Eigen::VectorXd &, const Eigen::VectorXd &>());
        Bind::DenseBaseBuild<IOScaled<Func>>(obj);
    }
};
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

namespace Tycho {
void VectorFunctionBuild(FunctionRegistry &reg, nb::module_ &m);
void VectorFunctionBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void VectorFunctionBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart3(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart4(FunctionRegistry &reg, nb::module_ &m);
void ArgsSegBuildPart5(FunctionRegistry &reg, nb::module_ &m);
void BulkOperationsBuild(FunctionRegistry &reg, nb::module_ &m);
void FreeFunctionsBuild(FunctionRegistry &reg, nb::module_ &m);
void MatrixFunctionBuild(nb::module_ &m);
} // namespace Tycho

void Tycho::VectorFunctionBuild(FunctionRegistry &reg, nb::module_ &m) {
    auto &mod = reg.getVectorFunctionsModule();

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using ELEM = Segment<-1, 1, -1>;

    //////////////////////////////////
    Bind::GenericBuild<Gen>(reg.vfuncx);
    Bind::GenericBuild<GenS>(reg.sfuncx);
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
    Bind::ConditionalBuild(mod);
    Bind::ComparativeBuild(mod);

    reg.Build_Register<PyVectorFunction<-1, -1>>(mod, "PyVectorFunction");
    reg.Build_Register<PyVectorFunction<-1, 1>>(mod, "PyScalarFunction");
    reg.Build_Register<NumbaVectorFunction<-1, -1>>(mod, "NumbaVectorFunction");
    reg.Build_Register<NumbaVectorFunction<-1, 1>>(mod, "NumbaScalarFunction");

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
