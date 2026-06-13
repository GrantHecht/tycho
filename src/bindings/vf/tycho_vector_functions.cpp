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
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "tycho_vector_functions.h"

#include "tycho/detail/vf/scaling/io_scaled.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

// TychoBind<IOScaled<Func>> specialization — must come after IOScaled.h
// and cannot be in CommonFunctionsBind.h since IOScaled.h is not included by CommonFunctions.h.
namespace tycho {
template <class Func> struct TychoBind<IOScaled<Func>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<IOScaled<Func>>(m, name);
        obj.def(nb::init<Func, const Eigen::VectorXd &, const Eigen::VectorXd &>(),
                R"doc(Construct an IO-scaled wrapper around a VectorFunction.

Each input component is multiplied by the corresponding entry of ``input_scales``
before being passed to the wrapped function, and each output component is
multiplied by the corresponding entry of ``output_scales`` after evaluation.
Jacobian rows are scaled by ``output_scales`` and columns by ``input_scales``;
the adjoint gradient and Hessian are transformed consistently.

Parameters
----------
func : VectorFunction
    The function whose input and output are to be scaled.
input_scales : array_like, shape (input_rows,)
    Per-component scale factors applied to the input before evaluation.
output_scales : array_like, shape (output_rows,)
    Per-component scale factors applied to the output after evaluation.

Returns
-------
IOScaled
    Wrapped function with scaled input and output.

Raises
------
ValueError
    If the length of ``input_scales`` or ``output_scales`` does not match the
    corresponding dimension of ``func``.
)doc");
        bind::DenseBaseBuild<IOScaled<Func>>(obj);
    }
};
} // namespace tycho

#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "utils/fmtlib.h"

// Out-of-class definitions for InterpTable binding functions.
// Included here after all InterpTable headers so all types are complete.
#ifdef TYCHO_PYTHON_BINDINGS
#include "interp_table_bind.h"
#endif

namespace tycho {
void vector_functions_build(FunctionRegistry &reg, nb::module_ &m);
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
} // namespace tycho

void tycho::vector_functions_build(FunctionRegistry &reg, nb::module_ &m) {
    auto &mod = reg.getVectorFunctionsModule();

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using ELEM = Segment<-1, 1, -1>;

    //////////////////////////////////
    bind::GenericBuild<Gen>(reg.vfuncx);
    bind::GenericBuild<GenS>(reg.sfuncx);
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
    bind::ConditionalBuild(mod);
    bind::ComparativeBuild(mod);

    reg.build_register<PyVectorFunction<-1, -1>>(mod, "PyVectorFunction");
    reg.build_register<PyVectorFunction<-1, 1>>(mod, "PyScalarFunction");
    reg.build_register<NumbaVectorFunction<-1, -1>>(mod, "NumbaVectorFunction");
    reg.build_register<NumbaVectorFunction<-1, 1>>(mod, "NumbaScalarFunction");

    reg.build_register<Constant<-1, -1>>(mod, "ConstantVector");
    reg.build_register<Constant<-1, 1>>(mod, "ConstantScalar");

    reg.build_register<IOScaled<Gen>>(mod, "IOScaled");

    nb::enum_<InterpType>(mod, "InterpType")
        .value("Cubic", InterpType::Cubic)
        .value("Linear", InterpType::Linear);

    InterpTable1DBuild(mod);
    InterpTable2DBuild(mod);
    InterpTable3DBuild(mod);
    InterpTable4DBuild(mod);

    mod.def("scalar_dynamic_stack_test", [](const std::vector<GenericFunction<-1, 1>> &funcs) {
        return GenericFunction<-1, -1>(DynamicStackedOutputs<GenericFunction<-1, 1>>{funcs});
    },
        R"doc(Internal testing helper: stack a list of scalar functions into a VectorFunction.

Constructs a ``DynamicStackedOutputs`` from a homogeneous list of scalar-output
(1-output-row) VectorFunctions and type-erases the result to ``VectorFunction``.
This function exists to exercise the dynamic-stacking code path from Python tests;
it is not part of the public API.

Parameters
----------
funcs : list[ScalarFunction]
    Non-empty list of scalar-output VectorFunctions sharing the same input size.

Returns
-------
VectorFunction
    A VectorFunction whose output is the vertical concatenation of all inputs.
)doc");

    mod.def("dynamic_stack_test", [](const std::vector<GenericFunction<-1, -1>> &funcs) {
        return GenericFunction<-1, -1>(DynamicStackedOutputs<GenericFunction<-1, -1>>{funcs});
    },
        R"doc(Internal testing helper: stack a list of VectorFunctions into a VectorFunction.

Constructs a ``DynamicStackedOutputs`` from a homogeneous list of multi-output
VectorFunctions and type-erases the result to ``VectorFunction``.  This function
exists to exercise the dynamic-stacking code path from Python tests; it is not
part of the public API.

Parameters
----------
funcs : list[VectorFunction]
    Non-empty list of VectorFunctions sharing the same input size.

Returns
-------
VectorFunction
    A VectorFunction whose output is the vertical concatenation of all inputs.
)doc");
}
