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

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

template <class FType> void DefineListEval(nb::class_<FType> &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    obj.def("__call__", [](const FType &fun, const std::vector<GenS> &funcs) {
        return FType(fun.eval(DynamicStack(funcs)));
    },
        R"doc(Compose this function by substituting a list of scalar functions as its input.

The argument functions are vertically stacked — their outputs are concatenated in
order to form a single input vector — and the result is used as the input to this
function.  Concretely, if ``funcs = [g0, g1, ...]`` and the shared domain of all
``gi`` is ``x``, then the returned function computes ``self([g0(x); g1(x); ...])``.

All functions in ``funcs`` must share the same input size, and their concatenated
output size must equal ``self.input_rows()``.

Parameters
----------
funcs : list[ScalarFunction]
    Homogeneous list of scalar-output (1-output-row) VectorFunctions whose
    stacked outputs form the input to this function.

Returns
-------
VectorFunction
    A new VectorFunction representing the composition.

Raises
------
ValueError
    If the stacked output size does not match ``self.input_rows()``, if the
    functions in ``funcs`` do not share a common input size, or if ``funcs``
    is empty.
)doc");
    obj.def("__call__", [](const FType &fun, const std::vector<Gen> &funcs) {
        return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType &fun, const Gen &first, nb::args x) {
        auto funcs = std::vector{first};
        auto funcsrest = ParsePythonArgs(x, first.input_rows());
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType &fun, double first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        Vector1<double> val;
        val[0] = first;
        auto funcs = std::vector{Gen(Constant<-1, 1>(funcsrest[0].input_rows(), val))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType &fun, Eigen::VectorXd first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        auto funcs = std::vector{Gen(Constant<-1, -1>(funcsrest[0].input_rows(), first))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return FType(fun.eval(DynamicStack(funcs)));
    });
}

void VectorFunctionBuildPart2(FunctionRegistry &reg, nb::module_ &m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    bind::DoubleMathBuild<GenS>(reg.sfuncx);
    bind::UnaryMathBuild<GenS>(reg.sfuncx);
    bind::BinaryMathBuild<GenS>(reg.sfuncx);
    bind::BinaryOperatorsBuild<GenS>(reg.sfuncx);
    bind::FunctionIndexingBuild<GenS>(reg.sfuncx);
    bind::ConditionalOperatorsBuild<GenS>(reg.sfuncx);

    ///////////////////////////////////////
    bind::DoubleMathBuild<Gen>(reg.vfuncx);
    bind::FunctionIndexingBuild<Gen>(reg.vfuncx);
    bind::BinaryOperatorsBuild<Gen>(reg.vfuncx);
    ///////////////////////////////////////

    DefineListEval(reg.vfuncx);
    DefineListEval(reg.sfuncx);
}

} // namespace tycho