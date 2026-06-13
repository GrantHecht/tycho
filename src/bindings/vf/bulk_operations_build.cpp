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

namespace tycho::vf {

GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>> &elems) {
    using Gen = GenericFunction<-1, -1>;
    return make_dynamic_stack<Gen, Gen>(elems);
}
GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>> &elems) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_stack<Gen, GenS>(elems);
}
GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>> &elems) {
    using Gen = GenericFunction<-1, -1>;
    return make_dynamic_sum<Gen, Gen>(elems);
}
GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>> &elems) {
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_sum<GenS, GenS>(elems);
}

} // namespace tycho::vf

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

void BulkOperationsBuild(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::BulkOperationsBuild(FunctionRegistry &reg, nb::module_ &m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    m.def("stack", [](const std::vector<GenS> &elems) { return DynamicStack(elems); },
          R"doc(Concatenate scalar-output VectorFunctions vertically into one VectorFunction.

Each function in *funcs* must share the same input dimension.  Their scalar
outputs are stacked in order to form a single multi-output VectorFunction
whose output dimension equals ``len(funcs)``.

Parameters
----------
funcs : list[VectorFunction]
    Scalar-output (output dimension 1) functions sharing the same input
    dimension.  The list must be non-empty.

Returns
-------
VectorFunction
    A VectorFunction whose output is the vertical concatenation of all
    inputs; its output dimension is ``len(funcs)``.

Raises
------
ValueError
    If any two functions have mismatched input dimensions, or the list is
    empty.

Examples
--------
>>> v = vf.Arguments(3)
>>> f = vf.stack([v[0], v[1] + v[2]])
>>> f.output_rows()
2
)doc");
    m.def("stack", [](const std::vector<Gen> &elems) { return DynamicStack(elems); });
    m.def("stack", [](const Gen &first, nb::args x) {
        auto funcs = std::vector{first};
        auto funcsrest = ParsePythonArgs(x, first.input_rows());
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicStack(funcs);
    });
    m.def("stack", [](double first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        Vector1<double> val;
        val[0] = first;
        auto funcs = std::vector{Gen(Constant<-1, 1>(funcsrest[0].input_rows(), val))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicStack(funcs);
    });
    m.def("stack", [](Eigen::VectorXd first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        auto funcs = std::vector{Gen(Constant<-1, -1>(funcsrest[0].input_rows(), first))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicStack(funcs);
    });
    ///////////////////////////////////////////////////////
    m.def("stack_scalar", [](const std::vector<GenS> &elems) { return DynamicStack(elems); },
          R"doc(Concatenate scalar-output VectorFunctions vertically into one VectorFunction.

Equivalent to :func:`stack` but restricted to scalar-output (output
dimension 1) functions.  All functions in *funcs* must share the same input
dimension; their outputs are concatenated in order.

Parameters
----------
funcs : list[VectorFunction]
    Scalar-output (output dimension 1) functions sharing the same input
    dimension.  The list must be non-empty.

Returns
-------
VectorFunction
    A VectorFunction whose output dimension equals ``len(funcs)`` formed by
    concatenating the scalar outputs of each element.

Raises
------
ValueError
    If any two functions have mismatched input dimensions, or the list is
    empty.
)doc");
    m.def("stack_scalar", [](const GenS &first, nb::args x) {
        auto funcs = std::vector{first};
        auto funcsrest = ParsePythonArgsScalar(x, first.input_rows());
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicStack(funcs);
    });
    m.def("stack_scalar", [](double first, nb::args x) {
        auto funcsrest = ParsePythonArgsScalar(x);
        Vector1<double> val;
        val[0] = first;
        auto funcs = std::vector{GenS(Constant<-1, 1>(funcsrest[0].input_rows(), val))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicStack(funcs);
    });
    ////////////////////////////////////////////////////////

    m.def("sum", [](const std::vector<GenS> &elems) { return DynamicSum(elems); },
          R"doc(Elementwise sum of several equal-output scalar VectorFunctions.

Each function in *funcs* must have the same input dimension and the same
output dimension (1).  The result evaluates ``f1(x) + f2(x) + ...``
elementwise; its Jacobian and higher derivatives are accumulated from all
addends.

Parameters
----------
funcs : list[VectorFunction]
    Scalar-output (output dimension 1) functions sharing the same input and
    output dimension.  The list must contain at least one element.

Returns
-------
VectorFunction
    A scalar-output VectorFunction whose value is the elementwise sum of all
    inputs.

Raises
------
ValueError
    If any two functions have mismatched input or output dimensions, or the
    list is empty.
)doc");
    m.def("sum", [](const std::vector<Gen> &elems) { return DynamicSum(elems); });

    m.def("sum", [](const GenS &first, nb::args x) {
        auto funcs = std::vector{first};
        auto funcsrest = ParsePythonArgsScalar(x);
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicSum(funcs);
    });
    m.def("sum", [](double first, nb::args x) {
        auto funcsrest = ParsePythonArgsScalar(x);
        Vector1<double> val;
        val[0] = first;
        auto funcs = std::vector{GenS(Constant<-1, 1>(funcsrest[0].input_rows(), val))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicSum(funcs);
    });

    m.def("sum", [](const Gen &first, nb::args x) {
        auto funcs = std::vector{first};
        auto funcsrest = ParsePythonArgs(x);
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicSum(funcs);
    });

    m.def("sum", [](Eigen::VectorXd first, nb::args x) {
        auto funcsrest = ParsePythonArgs(x);
        auto funcs = std::vector{Gen(Constant<-1, -1>(funcsrest[0].input_rows(), first))};
        for (const auto &f : funcsrest)
            funcs.push_back(f);
        return DynamicSum(funcs);
    });

    //////////////////////////////////////////////////////////

    m.def("stack", [](const std::vector<GenS> &elems) { return DynamicStack(elems); });
    m.def("stack_scalar", [](const std::vector<GenS> &elems) { return DynamicStack(elems); });
    m.def("stack", [](const std::vector<Gen> &elems) { return DynamicStack(elems); });

    m.def("sum", [](const std::vector<GenS> &elems) { return DynamicSum(elems); });
    m.def("sum_scalar", [](const std::vector<GenS> &elems) { return DynamicSum(elems); },
          R"doc(Elementwise sum of several scalar-output VectorFunctions.

Equivalent to :func:`sum` but restricted to scalar-output (output dimension
1) functions.  Each function in *funcs* must share the same input dimension;
the result evaluates ``f1(x) + f2(x) + ...`` and accumulates Jacobians and
higher derivatives from all addends.

Parameters
----------
funcs : list[VectorFunction]
    Scalar-output (output dimension 1) functions sharing the same input
    dimension.  The list must contain at least one element.

Returns
-------
VectorFunction
    A scalar-output VectorFunction whose value is the elementwise sum of all
    inputs.

Raises
------
ValueError
    If any two functions have mismatched input dimensions, or the list is
    empty.
)doc");
    m.def("sum", [](const std::vector<Gen> &elems) { return DynamicSum(elems); });

    m.def("sum_elems", &make_dynamic_sum<GenS, ELEM>,
          R"doc(Elementwise sum of several scalar segment-output VectorFunctions.

Sums a list of single-element segment VectorFunctions (each extracting one
output component) into a single scalar-output VectorFunction.  All segments
must share the same input dimension.  Useful for building scalar objectives
or constraints from individual state or control components.

Parameters
----------
funcs : list[VectorFunction]
    Single-element segment VectorFunctions sharing the same input dimension.
    The list must contain at least one element.

Returns
-------
VectorFunction
    A scalar-output VectorFunction whose value is the sum of all segment
    values.

Raises
------
ValueError
    If the list is empty or any two elements have mismatched input
    dimensions.
)doc");
    m.def("sum_elems", [](const std::vector<ELEM> &elems, const std::vector<double> &scales) {
        std::vector<Scaled<ELEM>> selems;
        for (int i = 0; i < elems.size(); i++) {
            selems.emplace_back(elems[i] * scales[i]);
        };
        return make_dynamic_sum<GenS, Scaled<ELEM>>(selems);
    });
}