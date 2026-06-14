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

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// TychoBind<T> specializations for CommonFunctions types.
// Also provides bind::SegBuild<Derived, PyClass> free function.
// Included from CommonFunctions/CommonFunctions.h under TYCHO_PYTHON_BINDINGS.

#include "dense_function_base_bind.h"

namespace tycho {

using namespace tycho::vf;

// ── bind::SegBuild
// ────────────────────────────────────────────────────────────────────────────────
namespace bind {

template <class Derived, class PyClass> void SegBuild(PyClass &obj) {
    bind::DoubleMathBuild<Derived>(obj);
    bind::UnaryMathBuild<Derived>(obj);
    bind::BinaryMathBuild<Derived>(obj);
    bind::BinaryOperatorsBuild<Derived>(obj);
    bind::FunctionIndexingBuild<Derived>(obj);
    bind::ConditionalOperatorsBuild<Derived>(obj);

    obj.def("tolist", [](const Derived &func) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (int i = 0; i < func.output_rows(); i++) {
            elems.push_back(func.coeff(i));
        }
        return elems;
    },
    R"doc(Split the function into a list of scalar element functions.

Returns one scalar-valued VectorFunction per output row, in order, each
extracting a single component of the original output vector.

Returns
-------
list of Element
    A list of ``output_rows`` scalar Element functions, where element *i*
    extracts output component *i*.
)doc");

    obj.def("tolist", [](const Derived &func, std::vector<int> coeffs) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (const auto &coeff : coeffs) {
            elems.push_back(func.coeff(coeff));
        }
        return elems;
    });

    obj.def("tolist", [](const Derived &func, std::vector<std::tuple<int, int>> seglist) {
        using ELEM = Segment<-1, 1, -1>;
        using SEG2 = Segment<-1, 2, -1>;
        using SEG3 = Segment<-1, 3, -1>;
        using SEG = Segment<-1, -1, -1>;

        std::vector<nb::object> segs;
        for (const auto &seg : seglist) {

            int start = std::get<0>(seg);
            int size = std::get<1>(seg);
            nb::object pyfun;
            if (size == 1) {
                auto f = func.coeff(start);
                pyfun = nb::cast(f);
            } else if (size == 2) {
                auto f = func.template segment<2>(start);
                pyfun = nb::cast(f);
            } else if (size == 3) {
                auto f = func.template segment<3>(start);
                pyfun = nb::cast(f);
            } else {
                auto f = func.segment(start, size);
                pyfun = nb::cast(f);
            }

            segs.push_back(pyfun);
        }
        return segs;
    });
}

} // namespace bind

// ── Constant
// ──────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR> struct TychoBind<Constant<IR, OR>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Constant<IR, OR>>(m, name);
        obj.def(nb::init<int, Eigen::VectorXd>(),
                R"doc(Construct a constant VectorFunction with a given input size and fixed output vector.

Parameters
----------
input_rows : int
    Number of input rows (dimension of the input vector).
value : array_like, shape (output_rows,)
    Fixed output vector returned for every input. Its length sets the output
    dimension.
)doc");
        bind::DenseBaseBuild<Constant<IR, OR>>(obj);
    }
};

// ── FunctionDotProduct_Impl
// ───────────────────────────────────────────────────────────────────────
template <class Derived, class Func1, class Func2>
struct TychoBind<FunctionDotProduct_Impl<Derived, Func1, Func2>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<Func1, Func2>(),
                R"doc(Construct the dot-product function from two operand functions.

Parameters
----------
f1 : VectorFunction
    First operand.
f2 : VectorFunction
    Second operand with matching dimensions.
)doc");
        bind::DenseBaseBuild<Derived>(obj);
    }
};

// ── NestedFunction
// ────────────────────────────────────────────────────────────────────────────────
template <class OuterFunc, class InnerFunc> struct TychoBind<NestedFunction<OuterFunc, InnerFunc>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<NestedFunction<OuterFunc, InnerFunc>>(m, name);
        obj.def(nb::init<>(),
                R"doc(Construct an empty NestedFunction (inner and outer functions set later).
)doc");
        obj.def(nb::init<OuterFunc, InnerFunc>());
        bind::DenseBaseBuild<NestedFunction<OuterFunc, InnerFunc>>(obj);
    }
};

// ── NormalizedPower_Impl
// ──────────────────────────────────────────────────────────────────────────
template <class Derived, int IR, int PW> struct TychoBind<NormalizedPower_Impl<Derived, IR, PW>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<int>(),
                R"doc(Construct a normalized power function for a given input size.

Parameters
----------
input_rows : int
    Number of input rows (dimension of the input vector).
)doc");
        if constexpr (IR > 0) {
            obj.def(nb::init<>());
        }
        bind::DenseBaseBuild<Derived>(obj);
    }
};

// ── IntegralNorm_Impl
// ─────────────────────────────────────────────────────────────────────────────
template <class Derived, int USZ, int Power>
struct TychoBind<IntegralNorm_Impl<Derived, USZ, Power>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<int>(),
                R"doc(Construct an integral norm function for a given input size.

Parameters
----------
input_rows : int
    Number of input rows (dimension of the input vector).
)doc");
        if constexpr (USZ > 0) {
            obj.def(nb::init<>());
        }
        bind::DenseBaseBuild<Derived>(obj);
    }
};

// ── ParsedInput
// ───────────────────────────────────────────────────────────────────────────────────
template <class Func, int IRC, int ORC> struct TychoBind<ParsedInput<Func, IRC, ORC>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<ParsedInput<Func, IRC, ORC>>(m, name);
        obj.def(nb::init<Func, const Eigen::VectorXi &, int>(),
                R"doc(Construct a ParsedInput by selecting a subset of input components.

Parameters
----------
func : VectorFunction
    Wrapped function to evaluate on the selected input components.
indices : array_like of int, shape (selected,)
    Indices of input components to extract and pass to *func*.
total_input_rows : int
    Total number of rows in the outer input vector.
)doc");
        bind::DenseBaseBuild<ParsedInput<Func, IRC, ORC>>(obj);
    }
};

// ── Arguments
// ─────────────────────────────────────────────────────────────────────────────────────
template <int IR_OR> struct TychoBind<Arguments<IR_OR>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Arguments<IR_OR>>(m, name);
        obj.def(nb::init<int>(),
                R"doc(Construct an identity-passthrough VectorFunction for a given input size.

``Arguments(n)`` is the identity function on ``n``-dimensional inputs: its
output is the full input vector. It is the usual starting point for building
VectorFunction expressions — index into it to form element or segment functions.

Parameters
----------
input_rows : int
    Number of input rows (dimension of the input/output vector).
)doc");
        obj.def("constant", [](const Arguments<IR_OR> &a, Eigen::VectorXd v) {
            return GenericFunction<-1, -1>(Constant<-1, -1>(a.input_rows(), v));
        },
        R"doc(Build a constant VectorFunction whose output is a fixed vector.

Creates a VectorFunction that returns the same vector *v* for any input,
with input dimension matching this ``Arguments`` object. The Jacobian and
adjoint Hessian of a constant function are identically zero.

Parameters
----------
v : array_like, shape (output_rows,)
    Fixed output vector to return at every input point.

Returns
-------
GenericFunction
    Constant function with ``input_rows`` matching this object and
    ``output_rows`` equal to ``len(v)``.

Examples
--------
>>> import numpy as np
>>> from tychopy import vector_functions as vf
>>> x = vf.Arguments(3)
>>> c = x.constant(np.array([1.0, 2.0, 3.0]))
>>> c.compute(np.zeros(3))
array([1., 2., 3.])
)doc");
        obj.def("constant", [](const Arguments<IR_OR> &a, double v) {
            Eigen::Matrix<double, 1, 1> vv;
            vv[0] = v;
            return GenericFunction<-1, 1>(Constant<-1, 1>(a.input_rows(), vv));
        });
        bind::DenseBaseBuild<Arguments<IR_OR>>(obj);
        bind::SegBuild<Arguments<IR_OR>>(obj);
    }
};

// ── Segment
// ───────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR, int ST> struct TychoBind<Segment<IR, OR, ST>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Segment<IR, OR, ST>>(m, name);
        obj.def(nb::init<int, int, int>(),
                R"doc(Construct a Segment by specifying input size, segment length, and start index.

Parameters
----------
input_rows : int
    Total number of input rows.
output_rows : int
    Number of consecutive input components to extract (segment length).
start : int
    Index of the first input component to extract.
)doc");
        bind::DenseBaseBuild<Segment<IR, OR, ST>>(obj);
        bind::SegBuild<Segment<IR, OR, ST>>(obj);
    }
};

// ── TwoFunctionSum_Impl
// ───────────────────────────────────────────────────────────────────────────
template <class Derived, class Func1, class Func2, bool DoDifference>
struct TychoBind<TwoFunctionSum_Impl<Derived, Func1, Func2, DoDifference>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name);
        obj.def(nb::init<Func1, Func2>(),
                R"doc(Construct the sum (or difference) of two VectorFunctions.

Parameters
----------
f1 : VectorFunction
    First operand.
f2 : VectorFunction
    Second operand with matching input and output dimensions.
)doc");
        bind::DenseBaseBuild<Derived>(obj);
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
