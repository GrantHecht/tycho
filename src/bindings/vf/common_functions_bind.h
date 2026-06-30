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
    },
    R"doc(Split selected output components into a list of scalar element functions.

Returns one scalar-valued VectorFunction for each index in *coeffs*, each
extracting the corresponding output component.

Parameters
----------
coeffs : list of int
    Output-component indices to extract, in the requested order.

Returns
-------
list of Element
    A list of scalar Element functions, one per entry in *coeffs*.
)doc");

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
    },
    R"doc(Split the function into a mixed list of element and segment functions.

Each entry in *seglist* is a ``(start, size)`` pair. A pair with ``size == 1``
produces a scalar Element function; ``size == 2`` or ``3`` produces a
fixed-width Segment; any other size produces a dynamic-length Segment.

Parameters
----------
seglist : list of (int, int)
    List of ``(start, size)`` pairs specifying which contiguous slices of the
    output vector to extract.

Returns
-------
list of Element or Segment
    A heterogeneous list of VectorFunctions, one per entry in *seglist*.
)doc");
}

} // namespace bind

// ── Constant
// ──────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR> struct TychoBind<Constant<IR, OR>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Constant<IR, OR>>(m, name, nb::pooled(128),
            R"doc(A VectorFunction that always returns the same fixed output vector.

Regardless of the input, ``Constant`` evaluates to a compile-time or
runtime-specified constant vector. Its Jacobian and Hessian are identically
zero. Construct one via ``Arguments.constant(v)`` or directly with
``ConstantVector(input_rows, v)`` / ``ConstantScalar(input_rows, v)``.
)doc");
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
        auto obj = nb::class_<Derived>(m, name,
            R"doc(Elementwise dot product of two VectorFunctions evaluated on the same input.

Given two vector-valued functions ``f1`` and ``f2`` with matching input and
output dimensions, this VectorFunction computes the scalar ``f1(x) · f2(x)``
for every input *x*. Obtain one via the ``dot`` free function in
``tychopy.vector_functions``.
)doc");
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
        auto obj = nb::class_<NestedFunction<OuterFunc, InnerFunc>>(m, name,
            R"doc(Function composition: the outer function applied to the output of the inner function.

``NestedFunction(outer, inner)`` represents the composition ``outer(inner(x))``.
Jacobians and Hessians are propagated through the chain rule automatically.
Construct one by passing a VectorFunction as the argument to another —
``outer(inner)`` or ``outer.eval(inner)`` — or by using the two-argument
constructor directly.
)doc");
        obj.def(nb::init<>(),
                R"doc(Construct an empty NestedFunction (inner and outer functions set later).
)doc");
        obj.def(nb::init<OuterFunc, InnerFunc>(),
                R"doc(Construct a NestedFunction from explicit outer and inner functions.

Parameters
----------
outer : VectorFunction
    The outer function; receives the output of *inner* as its input.
inner : VectorFunction
    The inner function; receives the original input *x*.
)doc");
        bind::DenseBaseBuild<NestedFunction<OuterFunc, InnerFunc>>(obj);
    }
};

// ── NormalizedPower_Impl
// ──────────────────────────────────────────────────────────────────────────
template <class Derived, int IR, int PW> struct TychoBind<NormalizedPower_Impl<Derived, IR, PW>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived>(m, name,
            R"doc(Input vector divided by an integer power of its Euclidean norm.

Computes ``x / ||x||_2^PW`` for a compile-time positive integer exponent
*PW* — the unit vector when *PW* = 1, the vector divided by its squared norm
when *PW* = 2, and so on. Obtain one via the ``normalized`` (alias
``normalize``), ``normalized_power2``, ``normalized_power3``,
``normalized_power4``, or ``normalized_power5`` free functions in
``tychopy.vector_functions``.
)doc");
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
        auto obj = nb::class_<Derived>(m, name,
            R"doc(Integer power of the Euclidean (L2) norm of a vector-valued input.

Computes ``||x||_2^Power`` for a compile-time integer exponent *Power*
(scalar output). Used as an objective or constraint term whenever a norm
raised to a fixed power is required. Obtain one via the ``norm``,
``squared_norm``, ``cubed_norm``, ``inverse_norm``,
``inverse_squared_norm``, ``inverse_cubed_norm``, or ``inverse_four_norm``
free functions in ``tychopy.vector_functions``.
)doc");
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
        auto obj = nb::class_<ParsedInput<Func, IRC, ORC>>(m, name,
            R"doc(A VectorFunction that selects a subset of input components before evaluation.

``ParsedInput(func, indices, total_input_rows)`` wraps *func* so that it
receives only the input components at the specified *indices*, taken from a
larger input vector of length *total_input_rows*. Jacobian columns are
re-mapped to the correct positions in the outer input space. Typically
produced via ``func.eval(ir, v)`` where ``ir`` is the ambient input size and
``v`` is an integer index vector mapping each of ``func``'s inputs to a
position in the outer input vector.
)doc");
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
        auto obj = nb::class_<Arguments<IR_OR>>(m, name, nb::pooled(128),
            R"doc(The identity VectorFunction — the symbolic input vector for building expressions.

``Arguments(n)`` is the identity map on ``n``-dimensional inputs: it returns
its input unchanged. It is the standard starting point for constructing
VectorFunction expressions — use indexing (``x[i]``, ``x[i:j]``) or methods
such as ``x.constant(v)`` to derive sub-expressions and build up dynamics,
constraints, and objectives.
)doc");
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
        },
        R"doc(Build a scalar constant VectorFunction whose output is a fixed scalar.

Creates a scalar-output VectorFunction that returns *v* for any input, with
input dimension matching this ``Arguments`` object.

Parameters
----------
v : float
    Fixed scalar output value.

Returns
-------
ScalarFunction
    Constant scalar function with ``input_rows`` matching this object.
)doc");
        bind::DenseBaseBuild<Arguments<IR_OR>>(obj);
        bind::SegBuild<Arguments<IR_OR>>(obj);
    }
};

// ── Segment
// ───────────────────────────────────────────────────────────────────────────────────────
template <int IR, int OR, int ST> struct TychoBind<Segment<IR, OR, ST>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Segment<IR, OR, ST>>(m, name, nb::pooled(128),
            R"doc(A contiguous sub-vector view of a VectorFunction's input.

``Segment(input_rows, output_rows, start)`` extracts ``output_rows``
consecutive components of the input vector beginning at index ``start``,
producing a new vector-valued VectorFunction. It is the standard mechanism for
slicing the symbolic input — obtained by indexing ``Arguments`` objects (e.g.
``x[2:5]``) or by calling ``.segment(start, size)`` on any VectorFunction.
)doc");
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
        auto obj = nb::class_<Derived>(m, name,
            R"doc(Elementwise sum or difference of two VectorFunctions on a shared input.

When ``DoDifference`` is false, evaluates ``f1(x) + f2(x)``; when true,
evaluates ``f1(x) - f2(x)``. Both operands must share the same input dimension
and output dimension. Produced automatically by the ``+`` and ``-`` operators
on VectorFunction expressions.
)doc");
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
