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

// Free function templates in namespace tycho::bind for GenericFunction,
// GenericComparative, and GenericConditional binding methods.
// Included from tycho_vector_functions.h under TYCHO_PYTHON_BINDINGS,
// after all three type definitions are complete.

#include "dense_function_base_bind.h"

namespace tycho {

using namespace tycho::vf;

namespace bind {

// ── GenericBuild
// ──────────────────────────────────────────────────────────────────────────────────
template <class Derived, class PYClass> void GenericBuild(PYClass &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<Derived::ORC == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    obj.def(nb::init<const Derived &>(),
            R"doc(Construct a GenericFunction as a shared-ownership copy of another.

Parameters
----------
src : GenericFunction
    Source function to copy. The new instance shares ownership of the
    underlying erased function (O(1) refcount increment — no deep clone).

Raises
------
ValueError
    If *src* is an empty (null) GenericFunction.
)doc");
    if constexpr (Derived::ORC == -1 && Derived::IRC == -1) {
        obj.def("__init__", [](Derived *self, const GenericFunction<Derived::IRC, 1> &src) {
            new (self) Derived(Derived::template PyCopy<GenericFunction<Derived::IRC, 1>>(src));
        },
        R"doc(Construct a fully-dynamic GenericFunction from a scalar-output GenericFunction.

Absorbs a scalar-valued (output rows = 1) GenericFunction into a fully-dynamic
``GenericFunction[-1, -1]`` by cloning it into dynamic storage.

Parameters
----------
src : GenericFunction
    Scalar-output source function to promote. Must be non-empty.

Raises
------
ValueError
    If *src* is an empty (null) GenericFunction.
)doc");
    }

    bind::DenseBaseBuild<Derived>(obj);
}

// ── MinMaxBuild
// ───────────────────────────────────────────────────────────────────────────────────
template <class PYCLASS> void MinMaxBuild(PYCLASS &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using GenComp = GenericComparative<-1>;

    // =========================================================================
    // ONE ARG BINDINGS

    // =========================================================================
    // TWO ARG BINDINGS
    // 1D Output Maximization Bindings
    obj.def("max", [](const GenS &f1, const GenS &f2) {
        return GenS(ComparativeFunction<GenS, GenS>(ComparativeFlags::MaxFlag, f1, f2));
    },
    R"doc(Return a VectorFunction that evaluates the pointwise maximum of two scalar functions.

Both operands must be scalar-valued (output rows = 1) and share the same input
dimension. At each evaluation point the result equals ``max(f1(x), f2(x))``.
The derivative is taken from whichever branch is selected, so the result is
piecewise differentiable across the switching boundary.

Parameters
----------
f1 : GenericFunction (scalar)
    First scalar operand.
f2 : GenericFunction (scalar)
    Second scalar operand.

Returns
-------
GenericFunction (scalar)
    Scalar function returning the pointwise maximum.

Raises
------
ValueError
    If *f1* and *f2* have different input dimensions.
)doc");
    obj.def("max", [](double v1, const GenS &f2) {
        Vector1<double> v;
        v[0] = v1;
        Constant<-1, 1> f1(f2.input_rows(), v);
        return GenS(ComparativeFunction<Constant<-1, 1>, GenS>(ComparativeFlags::MaxFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the pointwise maximum of a scalar constant and a scalar function.

Parameters
----------
v1 : float
    Constant scalar to compare against.
f2 : GenericFunction (scalar)
    Scalar function operand.

Returns
-------
GenericFunction (scalar)
    Scalar function returning ``max(v1, f2(x))``.
)doc");
    obj.def("max", [](const GenS &f1, double v2) {
        Vector1<double> v;
        v[0] = v2;
        Constant<-1, 1> f2(f1.input_rows(), v);
        return GenS(ComparativeFunction<GenS, Constant<-1, 1>>(ComparativeFlags::MaxFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the pointwise maximum of a scalar function and a scalar constant.

Parameters
----------
f1 : GenericFunction (scalar)
    Scalar function operand.
v2 : float
    Constant scalar to compare against.

Returns
-------
GenericFunction (scalar)
    Scalar function returning ``max(f1(x), v2)``.
)doc");

    // ND Output Maximization Bindings
    obj.def("max", [](const Gen &f1, const Gen &f2) {
        return Gen(ComparativeFunction<Gen, Gen>(ComparativeFlags::MaxFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the element-wise maximum of two vector-valued functions.

Both operands must have the same input dimension and the same output
dimension. The element-wise maximum is taken independently for each output
component. Derivatives track whichever branch is selected per component.

Parameters
----------
f1 : GenericFunction
    First vector-valued operand.
f2 : GenericFunction
    Second vector-valued operand.

Returns
-------
GenericFunction
    Vector function returning the element-wise maximum of *f1* and *f2*.

Raises
------
ValueError
    If *f1* and *f2* have different input or output dimensions.
)doc");
    obj.def("max", [](Eigen::VectorXd v1, const Gen &f2) {
        Constant<-1, -1> f1(f2.input_rows(), v1);
        return Gen(ComparativeFunction<Constant<-1, -1>, Gen>(ComparativeFlags::MaxFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the element-wise maximum of a constant vector and a vector-valued function.

Parameters
----------
v1 : array_like
    Constant vector; must have the same length as ``f2.output_rows()``.
f2 : GenericFunction
    Vector-valued function operand.

Returns
-------
GenericFunction
    Vector function returning the element-wise maximum of *v1* and *f2(x)*.
)doc");
    obj.def("max", [](const Gen &f1, Eigen::VectorXd v2) {
        Constant<-1, -1> f2(f1.input_rows(), v2);
        return Gen(ComparativeFunction<Gen, Constant<-1, -1>>(ComparativeFlags::MaxFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the element-wise maximum of a vector-valued function and a constant vector.

Parameters
----------
f1 : GenericFunction
    Vector-valued function operand.
v2 : array_like
    Constant vector; must have the same length as ``f1.output_rows()``.

Returns
-------
GenericFunction
    Vector function returning the element-wise maximum of *f1(x)* and *v2*.
)doc");

    // 1D Output Minimization Bindings
    obj.def("min", [](const GenS &f1, const GenS &f2) {
        return GenS(ComparativeFunction<GenS, GenS>(ComparativeFlags::MinFlag, f1, f2));
    },
    R"doc(Return a VectorFunction that evaluates the pointwise minimum of two scalar functions.

Both operands must be scalar-valued (output rows = 1) and share the same input
dimension. At each evaluation point the result equals ``min(f1(x), f2(x))``.
The derivative is taken from whichever branch is selected, so the result is
piecewise differentiable across the switching boundary.

Parameters
----------
f1 : GenericFunction (scalar)
    First scalar operand.
f2 : GenericFunction (scalar)
    Second scalar operand.

Returns
-------
GenericFunction (scalar)
    Scalar function returning the pointwise minimum.

Raises
------
ValueError
    If *f1* and *f2* have different input dimensions.
)doc");
    obj.def("min", [](double v1, const GenS &f2) {
        Vector1<double> v;
        v[0] = v1;
        Constant<-1, 1> f1(f2.input_rows(), v);
        return GenS(ComparativeFunction<Constant<-1, 1>, GenS>(ComparativeFlags::MinFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the pointwise minimum of a scalar constant and a scalar function.

Parameters
----------
v1 : float
    Constant scalar to compare against.
f2 : GenericFunction (scalar)
    Scalar function operand.

Returns
-------
GenericFunction (scalar)
    Scalar function returning ``min(v1, f2(x))``.
)doc");
    obj.def("min", [](const GenS &f1, double v2) {
        Vector1<double> v;
        v[0] = v2;
        Constant<-1, 1> f2(f1.input_rows(), v);
        return GenS(ComparativeFunction<GenS, Constant<-1, 1>>(ComparativeFlags::MinFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the pointwise minimum of a scalar function and a scalar constant.

Parameters
----------
f1 : GenericFunction (scalar)
    Scalar function operand.
v2 : float
    Constant scalar to compare against.

Returns
-------
GenericFunction (scalar)
    Scalar function returning ``min(f1(x), v2)``.
)doc");

    // ND Output Minimization Bindings
    obj.def("min", [](const Gen &f1, const Gen &f2) {
        return Gen(ComparativeFunction<Gen, Gen>(ComparativeFlags::MinFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the element-wise minimum of two vector-valued functions.

Both operands must have the same input dimension and the same output
dimension. The element-wise minimum is taken independently for each output
component. Derivatives track whichever branch is selected per component.

Parameters
----------
f1 : GenericFunction
    First vector-valued operand.
f2 : GenericFunction
    Second vector-valued operand.

Returns
-------
GenericFunction
    Vector function returning the element-wise minimum of *f1* and *f2*.

Raises
------
ValueError
    If *f1* and *f2* have different input or output dimensions.
)doc");
    obj.def("min", [](Eigen::VectorXd v1, const Gen &f2) {
        Constant<-1, -1> f1(f2.input_rows(), v1);
        return Gen(ComparativeFunction<Constant<-1, -1>, Gen>(ComparativeFlags::MinFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the element-wise minimum of a constant vector and a vector-valued function.

Parameters
----------
v1 : array_like
    Constant vector; must have the same length as ``f2.output_rows()``.
f2 : GenericFunction
    Vector-valued function operand.

Returns
-------
GenericFunction
    Vector function returning the element-wise minimum of *v1* and *f2(x)*.
)doc");
    obj.def("min", [](const Gen &f1, Eigen::VectorXd v2) {
        Constant<-1, -1> f2(f1.input_rows(), v2);
        return Gen(ComparativeFunction<Gen, Constant<-1, -1>>(ComparativeFlags::MinFlag, f1, f2));
    },
    R"doc(Return a VectorFunction evaluating the element-wise minimum of a vector-valued function and a constant vector.

Parameters
----------
f1 : GenericFunction
    Vector-valued function operand.
v2 : array_like
    Constant vector; must have the same length as ``f1.output_rows()``.

Returns
-------
GenericFunction
    Vector function returning the element-wise minimum of *f1(x)* and *v2*.
)doc");

    // =========================================================================
    // THREE ARG BINDINGS
    // TODO: 3-argument min-max bindings
}

// ── ComparativeBuild
// ──────────────────────────────────────────────────────────────────────────────
inline void ComparativeBuild(nb::module_ &m) {
    using GenComp = GenericComparative<-1>;

    auto obj = nb::class_<GenComp>(m, "Comparative",
        R"doc(Type-erased predicate used as the branch selector for min/max VectorFunctions.

A ``Comparative`` wraps a pairwise min or max comparison between two
scalar-valued (or vector-valued) VectorFunctions behind a uniform boolean
interface. It is produced internally by the ``min`` and ``max``
methods on ``Comparative`` and is distinct from :class:`Conditional` so
that the two types are separately registered in the Python module.

At each evaluation point ``compute(x)`` returns the boolean result of the
underlying comparison (i.e. which operand is selected). The associated
``min`` / ``max`` combinators return a new VectorFunction (not a
``Comparative``) whose value and derivatives track the selected branch.

The ``Comparative`` object itself is rarely needed directly — it surfaces
only when you want to inspect the switching predicate independently of the
enclosing min/max function.
)doc");

    obj.def("compute",
            [](const GenComp &a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); },
            R"doc(Evaluate the comparative function at a given input vector.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector at which to evaluate the min/max function.

Returns
-------
bool
    ``True`` if the first operand (*f1*) is selected (i.e. ``f1(x)`` wins
    the min/max comparison); ``False`` if the second operand (*f2*) is
    selected.
)doc");

    bind::MinMaxBuild(obj);
}

// ── IfElseBuild
// ───────────────────────────────────────────────────────────────────────────────────
template <class PYCLASS> void IfElseBuild(PYCLASS &obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<-1>;

    obj.def("ifelse", [](const GenCon &test, const GenS &tf, const GenS &ff) {
        return GenS(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a scalar VectorFunction that selects between two scalar branches based on a predicate.

At each evaluation point the predicate *test* is checked. If true, *tf* is
evaluated; otherwise *ff* is evaluated. Derivatives are taken from whichever
branch is active, so the result is only piecewise differentiable across the
switching boundary.

All three arguments must share the same input dimension, and both branch
functions must be scalar-valued (output rows = 1).

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
tf : GenericFunction (scalar)
    Function evaluated when *test* is true.
ff : GenericFunction (scalar)
    Function evaluated when *test* is false.

Returns
-------
GenericFunction (scalar)
    Scalar function returning ``tf(x)`` when ``test(x)`` is true, else ``ff(x)``.

Raises
------
ValueError
    If *test*, *tf*, and *ff* do not share the same input dimension, or if
    *tf* and *ff* have different output dimensions.
)doc");

    obj.def("ifelse", [](const GenCon &test, double tfv, const GenS &ff) {
        Vector1<double> v;
        v[0] = tfv;
        Constant<-1, 1> tf(test.input_rows(), v);
        return GenS(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a scalar VectorFunction that returns a constant when the predicate is true, else a scalar function.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
tfv : float
    Constant value returned when *test* is true.
ff : GenericFunction (scalar)
    Function evaluated when *test* is false.

Returns
-------
GenericFunction (scalar)
    Scalar function returning *tfv* when ``test(x)`` is true, else ``ff(x)``.
)doc");
    obj.def("ifelse", [](const GenCon &test, const GenS &tf, double ffv) {
        Vector1<double> v;
        v[0] = ffv;
        Constant<-1, 1> ff(test.input_rows(), v);
        return GenS(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a scalar VectorFunction that returns a scalar function when the predicate is true, else a constant.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
tf : GenericFunction (scalar)
    Function evaluated when *test* is true.
ffv : float
    Constant value returned when *test* is false.

Returns
-------
GenericFunction (scalar)
    Scalar function returning ``tf(x)`` when ``test(x)`` is true, else *ffv*.
)doc");
    obj.def("ifelse", [](const GenCon &test, double tfv, double ffv) {
        Vector1<double> v1;
        v1[0] = tfv;
        Constant<-1, 1> tf(test.input_rows(), v1);
        Vector1<double> v2;
        v2[0] = ffv;
        Constant<-1, 1> ff(test.input_rows(), v2);
        return GenS(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a scalar VectorFunction that selects between two scalar constants based on a predicate.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
tfv : float
    Constant value returned when *test* is true.
ffv : float
    Constant value returned when *test* is false.

Returns
-------
GenericFunction (scalar)
    Scalar function returning *tfv* when ``test(x)`` is true, else *ffv*.
)doc");

    obj.def("ifelse", [](const GenCon &test, const Gen &tf, const Gen &ff) {
        return Gen(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a vector-valued VectorFunction that selects between two vector-valued branches based on a predicate.

At each evaluation point the predicate *test* is checked. If true, *tf* is
evaluated; otherwise *ff* is evaluated. Both branch functions must share the
same input and output dimensions as one another, and their input dimension
must match that of *test*. Derivatives are taken from the active branch
only, so the result is piecewise differentiable across the switching
boundary.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
tf : GenericFunction
    Vector-valued function evaluated when *test* is true.
ff : GenericFunction
    Vector-valued function evaluated when *test* is false.

Returns
-------
GenericFunction
    Vector function returning ``tf(x)`` when ``test(x)`` is true, else ``ff(x)``.

Raises
------
ValueError
    If *test*, *tf*, and *ff* do not share the same input dimension, or if
    *tf* and *ff* have different output dimensions.
)doc");

    obj.def("ifelse", [](const GenCon &test, Eigen::VectorXd v, const Gen &ff) {
        Constant<-1, -1> tf(test.input_rows(), v);
        return Gen(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a vector-valued VectorFunction that returns a constant vector when the predicate is true, else a vector-valued function.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
v : array_like
    Constant vector returned when *test* is true; must have the same length
    as ``ff.output_rows()``.
ff : GenericFunction
    Vector-valued function evaluated when *test* is false.

Returns
-------
GenericFunction
    Vector function returning *v* when ``test(x)`` is true, else ``ff(x)``.
)doc");
    obj.def("ifelse", [](const GenCon &test, const Gen &tf, Eigen::VectorXd v) {
        Constant<-1, -1> ff(test.input_rows(), v);
        return Gen(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a vector-valued VectorFunction that returns a vector-valued function when the predicate is true, else a constant vector.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
tf : GenericFunction
    Vector-valued function evaluated when *test* is true.
v : array_like
    Constant vector returned when *test* is false; must have the same length
    as ``tf.output_rows()``.

Returns
-------
GenericFunction
    Vector function returning ``tf(x)`` when ``test(x)`` is true, else *v*.
)doc");

    obj.def("ifelse", [](const GenCon &test, Eigen::VectorXd v1, Eigen::VectorXd v2) {
        Constant<-1, -1> tf(test.input_rows(), v1);
        Constant<-1, -1> ff(test.input_rows(), v2);
        return Gen(IfElseFunction{test, tf, ff});
    },
    R"doc(Build a vector-valued VectorFunction that selects between two constant vectors based on a predicate.

Parameters
----------
test : Conditional
    Predicate that selects the active branch.
v1 : array_like
    Constant vector returned when *test* is true.
v2 : array_like
    Constant vector returned when *test* is false; must have the same length
    as *v1*.

Returns
-------
GenericFunction
    Vector function returning *v1* when ``test(x)`` is true, else *v2*.
)doc");
}

// ── ConditionalBuild
// ──────────────────────────────────────────────────────────────────────────────
inline void ConditionalBuild(nb::module_ &m) {
    using GenCon = GenericConditional<-1>;

    auto obj = nb::class_<GenCon>(m, "Conditional",
        R"doc(Type-erased boolean predicate over an input vector.

A ``Conditional`` wraps any comparison or logical combination of
scalar-valued VectorFunctions behind a single uniform interface. It is the
predicate type that drives :func:`ifelse` branching.

Leaf predicates are formed by applying comparison operators (``<``, ``<=``,
``>``, ``>=``) to pairs of scalar VectorFunctions; these operators
return ``Conditional`` objects. Leaf predicates can then be combined with
``&`` (:meth:`__and__`) and ``|`` (:meth:`__or__`) to build compound
logical expressions. The resulting ``Conditional`` can be passed to
:func:`ifelse` to select between two VectorFunction branches.

``compute(x)`` evaluates the predicate at the input vector *x* and returns
the resulting ``bool``. Derivatives are not defined on ``Conditional``
itself — they are computed from whichever branch is active in the enclosing
:func:`ifelse` or ``min`` / ``max`` expression.
)doc");

    obj.def("compute",
            [](const GenCon &a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); },
            R"doc(Evaluate the conditional predicate at a given input vector.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector at which to evaluate the predicate.

Returns
-------
bool
    Result of the comparison or logical combination at *x*.
)doc");

    obj.def(
        "__and__",
        [](const GenCon &a, const GenCon &b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ANDFlag, b));
        },
        nb::is_operator(),
        R"doc(Combine two conditionals with logical AND.

Returns a new Conditional that is true only when both *self* and *other* are
true at the same input point.

Parameters
----------
other : Conditional
    Right-hand conditional to combine.

Returns
-------
Conditional
    Logical AND of the two predicates.
)doc");

    obj.def(
        "__or__",
        [](const GenCon &a, const GenCon &b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ORFlag, b));
        },
        nb::is_operator(),
        R"doc(Combine two conditionals with logical OR.

Returns a new Conditional that is true when at least one of *self* or *other*
is true at the same input point.

Parameters
----------
other : Conditional
    Right-hand conditional to combine.

Returns
-------
Conditional
    Logical OR of the two predicates.
)doc");

    bind::IfElseBuild(obj);
}

} // namespace bind
} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
