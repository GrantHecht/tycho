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

#include "tycho/detail/vf/operators/root_finder.h"
#include "tycho_vector_functions.h"
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

template <class T> void UnaryVectorOpBuild(nb::module_ &m) {

    auto UnaryOpLam = [](const T &cfun, const char *name) {
        nb::object fun = nb::cast(cfun);
        return fun.attr(name)();
    };
    m.def(
        "norm", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "norm"); },
        R"doc(Euclidean (L2) norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose output norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``||fun(x)||_2``.
)doc");
    m.def(
        "squared_norm", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "squared_norm"); },
        R"doc(Squared Euclidean norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose squared norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``||fun(x)||_2^2``.
)doc");
    m.def(
        "cubed_norm", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "cubed_norm"); },
        R"doc(Euclidean norm raised to the third power of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose cubed norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``||fun(x)||_2^3``.
)doc");

    m.def(
        "inverse_norm", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "inverse_norm"); },
        R"doc(Reciprocal Euclidean norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose reciprocal norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``1 / ||fun(x)||_2``.
)doc");
    m.def(
        "inverse_squared_norm",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "inverse_squared_norm"); },
        R"doc(Reciprocal squared Euclidean norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose reciprocal squared norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``1 / ||fun(x)||_2^2``.
)doc");
    m.def(
        "inverse_cubed_norm",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "inverse_cubed_norm"); },
        R"doc(Reciprocal cubed Euclidean norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose reciprocal cubed norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``1 / ||fun(x)||_2^3``.
)doc");
    m.def(
        "inverse_four_norm",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "inverse_four_norm"); },
        R"doc(Reciprocal fourth-power Euclidean norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction whose reciprocal fourth-power norm is computed.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``1 / ||fun(x)||_2^4``.
)doc");

    m.def(
        "normalize", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "normalized"); },
        R"doc(Unit-vector direction of a vector-valued VectorFunction.

Alias for :func:`normalized`.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction to normalise.

Returns
-------
VectorFunction
    Vector VectorFunction with the same output dimension as *fun*, evaluating
    ``fun(x) / ||fun(x)||_2``.
)doc");
    m.def(
        "normalized", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "normalized"); },
        R"doc(Unit-vector direction of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction to normalise.

Returns
-------
VectorFunction
    Vector VectorFunction with the same output dimension as *fun*, evaluating
    ``fun(x) / ||fun(x)||_2``.
)doc");
    m.def(
        "normalized_power2",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "normalized_power2"); },
        R"doc(Vector divided by the squared norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^2``.
)doc");
    m.def(
        "normalized_power3",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "normalized_power3"); },
        R"doc(Vector divided by the cubed norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^3``.
)doc");
    m.def(
        "normalized_power4",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "normalized_power4"); },
        R"doc(Vector divided by the fourth-power norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^4``.
)doc");
    m.def(
        "normalized_power5",
        [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "normalized_power5"); },
        R"doc(Vector divided by the fifth-power norm of a vector-valued VectorFunction.

Parameters
----------
fun : VectorFunction
    Vector-valued VectorFunction.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating ``fun(x) / ||fun(x)||_2^5``.
)doc");
}

template <class T> void UnaryScalarOpBuild(nb::module_ &m) {

    auto UnaryOpLam = [](const T &cfun, const char *name) {
        nb::object fun = nb::cast(cfun);
        return fun.attr(name)();
    };
    m.def(
        "sin", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "sin"); },
        R"doc(Elementwise sine of a scalar VectorFunction (argument in radians).

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``sin(fun(x))``.

Examples
--------
>>> x = vf.Arguments(1)[0]
>>> f = vf.sin(x)
>>> f.output_rows()
1
)doc");
    m.def(
        "cos", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "cos"); },
        R"doc(Elementwise cosine of a scalar VectorFunction (argument in radians).

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``cos(fun(x))``.
)doc");
    m.def(
        "tan", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "tan"); },
        R"doc(Elementwise tangent of a scalar VectorFunction (argument in radians).

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``tan(fun(x))``.
)doc");

    m.def(
        "sqrt", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "sqrt"); },
        R"doc(Elementwise square root of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The value must be
    non-negative at all evaluation points.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``sqrt(fun(x))``.
)doc");
    m.def(
        "exp", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "exp"); },
        R"doc(Elementwise natural exponential of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``exp(fun(x))``.
)doc");
    m.def(
        "log", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "log"); },
        R"doc(Elementwise natural logarithm of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The value must be
    strictly positive at all evaluation points.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``ln(fun(x))``.
)doc");
    m.def(
        "squared", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "squared"); },
        R"doc(Elementwise square of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``fun(x)**2``.
)doc");

    m.def(
        "arcsin", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "arcsin"); },
        R"doc(Elementwise arcsine of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The value must lie in
    ``[-1, 1]`` at all evaluation points.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``arcsin(fun(x))``, with result in
    ``[-pi/2, pi/2]``.
)doc");
    m.def(
        "arccos", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "arccos"); },
        R"doc(Elementwise arccosine of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The value must lie in
    ``[-1, 1]`` at all evaluation points.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``arccos(fun(x))``, with result in
    ``[0, pi]``.
)doc");
    m.def(
        "arctan", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "arctan"); },
        R"doc(Elementwise arctangent of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``arctan(fun(x))``, with result in
    ``(-pi/2, pi/2)``.
)doc");

    m.def(
        "sinh", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "sinh"); },
        R"doc(Elementwise hyperbolic sine of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``sinh(fun(x))``.
)doc");
    m.def(
        "cosh", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "cosh"); },
        R"doc(Elementwise hyperbolic cosine of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``cosh(fun(x))``.
)doc");
    m.def(
        "tanh", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "tanh"); },
        R"doc(Elementwise hyperbolic tangent of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``tanh(fun(x))``.
)doc");

    m.def(
        "pow", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "pow"); },
        R"doc(Elementwise real power of a scalar VectorFunction.

Delegates to the ``.pow(exponent)`` method on *fun*. Prefer using the method
form ``fun.pow(p)`` or the ``**`` operator directly, as the free-function form
does not forward the exponent argument.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The base value must be
    positive at all evaluation points for non-integer exponents.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``fun(x)**p``.
)doc");

    m.def(
        "arcsinh", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "arcsinh"); },
        R"doc(Elementwise inverse hyperbolic sine of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``arcsinh(fun(x))``.
)doc");
    m.def(
        "arccosh", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "arccosh"); },
        R"doc(Elementwise inverse hyperbolic cosine of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The value must be
    greater than or equal to 1 at all evaluation points.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``arccosh(fun(x))``.
)doc");
    m.def(
        "arctanh", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "arctanh"); },
        R"doc(Elementwise inverse hyperbolic tangent of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction. The value must lie
    strictly within ``(-1, 1)`` at all evaluation points.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``arctanh(fun(x))``.
)doc");

    m.def(
        "sign", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "sign"); },
        R"doc(Elementwise sign of a scalar VectorFunction.

The sign function returns -1 for negative values, 0 for zero, and +1 for
positive values. Its derivative is zero almost everywhere; Jacobian and
Hessian contributions are not propagated.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``sign(fun(x))`` in ``{-1, 0, 1}``.
)doc");
    m.def(
        "abs", [UnaryOpLam](const T &fun) { return UnaryOpLam(fun, "__abs__"); },
        R"doc(Elementwise absolute value of a scalar VectorFunction.

Parameters
----------
fun : VectorFunction
    Scalar-valued (output dimension 1) VectorFunction.

Returns
-------
VectorFunction
    A new scalar VectorFunction evaluating ``|fun(x)|``.
)doc");
}

void ProductBuild(nb::module_ &m) {

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    using ARGS = Arguments<-1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////Cross
    /// Product///////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto CrossOpLam = [](const auto &f1, const auto &f2) {
        nb::object fun1 = nb::cast(f1);
        nb::object fun2 = nb::cast(f2);
        return fun1.attr("cross")(fun2);
    };

    m.def(
        "cross",
        [CrossOpLam](const SEG3 &f1, const Vector3<double> &f2) { return CrossOpLam(f1, f2); },
        R"doc(3D cross product of two 3-vector quantities.

Each operand may be a VectorFunction or a constant 3-vector array, but at
least one must be a VectorFunction. Both operands must produce 3-element
output vectors. The accepted operand-type combinations are listed in the
overload signatures.

Parameters
----------
a, b : VectorFunction or array_like, shape (3,)
    The two 3-vector operands.

Returns
-------
VectorFunction
    A 3-vector VectorFunction evaluating the cross product ``a(x) x b(x)``.

Examples
--------
>>> v = vf.Arguments(3)
>>> f = vf.cross(v, v)       # zero vector (self cross product)
>>> f.output_rows()
3
)doc");
    m.def("cross",
          [CrossOpLam](const SEG &f1, const Vector3<double> &f2) { return CrossOpLam(f1, f2); });
    m.def("cross",
          [CrossOpLam](const Gen &f1, const Vector3<double> &f2) { return CrossOpLam(f1, f2); });

    m.def("cross", [CrossOpLam](const Vector3<double> &f2, const SEG3 &f1) {
        Vector3<double> f2tmp = -1.0 * f2;
        return CrossOpLam(f1, f2tmp);
    });
    m.def("cross", [CrossOpLam](const Vector3<double> &f2, const SEG &f1) {
        Vector3<double> f2tmp = -1.0 * f2;
        return CrossOpLam(f1, f2tmp);
    });
    m.def("cross", [CrossOpLam](const Vector3<double> &f2, const Gen &f1) {
        Vector3<double> f2tmp = -1.0 * f2;
        return CrossOpLam(f1, f2tmp);
    });

    m.def("cross", [CrossOpLam](const SEG3 &f1, const SEG3 &f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG3 &f1, const SEG &f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG3 &f1, const Gen &f2) { return CrossOpLam(f1, f2); });

    m.def("cross", [CrossOpLam](const SEG &f1, const SEG &f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG &f1, const SEG3 &f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG &f1, const Gen &f2) { return CrossOpLam(f1, f2); });

    m.def("cross", [CrossOpLam](const Gen &f1, const SEG3 &f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const Gen &f1, const SEG &f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const Gen &f1, const Gen &f2) { return CrossOpLam(f1, f2); });

    m.def(
        "doublecross",
        [](const SEG3 &seg1, const SEG3 &seg2, const SEG3 &seg3) {
            auto f1 = FunctionCrossProduct<SEG3, SEG3>(seg1, seg2);
            return Gen(FunctionCrossProduct<decltype(f1), SEG3>(f1, seg3));
        },
        R"doc(Double (iterated) cross product of three 3-vector quantities.

Computes ``(a x b) x c``, where each argument must produce a 3-element output
vector.

Parameters
----------
a, b, c : VectorFunction
    Three 3-vector-valued VectorFunctions.

Returns
-------
VectorFunction
    A 3-vector VectorFunction evaluating ``(a(x) x b(x)) x c(x)``.
)doc");
    m.def("doublecross", [](const Gen &seg1, const Gen &seg2, const Gen &seg3) {
        auto f1 = FunctionCrossProduct<Gen, Gen>(seg1, seg2);
        return Gen(FunctionCrossProduct<decltype(f1), Gen>(f1, seg3));
    });

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////cwiseProduct////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto cwiseProductOpLam = [](const auto &f1, const auto &f2) {
        nb::object fun1 = nb::cast(f1);
        nb::object fun2 = nb::cast(f2);
        return fun1.attr("cwise_product")(fun2);
    };

    m.def(
        "cwise_product",
        [cwiseProductOpLam](const SEG2 &f1, const Vector2<double> &f2) {
            return cwiseProductOpLam(f1, f2);
        },
        R"doc(Elementwise (Hadamard) product of two equal-length vector quantities.

Each operand may be a VectorFunction or a constant array, but at least one
must be a VectorFunction. Both operands must produce the same number of output
elements. The result at each index ``i`` is ``a_i(x) * b_i(x)``.

Parameters
----------
a, b : VectorFunction or array_like
    The two operands, each with the same output dimension.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating the element-wise product ``a(x) * b(x)``
    with the same output dimension as the inputs.
)doc");
    m.def("cwise_product", [cwiseProductOpLam](const SEG3 &f1, const Vector3<double> &f2) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product", [cwiseProductOpLam](const SEG &f1, const Vector3<double> &f2) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product", [cwiseProductOpLam](const Gen &f1, const Vector3<double> &f2) {
        return cwiseProductOpLam(f1, f2);
    });

    m.def("cwise_product", [cwiseProductOpLam](const Vector2<double> &f2, const SEG2 &f1) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product", [cwiseProductOpLam](const Vector3<double> &f2, const SEG3 &f1) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product", [cwiseProductOpLam](const VectorX<double> &f2, const SEG &f1) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product", [cwiseProductOpLam](const VectorX<double> &f2, const Gen &f1) {
        return cwiseProductOpLam(f1, f2);
    });

    m.def("cwise_product", [cwiseProductOpLam](const SEG2 &f1, const SEG2 &f2) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG2 &f1, const SEG &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG2 &f1, const Gen &f2) { return cwiseProductOpLam(f1, f2); });

    m.def("cwise_product", [cwiseProductOpLam](const SEG3 &f1, const SEG3 &f2) {
        return cwiseProductOpLam(f1, f2);
    });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG3 &f1, const SEG &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG3 &f1, const Gen &f2) { return cwiseProductOpLam(f1, f2); });

    m.def("cwise_product",
          [cwiseProductOpLam](const SEG &f1, const SEG &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG &f1, const SEG2 &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG &f1, const SEG3 &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const SEG &f1, const Gen &f2) { return cwiseProductOpLam(f1, f2); });

    m.def("cwise_product",
          [cwiseProductOpLam](const Gen &f1, const SEG2 &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const Gen &f1, const SEG3 &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const Gen &f1, const SEG &f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwise_product",
          [cwiseProductOpLam](const Gen &f1, const Gen &f2) { return cwiseProductOpLam(f1, f2); });

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////cwiseProduct////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto cwiseQuotientOpLam = [](const auto &f1, const auto &f2) {
        nb::object fun1 = nb::cast(f1);
        nb::object fun2 = nb::cast(f2);
        return fun1.attr("cwise_quotient")(fun2);
    };

    m.def(
        "cwise_quotient",
        [cwiseQuotientOpLam](const SEG2 &f1, const Vector2<double> &f2) {
            return cwiseQuotientOpLam(f1, f2);
        },
        R"doc(Elementwise quotient of two equal-length vector quantities.

Each operand may be a VectorFunction or a constant array, but at least one
must be a VectorFunction. Both operands must produce the same number of output
elements. The result at each index ``i`` is ``a_i(x) / b_i(x)``.

Parameters
----------
a, b : VectorFunction or array_like
    Numerator and denominator, each with the same output dimension. The
    denominator must be non-zero at all evaluation points.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating the element-wise quotient ``a(x) / b(x)``
    with the same output dimension as the inputs.
)doc");
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG3 &f1, const Vector3<double> &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG &f1, const Vector3<double> &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const Gen &f1, const Vector3<double> &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });

    m.def("cwise_quotient", [cwiseQuotientOpLam](const Vector2<double> &f1, const SEG2 &f2) {
        Constant<-1, -1> f1tmp(f2.input_rows(), f1);
        return cwiseQuotientOpLam(f1tmp, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const Vector3<double> &f1, const SEG3 &f2) {
        Constant<-1, -1> f1tmp(f2.input_rows(), f1);
        return cwiseQuotientOpLam(f1tmp, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const VectorX<double> &f1, const SEG &f2) {
        Constant<-1, -1> f1tmp(f2.input_rows(), f1);
        return cwiseQuotientOpLam(f1tmp, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const VectorX<double> &f1, const Gen &f2) {
        Constant<-1, -1> f1tmp(f2.input_rows(), f1);
        return cwiseQuotientOpLam(f1tmp, f2);
    });

    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG2 &f1, const SEG2 &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG2 &f1, const SEG &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG2 &f1, const Gen &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });

    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG3 &f1, const SEG3 &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG3 &f1, const SEG &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG3 &f1, const Gen &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });

    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG &f1, const SEG &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG &f1, const SEG2 &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG &f1, const SEG3 &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const SEG &f1, const Gen &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });

    m.def("cwise_quotient", [cwiseQuotientOpLam](const Gen &f1, const SEG2 &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const Gen &f1, const SEG3 &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const Gen &f1, const SEG &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwise_quotient", [cwiseQuotientOpLam](const Gen &f1, const Gen &f2) {
        return cwiseQuotientOpLam(f1, f2);
    });

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////Dot
    /// Product////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto dotOpLam = [](const auto &f1, const auto &f2) {
        nb::object fun1 = nb::cast(f1);
        nb::object fun2 = nb::cast(f2);
        return fun1.attr("dot")(fun2);
    };

    m.def(
        "dot", [dotOpLam](const SEG2 &f1, const Vector2<double> &f2) { return dotOpLam(f1, f2); },
        R"doc(Dot product of two equal-length vector quantities.

Each operand may be a VectorFunction or a constant array, but at least one must
be a VectorFunction, and both must have the same output dimension. The accepted
operand-type combinations are listed in the overload signatures.

Parameters
----------
a, b : VectorFunction or array_like, shape (n,)
    The two operands.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the dot product ``a(x) . b(x)``.

Examples
--------
>>> v = vf.Arguments(3)
>>> f = vf.dot(v, v)          # squared norm of the 3-vector
>>> f.output_rows()
1
)doc");
    m.def("dot",
          [dotOpLam](const SEG3 &f1, const Vector3<double> &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG &f1, const Vector3<double> &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen &f1, const Vector3<double> &f2) { return dotOpLam(f1, f2); });

    m.def("dot",
          [dotOpLam](const Vector2<double> &f2, const SEG2 &f1) { return dotOpLam(f1, f2); });
    m.def("dot",
          [dotOpLam](const Vector3<double> &f2, const SEG3 &f1) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const VectorX<double> &f2, const SEG &f1) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const VectorX<double> &f2, const Gen &f1) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const SEG2 &f1, const SEG2 &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG2 &f1, const SEG &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG2 &f1, const Gen &f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const SEG3 &f1, const SEG3 &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG3 &f1, const SEG &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG3 &f1, const Gen &f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const SEG &f1, const SEG &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG &f1, const SEG2 &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG &f1, const SEG3 &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG &f1, const Gen &f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const Gen &f1, const SEG2 &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen &f1, const SEG3 &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen &f1, const SEG &f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen &f1, const Gen &f2) { return dotOpLam(f1, f2); });
}

struct QuatRotation {

    static auto Definition() {
        auto args = Arguments<7>();
        auto q = args.head<4>();
        auto qinv = StackedOutputs{q.head<3>() * (-1.0), q.coeff<3>()};
        auto vq = args.tail<3>().padded_lower<1>();
        auto qvq = FunctionQuatProduct<decltype(q), decltype(vq)>{q, vq};
        auto expr = FunctionQuatProduct<decltype(qvq), decltype(qinv)>{qvq, qinv}.head<3>();
        return expr;
    }
};

void FreeFunctionsBuild(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::FreeFunctionsBuild(FunctionRegistry &reg, nb::module_ &m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    using ARGS = Arguments<-1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<-1>;

    /////////////////////////////////////////////////////////////////////

    m.def("normalize", [](const VectorX<double> &x) { return x.normalized(); });
    m.def("normalized", [](const VectorX<double> &x) { return x.normalized(); });

    /*
     These methods are already bound member functions for all valid asset vector functions in their
     build functions. Im simply calling them here thru pybind to prevent rebuilding those objects
    */

    UnaryVectorOpBuild<ARGS>(m);
    UnaryVectorOpBuild<SEG>(m);
    UnaryVectorOpBuild<SEG2>(m);
    UnaryVectorOpBuild<SEG3>(m);
    UnaryVectorOpBuild<Gen>(m);

    ProductBuild(m);

    ////////////////////////////////////////////////////////////////////////

    m.def(
        "scalar_root_finder",
        [](const GenS &fx, const GenS &dfx, int iter, double tol) {
            return GenS(ScalarRootFinder<GenS, GenS>{fx, dfx, iter, tol});
        },
        R"doc(Embed a scalar Newton root-finder as a VectorFunction.

Constructs a VectorFunction that, when evaluated, runs Newton iteration on
*fx* to find the root of the scalar residual with respect to the first input
element (the iteration variable). The converged root is returned as a scalar
output; its derivatives with respect to the remaining inputs are computed via
the implicit-function theorem.

The first element of the input vector is treated as the initial guess for the
iteration variable and is updated in place during each evaluation; the remaining
elements are differentiated-through parameters.

Parameters
----------
fx : VectorFunction
    Scalar-valued residual function whose root is sought.
dfx : VectorFunction
    Scalar-valued derivative of *fx* with respect to the iteration variable
    (first input element). When provided, this overload skips the automatic
    Jacobian evaluation inside the Newton loop.
iter : int
    Maximum number of Newton iterations.
tol : float
    Convergence tolerance on the absolute residual value.

Returns
-------
VectorFunction
    Scalar VectorFunction returning the converged root of *fx*.
)doc");
    m.def("scalar_root_finder", [](const GenS &fx, int iter, double tol) {
        return GenS(ScalarRootFinder<GenS, std::false_type>{fx, std::false_type(), iter, tol});
    });
    ////////////////////////////////////////////////////////////////////////

    m.def(
        "quat_product",
        [](const SEG &seg1, const SEG &seg2) {
            return Gen(FunctionQuatProduct<SEG, SEG>(seg1, seg2));
        },
        R"doc(Quaternion product of two 4-vector VectorFunctions.

Computes the Hamilton product ``q1 ⊗ q2`` where each operand is a
4-element quaternion in ``[x, y, z, w]`` storage order (vector part first,
scalar part last).

Parameters
----------
q1 : VectorFunction
    Left quaternion operand, output dimension 4.
q2 : VectorFunction
    Right quaternion operand, output dimension 4.

Returns
-------
VectorFunction
    A 4-vector VectorFunction evaluating the quaternion product ``q1(x) ⊗ q2(x)``.
)doc");
    m.def("quat_product", [](const SEG &seg1, const Gen &seg2) {
        return Gen(FunctionQuatProduct<SEG, Gen>(seg1, seg2));
    });
    m.def("quat_product", [](const Gen &seg1, const SEG &seg2) {
        return Gen(FunctionQuatProduct<Gen, SEG>(seg1, seg2));
    });
    m.def("quat_product", [](const Gen &seg1, const Gen &seg2) {
        return Gen(FunctionQuatProduct<Gen, Gen>(seg1, seg2));
    });

    m.def(
        "quat_rotate",
        [](const Gen &q, const Gen &v) {
            return Gen(QuatRotation::Definition().eval(stack(q, v)));
        },
        R"doc(Rotate a 3-vector by a unit quaternion.

Computes the active rotation ``q ⊗ [v; 0] ⊗ q*``, returning only the vector
part. The quaternion *q* must be stored in ``[x, y, z, w]`` order (vector part
first, scalar part last) and should be a unit quaternion at all evaluation
points.

Parameters
----------
q : VectorFunction
    Unit quaternion operand, output dimension 4, in ``[x, y, z, w]`` order.
v : VectorFunction or array_like, shape (3,)
    The 3-vector to rotate.

Returns
-------
VectorFunction
    A 3-vector VectorFunction evaluating the rotated vector ``q(x) * v(x)``.
)doc");
    m.def("quat_rotate", [](const Gen &q, const SEG3 &v) {
        return Gen(QuatRotation::Definition().eval(stack(q, v)));
    });
    m.def("quat_rotate", [](const Gen &q, const SEG &v) {
        return Gen(QuatRotation::Definition().eval(stack(q, v)));
    });
    m.def("quat_rotate", [](const Gen &q, const Vector3<double> &vec) {
        return Gen(QuatRotation::Definition().eval(stack(q, Constant<-1, 3>(q.input_rows(), vec))));
    });

    /////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////

    UnaryScalarOpBuild<ELEM>(m);
    UnaryScalarOpBuild<GenS>(m);

    bind::IfElseBuild(m);

    m.def(
        "arctan2",
        [](const GenS &y, const GenS &x) {
            return GenS(ArcTan2Op().eval(StackedOutputs{y, x}));
        },
        R"doc(Quadrant-aware arctangent of two scalar VectorFunctions.

Computes ``atan2(y(x), x(x))`` with analytic first and second derivatives.
The result lies in ``(-pi, pi]`` and correctly identifies the quadrant based
on the signs of both operands.

Parameters
----------
y : VectorFunction
    Scalar-valued VectorFunction for the opposite (sine-like) component.
x : VectorFunction
    Scalar-valued VectorFunction for the adjacent (cosine-like) component.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``atan2(y(x), x(x))``.
)doc");
    m.def("arctan2", [](const ELEM &y, const ELEM &x) {
        return GenS(ArcTan2Op().eval(StackedOutputs{y, x}));
    });
    m.def("arctan2", [](const ELEM &y, const GenS &x) {
        return GenS(ArcTan2Op().eval(StackedOutputs{y, x}));
    });
    m.def("arctan2", [](const GenS &y, const ELEM &x) {
        return GenS(ArcTan2Op().eval(StackedOutputs{y, x}));
    });

    m.def(
        "divtest",
        [](const Gen &a, const GenS &b) {
            return Gen(VectorScalarFunctionDivision<Gen, GenS>{a, b});
        },
        R"doc(Divide a vector VectorFunction by a scalar VectorFunction (internal testing utility).

Divides each element of *a* by the scalar *b*. This function exists primarily
to exercise the ``VectorScalarFunctionDivision`` implementation and is not
intended for production use.

Parameters
----------
a : VectorFunction
    Vector-valued VectorFunction (numerator).
b : VectorFunction
    Scalar-valued VectorFunction (denominator). Must be non-zero at all
    evaluation points.

Returns
-------
VectorFunction
    Vector VectorFunction evaluating ``a(x) / b(x)`` element-wise.
)doc");
    m.def("divtest", [](const Gen &a, const ELEM &b) {
        return Gen(VectorScalarFunctionDivision<Gen, ELEM>{a, b});
    });

    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
}