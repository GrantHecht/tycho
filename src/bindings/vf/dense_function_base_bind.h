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

// Free function templates implementing DenseFunctionBase binding helpers.
// These replace the out-of-class member function definitions that previously
// lived in DenseFunctionBase.h under TYCHO_PYTHON_BINDINGS.

namespace tycho::bind {

using namespace tycho::vf;

template <class Derived, class PYClass> void DenseBaseBuild(PYClass &obj) {

    using Gen = GenericFunction<-1, -1>;

    obj.def("input_rows", &Derived::input_rows,
            R"doc(Number of inputs this VectorFunction expects.

Returns
-------
int
    Length of the input vector ``x`` passed to :meth:`compute`, :meth:`jacobian`,
    and the adjoint methods.
)doc");
    obj.def("output_rows", &Derived::output_rows,
            R"doc(Number of scalar outputs this VectorFunction produces.

Returns
-------
int
    Length of the output vector ``f(x)`` returned by :meth:`compute`.
)doc");
    obj.def("name", &Derived::name,
            R"doc(Human-readable name of this VectorFunction.

The name is derived from the C++ type name of the underlying expression.
It is primarily used for diagnostics and display.

Returns
-------
str
    Type-based name string.
)doc");

    obj.def("input_domain", &Derived::input_domain,
            R"doc(Return the sparsity domain describing which input indices this function uses.

The input domain encodes which contiguous sub-ranges of the input vector
this function actually depends on.  Dynamic-size functions compute the domain
at construction time; static-size functions return the full range ``[0, IR)``.

Returns
-------
ndarray, shape (2, k), dtype int
    Column matrix with ``k`` sub-ranges.  Row 0 contains start indices,
    row 1 contains the corresponding lengths.
)doc");
    obj.def("is_linear", &Derived::is_linear,
            R"doc(Whether this VectorFunction is known to be linear at compile time.

A linear function has a constant Jacobian and zero Hessian.  PSIOPT uses
this flag to skip second-derivative computations for linear expressions.

Returns
-------
bool
    ``True`` if the function is linear, ``False`` otherwise.
)doc");

    // Generic lambda bodies — defined once, called from both the zero-copy
    // (ConstEigenRef/numpy) and the sequence-fallback (VectorXd) overloads.
    auto compute_body = [](const Derived &func, const auto &x) -> Eigen::VectorXd {
        if ((int)x.size() != func.input_rows())
            throw std::invalid_argument("Incorrectly sized input to function");
        Eigen::VectorXd fx(func.output_rows());
        fx.setZero();
        func.derived().compute(x, fx);
        return fx;
    };
    auto jacobian_body = [](const Derived &func, const auto &x) -> Eigen::MatrixXd {
        if ((int)x.size() != func.input_rows())
            throw std::invalid_argument("Incorrectly sized input to function");
        Eigen::MatrixXd jx(func.output_rows(), func.input_rows());
        jx.setZero();
        func.derived().jacobian(x, jx);
        return jx;
    };
    auto compute_jacobian_body = [](const Derived &func,
                                    const auto &x) -> std::tuple<Eigen::VectorXd, Eigen::MatrixXd> {
        if ((int)x.size() != func.input_rows())
            throw std::invalid_argument("Incorrectly sized input to function");
        Eigen::VectorXd fx(func.output_rows());
        fx.setZero();
        Eigen::MatrixXd jx(func.output_rows(), func.input_rows());
        jx.setZero();
        func.derived().compute_jacobian(x, fx, jx);
        return std::tuple{fx, jx};
    };
    auto adjointgradient_body = [](const Derived &func, const auto &x,
                                   const auto &lm) -> Eigen::VectorXd {
        if ((int)x.size() != func.input_rows())
            throw std::invalid_argument("Incorrectly sized input to function");
        if ((int)lm.size() != func.output_rows())
            throw std::invalid_argument("Incorrectly sized multiplier input to function");
        Eigen::VectorXd ax(func.input_rows());
        ax.setZero();
        func.derived().adjointgradient(x, ax, lm);
        return ax;
    };
    auto adjointhessian_body = [](const Derived &func, const auto &x,
                                  const auto &lm) -> Eigen::MatrixXd {
        if ((int)x.size() != func.input_rows())
            throw std::invalid_argument("Incorrectly sized input to function");
        if ((int)lm.size() != func.output_rows())
            throw std::invalid_argument("Incorrectly sized multiplier input to function");
        Eigen::MatrixXd hx(func.input_rows(), func.input_rows());
        hx.setZero();
        func.derived().adjointhessian(x, hx, lm);
        return hx;
    };
    auto computeall_body = [](const Derived &func, const auto &x, const auto &lm)
        -> std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> {
        if ((int)x.size() != func.input_rows())
            throw std::invalid_argument("Incorrectly sized input to function");
        if ((int)lm.size() != func.output_rows())
            throw std::invalid_argument("Incorrectly sized multiplier input to function");
        Eigen::VectorXd fx(func.output_rows());
        Eigen::MatrixXd jx(func.output_rows(), func.input_rows());
        Eigen::VectorXd gx(func.input_rows());
        Eigen::MatrixXd hx(func.input_rows(), func.input_rows());
        fx.setZero();
        jx.setZero();
        gx.setZero();
        hx.setZero();
        func.derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
        return std::tuple{fx, jx, gx, hx};
    };

    // Two overloads per method are required:
    //   1. ConstEigenRef<VectorXd> — zero-copy reference into a numpy array buffer.
    //      Uses nanobind's built-in Eigen caster; does NOT invoke our VectorXd type_caster.
    //   2. VectorXd (by value)   — invokes our custom type_caster which accepts Python
    //      lists, tuples, and other sequences via its from_python fallback path.
    // Without overload 1, every numpy call copies. Without overload 2, lists/tuples fail.
    obj.def("compute",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) { return compute_body(f, x); },
            R"doc(Evaluate the function at a point.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector.  Accepts a NumPy array (zero-copy) or any Python
    sequence (list, tuple) that can be converted to a 1-D float64 vector.

Returns
-------
ndarray, shape (output_rows,)
    Output vector ``f(x)``.

Raises
------
ValueError
    If ``x`` has the wrong length.
)doc");
    obj.def("compute", [=](const Derived &f, Eigen::VectorXd x) { return compute_body(f, x); });

    obj.def("__call__",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) { return compute_body(f, x); },
            R"doc(Evaluate the function at a point (``f(x)``).

Equivalent to :meth:`compute`.  Accepts a numeric vector or a VectorFunction
argument for functional composition — see :meth:`eval` for that overload.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector.  Accepts a NumPy array (zero-copy) or any Python sequence.

Returns
-------
ndarray, shape (output_rows,)
    Output vector ``f(x)``.
)doc");
    obj.def("__call__", [=](const Derived &f, Eigen::VectorXd x) { return compute_body(f, x); });

    obj.def("jacobian",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) {
                return jacobian_body(f, x);
            },
            R"doc(Evaluate the Jacobian of the function at a point.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector at which to evaluate the Jacobian.

Returns
-------
ndarray, shape (output_rows, input_rows)
    Jacobian matrix ``J(x)`` where entry ``(i, j)`` is
    ``∂f_i/∂x_j`` evaluated at ``x``.

Raises
------
ValueError
    If ``x`` has the wrong length.
)doc");
    obj.def("jacobian", [=](const Derived &f, Eigen::VectorXd x) { return jacobian_body(f, x); });

    obj.def("compute_jacobian",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x) {
                return compute_jacobian_body(f, x);
            },
            R"doc(Evaluate the function and its Jacobian simultaneously.

Computing both at once is cheaper than two separate calls because internal
temporary buffers are shared.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector.

Returns
-------
fx : ndarray, shape (output_rows,)
    Function value ``f(x)``.
jx : ndarray, shape (output_rows, input_rows)
    Jacobian ``J(x)``.

Raises
------
ValueError
    If ``x`` has the wrong length.
)doc");
    obj.def("compute_jacobian",
            [=](const Derived &f, Eigen::VectorXd x) { return compute_jacobian_body(f, x); });

    obj.def("adjointgradient",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x,
                ConstEigenRef<Eigen::VectorXd> lm) { return adjointgradient_body(f, x, lm); },
            R"doc(Compute the adjoint (Lagrange-multiplier-weighted) gradient.

Evaluates ``g = Jᵀ λ`` where ``J`` is the Jacobian of ``self`` at ``x`` and
``λ`` (``lm``) is the adjoint (multiplier) vector.  This is the first-order
sensitivity of the Lagrangian with respect to the inputs.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector at which to evaluate.
lm : array_like, shape (output_rows,)
    Adjoint (Lagrange multiplier) vector, one entry per output.

Returns
-------
ndarray, shape (input_rows,)
    Adjoint gradient vector ``Jᵀ λ``.

Raises
------
ValueError
    If ``x`` or ``lm`` has the wrong length.
)doc");
    obj.def("adjointgradient", [=](const Derived &f, Eigen::VectorXd x, Eigen::VectorXd lm) {
        return adjointgradient_body(f, x, lm);
    });

    obj.def("adjointhessian",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x,
                ConstEigenRef<Eigen::VectorXd> lm) { return adjointhessian_body(f, x, lm); },
            R"doc(Compute the adjoint (Lagrange-multiplier-weighted) Hessian.

Evaluates the symmetric matrix ``H = Σ_k λ_k ∇²f_k(x)``, i.e. the
multiplier-weighted sum of the per-output Hessians.  This is the
second-order term of the Lagrangian's Hessian with respect to the inputs.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector at which to evaluate.
lm : array_like, shape (output_rows,)
    Adjoint (Lagrange multiplier) vector, one entry per output.

Returns
-------
ndarray, shape (input_rows, input_rows)
    Adjoint Hessian matrix (symmetric).

Raises
------
ValueError
    If ``x`` or ``lm`` has the wrong length.
)doc");
    obj.def("adjointhessian", [=](const Derived &f, Eigen::VectorXd x, Eigen::VectorXd lm) {
        return adjointhessian_body(f, x, lm);
    });

    obj.def("computeall",
            [=](const Derived &f, ConstEigenRef<Eigen::VectorXd> x,
                ConstEigenRef<Eigen::VectorXd> lm) { return computeall_body(f, x, lm); },
            R"doc(Evaluate the function value, Jacobian, adjoint gradient, and adjoint Hessian in one call.

All four quantities are computed together, sharing internal buffers, which is
more efficient than invoking them separately.  This is the entry point used
by PSIOPT during second-order optimization.

Parameters
----------
x : array_like, shape (input_rows,)
    Input vector at which to evaluate.
lm : array_like, shape (output_rows,)
    Adjoint (Lagrange multiplier) vector used to form the weighted gradient
    and Hessian.

Returns
-------
fx : ndarray, shape (output_rows,)
    Function value ``f(x)``.
jx : ndarray, shape (output_rows, input_rows)
    Jacobian ``J(x)``.
gx : ndarray, shape (input_rows,)
    Adjoint gradient ``Jᵀ λ``.
hx : ndarray, shape (input_rows, input_rows)
    Adjoint Hessian ``Σ_k λ_k ∇²f_k(x)``.

Raises
------
ValueError
    If ``x`` or ``lm`` has the wrong length.
)doc");
    obj.def("computeall", [=](const Derived &f, Eigen::VectorXd x, Eigen::VectorXd lm) {
        return computeall_body(f, x, lm);
    });

    obj.def("vf", &Derived::template make_generic<GenericFunction<-1, -1>>,
            R"doc(Type-erase this expression into a generic VectorFunction.

Wraps the current expression in a fully dynamic ``GenericFunction`` that can
be stored, passed to functions expecting a ``GenericFunction``, or used as an
operand in mixed-type expressions without exposing the underlying expression
template type.

Returns
-------
GenericFunction
    A dynamically-typed wrapper around this function.
)doc");

    constexpr int OR = Derived::ORC;
    if constexpr (OR == 1) {
        obj.def("sf", &Derived::template make_generic<GenericFunction<-1, 1>>,
                R"doc(Type-erase this scalar-output expression into a scalar GenericFunction.

Available only on VectorFunctions with a single output row.  Equivalent to
:meth:`vf` but the returned wrapper carries the compile-time knowledge that
the output dimension is 1, which enables scalar-specific operations.

Returns
-------
ScalarFunction
    A dynamically-typed scalar-output wrapper around this function.
)doc");
    }
}

template <class Derived, class PYClass> void DoubleMathBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    // Prevents numpy from overriding __radd__ and __rmul__
    obj.attr("__array_ufunc__") = nb::none();

    obj.def(
        "__add__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(a + b); },
        nb::is_operator(),
        R"doc(Add a constant vector to this VectorFunction (``self + b``).

Parameters
----------
b : array_like, shape (output_rows,)
    Constant offset to add element-wise to the output of ``self``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) + b``.
)doc");
    obj.def(
        "__radd__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(a + b); },
        nb::is_operator());
    obj.def(
        "__sub__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(a - b); },
        nb::is_operator(),
        R"doc(Subtract a constant vector from this VectorFunction (``self - b``).

Parameters
----------
b : array_like, shape (output_rows,)
    Constant to subtract element-wise from the output of ``self``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) - b``.
)doc");
    obj.def(
        "__rsub__", [](const Derived &a, Eigen::VectorXd b) { return BinGen(b - a); },
        nb::is_operator());

    obj.def("__mul__", [](const Derived &a, double b) { return BinGen(a * b); }, nb::is_operator(),
            R"doc(Scale this VectorFunction by a scalar constant (``self * b``).

Parameters
----------
b : float
    Scalar multiplier applied to every output component.

Returns
-------
VectorFunction
    Expression evaluating ``b * self(x)``.
)doc");
    obj.def(
        "__rmul__", [](const Derived &a, double b) { return BinGen(a * b); }, nb::is_operator());

    obj.def("__neg__", [](const Derived &a) { return BinGen(a * (-1.0)); }, nb::is_operator(),
            R"doc(Negate this VectorFunction (``-self``).

Returns
-------
VectorFunction
    Expression evaluating ``-self(x)``.
)doc");

    obj.def(
        "__truediv__", [](const Derived &a, double b) { return BinGen(a * (1.0 / b)); },
        nb::is_operator(),
        R"doc(Divide this VectorFunction by a scalar constant (``self / b``).

A separate overload accepts a scalar VectorFunction divisor.

Parameters
----------
b : float
    Constant divisor; every output component is divided by ``b``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) / b``.
)doc");
    obj.def(
        "__truediv__", [](const Derived &a, const Segment<-1, 1, -1> &b) { return BinGen(a / b); },
        nb::is_operator());

    if constexpr (OR == 1) { // Scalars
        obj.def(
            "__add__", [](const Derived &a, double b) { return BinGen(a + b); }, nb::is_operator());
        obj.def(
            "__radd__", [](const Derived &a, double b) { return BinGen(a + b); },
            nb::is_operator());
        obj.def(
            "__sub__", [](const Derived &a, double b) { return BinGen(a - b); }, nb::is_operator());
        obj.def(
            "__rsub__", [](const Derived &a, double b) { return BinGen(b - a); },
            nb::is_operator());
        obj.def(
            "__rtruediv__", [](const Derived &a, double b) { return BinGen(b / a); },
            nb::is_operator());

        obj.def(
            "__rmul__",
            [](const Derived &a, const Eigen::VectorXd &b) {
                return Gen(MatrixScaled<Derived, -1>(a, b));
            },
            nb::is_operator());

        obj.def(
            "__mul__",
            [](const Derived &a, const Eigen::VectorXd &b) {
                return Gen(MatrixScaled<Derived, -1>(a, b));
            },
            nb::is_operator());
    }
}

template <class Derived, class PYClass> void UnaryMathBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    if constexpr (OR != 1) {

        if constexpr (!std::is_same<PYClass, nb::module_>::value) {
            obj.def("sum", [](const Derived &a) { return GenS(a.sum()); },
                    R"doc(Sum all output components into a scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``Σ_i self(x)[i]``.
)doc");
        }

        obj.def("normalized_power3",
                [](const Derived &a, Eigen::VectorXd b) {
                    return Gen((a + b).template normalized_power<3>());
                },
                R"doc(Shifted and normalized-power-3 of this VectorFunction.

Computes ``(self(x) + b) / ||self(x) + b||³``.  This operation appears
frequently in gravitational and electrostatic force expressions.

Parameters
----------
b : array_like, shape (output_rows,)
    Constant offset added to the output of ``self`` before normalizing.

Returns
-------
VectorFunction
    Expression evaluating ``(self(x) + b) / ||self(x) + b||³``.
)doc");
        obj.def("normalized_power3", [](const Derived &a, Eigen::VectorXd b, double s) {
            return Gen(((a + b).template normalized_power<3>()) * s);
        });
        ////////////////////////////////////////////////////////////////////
        if constexpr (OR > 0) { // already constant size

            obj.def("norm", [](const Derived &a) { return GenS(a.norm()); },
                    R"doc(Euclidean norm of the output vector: ``||self(x)||``.

Returns
-------
ScalarFunction
    Expression evaluating ``||self(x)||₂``.
)doc");
            obj.def("squared_norm", [](const Derived &a) { return GenS(a.squared_norm()); },
                    R"doc(Squared Euclidean norm of the output vector: ``||self(x)||²``.

Returns
-------
ScalarFunction
    Expression evaluating ``||self(x)||₂²``.
)doc");
            obj.def("cubed_norm",
                    [](const Derived &a) { return GenS(a.template norm_power<3>()); },
                    R"doc(Euclidean norm of the output raised to the third power: ``||self(x)||³``.

Returns
-------
ScalarFunction
    Expression evaluating ``||self(x)||₂³``.
)doc");

            obj.def("inverse_norm", [](const Derived &a) { return GenS(a.inverse_norm()); },
                    R"doc(Reciprocal Euclidean norm of the output: ``1 / ||self(x)||``.

Returns
-------
ScalarFunction
    Expression evaluating ``1 / ||self(x)||₂``.
)doc");
            obj.def("inverse_squared_norm",
                    [](const Derived &a) { return GenS(a.inverse_squared_norm()); },
                    R"doc(Reciprocal squared Euclidean norm: ``1 / ||self(x)||²``.

Returns
-------
ScalarFunction
    Expression evaluating ``1 / ||self(x)||₂²``.
)doc");
            obj.def("inverse_cubed_norm",
                    [](const Derived &a) { return GenS(a.template inverse_norm_power<3>()); },
                    R"doc(Reciprocal cubed Euclidean norm: ``1 / ||self(x)||³``.

Returns
-------
ScalarFunction
    Expression evaluating ``1 / ||self(x)||₂³``.
)doc");
            obj.def("inverse_four_norm",
                    [](const Derived &a) { return GenS(a.template inverse_norm_power<4>()); },
                    R"doc(Reciprocal fourth-power Euclidean norm: ``1 / ||self(x)||⁴``.

Returns
-------
ScalarFunction
    Expression evaluating ``1 / ||self(x)||₂⁴``.
)doc");

            obj.def("normalized", [](const Derived &a) { return Gen(a.normalized()); },
                    R"doc(Normalize the output to unit Euclidean length: ``self(x) / ||self(x)||``.

Returns
-------
VectorFunction
    Expression evaluating the unit vector ``self(x) / ||self(x)||₂``.
)doc");

            obj.def("normalized_power2",
                    [](const Derived &a) { return Gen(a.template normalized_power<2>()); },
                    R"doc(Output divided by the squared Euclidean norm: ``self(x) / ||self(x)||²``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) / ||self(x)||₂²``.
)doc");

            obj.def("normalized_power4",
                    [](const Derived &a) { return Gen(a.template normalized_power<4>()); },
                    R"doc(Output divided by the fourth power of its Euclidean norm: ``self(x) / ||self(x)||⁴``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) / ||self(x)||₂⁴``.
)doc");
            obj.def("normalized_power5",
                    [](const Derived &a) { return Gen(a.template normalized_power<5>()); },
                    R"doc(Output divided by the fifth power of its Euclidean norm: ``self(x) / ||self(x)||⁵``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) / ||self(x)||₂⁵``.
)doc");

            obj.def("normalized_power3",
                    [](const Derived &a) { return Gen(a.template normalized_power<3>()); });

        } else {
            /// Try to fit these to constant size 2,3 if possible

            auto SizeSwitch = [](const auto &fun, auto Lam) {
                int orr = fun.output_rows();
                // if (orr == 2)     return Lam(fun, std::integral_constant<int, 2>());
                // else
                if (orr == 3)
                    return Lam(fun, std::integral_constant<int, 3>());
                return Lam(fun, std::integral_constant<int, -1>());
            };

            obj.def("norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(Norm<size.value>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("squared_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(SquaredNorm<size.value>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("cubed_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(NormPower<size.value, 3>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseNorm<size.value>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_squared_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseSquaredNorm<size.value>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_cubed_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseNormPower<size.value, 3>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("inverse_four_norm", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return GenS(InverseNormPower<size.value, 4>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(Normalized<size.value>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power2", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 2>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power3", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 3>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power4", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 4>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
            obj.def("normalized_power5", [SizeSwitch](const Derived &fun) {
                auto Lam = [](const Derived &funt, auto size) {
                    return Gen(NormalizedPower<size.value, 5>(funt.output_rows()).eval(funt));
                };
                return SizeSwitch(fun, Lam);
            });
        }
    }

    if constexpr (OR == 1) {

        obj.def("sin", [](const Derived &a) { return GenS(a.sin()); },
                R"doc(Sine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``sin(self(x))``.
)doc");
        obj.def("cos", [](const Derived &a) { return GenS(a.cos()); },
                R"doc(Cosine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``cos(self(x))``.
)doc");
        obj.def("tan", [](const Derived &a) { return GenS(a.tan()); },
                R"doc(Tangent of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``tan(self(x))``.
)doc");
        obj.def("sqrt", [](const Derived &a) { return GenS(a.sqrt()); },
                R"doc(Square root of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``sqrt(self(x))``.
)doc");
        obj.def("exp", [](const Derived &a) { return GenS(a.exp()); },
                R"doc(Natural exponential of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``exp(self(x))``.
)doc");
        obj.def("log", [](const Derived &e) { return GenS(CwiseLog<Derived>(e)); },
                R"doc(Natural logarithm of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``log(self(x))``.
)doc");
        obj.def("squared", [](const Derived &a) { return GenS(a.square()); },
                R"doc(Square of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``self(x)²``.
)doc");
        obj.def("arcsin", [](const Derived &e) { return GenS(CwiseArcSin<Derived>(e)); },
                R"doc(Inverse sine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``arcsin(self(x))``, in radians, range ``[-π/2, π/2]``.
)doc");
        obj.def("arccos", [](const Derived &e) { return GenS(CwiseArcCos<Derived>(e)); },
                R"doc(Inverse cosine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``arccos(self(x))``, in radians, range ``[0, π]``.
)doc");
        obj.def("arctan", [](const Derived &e) { return GenS(CwiseArcTan<Derived>(e)); },
                R"doc(Inverse tangent of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``arctan(self(x))``, in radians, range ``(-π/2, π/2)``.
)doc");

        obj.def("sinh", [](const Derived &e) { return GenS(CwiseSinH<Derived>(e)); },
                R"doc(Hyperbolic sine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``sinh(self(x))``.
)doc");
        obj.def("cosh", [](const Derived &e) { return GenS(CwiseCosH<Derived>(e)); },
                R"doc(Hyperbolic cosine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``cosh(self(x))``.
)doc");
        obj.def("tanh", [](const Derived &e) { return GenS(CwiseTanH<Derived>(e)); },
                R"doc(Hyperbolic tangent of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``tanh(self(x))``.
)doc");

        obj.def("arcsinh", [](const Derived &e) { return GenS(CwiseArcSinH<Derived>(e)); },
                R"doc(Inverse hyperbolic sine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``arcsinh(self(x))``.
)doc");
        obj.def("arccosh", [](const Derived &e) { return GenS(CwiseArcCosH<Derived>(e)); },
                R"doc(Inverse hyperbolic cosine of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``arccosh(self(x))``, defined for input ≥ 1.
)doc");
        obj.def("arctanh", [](const Derived &e) { return GenS(CwiseArcTanH<Derived>(e)); },
                R"doc(Inverse hyperbolic tangent of this scalar VectorFunction.

Returns
-------
ScalarFunction
    Expression evaluating ``arctanh(self(x))``, defined for input in ``(-1, 1)``.
)doc");

        obj.def("__abs__", [](const Derived &e) { return GenS(CwiseAbs<Derived>(e)); },
                R"doc(Absolute value of this scalar VectorFunction (``abs(self)``).

Returns
-------
ScalarFunction
    Expression evaluating ``|self(x)|``.
)doc");
        obj.def("sign", [](const Derived &e) { return GenS(SignFunction<Derived>(e)); },
                R"doc(Sign of this scalar VectorFunction: ``-1``, ``0``, or ``+1``.

Returns
-------
ScalarFunction
    Expression evaluating ``sign(self(x))``.
)doc");

        obj.def("pow",
                [](const Derived &e, double power) { return GenS(CwisePow<Derived>(e, power)); },
                R"doc(Raise this scalar VectorFunction to a real power.

Parameters
----------
power : float
    Exponent to raise ``self(x)`` to.

Returns
-------
ScalarFunction
    Expression evaluating ``self(x) ** power``.
)doc");

        if constexpr (!std::is_same<PYClass, nb::module_>::value) {
            obj.def(
                "__pow__", [](const Derived &a, double b) { return GenS(CwisePow<Derived>(a, b)); },
                nb::is_operator());
            obj.def(
                "__pow__",
                [](const Derived &a, int b) {
                    if (b == 1)
                        return GenS(a);
                    else if (b == 2)
                        return GenS(a.square());
                    return GenS(CwisePow<Derived>(a, b));
                },
                nb::is_operator());
        }
    }
}

template <class Derived, class PYClass> void FunctionIndexingBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    obj.def("padded_lower", [](const Derived &a, int lpad) { return Gen(a.padded_lower(lpad)); },
            R"doc(Append zero rows below the output of this VectorFunction.

Parameters
----------
lpad : int
    Number of zero rows to append after the last output component.

Returns
-------
VectorFunction
    Expression whose output is ``[self(x); 0, …, 0]`` with ``lpad`` trailing zeros.
)doc");
    obj.def("padded_upper", [](const Derived &a, int upad) { return Gen(a.padded_upper(upad)); },
            R"doc(Prepend zero rows above the output of this VectorFunction.

Parameters
----------
upad : int
    Number of zero rows to prepend before the first output component.

Returns
-------
VectorFunction
    Expression whose output is ``[0, …, 0; self(x)]`` with ``upad`` leading zeros.
)doc");
    obj.def("padded",
            [](const Derived &a, int upad, int lpad) { return Gen(a.padded(upad, lpad)); },
            R"doc(Pad the output of this VectorFunction with zeros above and below.

Parameters
----------
upad : int
    Number of zero rows prepended before the first output component.
lpad : int
    Number of zero rows appended after the last output component.

Returns
-------
VectorFunction
    Expression whose output is ``[0…; self(x); 0…]``.
)doc");

    constexpr bool is_seg = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

    using SegRetType = typename std::conditional<is_seg, SEG, GenericFunction<-1, -1>>::type;
    using ElemRetType = typename std::conditional<is_seg, ELEM, GenericFunction<-1, 1>>::type;

    obj.def("segment",
            [](const Derived &a, int start, int size) {
                return SegRetType(a.segment(start, size));
            },
            R"doc(Extract a contiguous sub-vector of the output.

Parameters
----------
start : int
    Zero-based index of the first output element to include.
size : int
    Number of consecutive output elements to extract.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[start : start + size]``.
)doc");
    obj.def("head", [](const Derived &a, int size) { return SegRetType(a.head(size)); },
            R"doc(Extract the first ``size`` output components.

Parameters
----------
size : int
    Number of leading output elements to keep.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[:size]``.
)doc");
    obj.def("tail", [](const Derived &a, int size) { return SegRetType(a.tail(size)); },
            R"doc(Extract the last ``size`` output components.

Parameters
----------
size : int
    Number of trailing output elements to keep.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[-size:]``.
)doc");

    obj.def("coeff", [](const Derived &a, int elem) { return ElemRetType(a.coeff(elem)); },
            R"doc(Extract a single scalar output component by index.

Parameters
----------
elem : int
    Zero-based index of the output element to select.

Returns
-------
ScalarFunction
    Expression whose output is the scalar ``self(x)[elem]``.
)doc");
    obj.def(
        "__getitem__", [](const Derived &a, int elem) { return ElemRetType(a.coeff(elem)); },
        nb::is_operator(),
        R"doc(Extract a single output element or a contiguous slice (``self[i]`` or ``self[i:j]``).

Parameters
----------
index : int or slice
    Integer index selects a single scalar output.  A ``slice`` selects a
    contiguous sub-vector (step must be 1 and the slice must be non-empty
    and forward-only).

Returns
-------
VectorFunction
    Scalar expression for integer index; sub-vector expression for a slice.

Raises
------
ValueError
    If the index is out of range, the slice is empty, backward, or non-contiguous.
)doc");
    obj.def(
        "__getitem__",
        [](const Derived &a, const nb::slice &slice) {
            auto [start, stop, step, slicelength] = slice.compute(a.output_rows());

            if (step != 1) {
                throw std::invalid_argument("Non continous slices not supported");
            }
            if (start >= a.output_rows()) {
                throw std::invalid_argument("Segment index out of bounds.");
            }
            if (start > stop) {
                throw std::invalid_argument("Backward indexing not supported.");
            }
            if (slicelength == 0) {
                throw std::invalid_argument("Slice length must be greater than 0.");
            }

            int start_ = start;
            int size_ = stop - start;

            return SegRetType(a.segment(start_, size_));
        },
        nb::is_operator());

    // if constexpr (OR != 1) {
    //     obj.def("colmatrix", [](const Derived& a, int rows, int cols) {
    //         Gen agen(a);
    //         return agen.colmatrix(rows, cols);
    //         });
    //     obj.def("rowmatrix", [](const Derived& a, int rows, int cols) {
    //         Gen agen(a);
    //         return agen.rowmatrix(rows, cols);
    //         });
    // }

    if constexpr (OR < 0 || OR > 2) {
        using Seg2RetType = typename std::conditional<is_seg, SEG2, GenericFunction<-1, -1>>::type;

        auto seg2 = [](const Derived &a, int start) {
            return Seg2RetType(a.template segment<2>(start));
        };
        auto head2 = [](const Derived &a) { return Seg2RetType(a.template segment<2>(0)); };
        auto tail2 = [](const Derived &a) {
            return Seg2RetType(a.template segment<2>(a.output_rows() - 2));
        };

        obj.def("segment_2", seg2,
                R"doc(Extract a 2-element sub-vector of the output starting at a given index.

Parameters
----------
start : int
    Zero-based index of the first output element to include.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[start : start + 2]``.
)doc");
        obj.def("head_2", head2,
                R"doc(Extract the first 2 output components.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[:2]``.
)doc");
        obj.def("tail_2", tail2,
                R"doc(Extract the last 2 output components.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[-2:]``.
)doc");
        obj.def("segment2", seg2);
        obj.def("head2", head2);
        obj.def("tail2", tail2);
    }
    if constexpr (OR < 0 || OR > 3) {
        using Seg3RetType = typename std::conditional<is_seg, SEG3, GenericFunction<-1, -1>>::type;
        auto seg3 = [](const Derived &a, int start) {
            return Seg3RetType(a.template segment<3>(start));
        };
        auto head3 = [](const Derived &a) { return Seg3RetType(a.template segment<3>(0)); };
        auto tail3 = [](const Derived &a) {
            return Seg3RetType(a.template segment<3>(a.output_rows() - 3));
        };
        obj.def("segment_3", seg3,
                R"doc(Extract a 3-element sub-vector of the output starting at a given index.

Parameters
----------
start : int
    Zero-based index of the first output element to include.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[start : start + 3]``.
)doc");
        obj.def("head_3", head3,
                R"doc(Extract the first 3 output components.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[:3]``.
)doc");
        obj.def("tail_3", tail3,
                R"doc(Extract the last 3 output components.

Returns
-------
VectorFunction
    Expression whose output is ``self(x)[-3:]``.
)doc");
        obj.def("segment3", seg3);
        obj.def("head3", head3);
        obj.def("tail3", tail3);
    }
}

template <class Derived, class PYClass> void BinaryMathBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    constexpr bool is_seg = std::is_same<Derived, SEG>::value;
    constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
    constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

    constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
    constexpr bool is_gen = std::is_same<Derived, Gen>::value;
    constexpr bool is_gens = std::is_same<Derived, GenS>::value;

    if constexpr (OR == 3 || OR == -1) {

        if constexpr (!is_seg)
            obj.def("cross",
                    [](const Derived &seg1, const SEG &seg2) {
                        return Gen(cross_product(seg1, seg2));
                    },
                    R"doc(3-D cross product of this VectorFunction with another.

Both operands must produce 3-element output vectors.  The result
is the standard right-hand-rule cross product ``self(x) × other(x)``.

Parameters
----------
other : VectorFunction or array_like, shape (3,)
    Right-hand operand.  Accepts a VectorFunction of any concrete type
    or a constant 3-element vector.

Returns
-------
VectorFunction
    3-element VectorFunction evaluating ``self(x) × other(x)``.
)doc");

        if constexpr (!is_seg3)
            obj.def("cross", [](const Derived &seg1, const SEG3 &seg2) {
                return Gen(cross_product(seg1, seg2));
            });
        if constexpr (!is_gen)
            obj.def("cross", [](const Derived &seg1, const Gen &seg2) {
                return Gen(cross_product(seg1, seg2));
            });

        obj.def("cross", [](const Derived &seg1, const Vector3<double> &seg2) {
            return Gen(cross_product(seg1, Constant<-1, 3>(seg1.input_rows(), seg2)));
        });

        obj.def("cross", [](const Derived &seg1, const Derived &seg2) {
            return Gen(cross_product(seg1, seg2));
        });
    }

    if constexpr (OR != 1) {
        if constexpr (!is_seg)
            obj.def("dot", [](const Derived &seg1, const SEG &seg2) {
                return GenS(dot_product(seg1, seg2));
            });

        if constexpr (!is_gen)
            obj.def("dot", [](const Derived &seg1, const Gen &seg2) {
                return GenS(dot_product(seg1, seg2));
            });

        ///////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        if constexpr (!is_seg)
            obj.def("cwise_product",
                    [](const Derived &seg1, const SEG &seg2) {
                        return Gen(CwiseFunctionProduct<Derived, SEG>(seg1, seg2));
                    },
                    R"doc(Element-wise product of this VectorFunction with another.

Parameters
----------
other : VectorFunction or array_like, shape (n,)
    Right-hand operand; must have the same output dimension as ``self``.
    Accepts a VectorFunction of any concrete type or a constant vector.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) .* other(x)`` element-wise.
)doc");
        if constexpr (!is_gen)
            obj.def("cwise_product", [](const Derived &seg1, const Gen &seg2) {
                return Gen(CwiseFunctionProduct<Derived, Gen>(seg1, seg2));
            });

        ///////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////

        if constexpr (!is_seg)
            obj.def("cwise_quotient",
                    [](const Derived &seg1, const SEG &seg2) {
                        return Gen(cwise_quotient(seg1, seg2));
                    },
                    R"doc(Element-wise quotient of this VectorFunction divided by another.

Parameters
----------
other : VectorFunction or array_like, shape (n,)
    Divisor; must have the same output dimension as ``self``.
    Accepts a VectorFunction of any concrete type or a constant vector.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) ./ other(x)`` element-wise.
)doc");
        if constexpr (!is_gen)
            obj.def("cwise_quotient", [](const Derived &seg1, const Gen &seg2) {
                return Gen(cwise_quotient(seg1, seg2));
            });
    }

    obj.def(
        "dot",
        [](const Derived &seg1, const Derived &seg2) { return GenS(dot_product(seg1, seg2)); },
        R"doc(Dot product of this VectorFunction with another equal-length one.

Parameters
----------
other : VectorFunction or array_like, shape (n,)
    Right-hand operand; must have the same output dimension as ``self``.

Returns
-------
ScalarFunction
    Expression evaluating ``self(x) · other(x)``.
)doc");
    obj.def("cwise_product", [](const Derived &seg1, const Derived &seg2) {
        return BinGen(CwiseFunctionProduct<Derived, Derived>(seg1, seg2));
    });

    obj.def("cwise_quotient", [](const Derived &seg1, const Derived &seg2) {
        return BinGen(cwise_quotient(seg1, seg2));
    });

    //////////////////////////////////////
    obj.def("dot", [](const Derived &seg1, const Eigen::VectorXd &seg2) {
        return GenS(dot_product(seg1, Constant<-1, Derived::ORC>(seg1.input_rows(), seg2)));
    });

    obj.def("cwise_product", [](const Derived &seg1, const Eigen::VectorXd &seg2) {
        return BinGen(RowScaled<Derived>(seg1, seg2));
    });

    obj.def("cwise_quotient", [](const Derived &seg1, const Eigen::VectorXd &seg2) {
        Eigen::VectorXd seg2i = seg2.cwiseInverse();
        return BinGen(RowScaled<Derived>(seg1, seg2i));
    });
}

template <class Derived, class PYClass> void BinaryOperatorsBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using ARGS = Arguments<-1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    constexpr bool is_seg = std::is_same<Derived, SEG>::value;
    constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
    constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

    constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
    constexpr bool is_gen = std::is_same<Derived, Gen>::value;
    constexpr bool is_gens = std::is_same<Derived, GenS>::value;

    constexpr bool is_arglike = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

    if constexpr (!is_arglike) {

        obj.def("eval", [](const Derived &a, const ELEM &b) { return BinGen(a.eval(b)); },
                R"doc(Compose this VectorFunction with an inner VectorFunction: ``self(g(.))``.

Substitutes the inner function ``g`` into the input of ``self``, forming
the composition ``x ↦ self(g(x))``.  The inner function's output dimension
must match ``self``'s input dimension.

Parameters
----------
g : VectorFunction
    Inner function whose output feeds ``self``'s input.  Accepts all
    concrete VectorFunction types (scalar, segment, generic, etc.).

Returns
-------
VectorFunction
    Expression evaluating ``self(g(x))``.
)doc");
        obj.def("eval", [](const Derived &a, const SEG &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const SEG2 &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const SEG3 &b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived &a, const Gen &b) { return BinGen(a.eval(b)); });

        /////////////////////////////////////////////////////
        obj.def(
            "__call__", [](const Derived &a, const Gen &b) { return BinGen(a.eval(b)); },
            nb::is_operator(),
            R"doc(Compose this VectorFunction with an inner VectorFunction (``self(g)``).

When called with a VectorFunction argument rather than a numeric vector,
returns the functional composition ``x ↦ self(g(x))`` rather than evaluating
``self`` at a point.  See :meth:`eval` for the non-operator form.

Parameters
----------
g : VectorFunction
    Inner function to substitute into the input of ``self``.

Returns
-------
VectorFunction
    Expression evaluating ``self(g(x))``.
)doc");
        obj.def(
            "__call__", [](const Derived &a, const ELEM &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        obj.def("__call__", [](const Derived &a, const SEG &b) { return BinGen(a.eval(b)); });
        obj.def(
            "__call__", [](const Derived &a, const SEG2 &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        obj.def(
            "__call__", [](const Derived &a, const SEG3 &b) { return BinGen(a.eval(b)); },
            nb::is_operator());
        ///////////////////////////////////////////////////////////////
        obj.def("eval", [](const Derived &a, int ir, Eigen::VectorXi v) {
            return BinGen(ParsedInput<Derived, -1, Derived::ORC>(a, v, ir));
        });
    }

    obj.def(
        "apply",
        [](const Derived &a, const Gen &b) {
            // ``g(self(x))``: g (= b) is the outer function, self (= a) the inner.
            return Gen(b.eval(a));
        },
        nb::is_operator(),
        R"doc(Apply an outer VectorFunction to the output of this one: ``g(self(.))``.

Forms the composition ``x ↦ g(self(x))``, with ``self`` as the *inner*
function and ``g`` as the *outer* function.  This is the reverse of
:meth:`eval`; use it when you have an outer function ``g`` and want to
pipe ``self`` through it.

Parameters
----------
g : VectorFunction
    Outer function; its input dimension must match ``self``'s output dimension.

Returns
-------
VectorFunction
    Expression evaluating ``g(self(x))``.
)doc");
    obj.def(
        "apply",
        [](const Derived &a, const GenS &b) {
            // Scalar outer function: ``g(self(x))`` with scalar result.
            return GenS(b.eval(a));
        },
        nb::is_operator());

    obj.def(
        "__add__", [](const Derived &a, const Derived &b) { return BinGen(a + b); },
        nb::is_operator(),
        R"doc(Add two VectorFunctions of the same type (``self + other``).

Parameters
----------
other : VectorFunction
    Right-hand operand; must have the same input and output dimensions as ``self``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) + other(x)``.
)doc");
    obj.def(
        "__sub__", [](const Derived &a, const Derived &b) { return BinGen(a - b); },
        nb::is_operator(),
        R"doc(Subtract two VectorFunctions of the same type (``self - other``).

Parameters
----------
other : VectorFunction
    Right-hand operand; must have the same input and output dimensions as ``self``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) - other(x)``.
)doc");

    obj.def(
        "__mul__", [](const Derived &a, const ELEM &b) { return BinGen(a * b); },
        nb::is_operator(),
        R"doc(Scale this VectorFunction by a scalar-valued VectorFunction (``self * s``).

Parameters
----------
s : ScalarFunction
    A single-output VectorFunction; its runtime value scales every output
    component of ``self``.

Returns
-------
VectorFunction
    Expression evaluating ``s(x) * self(x)`` element-wise.
)doc");
    obj.def(
        "__mul__", [](const Derived &a, const GenS &b) { return BinGen(a * b); },
        nb::is_operator());

    obj.def(
        "__truediv__", [](const Derived &a, const ELEM &b) { return BinGen(a / b); },
        nb::is_operator(),
        R"doc(Divide this VectorFunction by a scalar-valued VectorFunction (``self / s``).

Parameters
----------
s : ScalarFunction
    A single-output VectorFunction; every output component of ``self`` is
    divided by the runtime value of ``s``.

Returns
-------
VectorFunction
    Expression evaluating ``self(x) / s(x)`` element-wise.
)doc");

    obj.def(
        "__truediv__", [](const Derived &a, const GenS &b) { return BinGen(a / b); },
        nb::is_operator());

    if constexpr (is_seg || is_elem || is_seg2 || is_seg3) { //
        obj.def(
            "__add__",
            [](const Derived &a, const BinGen &b) {
                return BinGen(TwoFunctionSum<Derived, BinGen>(a, b));
            },
            nb::is_operator());
        obj.def(
            "__sub__",
            [](const Derived &a, const BinGen &b) {
                return BinGen(FunctionDifference<Derived, BinGen>(a, b));
            },
            nb::is_operator());
    }

    if constexpr (OR != 1) {

        obj.def(
            "__rmul__", [](const Derived &a, const ELEM &b) { return BinGen(a * b); },
            nb::is_operator());

        obj.def(
            "__rmul__", [](const Derived &a, const GenS &b) { return BinGen(a * b); },
            nb::is_operator());
    }

    ////////////////////////////////////////////////////////////////
}

template <class Derived, class PYClass> void ConditionalOperatorsBuild(PYClass &obj) {
    constexpr int OR = Derived::ORC;
    constexpr int IR = Derived::IRC;
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<IR>;

    if constexpr (OR == 1) {
        obj.def(
            "__lt__", [](const Derived &a, double b) { return GenCon(a < b); }, nb::is_operator());
        obj.def(
            "__gt__", [](const Derived &a, double b) { return GenCon(a > b); }, nb::is_operator());
        obj.def(
            "__rlt__", [](const Derived &a, double b) { return GenCon(a > b); }, nb::is_operator());
        obj.def(
            "__rgt__", [](const Derived &a, double b) { return GenCon(a < b); }, nb::is_operator());
        obj.def(
            "__le__", [](const Derived &a, double b) { return GenCon(a <= b); }, nb::is_operator());
        obj.def(
            "__ge__", [](const Derived &a, double b) { return GenCon(a >= b); }, nb::is_operator());

        obj.def(
            "__lt__", [](const Derived &a, const Derived &b) { return GenCon(a < b); },
            nb::is_operator());
        obj.def(
            "__gt__", [](const Derived &a, const Derived &b) { return GenCon(a > b); },
            nb::is_operator());
        obj.def(
            "__le__", [](const Derived &a, const Derived &b) { return GenCon(a <= b); },
            nb::is_operator());
        obj.def(
            "__ge__", [](const Derived &a, const Derived &b) { return GenCon(a >= b); },
            nb::is_operator());

        if constexpr (std::is_same<Derived, ELEM>::value) {
            obj.def(
                "__lt__", [](const Derived &a, const GenS &b) { return GenCon(a < b); },
                nb::is_operator());
            obj.def(
                "__gt__", [](const Derived &a, const GenS &b) { return GenCon(a > b); },
                nb::is_operator());
            obj.def(
                "__le__", [](const Derived &a, const GenS &b) { return GenCon(a <= b); },
                nb::is_operator());
            obj.def(
                "__ge__", [](const Derived &a, const GenS &b) { return GenCon(a >= b); },
                nb::is_operator());
        }
    }
}

} // namespace tycho::bind

#endif // TYCHO_PYTHON_BINDINGS
