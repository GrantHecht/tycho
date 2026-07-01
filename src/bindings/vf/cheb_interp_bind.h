// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class ChebTable binding. Included from tycho_vector_functions.cpp
// after the ChebTable header so all types are complete.

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

inline void ChebInterpBuild(nb::module_ &m) {
    using MatType = ChebTable::MatType;
    auto obj = nb::class_<ChebTable>(m, "ChebTable",
        R"doc(Chebyshev (2nd-kind / DCT-I) interpolant VectorFunction — 1-D and N-D.

Stores Chebyshev coefficients of one or more output channels and evaluates them
via Clenshaw recurrence with analytic first and second derivatives.  Build from
values sampled at :py:meth:`cheb_points`, then compose with VectorFunction
arguments via ``__call__`` for use in optimal-control problems.

Derivative-series coefficients are precomputed once at construction time, so
evaluation at solve time incurs no per-call allocation or recurrence work.
)doc");

    // ---- 1-D cheb_points(n, lb, ub) -> ndarray ----
    obj.def_static(
        "cheb_points",
        [](int order, double lb, double ub) { return ChebTable::cheb_points(order, lb, ub); },
        nb::arg("order"), nb::arg("lb"), nb::arg("ub"),
        R"doc(Return the order+1 second-kind Chebyshev nodes on [lb, ub] (1-D).

The nodes are ordered so that ``t[0] = ub`` and ``t[order] = lb`` (i.e. the
cosine mapping places the first node at the right endpoint).  Pass the returned
array to :py:meth:`from_values` to build a :py:class:`ChebTable`.

Parameters
----------
order : int
    Polynomial order ``n >= 1``.  The returned array has ``n+1`` elements.
lb : float
    Lower bound of the physical domain.
ub : float
    Upper bound of the physical domain.  Must satisfy ``ub > lb``.

Returns
-------
numpy.ndarray, shape (order+1,)
    Second-kind Chebyshev node coordinates on ``[lb, ub]``.
)doc");

    // ---- N-D cheb_points(orders, lb, ub) -> list[ndarray] ----
    obj.def_static(
        "cheb_points",
        [](const std::vector<int> &orders, const Eigen::VectorXd &lb,
           const Eigen::VectorXd &ub) -> nb::list {
            auto per_axis = ChebTable::cheb_points(orders, lb, ub);
            nb::list result;
            for (auto &v : per_axis) result.append(v);
            return result;
        },
        nb::arg("orders"), nb::arg("lb"), nb::arg("ub"),
        R"doc(Return per-axis Chebyshev nodes for an N-D tensor-product grid.

Parameters
----------
orders : list[int]
    Per-axis polynomial orders.  ``orders[d]`` nodes per axis are ``orders[d]+1``.
lb : array_like, shape (D,)
    Per-axis lower bounds.
ub : array_like, shape (D,)
    Per-axis upper bounds.  Must satisfy ``ub[d] > lb[d]`` for all d.

Returns
-------
list of numpy.ndarray
    ``orders[d]+1`` node coordinates for each axis d.
)doc");

    // ---- 1-D from_values(grid_values olen×(n+1), lb, ub, order, nthreads) ----
    obj.def_static(
        "from_values",
        [](const MatType &grid_values, double lb, double ub, int order, int nthreads) {
            return ChebTable::from_values(grid_values, lb, ub, order, nthreads);
        },
        nb::arg("grid_values"), nb::arg("lb"), nb::arg("ub"), nb::arg("order"),
        nb::arg("nthreads") = 1,
        R"doc(Build a ChebTable from values sampled at the Chebyshev nodes (1-D).

The values must be sampled at the ``order+1`` nodes returned by
:py:meth:`cheb_points`.  Coefficients are computed via a DCT-I transform
(pocketfft) and derivative-series coefficients are precomputed immediately,
so subsequent evaluation calls incur no per-call allocation.

Parameters
----------
grid_values : array_like, shape (olen, order+1)
    Sampled values.  Rows are output channels; columns correspond to the
    ``order+1`` Chebyshev nodes in the order returned by :py:meth:`cheb_points`.
lb : float
    Lower bound of the physical domain.
ub : float
    Upper bound of the physical domain.  Must satisfy ``ub > lb``.
order : int
    Polynomial order ``n >= 1``.
nthreads : int, optional
    Thread count forwarded to pocketfft for the DCT-I transform.  Defaults
    to 1; pass 0 to use all available cores.  Negative values raise
    ``ValueError``.

Returns
-------
ChebTable
    Constructed interpolant ready for evaluation or composition.
)doc");

    // ---- N-D from_values(grid_values_flat tsize×olen, lb, ub, orders, nthreads) ----
    obj.def_static(
        "from_values",
        [](const MatType &grid_values_flat, const Eigen::VectorXd &lb,
           const Eigen::VectorXd &ub, const std::vector<int> &orders,
           int nthreads) -> ChebTable {
            return ChebTable::from_values(grid_values_flat, lb, ub, orders, nthreads);
        },
        nb::arg("grid_values_flat"), nb::arg("lb"), nb::arg("ub"), nb::arg("orders"),
        nb::arg("nthreads") = 1,
        R"doc(Build a ChebTable from values on an N-D tensor-product Chebyshev grid.

Parameters
----------
grid_values_flat : array_like, shape (tsize, olen)
    Flattened grid values in row-major (axis-0-outer) order, where
    ``tsize = prod(orders[d]+1)``.  Columns are output channels.
    Obtain the grid from :py:meth:`cheb_points` and flatten row-major.
lb : array_like, shape (D,)
    Per-axis lower bounds.
ub : array_like, shape (D,)
    Per-axis upper bounds.
orders : list[int]
    Per-axis polynomial orders.
nthreads : int, optional
    Thread count for the DCT-I transforms.  Defaults to 1.

Returns
-------
ChebTable
    N-D Chebyshev interpolant ready for evaluation or composition.
)doc");

    // ---- eval(double) for 1-D ----
    obj.def("eval",
            [](const ChebTable &self, double t) { return self.eval(t); },
            nb::arg("t"),
            R"doc(Evaluate all output channels at scalar coordinate t (1-D).

Parameters
----------
t : float
    Query coordinate.  Values outside ``[lb, ub]`` are clamped to the
    nearest endpoint (no extrapolation error is raised).

Returns
-------
numpy.ndarray, shape (olen,)
    Interpolated channel values at ``t``.
)doc");

    // ---- eval(VectorXd) for N-D ----
    obj.def("eval",
            [](const ChebTable &self, const Eigen::VectorXd &x) { return self.eval(x); },
            nb::arg("x"),
            R"doc(Evaluate all output channels at a physical point x (N-D).

Parameters
----------
x : array_like, shape (D,)
    Query coordinates.  Each coordinate is clamped to its axis interval.

Returns
-------
numpy.ndarray, shape (olen,)
    Interpolated channel values at ``x``.
)doc");

    // ---- __call__(double) numeric evaluation ----
    obj.def("__call__",
            [](const ChebTable &self, double t) { return self.eval(t); },
            nb::arg("t"), nb::is_operator(),
            R"doc(Evaluate all output channels at scalar coordinate t (numeric call).

Equivalent to :py:meth:`eval`.
)doc");

    obj.def("eval_deriv1", &ChebTable::eval_deriv1, nb::arg("t"),
            R"doc(Evaluate value and first derivative at t.

Parameters
----------
t : float
    Query coordinate within or clamped to ``[lb, ub]``.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    ``(value, deriv1)`` where both arrays have shape ``(olen,)``.
    ``deriv1`` contains ``df/dt`` for each output channel.
)doc");

    obj.def("eval_deriv2", &ChebTable::eval_deriv2, nb::arg("t"),
            R"doc(Evaluate value, first derivative, and second derivative at t.

Parameters
----------
t : float
    Query coordinate within or clamped to ``[lb, ub]``.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    ``(value, deriv1, deriv2)`` where all arrays have shape ``(olen,)``.
    ``deriv1`` contains ``df/dt`` and ``deriv2`` contains ``d²f/dt²`` for
    each output channel.
)doc");

    obj.def("coeff_tail_norm", &ChebTable::coeff_tail_norm,
            R"doc(Per-channel norm of the trailing-half Chebyshev coefficients (1-D).

Returns the L2 norm of the upper half of the coefficient series for each
output channel.  Used by the adaptive construction loop
(:py:func:`cheb_adaptive`) as a convergence indicator: when this norm is
small relative to the function scale the series has converged.

Returns
-------
numpy.ndarray, shape (olen,)
    Tail norms; one entry per output channel.
)doc");

    obj.def_prop_ro("output_dim", &ChebTable::output_dim,
                    R"doc(Number of output channels (olen).)doc");
    obj.def_prop_ro("input_dim", &ChebTable::input_dim,
                    R"doc(Number of input dimensions (1 for 1-D tables, D for N-D).)doc");
    obj.def_prop_ro("order", &ChebTable::order,
                    R"doc(Polynomial order n for 1-D tables (orders()[0]).)doc");
    obj.def_prop_ro("orders", [](const ChebTable &self) -> nb::list {
        nb::list result;
        for (int o : self.orders()) result.append(o);
        return result;
    }, R"doc(Per-axis polynomial orders as a list of int.)doc");

    // ---- Compose overloads: 1-D (scalar GenericFunction arg) ----
    obj.def(
        "__call__",
        [](std::shared_ptr<ChebTable> &self, const GenericFunction<-1, 1> &t) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self).eval(t));
        },
        R"doc(Compose the table with a scalar VectorFunction to produce a new VectorFunction.

Parameters
----------
t : VectorFunction (scalar output)
    Scalar-output VectorFunction whose result is used as the query coordinate.

Returns
-------
VectorFunction
    VectorFunction evaluating all channels at ``t(x)``.  Output size equals
    ``output_dim`` at runtime.
)doc");

    // ---- Compose overloads: 1-D (scalar Segment arg) ----
    obj.def(
        "__call__",
        [](std::shared_ptr<ChebTable> &self, const Segment<-1, 1, -1> &t) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self).eval(t));
        },
        R"doc(Compose the table with a scalar Segment to produce a new VectorFunction.

Parameters
----------
t : Segment (scalar output)
    Scalar-output ``Segment`` (a single element of a state/control vector)
    whose value is used as the query coordinate.

Returns
-------
VectorFunction
    VectorFunction evaluating all channels at the segment value.  Output size
    equals ``output_dim`` at runtime.
)doc");

    // ---- Compose overloads: 2-D (two scalar GenericFunction args) ----
    // Use ChebFunction<-1> (fully dynamic) to avoid static size mismatch in
    // ChebFunction's 1-D branch when IR is a fixed compile-time constant.
    obj.def(
        "__call__",
        [](std::shared_ptr<ChebTable> &self, const GenericFunction<-1, 1> &x,
           const GenericFunction<-1, 1> &y) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self).eval(stack(x, y)));
        },
        R"doc(Compose the table with two scalar VectorFunctions (2-D input).

Parameters
----------
x : VectorFunction (scalar output)
    VectorFunction providing the first coordinate.
y : VectorFunction (scalar output)
    VectorFunction providing the second coordinate.

Returns
-------
VectorFunction
    VectorFunction evaluating all channels at ``(x(v), y(v))``.
)doc");

    // ---- Compose overloads: 2-D (two scalar Segment args) ----
    obj.def(
        "__call__",
        [](std::shared_ptr<ChebTable> &self, const Segment<-1, 1, -1> &x,
           const Segment<-1, 1, -1> &y) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self).eval(stack(x, y)));
        },
        R"doc(Compose the table with two scalar Segments (2-D input).

Parameters
----------
x : Segment (scalar output)
y : Segment (scalar output)

Returns
-------
VectorFunction
    VectorFunction evaluating all channels at the two segment values.
)doc");

    // ---- Compose overloads: 3-D (three scalar GenericFunction args) ----
    obj.def(
        "__call__",
        [](std::shared_ptr<ChebTable> &self, const GenericFunction<-1, 1> &x,
           const GenericFunction<-1, 1> &y, const GenericFunction<-1, 1> &z) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self).eval(stack(x, y, z)));
        },
        R"doc(Compose the table with three scalar VectorFunctions (3-D input).

Parameters
----------
x : VectorFunction (scalar output)
y : VectorFunction (scalar output)
z : VectorFunction (scalar output)

Returns
-------
VectorFunction
    VectorFunction evaluating all channels at ``(x(v), y(v), z(v))``.
)doc");

    // ---- Compose overloads: 3-D (three scalar Segment args) ----
    obj.def(
        "__call__",
        [](std::shared_ptr<ChebTable> &self, const Segment<-1, 1, -1> &x,
           const Segment<-1, 1, -1> &y, const Segment<-1, 1, -1> &z) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self).eval(stack(x, y, z)));
        },
        R"doc(Compose the table with three scalar Segments (3-D input).

Parameters
----------
x : Segment (scalar output)
y : Segment (scalar output)
z : Segment (scalar output)

Returns
-------
VectorFunction
    VectorFunction evaluating all channels at the three segment values.
)doc");

    obj.def(
        "vf",
        [](std::shared_ptr<ChebTable> &self) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self));
        },
        R"doc(Return a VectorFunction backed by this table (any dimension).

For a 1-D table, takes a single-element input.
For an N-D table, takes an N-element input vector.

Returns
-------
VectorFunction
    input_dim-in, ``output_dim``-out VectorFunction backed by this table.
)doc");
}

}  // namespace tycho
#endif  // TYCHO_PYTHON_BINDINGS
