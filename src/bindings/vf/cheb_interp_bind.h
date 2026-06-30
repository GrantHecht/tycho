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
        R"doc(1-D Chebyshev (2nd-kind / DCT-I) interpolant VectorFunction.

Stores Chebyshev coefficients of one or more output channels and evaluates them
via Clenshaw recurrence with analytic first and second derivatives.  Build from
values sampled at :py:meth:`cheb_points`, then compose with a scalar VectorFunction
via ``__call__`` for use in optimal-control problems.

Derivative-series coefficients are precomputed once at construction time, so
evaluation at solve time incurs no per-call allocation or recurrence work.
)doc");

    obj.def_static(
        "cheb_points", &ChebTable::cheb_points, nb::arg("order"), nb::arg("lb"), nb::arg("ub"),
        R"doc(Return the order+1 second-kind Chebyshev nodes on [lb, ub].

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

    obj.def_static(
        "from_values", &ChebTable::from_values, nb::arg("grid_values"), nb::arg("lb"),
        nb::arg("ub"), nb::arg("order"), nb::arg("nthreads") = 1,
        R"doc(Build a ChebTable from values sampled at the Chebyshev nodes.

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

    obj.def("eval", &ChebTable::eval, nb::arg("t"),
            R"doc(Evaluate all output channels at scalar coordinate t.

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

    obj.def("__call__", &ChebTable::eval, nb::arg("t"), nb::is_operator(),
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
            R"doc(Per-channel norm of the trailing-half Chebyshev coefficients.

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
    obj.def_prop_ro("order", &ChebTable::order,
                    R"doc(Polynomial order n (number of nodes = order + 1).)doc");

    // Compose with a scalar VectorFunction -> new VectorFunction.
    // ChebFunction always has dynamic output rows (OR=-1), so we always
    // return GenericFunction<-1,-1>. The output_dim() at runtime determines
    // the actual number of channels.
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

    // Compose with a scalar Segment -> new VectorFunction.
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

    obj.def(
        "vf",
        [](std::shared_ptr<ChebTable> &self) {
            return GenericFunction<-1, -1>(ChebFunction<-1>(self));
        },
        R"doc(Return a scalar-in, output_dim-out VectorFunction backed by this table.

The returned VectorFunction takes a single-element input (the coordinate)
and produces an ``output_dim``-element output (all interpolated channels).

Returns
-------
VectorFunction
    Scalar-in, ``output_dim``-out VectorFunction backed by this table.
)doc");
}

}  // namespace tycho
#endif  // TYCHO_PYTHON_BINDINGS
