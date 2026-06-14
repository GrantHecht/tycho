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

// Out-of-class definitions of InterpTable binding functions.
// Included from src/Bindings/Tycho_VectorFunctions.cpp after all InterpTable
// headers have been included so all types are complete.

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

// ── InterpTable1D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable1DBuild(nb::module_ &m) {

    using MatType = InterpTable1D::MatType;
    auto obj = nb::class_<InterpTable1D>(m, "InterpTable1D",
        R"doc(Lookup-table VectorFunction that interpolates tabulated data over one independent variable.

``InterpTable1D`` stores a set of sample points and associated values and evaluates
them at arbitrary query points via cubic (Hermite) or linear interpolation.  The
table can hold either a single scalar channel (``vlen == 1``) or multiple parallel
channels stacked as rows in a matrix.  Once constructed it can be used as a plain
callable for immediate numeric evaluation, or composed with a scalar VectorFunction
(or a scalar ``Segment``) via ``__call__`` to produce a new VectorFunction suitable
for use in an optimal-control problem.  Derivatives are evaluated analytically.

Available interpolation kinds: ``"cubic"`` (default, C1 Hermite spline) and
``"linear"`` (piecewise linear, C0).
)doc");

    // String -> InterpType conversion happens here at the binding layer.
    obj.def(
        "__init__",
        [](InterpTable1D *self, const Eigen::VectorXd &ts, const Eigen::VectorXd &vs,
           const std::string &kind) { new (self) InterpTable1D(ts, vs, parse_interp_type(kind)); },
        nb::arg("ts"), nb::arg("vs"), nb::arg("kind") = std::string("cubic"),
        R"doc(Construct a 1-D interpolation table from a 1-D value array.

Parameters
----------
ts : array_like, shape (N,)
    Strictly ascending sample coordinates.  Must have at least 5 elements.
vs : array_like, shape (N,)
    Scalar values at each sample point.
kind : str, optional
    Interpolation method: ``"cubic"`` (default) or ``"linear"``.
)doc");

    obj.def(
        "__init__",
        [](InterpTable1D *self, const Eigen::VectorXd &ts, const MatType &vs, int axis,
           const std::string &kind) {
            new (self) InterpTable1D(ts, vs, axis, parse_interp_type(kind));
        },
        nb::arg("ts"), nb::arg("vs"), nb::arg("axis") = 0, nb::arg("kind") = std::string("cubic"),
        R"doc(Construct a 1-D interpolation table from a 2-D value matrix.

Parameters
----------
ts : array_like, shape (N,)
    Strictly ascending sample coordinates.  Must have at least 5 elements.
vs : array_like, shape (N, C) or (C, N)
    Multi-channel tabulated values.  ``axis`` selects which dimension
    contains the sample axis (``0`` for rows, ``1`` for columns).
axis : int, optional
    Axis of ``vs`` that corresponds to the sample coordinates.
    ``0`` (default) means rows are samples; ``1`` means columns are samples.
kind : str, optional
    Interpolation method: ``"cubic"`` (default) or ``"linear"``.
)doc");

    obj.def(
        "__init__",
        [](InterpTable1D *self, const std::vector<Eigen::VectorXd> &vts, int tvar,
           const std::string &kind) {
            new (self) InterpTable1D(vts, tvar, parse_interp_type(kind));
        },
        nb::arg("vts"), nb::arg("tvar") = -1, nb::arg("kind") = std::string("cubic"),
        R"doc(Construct a 1-D interpolation table from a list of value-plus-time vectors.

Each element of ``vts`` is a vector whose components are the channel values
together with the independent coordinate at that sample.  The coordinate
component is identified by ``tvar`` and stripped before storing the channels.

Parameters
----------
vts : list of array_like
    List of length N; each element is a 1-D array of the same length.
    One element per sample, with the independent coordinate embedded at
    position ``tvar``.  All vectors must be the same length (>= 2).
tvar : int, optional
    Index of the independent-coordinate component within each vector.
    Negative values count from the end (default ``-1`` selects the last
    component).
kind : str, optional
    Interpolation method: ``"cubic"`` (default) or ``"linear"``.
)doc");

    // InterpType-enum overloads — same semantics as the string overloads above.
    obj.def(nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, InterpType>(), nb::arg("ts"),
            nb::arg("vs"), nb::arg("kind") = InterpType::Cubic);

    obj.def(nb::init<const Eigen::VectorXd &, const MatType &, int, InterpType>(), nb::arg("ts"),
            nb::arg("vs"), nb::arg("axis") = 0, nb::arg("kind") = InterpType::Cubic);

    obj.def(nb::init<const std::vector<Eigen::VectorXd> &, int, InterpType>(), nb::arg("vts"),
            nb::arg("tvar") = -1, nb::arg("kind") = InterpType::Cubic);

    obj.def("interp", nb::overload_cast<double>(&InterpTable1D::interp, nb::const_),
            nb::arg("t"),
            R"doc(Evaluate the table at a single coordinate value.

Parameters
----------
t : float
    Query coordinate.  Must lie within the table's coordinate range.

Returns
-------
numpy.ndarray, shape (vlen,)
    Interpolated channel values at ``t``.

Raises
------
ValueError
    If ``t`` is outside the table's coordinate range.
)doc");
    obj.def("interp",
            nb::overload_cast<const Eigen::VectorXd &>(&InterpTable1D::interp, nb::const_),
            nb::arg("t_vals"),
            R"doc(Evaluate the table at multiple coordinate values.

Parameters
----------
t_vals : array_like, shape (M,)
    Query coordinates.  Each must lie within the table's coordinate range.

Returns
-------
numpy.ndarray, shape (vlen, M)
    Interpolated channel values; column ``i`` corresponds to ``t_vals[i]``.
)doc");

    obj.def("__call__", nb::overload_cast<double>(&InterpTable1D::interp, nb::const_),
            nb::is_operator(),
            R"doc(Evaluate the table numerically — scalar coordinate.

Equivalent to :py:meth:`interp(t) <InterpTable1D.interp>`.
)doc");
    obj.def("__call__",
            nb::overload_cast<const Eigen::VectorXd &>(&InterpTable1D::interp, nb::const_),
            nb::is_operator(),
            R"doc(Evaluate the table numerically — vector of coordinates.

Equivalent to :py:meth:`interp(t_vals) <InterpTable1D.interp>`.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable1D> &self, const GenericFunction<-1, 1> &t) {
        nb::object pyfun;
        if (self->vlen_ == 1) {
            auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
            pyfun = nb::cast(f);
        } else {
            auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
            pyfun = nb::cast(f);
        }
        return pyfun;
    },
        R"doc(Compose the table with a scalar VectorFunction to produce a new VectorFunction.

Parameters
----------
t : VectorFunction (scalar output)
    Scalar-output VectorFunction whose result is used as the query coordinate.

Returns
-------
VectorFunction
    If ``vlen == 1``: scalar VectorFunction evaluating the table at ``t(x)``.
    If ``vlen > 1``: vector VectorFunction (output size = ``vlen``) evaluating
    all channels at ``t(x)``.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable1D> &self, const Segment<-1, 1, -1> &t) {
        nb::object pyfun;

        if (self->vlen_ == 1) {
            auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
            pyfun = nb::cast(f);
        } else {
            auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
            pyfun = nb::cast(f);
        }
        return pyfun;
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
    Scalar or vector VectorFunction as described in the ``GenericFunction``
    overload above, depending on ``vlen``.
)doc");

    obj.def("interp_deriv1", &InterpTable1D::interp_deriv1, nb::arg("t"),
            R"doc(Evaluate the table and its first derivative at a single coordinate.

Parameters
----------
t : float
    Query coordinate within the table's range.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    ``(v, dv_dt)`` where both arrays have shape ``(vlen,)``.
    ``v`` is the interpolated value and ``dv_dt`` its derivative with respect
    to ``t``.
)doc");
    obj.def("interp_deriv2", &InterpTable1D::interp_deriv2, nb::arg("t"),
            R"doc(Evaluate the table, its first derivative, and its second derivative.

Parameters
----------
t : float
    Query coordinate within the table's range.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    ``(v, dv_dt, d2v_dt2)`` where all arrays have shape ``(vlen,)``.
    For linear interpolation the second derivative is not computed; its
    contents are undefined — do not rely on them being zero.
)doc");

    obj.def("sf", [](std::shared_ptr<InterpTable1D> &self) {
        if (self->vlen_ != 1) {
            throw std::invalid_argument(
                "InterpTable1D storing Vector data cannot be converted to Scalar Function.");
        }
        return GenericFunction<-1, 1>(InterpFunction1D<1>(self));
    },
        R"doc(Return a scalar VectorFunction that wraps this table.

The returned VectorFunction takes a single-element input (the coordinate)
and produces a single-element output (the interpolated value).  Raises
``ValueError`` if the table stores more than one channel (``vlen > 1``);
use :py:meth:`vf` in that case.

Returns
-------
VectorFunction
    Scalar-in, scalar-out VectorFunction backed by this table.
)doc");
    obj.def("vf", [](std::shared_ptr<InterpTable1D> &self) {
        return GenericFunction<-1, -1>(InterpFunction1D<-1>(self));
    },
        R"doc(Return a vector VectorFunction that wraps this table.

The returned VectorFunction takes a single-element input (the coordinate)
and produces a ``vlen``-element output (all interpolated channels).  Works
for both scalar (``vlen == 1``) and multi-channel tables.

Returns
-------
VectorFunction
    Scalar-in, ``vlen``-out VectorFunction backed by this table.
)doc");
}

// ── InterpTable2D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable2DBuild(nb::module_ &m) {
    using MatType = InterpTable2D::MatType;
    auto obj = nb::class_<InterpTable2D>(m, "InterpTable2D",
        R"doc(Lookup-table VectorFunction that interpolates tabulated scalar data over two independent variables.

``InterpTable2D`` stores a 2-D grid of scalar samples and evaluates them at
arbitrary ``(x, y)`` query points via bicubic or bilinear interpolation.  The
output is always a single scalar.  Once constructed the table can be used as a
plain callable for immediate numeric evaluation, or composed with two scalar
VectorFunctions (or a 2-element ``Segment``) via ``__call__`` to produce a new
scalar VectorFunction suitable for use in an optimal-control problem.  Derivatives
are evaluated analytically.

Available interpolation kinds: ``"cubic"`` (default, bicubic) and ``"linear"``
(bilinear).
)doc");

    obj.def(
        "__init__",
        [](InterpTable2D *self, const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
           const MatType &z, const std::string &kind) {
            new (self) InterpTable2D(xs, ys, z, parse_interp_type(kind));
        },
        nb::arg("xs"), nb::arg("ys"), nb::arg("z"), nb::arg("kind") = std::string("cubic"),
        R"doc(Construct a 2-D interpolation table.

Parameters
----------
xs : array_like, shape (Nx,)
    Strictly ascending x-coordinates.  Must have at least 5 elements.
    Correspond to **columns** of ``z``.
ys : array_like, shape (Ny,)
    Strictly ascending y-coordinates.  Must have at least 5 elements.
    Correspond to **rows** of ``z``.
z : array_like, shape (Ny, Nx)
    Scalar values on the ``(xs, ys)`` grid in row-major order.
kind : str, optional
    Interpolation method: ``"cubic"`` (default) or ``"linear"``.
)doc");

    // InterpType-enum overload — same semantics as the string overload above.
    obj.def(
        nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, const MatType &, InterpType>(),
        nb::arg("xs"), nb::arg("ys"), nb::arg("z"), nb::arg("kind") = InterpType::Cubic);

    obj.def("interp", nb::overload_cast<double, double>(&InterpTable2D::interp, nb::const_),
            nb::arg("x"), nb::arg("y"),
            R"doc(Evaluate the table at a single ``(x, y)`` point.

Parameters
----------
x : float
    X-coordinate within the table's x range.
y : float
    Y-coordinate within the table's y range.

Returns
-------
float
    Interpolated scalar value at ``(x, y)``.

Raises
------
ValueError
    If ``x`` or ``y`` is outside the table's range.
)doc");
    obj.def("interp", nb::overload_cast<const MatType &, const MatType &>(&InterpTable2D::interp,
                                                                          nb::const_),
            nb::arg("x_vals"), nb::arg("y_vals"),
            R"doc(Evaluate the table element-wise over two matrices of coordinates.

Parameters
----------
x_vals : array_like, shape (M, K)
    X-coordinates of the query points.
y_vals : array_like, shape (M, K)
    Y-coordinates of the query points; must have the same shape as ``x_vals``.

Returns
-------
numpy.ndarray, shape (M, K)
    Interpolated scalar values at each ``(x_vals[i,j], y_vals[i,j])``.
)doc");

    obj.def("interp_deriv1", &InterpTable2D::interp_deriv1, nb::arg("x"), nb::arg("y"),
            R"doc(Evaluate the table and its first partial derivatives at ``(x, y)``.

Parameters
----------
x : float
    X-coordinate within the table's range.
y : float
    Y-coordinate within the table's range.

Returns
-------
tuple[float, numpy.ndarray]
    ``(z, dz_dxy)`` where ``dz_dxy`` is a length-2 array containing
    ``[dz/dx, dz/dy]``.
)doc");
    obj.def("interp_deriv2", &InterpTable2D::interp_deriv2, nb::arg("x"), nb::arg("y"),
            R"doc(Evaluate the table, first, and second partial derivatives at ``(x, y)``.

Parameters
----------
x : float
    X-coordinate within the table's range.
y : float
    Y-coordinate within the table's range.

Returns
-------
tuple[float, numpy.ndarray, numpy.ndarray]
    ``(z, dz_dxy, d2z_dxy2)`` where ``dz_dxy`` has shape ``(2,)`` containing
    ``[dz/dx, dz/dy]``, and ``d2z_dxy2`` is a ``(2, 2)`` symmetric Hessian
    matrix.  For linear interpolation the diagonal second derivatives
    (d²z/dx², d²z/dy²) are zero, but the off-diagonal cross-derivative
    (d²z/dxdy) is in general non-zero.
)doc");

    obj.def("find_elem", &InterpTable2D::find_elem, nb::arg("vals"), nb::arg("v"),
            R"doc(Binary-search helper: find the raw interval index for ``v`` in ``vals``.

Parameters
----------
vals : array_like, shape (N,)
    Sorted coordinate vector to search (e.g. the internal ``xs_`` or ``ys_``
    array).
v : float
    Value to locate.

Returns
-------
int
    Raw upper-bound-based index: ``upper_bound(vals, v) - vals.begin() - 1``.
    This may be negative or greater than ``N-2`` when ``v`` is outside the
    sample range; no clamping is applied here.  Clamping to ``[0, N-2]``
    happens later in ``get_xyelems``.
)doc");

    obj.def("__call__", nb::overload_cast<double, double>(&InterpTable2D::interp, nb::const_),
            nb::is_operator(),
            R"doc(Evaluate the table numerically at a single ``(x, y)`` point.

Equivalent to :py:meth:`interp(x, y) <InterpTable2D.interp>`.
)doc");
    obj.def("__call__",
            nb::overload_cast<const MatType &, const MatType &>(&InterpTable2D::interp, nb::const_),
            nb::is_operator(),
            R"doc(Evaluate the table numerically over matrices of coordinates.

Equivalent to :py:meth:`interp(x_vals, y_vals) <InterpTable2D.interp>`.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable2D> &self, const GenericFunction<-1, 1> &x,
                           const GenericFunction<-1, 1> &y) {
        return GenericFunction<-1, 1>(InterpFunction2D(self).eval(stack(x, y)));
    },
        R"doc(Compose the table with two scalar VectorFunctions to produce a new VectorFunction.

Parameters
----------
x : VectorFunction (scalar output)
    VectorFunction providing the x-coordinate.
y : VectorFunction (scalar output)
    VectorFunction providing the y-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``table(x(v), y(v))``.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable2D> &self, const Segment<-1, 1, -1> &x,
                           const Segment<-1, 1, -1> &y) {
        return GenericFunction<-1, 1>(InterpFunction2D(self).eval(stack(x, y)));
    },
        R"doc(Compose the table with two scalar Segments to produce a new VectorFunction.

Parameters
----------
x : Segment (scalar output)
    Scalar ``Segment`` providing the x-coordinate.
y : Segment (scalar output)
    Scalar ``Segment`` providing the y-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at the two segment values.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable2D> &self, const Segment<-1, 2, -1> &xy) {
        return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
    },
        R"doc(Compose the table with a 2-element Segment to produce a new VectorFunction.

Parameters
----------
xy : Segment (2-element output)
    Two-element ``Segment`` whose first element is the x-coordinate and
    second element is the y-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at ``(xy[0], xy[1])``.
)doc");

    obj.def("__call__",
            [](std::shared_ptr<InterpTable2D> &self, const GenericFunction<-1, -1> &xy) {
                return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
            },
            R"doc(Compose the table with a 2-output VectorFunction to produce a new VectorFunction.

Parameters
----------
xy : VectorFunction (2-element output)
    Two-output VectorFunction whose first output is the x-coordinate and
    second output is the y-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at ``(xy(v)[0], xy(v)[1])``.
)doc");

    obj.def("sf", [](std::shared_ptr<InterpTable2D> &self) {
        return GenericFunction<-1, 1>(InterpFunction2D(self));
    },
        R"doc(Return a scalar VectorFunction that wraps this table.

The returned VectorFunction takes a 2-element input ``[x, y]`` and returns
a single scalar output.  Equivalent to :py:meth:`vf` for 2-D tables.

Returns
-------
VectorFunction
    2-in, 1-out VectorFunction backed by this table.
)doc");
    obj.def("vf", [](std::shared_ptr<InterpTable2D> &self) {
        return GenericFunction<-1, -1>(InterpFunction2D(self));
    },
        R"doc(Return a vector VectorFunction that wraps this table.

The returned VectorFunction takes a 2-element input ``[x, y]`` and returns
a single scalar output (output size is dynamic, but 1 at runtime).  Use
:py:meth:`sf` when a statically-typed 1-output ``GenericFunction<-1,1>``
is required.

Returns
-------
VectorFunction
    2-in, dynamic-out VectorFunction backed by this table (1 output at runtime).
)doc");
}

// ── InterpTable3D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable3DBuild(nb::module_ &m) {

    auto obj = nb::class_<InterpTable3D>(m, "InterpTable3D",
        R"doc(Lookup-table VectorFunction that interpolates tabulated scalar data over three independent variables.

``InterpTable3D`` stores a 3-D grid of scalar samples (a rank-3 tensor) and
evaluates them at arbitrary ``(x, y, z)`` query points via tricubic or trilinear
interpolation.  The output is always a single scalar.  Once constructed the table
can be used as a plain callable for immediate numeric evaluation, or composed with
three scalar VectorFunctions (or a 3-element ``Segment``) via ``__call__`` to
produce a new scalar VectorFunction suitable for use in an optimal-control problem.
Derivatives are evaluated analytically.

Available interpolation kinds: ``"cubic"`` (default, tricubic) and ``"linear"``
(trilinear).  The ``cache`` option pre-computes and stores the per-cell polynomial
coefficients to accelerate repeated evaluations at the cost of additional memory.
)doc");

    obj.def(
        "__init__",
        [](InterpTable3D *self, const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
           const Eigen::VectorXd &zs, const Eigen::Tensor<double, 3> &fs, const std::string &kind,
           bool cache) {
            new (self) InterpTable3D(xs, ys, zs, fs, parse_interp_type(kind), cache);
        },
        nb::arg("xs"), nb::arg("ys"), nb::arg("zs"), nb::arg("fs"),
        nb::arg("kind") = std::string("cubic"), nb::arg("cache") = false,
        R"doc(Construct a 3-D interpolation table.

Parameters
----------
xs : array_like, shape (Nx,)
    Strictly ascending x-coordinates.  Must have at least 5 elements.
    Correspond to the first dimension of ``fs``.
ys : array_like, shape (Ny,)
    Strictly ascending y-coordinates.  Must have at least 5 elements.
    Correspond to the second dimension of ``fs``.
zs : array_like, shape (Nz,)
    Strictly ascending z-coordinates.  Must have at least 5 elements.
    Correspond to the third dimension of ``fs``.
fs : array_like, shape (Nx, Ny, Nz)
    Scalar values on the ``(xs, ys, zs)`` grid in NumPy meshgrid ``ij``
    format (first index = x, second = y, third = z).
kind : str, optional
    Interpolation method: ``"cubic"`` (default) or ``"linear"``.
cache : bool, optional
    If ``True``, pre-compute and cache all per-cell polynomial coefficients
    at construction time.  Speeds up repeated queries; uses more memory.
    Default is ``False``.
)doc");

    // InterpType-enum overload — same semantics as the string overload above.
    obj.def(nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &,
                     const Eigen::Tensor<double, 3> &, InterpType, bool>(),
            nb::arg("xs"), nb::arg("ys"), nb::arg("zs"), nb::arg("fs"),
            nb::arg("kind") = InterpType::Cubic, nb::arg("cache") = false);

    obj.def("interp",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp, nb::const_),
            nb::arg("x"), nb::arg("y"), nb::arg("z"),
            R"doc(Evaluate the table at a single ``(x, y, z)`` point.

Parameters
----------
x : float
    X-coordinate within the table's x range.
y : float
    Y-coordinate within the table's y range.
z : float
    Z-coordinate within the table's z range.

Returns
-------
float
    Interpolated scalar value at ``(x, y, z)``.

Raises
------
ValueError
    If any coordinate is outside the table's range.
)doc");
    obj.def("interp_deriv1",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp_deriv1, nb::const_),
            nb::arg("x"), nb::arg("y"), nb::arg("z"),
            R"doc(Evaluate the table and its first partial derivatives at ``(x, y, z)``.

Parameters
----------
x : float
    X-coordinate within the table's range.
y : float
    Y-coordinate within the table's range.
z : float
    Z-coordinate within the table's range.

Returns
-------
tuple[float, numpy.ndarray]
    ``(f, df_dxyz)`` where ``df_dxyz`` has shape ``(3,)`` containing
    ``[df/dx, df/dy, df/dz]``.
)doc");
    obj.def("interp_deriv2",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp_deriv2, nb::const_),
            nb::arg("x"), nb::arg("y"), nb::arg("z"),
            R"doc(Evaluate the table, first, and second partial derivatives at ``(x, y, z)``.

Parameters
----------
x : float
    X-coordinate within the table's range.
y : float
    Y-coordinate within the table's range.
z : float
    Z-coordinate within the table's range.

Returns
-------
tuple[float, numpy.ndarray, numpy.ndarray]
    ``(f, df_dxyz, d2f_dxyz2)`` where ``df_dxyz`` has shape ``(3,)`` and
    ``d2f_dxyz2`` is a symmetric ``(3, 3)`` Hessian matrix.
)doc");

    obj.def("__call__",
            nb::overload_cast<double, double, double>(&InterpTable3D::interp, nb::const_),
            nb::is_operator(),
            R"doc(Evaluate the table numerically at a single ``(x, y, z)`` point.

Equivalent to :py:meth:`interp(x, y, z) <InterpTable3D.interp>`.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable3D> &self, const GenericFunction<-1, 1> &x,
                           const GenericFunction<-1, 1> &y, const GenericFunction<-1, 1> &z) {
        return GenericFunction<-1, 1>(InterpFunction3D(self).eval(stack(x, y, z)));
    },
        R"doc(Compose the table with three scalar VectorFunctions to produce a new VectorFunction.

Parameters
----------
x : VectorFunction (scalar output)
    VectorFunction providing the x-coordinate.
y : VectorFunction (scalar output)
    VectorFunction providing the y-coordinate.
z : VectorFunction (scalar output)
    VectorFunction providing the z-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``table(x(v), y(v), z(v))``.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable3D> &self, const Segment<-1, 1, -1> &x,
                           const Segment<-1, 1, -1> &y, const Segment<-1, 1, -1> &z) {
        return GenericFunction<-1, 1>(InterpFunction3D(self).eval(stack(x, y, z)));
    },
        R"doc(Compose the table with three scalar Segments to produce a new VectorFunction.

Parameters
----------
x : Segment (scalar output)
    Scalar ``Segment`` providing the x-coordinate.
y : Segment (scalar output)
    Scalar ``Segment`` providing the y-coordinate.
z : Segment (scalar output)
    Scalar ``Segment`` providing the z-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at the three segment values.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable3D> &self, const Segment<-1, 3, -1> &xyz) {
        return GenericFunction<-1, 1>(InterpFunction3D(self).eval(xyz));
    },
        R"doc(Compose the table with a 3-element Segment to produce a new VectorFunction.

Parameters
----------
xyz : Segment (3-element output)
    Three-element ``Segment`` whose components supply the ``(x, y, z)``
    coordinates in order.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at ``(xyz[0], xyz[1], xyz[2])``.
)doc");

    obj.def("__call__",
            [](std::shared_ptr<InterpTable3D> &self, const GenericFunction<-1, -1> &xyz) {
                return GenericFunction<-1, 1>(InterpFunction3D(self).eval(xyz));
            },
            R"doc(Compose the table with a 3-output VectorFunction to produce a new VectorFunction.

Parameters
----------
xyz : VectorFunction (3-element output)
    Three-output VectorFunction whose outputs supply the ``(x, y, z)``
    coordinates in order.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at ``(xyz(v)[0], xyz(v)[1], xyz(v)[2])``.
)doc");

    obj.def("sf", [](std::shared_ptr<InterpTable3D> &self) {
        return GenericFunction<-1, 1>(InterpFunction3D(self));
    },
        R"doc(Return a scalar VectorFunction that wraps this table.

The returned VectorFunction takes a 3-element input ``[x, y, z]`` and returns
a single scalar output.

Returns
-------
VectorFunction
    3-in, 1-out VectorFunction backed by this table.
)doc");
    obj.def("vf", [](std::shared_ptr<InterpTable3D> &self) {
        return GenericFunction<-1, -1>(InterpFunction3D(self));
    },
        R"doc(Return a vector VectorFunction that wraps this table (output size 1).

The returned VectorFunction takes a 3-element input ``[x, y, z]`` and returns
a single scalar output (output size 1).

Returns
-------
VectorFunction
    3-in, 1-out VectorFunction backed by this table.
)doc");
}

// ── InterpTable4D
// ─────────────────────────────────────────────────────────────────────────────────
inline void InterpTable4DBuild(nb::module_ &m) {

    auto obj = nb::class_<InterpTable4D>(m, "InterpTable4D",
        R"doc(Lookup-table VectorFunction that interpolates tabulated scalar data over four independent variables.

``InterpTable4D`` stores a 4-D grid of scalar samples (a rank-4 tensor) and
evaluates them at arbitrary ``(x, y, z, w)`` query points via quadricubic or
quadrilinear interpolation.  The output is always a single scalar.  Once
constructed the table can be used as a plain callable for immediate numeric
evaluation, or composed with four scalar VectorFunctions (or a 4-element
``Segment``) via ``__call__`` to produce a new scalar VectorFunction suitable
for use in an optimal-control problem.  Derivatives are evaluated analytically.

Available interpolation kinds: ``"cubic"`` (default, quadricubic) and ``"linear"``
(quadrilinear).  The ``cache`` option pre-computes and stores the per-cell
polynomial coefficients to accelerate repeated evaluations at the cost of
additional memory.
)doc");

    obj.def(
        "__init__",
        [](InterpTable4D *self, const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
           const Eigen::VectorXd &zs, const Eigen::VectorXd &ws, const Eigen::Tensor<double, 4> &fs,
           const std::string &kind, bool cache) {
            new (self) InterpTable4D(xs, ys, zs, ws, fs, parse_interp_type(kind), cache);
        },
        nb::arg("xs"), nb::arg("ys"), nb::arg("zs"), nb::arg("ws"), nb::arg("fs"),
        nb::arg("kind") = std::string("cubic"), nb::arg("cache") = false,
        R"doc(Construct a 4-D interpolation table.

Parameters
----------
xs : array_like, shape (Nx,)
    Strictly ascending x-coordinates.  Must have at least 5 elements.
    Correspond to the first dimension of ``fs``.
ys : array_like, shape (Ny,)
    Strictly ascending y-coordinates.  Must have at least 5 elements.
    Correspond to the second dimension of ``fs``.
zs : array_like, shape (Nz,)
    Strictly ascending z-coordinates.  Must have at least 5 elements.
    Correspond to the third dimension of ``fs``.
ws : array_like, shape (Nw,)
    Strictly ascending w-coordinates.  Must have at least 5 elements.
    Correspond to the fourth dimension of ``fs``.
fs : array_like, shape (Nx, Ny, Nz, Nw)
    Scalar values on the ``(xs, ys, zs, ws)`` grid in NumPy meshgrid ``ij``
    format (first index = x, second = y, third = z, fourth = w).
kind : str, optional
    Interpolation method: ``"cubic"`` (default) or ``"linear"``.
cache : bool, optional
    If ``True``, pre-compute and cache all per-cell polynomial coefficients
    at construction time.  Speeds up repeated queries; uses more memory.
    Default is ``False``.
)doc");

    // InterpType-enum overload — same semantics as the string overload above.
    obj.def(nb::init<const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &,
                     const Eigen::VectorXd &, const Eigen::Tensor<double, 4> &, InterpType, bool>(),
            nb::arg("xs"), nb::arg("ys"), nb::arg("zs"), nb::arg("ws"), nb::arg("fs"),
            nb::arg("kind") = InterpType::Cubic, nb::arg("cache") = false);

    obj.def("interp",
            nb::overload_cast<double, double, double, double>(&InterpTable4D::interp, nb::const_),
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"),
            R"doc(Evaluate the table at a single ``(x, y, z, w)`` point.

Parameters
----------
x : float
    X-coordinate within the table's x range.
y : float
    Y-coordinate within the table's y range.
z : float
    Z-coordinate within the table's z range.
w : float
    W-coordinate within the table's w range.

Returns
-------
float
    Interpolated scalar value at ``(x, y, z, w)``.

Raises
------
ValueError
    If any coordinate is outside the table's range.
)doc");
    obj.def("interp_deriv1", nb::overload_cast<double, double, double, double>(
                                 &InterpTable4D::interp_deriv1, nb::const_),
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"),
            R"doc(Evaluate the table and its first partial derivatives at ``(x, y, z, w)``.

Parameters
----------
x : float
    X-coordinate within the table's range.
y : float
    Y-coordinate within the table's range.
z : float
    Z-coordinate within the table's range.
w : float
    W-coordinate within the table's range.

Returns
-------
tuple[float, numpy.ndarray]
    ``(f, df_dxyzw)`` where ``df_dxyzw`` has shape ``(4,)`` containing
    ``[df/dx, df/dy, df/dz, df/dw]``.
)doc");
    obj.def("interp_deriv2", nb::overload_cast<double, double, double, double>(
                                 &InterpTable4D::interp_deriv2, nb::const_),
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"),
            R"doc(Evaluate the table, first, and second partial derivatives at ``(x, y, z, w)``.

Parameters
----------
x : float
    X-coordinate within the table's range.
y : float
    Y-coordinate within the table's range.
z : float
    Z-coordinate within the table's range.
w : float
    W-coordinate within the table's range.

Returns
-------
tuple[float, numpy.ndarray, numpy.ndarray]
    ``(f, df_dxyzw, d2f_dxyzw2)`` where ``df_dxyzw`` has shape ``(4,)``
    and ``d2f_dxyzw2`` is a symmetric ``(4, 4)`` Hessian matrix.
)doc");

    obj.def("__call__",
            nb::overload_cast<double, double, double, double>(&InterpTable4D::interp, nb::const_),
            nb::is_operator(),
            R"doc(Evaluate the table numerically at a single ``(x, y, z, w)`` point.

Equivalent to :py:meth:`interp(x, y, z, w) <InterpTable4D.interp>`.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable4D> &self, const GenericFunction<-1, 1> &x,
                           const GenericFunction<-1, 1> &y, const GenericFunction<-1, 1> &z,
                           const GenericFunction<-1, 1> &w) {
        return GenericFunction<-1, 1>(InterpFunction4D(self).eval(stack(x, y, z, w)));
    },
        R"doc(Compose the table with four scalar VectorFunctions to produce a new VectorFunction.

Parameters
----------
x : VectorFunction (scalar output)
    VectorFunction providing the x-coordinate.
y : VectorFunction (scalar output)
    VectorFunction providing the y-coordinate.
z : VectorFunction (scalar output)
    VectorFunction providing the z-coordinate.
w : VectorFunction (scalar output)
    VectorFunction providing the w-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating ``table(x(v), y(v), z(v), w(v))``.
)doc");

    obj.def("__call__",
            [](std::shared_ptr<InterpTable4D> &self, const Segment<-1, 1, -1> &x,
               const Segment<-1, 1, -1> &y, const Segment<-1, 1, -1> &z, const Segment<-1, 1, -1> &w

            ) { return GenericFunction<-1, 1>(InterpFunction4D(self).eval(stack(x, y, z, w))); },
            R"doc(Compose the table with four scalar Segments to produce a new VectorFunction.

Parameters
----------
x : Segment (scalar output)
    Scalar ``Segment`` providing the x-coordinate.
y : Segment (scalar output)
    Scalar ``Segment`` providing the y-coordinate.
z : Segment (scalar output)
    Scalar ``Segment`` providing the z-coordinate.
w : Segment (scalar output)
    Scalar ``Segment`` providing the w-coordinate.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at the four segment values.
)doc");

    obj.def("__call__", [](std::shared_ptr<InterpTable4D> &self, const Segment<-1, -1, -1> &xyzw) {
        return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
    },
        R"doc(Compose the table with a dynamically-sized Segment to produce a new VectorFunction.

Parameters
----------
xyzw : Segment (dynamic size)
    A ``Segment`` whose first four components supply the ``(x, y, z, w)``
    coordinates in order.  The size is not checked at compile time; the
    caller is responsible for ensuring at least four components are present.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at
    ``(xyzw[0], xyzw[1], xyzw[2], xyzw[3])``.
)doc");

    obj.def("__call__",
            [](std::shared_ptr<InterpTable4D> &self, const GenericFunction<-1, -1> &xyzw) {
                return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
            },
            R"doc(Compose the table with a 4-output VectorFunction to produce a new VectorFunction.

Parameters
----------
xyzw : VectorFunction (4-element output)
    Four-output VectorFunction whose outputs supply the ``(x, y, z, w)``
    coordinates in order.

Returns
-------
VectorFunction
    Scalar VectorFunction evaluating the table at
    ``(xyzw(v)[0], xyzw(v)[1], xyzw(v)[2], xyzw(v)[3])``.
)doc");

    obj.def("sf", [](std::shared_ptr<InterpTable4D> &self) {
        return GenericFunction<-1, 1>(InterpFunction4D(self));
    },
        R"doc(Return a scalar VectorFunction that wraps this table.

The returned VectorFunction takes a 4-element input ``[x, y, z, w]`` and
returns a single scalar output.

Returns
-------
VectorFunction
    4-in, 1-out VectorFunction backed by this table.
)doc");
    obj.def("vf", [](std::shared_ptr<InterpTable4D> &self) {
        return GenericFunction<-1, -1>(InterpFunction4D(self));
    },
        R"doc(Return a vector VectorFunction that wraps this table (output size 1).

The returned VectorFunction takes a 4-element input ``[x, y, z, w]`` and
returns a single scalar output (output size 1).

Returns
-------
VectorFunction
    4-in, 1-out VectorFunction backed by this table.
)doc");
}

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
