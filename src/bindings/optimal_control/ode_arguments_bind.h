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

#include "dense_function_base_bind.h"
#include "function_registry.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

template <int _XV, int _UV, int _PV> struct TychoBind<ODEArguments<_XV, _UV, _PV>> {
    using Derived = ODEArguments<_XV, _UV, _PV>;
    using Base = typename Derived::Base;
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Derived, Base>(
            m, name,
            R"doc(Symbolic argument vector for defining ODE right-hand sides.

``ODEArguments(Xvars, Uvars, Pvars)`` constructs the ordered input
``[x, t, u, p]`` of an ODE -- the state ``x`` (size ``Xvars``), time ``t``,
control ``u`` (size ``Uvars``), and ODE parameters ``p`` (size ``Pvars``).
The accessors below slice out named sub-vectors and individual scalars as
:class:`~tychopy.vector_functions.VectorFunction` expressions, which are then
combined arithmetically to build the dynamics.

Examples
--------
Build the ODE argument set and slice out named state/control expressions::

    XtU = ODEArguments(3, 1)        # 3 states, 1 control
    x, y, v = XtU.x_vec().tolist()
    theta = XtU.u_var(0)
)doc");
        obj.def(nb::init<int, int, int>(),
                R"doc(Construct with explicit state, control, and parameter counts.

Parameters
----------
Xvars : int
    Number of state variables.
Uvars : int
    Number of control variables.
Pvars : int
    Number of ODE parameters.
)doc");
        obj.def(nb::init<int, int>(),
                R"doc(Construct with state and control counts (no ODE parameters).

Parameters
----------
Xvars : int
    Number of state variables.
Uvars : int
    Number of control variables.
)doc");
        obj.def(nb::init<int>(), R"doc(Construct with state count only (no controls or parameters).

Parameters
----------
Xvars : int
    Number of state variables.
)doc");

        bind::DenseBaseBuild<Derived>(obj);

        obj.def(
            "x_vec", [](const Derived &a) { return a.segment(0, a.x_vars()); },
            R"doc(State sub-vector ``x``.

Returns
-------
VectorFunction
    Expression selecting the state block, of size ``Xvars``.
)doc");
        obj.def(
            "x_var", [](const Derived &a, int i) { return a.segment(0, a.x_vars()).coeff(i); },
            R"doc(A single state variable.

Parameters
----------
i : int
    Zero-based index into the state block.

Returns
-------
VectorFunction
    Scalar expression for state ``x[i]``.
)doc");
        obj.def(
            "xt_vec", [](const Derived &a) { return a.segment(0, a.xt_vars()); },
            R"doc(Combined state-and-time sub-vector ``[x, t]``.

Returns
-------
VectorFunction
    Expression of size ``Xvars + 1``.
)doc");
        obj.def(
            "t_var", [](const Derived &a) { return a.coeff(a.t_var()); },
            R"doc(The time variable ``t``.

Returns
-------
VectorFunction
    Scalar expression for time.
)doc");
        obj.def(
            "u_vec", [](const Derived &a) { return a.segment(a.xt_vars(), a.u_vars()); },
            R"doc(Control sub-vector ``u``.

Returns
-------
VectorFunction
    Expression selecting the control block, of size ``Uvars``.
)doc");
        obj.def(
            "u_var",
            [](const Derived &a, int i) { return a.segment(a.xt_vars(), a.u_vars()).coeff(i); },
            R"doc(A single control variable.

Parameters
----------
i : int
    Zero-based index into the control block.

Returns
-------
VectorFunction
    Scalar expression for control ``u[i]``.
)doc");
        obj.def(
            "p_vec", [](const Derived &a) { return a.segment(a.xtu_vars(), a.p_vars()); },
            R"doc(ODE-parameter sub-vector ``p``.

Returns
-------
VectorFunction
    Expression selecting the parameter block, of size ``Pvars``.
)doc");
        obj.def(
            "p_var",
            [](const Derived &a, int i) { return a.segment(a.xtu_vars(), a.p_vars()).coeff(i); },
            R"doc(A single ODE parameter.

Parameters
----------
i : int
    Zero-based index into the parameter block.

Returns
-------
VectorFunction
    Scalar expression for parameter ``p[i]``.
)doc");
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
