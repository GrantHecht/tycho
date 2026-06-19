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

// Free-function binding helpers for StaticODE_Expression, ODEBase, and GenericODE.

#include "dense_function_base_bind.h"
#include "ode_sizes_bind.h"

namespace tycho::bind {

using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

template <class Derived, class ExprImpl, class... Ts>
void StaticODE_ExpressionBuild(nb::module_ &m, const char *name) {
    auto obj = nb::class_<Derived>(m, name).def(nb::init<Ts...>());
    DenseBaseBuild<Derived>(obj);
    obj.def("phase", [](const Derived &od, TranscriptionModes Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
    });
    IntegratorBuildConstructors<Derived>(obj);
}

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
void BuildODEModule(const char *name, nb::module_ &mod, FunctionRegistry &reg) {
    auto odemod = mod.def_submodule(name);
    reg.template build_register<Derived>(odemod, "ode");
    reg.template build_register<Integrator<Derived>>(odemod, "integrator");
    TychoBind<ODEPhase<Derived>>::build(odemod);
}

template <class BaseType, class Derived, int _XV, int _UV, int _PV>
void BuildODEModule(const char *name, FunctionRegistry &reg) {
    BuildODEModule<BaseType, Derived, _XV, _UV, _PV>(name, reg.mod, reg);
}

template <class BaseType, int _XV, int _UV, int _PV>
void BuildGenODEModule(const char *name, nb::module_ &mod, FunctionRegistry &reg) {
    using Derived = GenericODE<BaseType, _XV, _UV, _PV>;
    auto odemod = mod.def_submodule(name);

    auto obj = nb::class_<Derived>(odemod, "ode");
    obj.def(nb::init<BaseType, int, int, int>(),
            R"doc(Construct a type-erased ODE with explicit state, control, and parameter sizes.

Parameters
----------
f : VectorFunction
    The dynamics function mapping ``[x, t, u, p]`` to ``xdot``.
xv : int
    Number of state variables.
uv : int
    Number of control variables.
pv : int
    Number of parameter variables.

Raises
------
ValueError
    If ``f``'s output size does not equal ``xv``, or its input size does not
    equal ``xv + uv + pv + 1``.
)doc");
    obj.def(nb::init<BaseType, int, int>(), "");
    obj.def(nb::init<BaseType, int>(), "");
    obj.def(nb::init<BaseType>(), "");
    TychoBind<ODEPhase<Derived>>::build(odemod);
    obj.def(
        "phase",
        [](const Derived &od, TranscriptionModes Tmode) {
            return std::make_shared<ODEPhase<Derived>>(od, Tmode);
        },
        R"doc(Create an optimal control phase with no initial trajectory.

Parameters
----------
mode : TranscriptionModes or str
    Transcription scheme: ``"LGL3"``, ``"LGL5"``, ``"LGL7"``, ``"Trapezoidal"``,
    or ``"CentralShooting"``.

Returns
-------
Phase
    A new phase bound to this ODE and the chosen transcription scheme.
)doc");
    obj.def(
        "phase",
        [](const Derived &od, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd> &Traj,
           int numdef) { return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef); },
        "");

    obj.def(
        "phase",
        [](const Derived &od, std::string Tmode) {
            return std::make_shared<ODEPhase<Derived>>(od, Tmode);
        },
        "");

    obj.def(
        "phase",
        [](const Derived &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj) {
            return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj);
        },
        "");

    obj.def(
        "phase",
        [](const Derived &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
           int numdef) { return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef); },
        "");
    obj.def(
        "phase",
        [](const Derived &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
           int numdef, bool LerpIG) {
            return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef, LerpIG);
        },
        "");

    nb::implicitly_convertible<Derived, GenericFunction<-1, -1>>();
    reg.vfuncx.def("__init__", [](GenericFunction<-1, -1> *self, const Derived &ode) {
        new (self) GenericFunction<-1, -1>(ode.func_);
    });

    IntegratorBuildConstructors<Derived>(obj);

    ODESizeBuild<_XV, _UV, _PV, Derived>(obj);

    obj.def(
        "vf", [](const Derived &od) { return od.func_; },
        R"doc(Return the underlying type-erased dynamics VectorFunction.

Returns
-------
VectorFunction
    The type-erased dynamics function held by this ODE, mapping
    ``[x, t, u, p]`` to ``xdot``.
)doc");

    obj.def(
        "shooting_defect",
        [](const Derived &ode, const Integrator<Derived> &integ) {
            auto shooter = CentralShootingDefect<Derived, Integrator<Derived>>(ode, integ);
            return GenericFunction<-1, -1>(shooter);
        },
        R"doc(Build a central-shooting defect VectorFunction for this ODE.

Constructs a ``CentralShootingDefect`` that, given two boundary states and
(optionally) shared parameters, integrates both arcs to the interval midpoint
and returns the mismatch. Suitable for use as a path constraint in a
``CentralShooting`` transcription phase, or directly in an NLP.

Parameters
----------
integrator : Integrator
    A fixed-step integrator built from this ODE (see
    :meth:`ODEBase.integrator`).

Returns
-------
VectorFunction
    A type-erased VectorFunction whose input is
    ``[x_left, t_left, u_left, x_right, t_right, u_right, p]``
    and whose output is the state mismatch at the midpoint.
)doc");
}

template <class BaseType, int _XV, int _UV, int _PV>
void BuildGenODEIntegrator(const char *name, nb::module_ &mod, FunctionRegistry &reg) {
    using Derived = GenericODE<BaseType, _XV, _UV, _PV>;
    auto odemod = nb::borrow<nb::module_>(mod.attr(name));
    reg.template build_register<Integrator<Derived>>(odemod, "integrator");
}

} // namespace tycho::bind

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

template <int OR> struct TychoBind<InterpFunction<OR>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<InterpFunction<OR>>(m, name);
        obj.doc() = R"doc(VectorFunction that interpolates selected trajectory variables over time.

Maps a scalar time input to the interpolated values of the requested
state/time/control variables read from an :class:`LGLInterpTable`, with
analytic first and second derivatives.  The compile-time output dimension
``OR`` is the number of variables interpolated; use the dynamic variant
``InterpFunction`` (``OR == -1``) when the selection is not known at
compile time.

Notes
-----
Instances are typically obtained from a solved phase's interpolation table
rather than constructed directly.  The ``__call__`` operator composes this
function with a scalar VectorFunction of time so that the interpolated
trajectory can be embedded directly into NLP expressions.

See Also
--------
LGLInterpTable, PhaseInterface
)doc";
        if constexpr (OR == -1) {
            obj.def(nb::init<std::shared_ptr<LGLInterpTable>, Eigen::VectorXi>(),
                    R"doc(Construct interpolating an explicit subset of trajectory variables.

Parameters
----------
table : LGLInterpTable
    The trajectory interpolation table to read from.
vars : numpy.ndarray of int
    Indices (into the packed ``[x, t, u]`` node vector) of the variables
    to interpolate.

Raises
------
ValueError
    If any index in ``vars`` exceeds the table's ``xtu_vars`` dimension.
)doc");
        } else {
            obj.def(nb::init<std::shared_ptr<LGLInterpTable>>(),
                    R"doc(Construct interpolating all state variables of the trajectory table.

Parameters
----------
table : LGLInterpTable
    The trajectory interpolation table to read from.  Its state dimension
    must match the compile-time output size ``OR``, and it must contain no
    control variables.

Raises
------
ValueError
    If the table's state size differs from ``OR`` or it has control variables.
)doc");
        }
        bind::DenseBaseBuild<InterpFunction<OR>>(obj);
        obj.def(
            "__call__",
            [](const InterpFunction<OR> &self, const GenericFunction<-1, 1> &t) {
                return GenericFunction<-1, -1>(self.eval(t));
            },
            R"doc(Compose this interpolant with a scalar-valued time function.

Parameters
----------
t : ScalarFunction
    A scalar VectorFunction whose single output is interpreted as time.

Returns
-------
VectorFunction
    A new VectorFunction that evaluates the interpolant at the time
    produced by ``t``.
)doc");
        obj.def(
            "__call__",
            [](const InterpFunction<OR> &self, const Segment<-1, 1, -1> &t) {
                return GenericFunction<-1, -1>(self.eval(t));
            },
            "");
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
