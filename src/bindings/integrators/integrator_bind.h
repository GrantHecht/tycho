// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt).
// =============================================================================

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// TychoBind<Integrator<DODE>> specialization + IntegratorBuildConstructors
// free function (nanobind binding for the Integrator template).

#include "dense_function_base_bind.h"

namespace tycho {

using namespace tycho::vf;
using namespace tycho::integrators;

namespace bind {

inline IVPAlg parse_ivp_alg(const std::string &str) {
    if (str == "DOPRI54" || str == "DP54")
        return IVPAlg::DOPRI54;
    if (str == "DOPRI87" || str == "DP87")
        return IVPAlg::DOPRI87;
    if (str == "Tsit5")
        return IVPAlg::Tsit5;
    if (str == "BS3")
        return IVPAlg::BS3;
    if (str == "BS5")
        return IVPAlg::BS5;
    if (str == "Vern7")
        return IVPAlg::Vern7;
    if (str == "Vern8")
        return IVPAlg::Vern8;
    if (str == "Vern9")
        return IVPAlg::Vern9;
    throw std::invalid_argument(fmt::format(
        "Unknown IVP algorithm: '{}'; accepted values: DOPRI54, DP54, DOPRI87, DP87, Tsit5, BS3, "
        "BS5, Vern7, Vern8, Vern9",
        str));
}

inline IVPController parse_ivp_controller(const std::string &str) {
    if (str == "I")
        return IVPController::I;
    if (str == "PI")
        return IVPController::PI;
    if (str == "PID")
        return IVPController::PID;
    throw std::invalid_argument(
        fmt::format("Unknown IVPController: '{}'; accepted values: I, PI, PID", str));
}

inline ErrorNormType parse_error_norm(const std::string &str) {
    if (str == "RMS")
        return ErrorNormType::RMS;
    if (str == "MAX")
        return ErrorNormType::MAX;
    throw std::invalid_argument(
        fmt::format("Unknown ErrorNormType: '{}'; accepted values: RMS, MAX", str));
}

template <class DODE, class PyDODE> void IntegratorBuildConstructors(PyDODE &obj) {
    constexpr const char *kIntegratorDoc = R"doc(Create an adaptive-step Integrator bound to this ODE.

Parameters
----------
dt : float
    Initial (nominal) step size.  Must be positive.  The adaptive controller
    may grow or shrink it after the first step; this value is used as the
    starting guess and is also consulted when ``adaptive=False``.
method : IVPAlg or str, optional
    Runge-Kutta algorithm to use.  Accepted strings: ``"DOPRI54"``,
    ``"DP54"``, ``"DOPRI87"``, ``"DP87"``, ``"Tsit5"``, ``"BS3"``,
    ``"BS5"``, ``"Vern7"``, ``"Vern8"``, ``"Vern9"``.  Defaults to
    ``IVPAlg.DOPRI87`` when omitted.
control_fn : VectorFunction or LGLInterpTable, optional
    Control law evaluated at each step (only for ODEs with control inputs).
var_locs : int, array-like of int, str, or list of str, optional
    Indices (or named variables) in the ODE input vector that the control
    law writes to.  Required when ``control_fn`` is a ``VectorFunction``
    that does not already carry location information.

Returns
-------
Integrator
    An ``Integrator`` object configured to integrate this ODE.

Examples
--------
>>> integ = ode.integrator(0.1)
>>> integ = ode.integrator("Tsit5", 0.05)
>>> integ = ode.integrator(0.1, ctrl_fn, range(0, 3))
)doc";

    obj.def("integrator", [](const DODE &od, double ds) { return Integrator<DODE>(od, ds); },
            kIntegratorDoc);

    // IVPAlg enum overload
    obj.def("integrator",
            [](const DODE &od, IVPAlg meth, double ds) { return Integrator<DODE>(od, meth, ds); },
            kIntegratorDoc);
    // String overload — converts to IVPAlg at the binding layer
    obj.def("integrator",
            [](const DODE &od, std::string meth, double ds) {
                return Integrator<DODE>(od, parse_ivp_alg(meth), ds);
            },
            kIntegratorDoc);
    if constexpr (DODE::UV != 0) {
        obj.def("integrator",
                [](const DODE &od, IVPAlg meth, double ds,
                   const typename Integrator<DODE>::ControllerType &u,
                   const typename Integrator<DODE>::ControlIndexType &v) {
                    return Integrator<DODE>(od, meth, ds, u, v);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds,
                   const typename Integrator<DODE>::ControllerType &u,
                   const typename Integrator<DODE>::ControlIndexType &v) {
                    return Integrator<DODE>(od, parse_ivp_alg(meth), ds, u, v);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, IVPAlg meth, double ds, std::shared_ptr<LGLInterpTable> u,
                   const Eigen::VectorXi &v) { return Integrator<DODE>(od, meth, ds, u, v); },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u,
                   const Eigen::VectorXi &v) {
                    return Integrator<DODE>(od, parse_ivp_alg(meth), ds, u, v);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, IVPAlg meth, double ds, std::shared_ptr<LGLInterpTable> u) {
                    return Integrator<DODE>(od, meth, ds, u);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u) {
                    return Integrator<DODE>(od, parse_ivp_alg(meth), ds, u);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, double ds, const typename Integrator<DODE>::ControllerType &u,
                   const typename Integrator<DODE>::ControlIndexType &v) {
                    return Integrator<DODE>(od, ds, u, v);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, double ds, const Eigen::VectorXd &v) {
                    return Integrator<DODE>(od, ds, v);
                },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, double ds, std::shared_ptr<LGLInterpTable> u,
                   const Eigen::VectorXi &v) { return Integrator<DODE>(od, ds, u, v); },
                kIntegratorDoc);
        obj.def("integrator",
                [](const DODE &od, double ds, std::shared_ptr<LGLInterpTable> u) {
                    return Integrator<DODE>(od, ds, u);
                },
                kIntegratorDoc);
    }
}

} // namespace bind

template <class DODE> struct TychoBind<Integrator<DODE>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Integrator<DODE>>(m, name);

        using ODEStateD = typename DODE::template Input<double>;
        using IntegRet = typename Integrator<DODE>::IntegRet;
        using IntegEventRet = typename Integrator<DODE>::IntegEventRet;
        using DenseRet = typename Integrator<DODE>::DenseRet;
        using DenseEventRet = typename Integrator<DODE>::DenseEventRet;
        using STMRet = typename Integrator<DODE>::STMRet;
        using STMEventRet = typename Integrator<DODE>::STMEventRet;
        using EventPack = typename Integrator<DODE>::EventPack;

        obj.def(nb::init<const DODE &, IVPAlg, double>());
        obj.def(nb::init<const DODE &, double>());

        if constexpr (DODE::UV != 0) {
            obj.def(nb::init<const DODE &, IVPAlg, double,
                             const typename Integrator<DODE>::ControllerType &,
                             const typename Integrator<DODE>::ControlIndexType &>());
            obj.def(
                nb::init<const DODE &, double, const typename Integrator<DODE>::ControllerType &,
                         const typename Integrator<DODE>::ControlIndexType &>());
            obj.def(nb::init<const DODE &, double, const Eigen::VectorXd &>());

            obj.def(nb::init<const DODE &, IVPAlg, double, std::shared_ptr<LGLInterpTable>,
                             const Eigen::VectorXi &>());
            obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>,
                             const Eigen::VectorXi &>());

            obj.def(nb::init<const DODE &, IVPAlg, double, std::shared_ptr<LGLInterpTable>>());
            obj.def(nb::init<const DODE &, double, std::shared_ptr<LGLInterpTable>>());
        }

        obj.def("integrate",
                (IntegRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate,
                nb::arg("xt0_up"), nb::arg("tf"),
                R"doc(Integrate from an initial condition to a final time.

Parameters
----------
xt0_up : array_like
    Concatenated state-and-time (and optional parameter) vector at the start
    of the integration interval, i.e. ``[x0..., t0]``.
tf : float
    Final time to integrate to.

Returns
-------
numpy.ndarray
    Final ODE state-and-time vector at ``tf``.
)doc");

        obj.def("integrate",
                (IntegEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                     const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"),
                R"doc(Integrate from an initial condition to a final time, stopping at events.

Each element of ``events`` is an :class:`EventPack` (or a 3-tuple
``(event_fn, direction, stop_count)``):

* ``event_fn`` — scalar :class:`VectorFunction` evaluated along the
  trajectory; a root (sign change) triggers the event.
* ``direction`` — ``0`` for any crossing, ``+1`` for rising only,
  ``-1`` for falling only.
* ``stop_count`` — number of event detections after which integration
  stops early.  ``0`` means never stop.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Nominal final time (may be reached early if an event halts integration).
events : list of EventPack or list of tuple
    Event specifications.

Returns
-------
tuple
    ``(final_state, event_locs)`` where ``event_locs`` is a list-of-lists
    of optional state vectors, one inner list per event.
)doc");

        obj.def("integrate",
                (std::vector<IntegRet> (Integrator<DODE>::*)(const std::vector<ODEStateD> &,
                                                             const Eigen::VectorXd &) const) &
                    Integrator<DODE>::integrate,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch of initial conditions serially.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.

Returns
-------
list of numpy.ndarray
    Final state-and-time vector for each initial condition.
)doc");

        obj.def("integrate_stm",
                (std::vector<STMRet> (Integrator<DODE>::*)(const std::vector<ODEStateD> &,
                                                           const Eigen::VectorXd &) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch with the state-transition matrix (STM), serially.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.

Returns
-------
list of tuple
    Each element is ``(final_state, STM)`` where ``STM`` is the
    ``(n x n)`` state-transition matrix at the final time.
)doc");

        obj.def("integrate_stm2", &Integrator<DODE>::integrate_stm2, nb::arg("xt0_ups"),
                nb::arg("tfs"), nb::arg("lfs"), nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch with the STM and second-order sensitivity (Hessian), serially.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
lfs : list of array_like
    Adjoint (co-state) vectors at the final time, one per trajectory.
    Used to compute the adjoint-vector product for the Hessian.

Returns
-------
list of tuple
    Each element is ``(final_state, STM, Hessian)`` — the final state,
    state-transition matrix, and second-order sensitivity matrix.
)doc");

        obj.def("integrate_parallel",
                (std::vector<IntegRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch of initial conditions in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
threads : int
    Number of threads to use.

Returns
-------
list of numpy.ndarray
    Final state-and-time vector for each initial condition.
)doc");

        obj.def("integrate_parallel",
                (std::vector<IntegEventRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("events"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch in parallel, stopping at events.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).
threads : int
    Number of threads to use.

Returns
-------
list of tuple
    Each element is ``(final_state, event_locs)`` — the final state and
    per-event crossing locations.
)doc");

        ////////////////////////////////////////////////////////////////////////////

        obj.def("integrate_dense",
                (DenseRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"),
                R"doc(Integrate and return the full trajectory.

Each element of the returned list is the ODE state-and-time vector at one
of the adaptive steps taken by the integrator.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Final time.

Returns
-------
list of numpy.ndarray
    Trajectory: state-and-time vector at each accepted step, including
    the final state at ``tf``.
)doc");

        obj.def("integrate_dense",
                (DenseRet (Integrator<DODE>::*)(const ODEStateD &, double, int) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("n"),
                R"doc(Integrate and return the trajectory at ``n`` evenly-spaced output times.

The integrator takes adaptive steps internally; the dense output (polynomial
interpolant of the RK solution) is then evaluated at ``n`` equally-spaced
times from ``t0`` to ``tf``.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Final time.
n : int
    Number of output states (including the initial and final points).

Returns
-------
list of numpy.ndarray
    Trajectory with exactly ``n`` state-and-time vectors.
)doc");

        obj.def(
            "integrate_dense",
            [](const Integrator<DODE> &integ, const ODEStateD &x0, double tf, int n,
               nb::object pyfunc) {
                return integ.integrate_dense(x0, tf, n,
                                             [pyfunc](ConstEigenRef<Eigen::VectorXd> x) -> bool {
                                                 // pyfunc(x) raises nb::python_error if the Python
                                                 // callable itself raises (nanobind converts the
                                                 // Python exception). PyObject_IsTrue accepts any
                                                 // truthy object — including numpy scalar bools
                                                 // that users commonly return (e.g. `x[1] < 0`). An
                                                 // error return (-1) from PyObject_IsTrue leaves
                                                 // the Python error indicator set; propagate as
                                                 // nb::python_error so the caller sees a clean
                                                 // exception instead of a silent truncation (the
                                                 // coerce -1→true trap).
                                                 nb::object result = pyfunc(x);
                                                 int truth = PyObject_IsTrue(result.ptr());
                                                 if (truth == -1)
                                                     throw nb::python_error();
                                                 return truth != 0;
                                             });
            },
            nb::arg("xt0_up"), nb::arg("tf"), nb::arg("n"), nb::arg("stop_func"),
            R"doc(Integrate with ``n`` output states, stopping when a predicate returns true.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Nominal final time.
n : int
    Number of evenly-spaced output states requested.
stop_func : callable
    Called after each accepted step with the current state-and-time vector.
    Integration stops early when it returns a truthy value.

Returns
-------
list of numpy.ndarray
    Trajectory up to the stopping point or ``tf``, with up to ``n`` states.
)doc");

        obj.def("integrate_dense",
                (DenseEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                     const std::vector<EventPack> &, bool) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"),
                nb::arg("alloutput") = false,
                R"doc(Integrate with dense output and event detection.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Nominal final time.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).
alloutput : bool, optional
    If ``True``, return the full dense trajectory in addition to event
    locations.  If ``False`` (default), return only the final state.

Returns
-------
tuple
    ``(trajectory, event_locs)`` where ``trajectory`` is a list of
    state-and-time vectors and ``event_locs`` is a list-of-lists of
    optional state vectors at detected event crossings.
)doc");

        obj.def("integrate_dense",
                (DenseEventRet (Integrator<DODE>::*)(const ODEStateD &, double, int,
                                                     const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate_dense,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("nstates"), nb::arg("events"),
                R"doc(Integrate with ``nstates`` evenly-spaced output states and event detection.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Nominal final time.
nstates : int
    Number of evenly-spaced output states.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).

Returns
-------
tuple
    ``(trajectory, event_locs)`` where ``trajectory`` contains exactly
    ``nstates`` state-and-time vectors and ``event_locs`` is a list-of-lists
    of optional state vectors at detected event crossings.
)doc");

        ////////////////////////////////////////////////////////////////////////////

        obj.def("integrate_dense_parallel",
                (std::vector<DenseRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_dense_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch with dense output in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
threads : int
    Number of threads to use.

Returns
-------
list of list of numpy.ndarray
    Dense trajectory for each initial condition.
)doc");

        obj.def(
            "integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator<DODE>::*)(
                const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_dense_parallel,
            nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("events"), nb::arg("threads"),
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(Integrate a batch with dense output and event detection in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).
threads : int
    Number of threads to use.

Returns
-------
list of tuple
    Each element is ``(trajectory, event_locs)`` for one initial condition.
)doc");

        obj.def("integrate_dense_parallel",
                (std::vector<DenseRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<int> &, int))&Integrator<DODE>::integrate_dense_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("ns"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch with per-trajectory output-state counts in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
ns : list of int
    Number of evenly-spaced output states for each trajectory.
threads : int
    Number of threads to use.

Returns
-------
list of list of numpy.ndarray
    Dense trajectory for each initial condition, with the requested number
    of output states.
)doc");

        obj.def(
            "integrate_dense_parallel",
            (std::vector<DenseEventRet> (Integrator<DODE>::*)(
                const std::vector<ODEStateD> &, const Eigen::VectorXd &, const std::vector<int> &,
                const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_dense_parallel,
            nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("ns"), nb::arg("events"),
            nb::arg("threads"), nb::call_guard<nb::gil_scoped_release>(),
            R"doc(Integrate a batch with per-trajectory output counts and event detection in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
ns : list of int
    Number of evenly-spaced output states per trajectory.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).
threads : int
    Number of threads to use.

Returns
-------
list of tuple
    Each element is ``(trajectory, event_locs)`` for one initial condition.
)doc");

        /////////////////////////////////////////////////////

        obj.def("integrate_stm",
                (STMRet (Integrator<DODE>::*)(const ODEStateD &, double) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("xt0_up"), nb::arg("tf"),
                R"doc(Integrate a single trajectory and return the state-transition matrix (STM).

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Final time.

Returns
-------
tuple
    ``(final_state, STM)`` — the final state-and-time vector and the
    ``(n x n)`` state-transition matrix ``d(x_f)/d(x_0)``.
)doc");
        obj.def("integrate_stm",
                (STMEventRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                                   const std::vector<EventPack> &) const) &
                    Integrator<DODE>::integrate_stm,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("events"),
                R"doc(Integrate a single trajectory with STM and event detection.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Nominal final time.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).

Returns
-------
tuple
    ``(final_state, STM, event_locs)`` — final state, state-transition
    matrix, and per-event crossing locations.
)doc");
        obj.def("integrate_stm_parallel",
                (STMRet (Integrator<DODE>::*)(const ODEStateD &, double,
                                              int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("xt0_up"), nb::arg("tf"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a single trajectory with STM using parallel columns.

Computes the state-transition matrix by propagating each column of the
identity perturbation in parallel.

Parameters
----------
xt0_up : array_like
    Initial state-and-time vector.
tf : float
    Final time.
threads : int
    Number of threads to use for parallel column propagation.

Returns
-------
tuple
    ``(final_state, STM)`` — the final state and state-transition matrix.
)doc");
        obj.def("integrate_stm_parallel",
                (std::vector<STMRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch with STM in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
threads : int
    Number of threads to use.

Returns
-------
list of tuple
    Each element is ``(final_state, STM)`` for one initial condition.
)doc");
        obj.def("integrate_stm_parallel",
                (std::vector<STMEventRet> (Integrator<DODE>::*)(
                    const std::vector<ODEStateD> &, const Eigen::VectorXd &,
                    const std::vector<EventPack> &, int))&Integrator<DODE>::integrate_stm_parallel,
                nb::arg("xt0_ups"), nb::arg("tfs"), nb::arg("events"), nb::arg("threads"),
                nb::call_guard<nb::gil_scoped_release>(),
                R"doc(Integrate a batch with STM and event detection in parallel.

Parameters
----------
xt0_ups : list of array_like
    Batch of initial state-and-time vectors.
tfs : numpy.ndarray
    Final times, one per initial condition.
events : list of EventPack or list of tuple
    Event specifications (see :meth:`integrate` with events for the format).
threads : int
    Number of threads to use.

Returns
-------
list of tuple
    Each element is ``(final_state, STM, event_locs)`` for one initial condition.
)doc");

        /////////////////////////////////////////////////////

        bind::DenseBaseBuild<Integrator<DODE>>(obj);

        obj.def_rw("enable_vectorization", &Integrator<DODE>::enable_vectorization_,
                   R"doc(Enable SIMD-vectorized batch integration when available (bool, default ``True``).

When ``True``, batch integration methods pack multiple trajectories into
SuperScalar (SIMD) lanes and evaluate the ODE and RK stages in parallel
across the batch.  Only has an effect when the underlying ODE supports
vectorisation (``is_vectorizable = True``).
)doc");

        // def_step_size routes through set_initial_step_size so Python writes
        // respect the documented contract (flips HW auto-initdt off, rejects
        // non-positive values).
        obj.def_prop_rw(
            "def_step_size", [](const Integrator<DODE> &self) { return self.def_step_size_; },
            [](Integrator<DODE> &self, double h) { self.set_initial_step_size(h); },
            R"doc(Nominal initial step size (float).

Setting this property calls :meth:`set_initial_step_size`, which disables the
Hairer-Wanner automatic initial step-size selection and validates that ``h``
is positive.
)doc");
        obj.def_prop_rw(
            "max_step_change",
            [](const Integrator<DODE> &self) { return self.get_max_step_change(); },
            [](Integrator<DODE> &self, double v) { self.set_max_step_change(v); },
            R"doc(Maximum per-step growth ratio ``|dt_new / dt_old|`` (float, default ``10.0``).

Limits how aggressively the adaptive controller can increase the step size
between consecutive accepted steps.  Must be finite and greater than 1.
)doc");

        obj.def_rw("adaptive", &Integrator<DODE>::adaptive_,
                   R"doc(Enable adaptive step-size control (bool, default ``True``).

When ``True`` the step-size controller (see :meth:`set_controller`) adjusts
``dt`` after each accepted step to target the requested error tolerances.
When ``False`` a fixed step size equal to :attr:`def_step_size` is used.
)doc");
        // abs_tols routes through set_abs_tols so Python writes validate size
        // and non-negativity.
        obj.def_prop_rw(
            "abs_tols", [](const Integrator<DODE> &self) { return self.get_abs_tols(); },
            [](Integrator<DODE> &self, typename Integrator<DODE>::template ODEDeriv<double> tol) {
                self.set_abs_tols(tol);
            },
            R"doc(Per-state absolute error tolerances (array, default ``1e-12`` per element).

Setting this property calls :meth:`set_abs_tols`, which validates that the
vector has the correct length and that all values are non-negative.
)doc");

        obj.def("set_abs_tol", &Integrator<DODE>::set_abs_tol,
                R"doc(Set a single absolute tolerance for all ODE states.

Parameters
----------
tol : float
    Absolute tolerance applied uniformly to every state component.
    Must be non-negative.
)doc");
        obj.def("set_abs_tols", &Integrator<DODE>::set_abs_tols,
                R"doc(Set per-state absolute tolerances.

Parameters
----------
tols : array_like
    Vector of absolute tolerances, one per ODE state.  All values must be
    non-negative and the length must match the ODE state dimension.
)doc");
        obj.def("get_abs_tols", &Integrator<DODE>::get_abs_tols,
                R"doc(Return the current per-state absolute tolerances.

Returns
-------
numpy.ndarray
    Absolute tolerance vector (length equals the ODE state dimension).
)doc");

        obj.def("set_rel_tol", &Integrator<DODE>::set_rel_tol,
                R"doc(Set a single relative tolerance for all ODE states.

Parameters
----------
tol : float
    Relative tolerance applied uniformly to every state component.
    Must be non-negative.
)doc");
        obj.def("set_rel_tols", &Integrator<DODE>::set_rel_tols,
                R"doc(Set per-state relative tolerances.

Parameters
----------
tols : array_like
    Vector of relative tolerances, one per ODE state.  All values must be
    non-negative and the length must match the ODE state dimension.
)doc");
        obj.def("get_rel_tols", &Integrator<DODE>::get_rel_tols,
                R"doc(Return the current per-state relative tolerances.

Returns
-------
numpy.ndarray
    Relative tolerance vector (length equals the ODE state dimension).
)doc");

        obj.def("set_initial_step_size", &Integrator<DODE>::set_initial_step_size, nb::arg("h"),
                R"doc(Set the initial step size and disable automatic step-size initialisation.

Calling this method fixes the first step to ``h`` and turns off the
Hairer-Wanner automatic initial step-size selection.  The adaptive controller
continues to adjust the step after the first accepted step.

Parameters
----------
h : float
    Initial step size.  Must be positive and finite.
)doc");
        obj.def("set_max_steps", &Integrator<DODE>::set_max_steps, nb::arg("n"),
                R"doc(Set the maximum number of accepted steps per integration call.

If the integrator reaches this limit before ``tf``, it raises an error.

Parameters
----------
n : int
    Maximum number of accepted steps (must be positive).
)doc");
        obj.def("get_max_steps", &Integrator<DODE>::get_max_steps,
                R"doc(Return the current maximum-steps limit.

Returns
-------
int
    Maximum number of accepted steps per integration call.
)doc");

        obj.def_prop_rw(
            "event_tol", [](const Integrator<DODE> &self) { return self.get_event_tol(); },
            [](Integrator<DODE> &self, double v) { self.set_event_tol(v); },
            R"doc(Absolute time tolerance for event root-finding (float, default ``1e-12``).

The bisection / secant loop that locates an event crossing stops when the
bracketing interval width falls below this value.
)doc");
        obj.def_prop_rw(
            "max_event_iters",
            [](const Integrator<DODE> &self) { return self.get_max_event_iters(); },
            [](Integrator<DODE> &self, int n) { self.set_max_event_iters(n); },
            R"doc(Maximum iterations for event root-finding (int, default ``50``).

If the root-finding loop has not converged within this many iterations and
the interval has not reached :attr:`event_tol`, integration stops at the
best current estimate of the event location.
)doc");
        obj.def_rw("vectorize_batch_calls", &Integrator<DODE>::vectorize_batch_calls_,
                   R"doc(Use vectorized (SuperScalar) evaluation for batch integration calls (bool, default ``True``).

When ``True`` and :attr:`enable_vectorization` is also ``True``, batch
methods pack multiple trajectories into SIMD lanes per call.  Disable
to force scalar per-trajectory evaluation (useful for debugging or when
the ODE is not vectorisable).
)doc");

        // Controller + stats + HW-initdt API
        obj.def("set_controller", &Integrator<DODE>::set_controller,
                R"doc(Set the adaptive step-size controller type.

Parameters
----------
controller : IVPController
    Step-size controller enum value: ``IVPController.I``,
    ``IVPController.PI``, or ``IVPController.PID``.
)doc");
        obj.def("set_controller", [](Integrator<DODE> &self, const std::string &s) {
            self.set_controller(bind::parse_ivp_controller(s));
        },
                R"doc(Set the adaptive step-size controller by name.

Parameters
----------
controller : str
    Controller name: ``"I"``, ``"PI"``, or ``"PID"``.
)doc");
        obj.def("get_controller", &Integrator<DODE>::get_controller,
                R"doc(Return the current step-size controller type.

Returns
-------
IVPController
    The active controller enum value.
)doc");

        obj.def("set_error_norm", &Integrator<DODE>::set_error_norm,
                R"doc(Set the error norm used by the adaptive controller.

Parameters
----------
norm : ErrorNormType
    Error norm enum value: ``ErrorNormType.RMS`` (root-mean-square) or
    ``ErrorNormType.MAX`` (maximum absolute value).
)doc");
        obj.def("set_error_norm", [](Integrator<DODE> &self, const std::string &s) {
            self.set_error_norm(bind::parse_error_norm(s));
        },
                R"doc(Set the error norm by name.

Parameters
----------
norm : str
    Error norm name: ``"RMS"`` (root-mean-square) or ``"MAX"``
    (maximum absolute value).
)doc");
        obj.def("get_error_norm", &Integrator<DODE>::get_error_norm,
                R"doc(Return the current error norm type.

Returns
-------
ErrorNormType
    The active error norm enum value.
)doc");

        obj.def("get_naccept", &Integrator<DODE>::get_naccept,
                R"doc(Return the number of accepted steps from the last integration call.

Returns
-------
int
    Accepted-step count.  Reset at the start of each integration call.
)doc");
        obj.def("get_nreject", &Integrator<DODE>::get_nreject,
                R"doc(Return the number of rejected steps from the last integration call.

Returns
-------
int
    Rejected-step count (steps where the error estimate exceeded the
    tolerance).  Reset at the start of each integration call.
)doc");
        obj.def("get_failed_event_count", &Integrator<DODE>::get_failed_event_count,
                R"doc(Return the number of events that failed to converge in the last call.

An event "fails" when the root-finding loop reaches :attr:`max_event_iters`
without meeting :attr:`event_tol`.  The integrator continues from the best
available estimate.

Returns
-------
int
    Number of non-converged event localisations in the last call.
)doc");

        obj.def("set_auto_initial_dt", &Integrator<DODE>::set_auto_initial_dt,
                R"doc(Enable or disable automatic initial step-size selection.

When enabled (the default), the integrator uses the Hairer-Wanner algorithm
to estimate a good first step based on the ODE Lipschitz constant.  Calling
:meth:`set_initial_step_size` or writing :attr:`def_step_size` disables this
automatically.

Parameters
----------
on : bool
    ``True`` to enable automatic selection; ``False`` to use
    :attr:`def_step_size` as the first step.
)doc");
        obj.def("get_auto_initial_dt", &Integrator<DODE>::get_auto_initial_dt,
                R"doc(Return whether automatic initial step-size selection is enabled.

Returns
-------
bool
    ``True`` if the Hairer-Wanner initial step selection is active.
)doc");
    }
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
