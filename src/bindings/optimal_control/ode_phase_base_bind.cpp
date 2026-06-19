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

#include "ode_phase_bind.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using VectorFunctionalX = GenericFunction<-1, -1>;
using ScalarFunctionalX = GenericFunction<-1, 1>;
using StateConstraint = StateFunction<VectorFunctionalX>;
using StateObjective = StateFunction<ScalarFunctionalX>;

void TychoBind<ODEPhaseBase>::build(nb::module_ &m) {
    auto obj = nb::class_<ODEPhaseBase, OptimizationProblemBase>(m, "PhaseInterface");
    obj.doc() = R"doc(A single phase of an optimal control problem.

A phase couples an ODE, a discretized trajectory, and the boundary values,
bounds, constraints, and objectives that transcribe a continuous optimal
control problem into a finite-dimensional nonlinear program. The
``transcription_mode`` selects how the dynamics are enforced between nodes:
direct collocation (the LGL schemes and ``Trapezoidal``) or central shooting
(``CentralShooting``), which numerically propagates both bounding nodes of each
interval to its midpoint and constrains the two results to agree. Phases are not
constructed directly; build one from an ODE with
``ode.phase(transcription_mode, traj, num_segments)`` (see
:class:`~tychopy.optimal_control.ODEBase`), then attach problem data with the
``add_*`` methods below.

Variables are addressed within a region of the phase. A *region* (see
:class:`PhaseRegionFlags`) selects which discretized state(s) a function or
bound applies to -- e.g. ``"Front"`` (initial node), ``"Back"`` (final node),
or ``"Path"`` (every node). Region selectors accept either the enum value or
its string name. Variable indices refer to the packed node layout
``[x, t, u, p]``.

Once configured, solve the phase with :meth:`optimize` (or :meth:`solve`),
then retrieve the result with :meth:`return_traj`.

Notes
-----
:meth:`optimize`, :meth:`solve`, and the ``optimizer`` handle are inherited
from the shared optimization-problem base; ``phase.optimizer`` exposes the
underlying PSIOPT solver (e.g. ``phase.optimizer.set_print_level(0)``).

Examples
--------
Build a phase, constrain it, solve, and read the trajectory::

    phase = ode.phase("LGL3", Xs, 32)
    phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
    phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
    phase.add_boundary_value("Back", [0, 1], [xf, yf])
    phase.add_delta_time_objective(1.0)
    phase.optimize()
    traj = phase.return_traj()
)doc";

    auto set_mesh_error_estimator = [](ODEPhaseBase &self, nb::object val) {
        if (nb::isinstance<MeshErrorEstimators>(val))
            self.mesh_error_estimator_ = nb::cast<MeshErrorEstimators>(val);
        else if (nb::isinstance<nb::str>(val))
            self.mesh_error_estimator_ = strto_MeshErrorEstimator(nb::cast<std::string>(val));
        else
            throw nb::type_error("expected MeshErrorEstimators enum or str");
    };
    auto set_mesh_error_aggregation = [](MeshErrorAggregation &target, nb::object val,
                                         const char *name) {
        if (nb::isinstance<MeshErrorAggregation>(val))
            target = nb::cast<MeshErrorAggregation>(val);
        else if (nb::isinstance<nb::str>(val))
            target = strto_MeshErrorAggregation(nb::cast<std::string>(val));
        else
            throw nb::type_error(
                fmt::format("expected MeshErrorAggregation enum or str for {}", name).c_str());
    };

    obj.def("enable_vectorization", &ODEPhaseBase::enable_vectorization,
            R"doc(Enable or disable SuperScalar (vectorized) function evaluation.

Parameters
----------
b : bool
    Pass ``True`` to enable vectorization (the default), ``False`` to disable.
)doc");

    obj.def(
        "set_traj",
        nb::overload_cast<const std::vector<Eigen::VectorXd> &, Eigen::VectorXd, Eigen::VectorXi>(
            &ODEPhaseBase::set_traj),
        "");

    obj.def("set_traj", nb::overload_cast<const std::vector<Eigen::VectorXd> &, Eigen::VectorXd,
                                          Eigen::VectorXi, bool>(&ODEPhaseBase::set_traj));

    obj.def("set_traj",
            nb::overload_cast<const std::vector<Eigen::VectorXd> &, int>(&ODEPhaseBase::set_traj),
            "");

    obj.def("set_traj", nb::overload_cast<const std::vector<Eigen::VectorXd> &, int, bool>(
                            &ODEPhaseBase::set_traj));

    obj.def("set_traj",
            nb::overload_cast<const std::vector<Eigen::VectorXd> &>(&ODEPhaseBase::set_traj),
            R"doc(Set (or replace) the phase's initial-guess trajectory.

Parameters
----------
mesh : list of numpy.ndarray
    Trajectory nodes, each packed as ``[x, t, u, p]``, in increasing time
    order. This becomes the initial guess that the optimizer refines.

Notes
-----
Additional overloads accept a defect-segment count or an explicit
defects-binding/defects-per-block specification; see the C++ reference.
)doc");

    obj.def("switch_transcription_mode",
            nb::overload_cast<TranscriptionModes, VectorXd, VectorXi>(
                &ODEPhaseBase::switch_transcription_mode),
            R"doc(Switch the transcription scheme and reload the trajectory onto a new mesh.

Parameters
----------
mode : TranscriptionModes or str
    The new transcription scheme (e.g. ``"LGL3"``, ``"LGL5"``, ``"Trapezoidal"``,
    ``"CentralShooting"``).
bin_times : numpy.ndarray, optional
    Normalized defect-bin boundary times in ``[0, 1]``.
bin_segments : numpy.ndarray of int, optional
    Number of defect intervals to place in each bin.

Notes
-----
If ``bin_times`` and ``bin_segments`` are omitted the current mesh spacing is
preserved. Invalidates the cached transcription and post-solve data.
)doc");
    obj.def("switch_transcription_mode",
            nb::overload_cast<TranscriptionModes>(&ODEPhaseBase::switch_transcription_mode), "");

    obj.def("switch_transcription_mode",
            nb::overload_cast<std::string, VectorXd, VectorXi>(
                &ODEPhaseBase::switch_transcription_mode),
            "");
    obj.def("switch_transcription_mode",
            nb::overload_cast<std::string>(&ODEPhaseBase::switch_transcription_mode), "");

    obj.def("transcribe", nb::overload_cast<bool, bool>(&ODEPhaseBase::transcribe),
            R"doc(Transcribe the phase into its underlying nonlinear program.

Builds the transcribed NLP (defect constraints, user constraints, and
objectives) from the current configuration. This is performed automatically
by :meth:`optimize`/:meth:`solve`; call it explicitly only to inspect the
transcribed problem or to force a rebuild.

Parameters
----------
showstats : bool
    Print transcription statistics (variable/constraint counts).
showfuns : bool
    Print the set of transcribed functions.
)doc");

    obj.def("refine_traj_manual", nb::overload_cast<int>(&ODEPhaseBase::refine_traj_manual),
            R"doc(Resample the current trajectory onto a new number of segments.

Parameters
----------
num : int
    New segment count to interpolate the existing trajectory onto.
)doc");
    obj.def("refine_traj_manual",
            nb::overload_cast<VectorXd, VectorXi>(&ODEPhaseBase::refine_traj_manual),
            R"doc(Resample the current trajectory onto an explicit mesh specification.

Parameters
----------
bin_times : numpy.ndarray
    Normalized bin boundary times in ``[0, 1]``.
bin_segments : numpy.ndarray of int
    Number of segments to place in each bin.
)doc");
    obj.def("refine_traj_equal", &ODEPhaseBase::refine_traj_equal, nb::arg("n"),
            R"doc(Resample the current trajectory onto an equally-spaced mesh.

Parameters
----------
n : int
    Number of equally-spaced segments (defect intervals) to divide the mesh
    into.

Returns
-------
list of numpy.ndarray
    The re-interpolated trajectory on the new equally-spaced mesh.
)doc");

    obj.def("set_static_params",
            nb::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::set_static_params),
            R"doc(Set the static (non-ODE) parameter vector and its per-element scaling units.

Parameters
----------
parm : numpy.ndarray
    Static-parameter values.
units : numpy.ndarray
    Per-parameter scaling units; must be the same length as ``parm``.

Notes
-----
Calling this overload with a ``units`` vector replaces any previously
registered static parameters and invalidates the cached transcription.
An overload that accepts only ``parm`` (with unit scaling) is also available.
)doc");
    obj.def("set_static_params", nb::overload_cast<VectorXd>(&ODEPhaseBase::set_static_params), "");

    obj.def("add_static_params",
            nb::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::add_static_params),
            R"doc(Append additional static parameters and their per-element scaling units.

Parameters
----------
parm : numpy.ndarray
    Static-parameter values to append.
units : numpy.ndarray
    Per-parameter scaling units for the appended values; must match ``parm`` in size.

Notes
-----
An overload without ``units`` (unit scaling assumed) is also available.
)doc");
    obj.def("add_static_params", nb::overload_cast<VectorXd>(&ODEPhaseBase::add_static_params), "");
    obj.def("add_static_param_vgroups",
            nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                &ODEPhaseBase::add_static_param_vgroups),
            R"doc(Merge additional named static-parameter variable groups into the phase.

Parameters
----------
spidxs : dict[str, numpy.ndarray of int]
    Map of group name to static-parameter index array. Existing groups with the
    same name are overwritten.
)doc");
    obj.def("set_static_param_vgroups",
            nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                &ODEPhaseBase::set_static_param_vgroups),
            R"doc(Replace all named static-parameter variable groups.

Parameters
----------
spidxs : dict[str, numpy.ndarray of int]
    Map of group name to static-parameter index array; replaces any previously
    registered groups entirely.
)doc");
    obj.def("add_static_param_vgroup",
            nb::overload_cast<Eigen::VectorXi, std::string>(&ODEPhaseBase::add_static_param_vgroup),
            R"doc(Register a named group of static-parameter indices.

Parameters
----------
idx : numpy.ndarray of int
    Static-parameter indices that belong to this group.
key : str
    Name of the group (used with string-based variable selectors).
)doc");
    obj.def("add_static_param_vgroup",
            nb::overload_cast<int, std::string>(&ODEPhaseBase::add_static_param_vgroup), "");

    obj.def("set_control_mode", nb::overload_cast<ControlModes>(&ODEPhaseBase::set_control_mode),
            R"doc(Set the control representation used during transcription.

Parameters
----------
mode : ControlModes or str
    Control mode, e.g. ``"FirstOrderSpline"`` (default) or ``"BlockConstant"``.
    Invalidates the cached transcription.
)doc");
    obj.def("set_control_mode", nb::overload_cast<std::string>(&ODEPhaseBase::set_control_mode),
            "");

    obj.def("set_integral_mode", &ODEPhaseBase::set_integral_mode,
            R"doc(Set the quadrature rule used for integral objectives and parameter functions.

Parameters
----------
mode : IntegralModes
    Integral quadrature mode. Invalidates the cached transcription.
)doc");

    obj.def("sub_static_params", &ODEPhaseBase::sub_static_params,
            R"doc(Update static-parameter values in-place without triggering re-transcription.

If the size of ``parm`` matches the current static-parameter count the values are
substituted directly; otherwise the parameters are fully replaced (which does
invalidate the transcription).

Parameters
----------
parm : numpy.ndarray
    New static-parameter values.
)doc");

    obj.def("sub_variables",
            nb::overload_cast<PhaseRegionFlags, VectorXi, VectorXd>(&ODEPhaseBase::sub_variables),
            R"doc(Substitute fixed values into selected trajectory variables at a region.

Parameters
----------
phase_region : PhaseRegionFlags or str
    The region (e.g. ``"Front"``, ``"Back"``, ``"Path"``) whose variables are
    updated.
indices : numpy.ndarray of int
    Variable indices (into the packed ``[x, t, u, p]`` layout) to overwrite.
vals : numpy.ndarray
    Replacement values; must match ``indices`` in length.
)doc");
    obj.def("sub_variable",
            nb::overload_cast<PhaseRegionFlags, int, double>(&ODEPhaseBase::sub_variable),
            R"doc(Substitute a fixed value into a single trajectory variable at a region.

Parameters
----------
phase_region : PhaseRegionFlags or str
    The region whose variable is updated.
var : int
    Variable index (into the packed ``[x, t, u, p]`` layout) to overwrite.
val : float
    Replacement value.
)doc");

    obj.def("sub_variables",
            nb::overload_cast<std::string, VectorXi, VectorXd>(&ODEPhaseBase::sub_variables), "");
    obj.def("sub_variable",
            nb::overload_cast<std::string, int, double>(&ODEPhaseBase::sub_variable), "");

    obj.def("return_traj", &ODEPhaseBase::return_traj,
            R"doc(Return the current discretized trajectory.

Returns
-------
list of numpy.ndarray
    One packed state vector ``[x, t, u, p]`` per node, in time order. After a
    successful :meth:`optimize`/:meth:`solve` this is the optimized solution.

Examples
--------
Solve the phase and read back the optimized trajectory::

    phase.optimize()
    traj = phase.return_traj()
)doc");
    obj.def("return_traj_range", &ODEPhaseBase::return_traj_range,
            R"doc(Return the trajectory resampled to ``num`` points over a time range.

Parameters
----------
num : int
    Number of output samples.
tl : float
    Lower time bound.
th : float
    Upper time bound.

Returns
-------
list of numpy.ndarray
    ``num`` interpolated node vectors spanning ``[tl, th]``.
)doc");
    obj.def("return_traj_range_nd", &ODEPhaseBase::return_traj_range_nd,
            R"doc(Resample the trajectory over a non-dimensional (fractional) time range.

Resamples the trajectory into ``num`` equally-spaced segments spanning the
fractional time range ``[tl, th]``, where ``0.0`` is the start of the
trajectory and ``1.0`` is the end. Nodes within each segment follow the
transcription's LGL node spacing, so the returned list is longer than ``num``.

Parameters
----------
num : int
    Number of equally-spaced segments to divide the range into.
tl : float
    Lower non-dimensional (fractional) time bound in ``[0, 1]``.
th : float
    Upper non-dimensional (fractional) time bound in ``[0, 1]``.

Returns
-------
list of numpy.ndarray
    The resampled node vectors spanning the fractional range ``[tl, th]``.
)doc");
    obj.def("return_traj_table", &ODEPhaseBase::return_traj_table,
            R"doc(Return an LGL interpolation table built from the exact current trajectory.

Returns
-------
LGLInterpTable
    An interpolation table loaded with the exact (non-interpolated) trajectory
    data. Useful for custom dense output queries.
)doc");

    obj.def("return_costate_traj", &ODEPhaseBase::return_costate_traj,
            R"doc(Return the costate (adjoint/dual) trajectory.

Returns
-------
list of numpy.ndarray
    One costate vector per trajectory node (same count as :meth:`return_traj`),
    each of length ``x_vars + 1`` -- the ``x_vars`` costate components plus a
    time entry at the time index. Costates are recovered from the
    dynamics-defect Lagrange multipliers, which are defined at the defect
    points and then linearly interpolated (extrapolated at the endpoints) onto
    the trajectory nodes. Only meaningful after a successful
    :meth:`optimize`/:meth:`solve`.
)doc");
    obj.def("return_traj_error", &ODEPhaseBase::return_traj_error,
            R"doc(Return the discretization-error estimate of the current trajectory.

Returns
-------
list of numpy.ndarray
    One error vector per interior collocation point (``num_card_states - 1``
    per defect interval, so the count is the number of trajectory nodes minus
    one). Each vector has length ``x_vars + 1`` -- the ``x_vars`` per-state
    error components plus a time entry. Only available after a successful
    :meth:`optimize`/:meth:`solve` (raises ``RuntimeError`` otherwise).
)doc");

    obj.def("return_u_spline_con_lmults", &ODEPhaseBase::return_u_spline_con_lmults,
            R"doc(Return the control-spline constraint Lagrange multipliers from the last solve.

Returns
-------
list of numpy.ndarray
    Per-node multiplier vectors for the control-spline continuity constraint,
    or an empty list if no control-spline constraint is active. Only meaningful
    after a successful :meth:`optimize`/:meth:`solve`.
)doc");
    obj.def("return_u_spline_con_vals", &ODEPhaseBase::return_u_spline_con_vals,
            R"doc(Return the control-spline constraint residual values from the last solve.

Returns
-------
list of numpy.ndarray
    Per-node residual vectors for the control-spline continuity constraint,
    or an empty list if no control-spline constraint is active. Only meaningful
    after a successful :meth:`optimize`/:meth:`solve`.
)doc");

    obj.def("return_equal_con_lmults", &ODEPhaseBase::return_equal_con_lmults,
            R"doc(Return the Lagrange multipliers of an equality constraint from the last solve.

Parameters
----------
index : int
    The equality-constraint index returned by :meth:`add_equal_con` (or a
    similar ``add_*`` call).

Returns
-------
list of numpy.ndarray
    Per-application multiplier vectors for the constraint. Only meaningful
    after a successful :meth:`optimize`/:meth:`solve`.
)doc");
    obj.def("return_equal_con_vals", &ODEPhaseBase::return_equal_con_vals,
            R"doc(Return the residual values of an equality constraint from the last solve.

Parameters
----------
index : int
    The equality-constraint index.

Returns
-------
list of numpy.ndarray
    Per-application residual vectors. Only meaningful after a successful
    :meth:`optimize`/:meth:`solve`.
)doc");
    obj.def("return_equal_con_scales", &ODEPhaseBase::return_equal_con_scales,
            R"doc(Return the output scales of a registered equality constraint.

Parameters
----------
index : int
    The equality-constraint index.

Returns
-------
numpy.ndarray
    Per-output scale vector for the constraint.
)doc");

    obj.def("return_inequal_con_lmults", &ODEPhaseBase::return_inequal_con_lmults,
            R"doc(Return the Lagrange multipliers of an inequality constraint from the last solve.

Parameters
----------
index : int
    The inequality-constraint index returned by :meth:`add_inequal_con` (or a
    similar ``add_*`` call).

Returns
-------
list of numpy.ndarray
    Per-application multiplier vectors. Only meaningful after a successful
    :meth:`optimize`/:meth:`solve`.
)doc");
    obj.def("return_inequal_con_vals", &ODEPhaseBase::return_inequal_con_vals,
            R"doc(Return the residual values of an inequality constraint from the last solve.

Parameters
----------
index : int
    The inequality-constraint index.

Returns
-------
list of numpy.ndarray
    Per-application residual vectors. Only meaningful after a successful
    :meth:`optimize`/:meth:`solve`.
)doc");
    obj.def("return_inequal_con_scales", &ODEPhaseBase::return_inequal_con_scales,
            R"doc(Return the output scales of a registered inequality constraint.

Parameters
----------
index : int
    The inequality-constraint index.

Returns
-------
numpy.ndarray
    Per-output scale vector for the constraint.
)doc");

    obj.def("return_integral_objective_scales", &ODEPhaseBase::return_integral_objective_scales,
            R"doc(Return the output scales of a registered integral objective.

Parameters
----------
index : int
    The integral-objective index returned by :meth:`add_integral_objective`.

Returns
-------
numpy.ndarray
    Per-output scale vector for the objective.
)doc");
    obj.def("return_integral_param_function_scales",
            &ODEPhaseBase::return_integral_param_function_scales,
            R"doc(Return the output scales of a registered integral-parameter function.

Parameters
----------
index : int
    The integral-parameter-function index returned by
    :meth:`add_integral_param_function`.

Returns
-------
numpy.ndarray
    Per-output scale vector for the function.
)doc");
    obj.def("return_state_objective_scales", &ODEPhaseBase::return_state_objective_scales,
            R"doc(Return the output scales of a registered state (terminal) objective.

Parameters
----------
index : int
    The state-objective index returned by :meth:`add_state_objective`.

Returns
-------
numpy.ndarray
    Per-output scale vector for the objective.
)doc");
    obj.def("return_ode_output_scales", &ODEPhaseBase::return_ode_output_scales,
            R"doc(Return the output scales applied to the ODE dynamics (defect) equations.

Returns
-------
numpy.ndarray
    Per-state-derivative output scale vector, derived from the current unit
    scaling of the state and time variables.
)doc");

    obj.def("return_static_params", &ODEPhaseBase::return_static_params,
            R"doc(Return the current static-parameter vector.

Returns
-------
numpy.ndarray
    The active static-parameter values. After a successful
    :meth:`optimize`/:meth:`solve` this reflects the optimized values.
)doc");

    obj.def("test_partitions", &ODEPhaseBase::test_partitions,
            R"doc(Benchmark threaded function partitioning over a range of thread counts.

.. note::
   Internal diagnostic / benchmarking helper. Not part of the normal
   problem-solving workflow; most users never need to call it.

Parameters
----------
i : int
    First partition count to test.
j : int
    Last partition count to test.
n : int
    Number of timing repetitions per count.
)doc");

    obj.def("remove_equal_con", &ODEPhaseBase::remove_equal_con,
            R"doc(Remove a previously added equality constraint by index.

Parameters
----------
index : int
    The constraint index to remove, or ``-1`` to remove the most recently added.
)doc");
    obj.def("remove_inequal_con", &ODEPhaseBase::remove_inequal_con,
            R"doc(Remove a previously added inequality constraint by index.

Parameters
----------
index : int
    The constraint index to remove, or ``-1`` to remove the most recently added.
)doc");
    obj.def("remove_state_objective", &ODEPhaseBase::remove_state_objective,
            R"doc(Remove a previously added state (terminal) objective by index.

Parameters
----------
index : int
    The objective index to remove, or ``-1`` to remove the most recently added.
)doc");
    obj.def("remove_integral_objective", &ODEPhaseBase::remove_integral_objective,
            R"doc(Remove a previously added integral (running-cost) objective by index.

Parameters
----------
index : int
    The objective index to remove, or ``-1`` to remove the most recently added.
)doc");
    obj.def("remove_integral_param_function", &ODEPhaseBase::remove_integral_param_function,
            R"doc(Remove a previously added integral-parameter function by index.

Parameters
----------
index : int
    The function index to remove, or ``-1`` to remove the most recently added.
)doc");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////
    ///// The New interface /////////////////////

    obj.def("add_equal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::add_equal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
            nb::arg("sp_vars"), nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add an equality constraint f=0 over a region with full variable bindings.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the constraint is evaluated.
func : VectorFunction
    Vector-valued constraint function; enforced to be zero.
xt_u_vars : int, sequence of int, or str
    State/time/control variable indices (or named group) passed to ``func``.
op_vars : int, sequence of int, or str
    ODE-parameter variable indices passed to ``func``.
sp_vars : int, sequence of int, or str
    Static-parameter variable indices passed to ``func``.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def("add_equal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::add_equal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("input_index"),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add an equality constraint f=0 over a region (state/time/control variables only).

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the constraint is evaluated.
func : VectorFunction
    Vector-valued constraint function; enforced to be zero.
input_index : int, sequence of int, or str
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
    ``func``.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def("add_boundary_value",
            nb::overload_cast<RegionType, VarIndexType, const std::variant<double, VectorXd> &,
                              ScaleType>(&ODEPhaseBase::add_boundary_value),
            nb::arg("phase_region"), nb::arg("index"), nb::arg("value"),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Fix selected region variables to given values (a boundary-value constraint).

Adds an equality constraint pinning the named variables -- evaluated in the
given region -- to ``value``. This is the usual way to set initial/final
conditions.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from, e.g. ``"Front"`` for the initial node
    or ``"Back"`` for the final node.
index : int or sequence of int
    Variable index or indices (into the packed ``[x, t, u, p]`` layout) to fix.
value : float or numpy.ndarray
    Target value(s). A scalar is broadcast to all selected variables; a vector
    must match the number of selected indices.
auto_scale : ScaleModes or str, optional
    Constraint output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``).
    Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.

Examples
--------
Pin the front state fully and the back position to a target::

    phase.add_boundary_value("Front", range(0, 4), [x0, y0, v0, 0])
    phase.add_boundary_value("Back", [0, 1], [xf, yf])
)doc");

    obj.def("add_delta_var_equal_con",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_delta_var_equal_con),
            nb::arg("var"), nb::arg("value"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Constrain the front-to-back change of a variable to a fixed value.

Parameters
----------
var : int or sequence of int
    Variable index (into the packed ``[x, t, u, p]`` layout).
value : float
    Target change (back value minus front value).
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def("add_delta_time_equal_con",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::add_delta_time_equal_con),
            nb::arg("value"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Constrain the total elapsed time of the phase to a fixed value.

Parameters
----------
value : float
    Target elapsed time (``t_back - t_front``).
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def("add_value_lock",
            nb::overload_cast<RegionType, VarIndexType, ScaleType>(&ODEPhaseBase::add_value_lock),
            nb::arg("reg"), nb::arg("vars"), nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Lock selected region variables to their current trajectory values.

Adds an equality constraint that pins the named variables to whatever values
they currently hold in the trajectory (useful to freeze a variable while
optimizing others).

Parameters
----------
reg : PhaseRegionFlags or str
    Region the variables are read from.
vars : int or sequence of int
    Variable index/indices to lock.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def("add_periodicity_con",
            nb::overload_cast<VarIndexType, ScaleType>(&ODEPhaseBase::add_periodicity_con),
            nb::arg("vars"), nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Enforce periodicity (front == back) on selected variables.

Parameters
----------
vars : int or sequence of int
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) that must
    match between the first and last trajectory node.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    //////////////////////////////////
    /////// inequal_cons_
    obj.def("add_inequal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::add_inequal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
            nb::arg("sp_vars"), nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add an inequality constraint f<=0 over a region with full variable bindings.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the constraint is evaluated.
func : VectorFunction
    Vector-valued constraint function; each output must be ≤ 0.
xt_u_vars : int, sequence of int, or str
    State/time/control variable indices (or named group) passed to ``func``.
op_vars : int, sequence of int, or str
    ODE-parameter variable indices passed to ``func``.
sp_vars : int, sequence of int, or str
    Static-parameter variable indices passed to ``func``.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def(
        "add_inequal_con",
        nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, ScaleType>(
            &ODEPhaseBase::add_inequal_con),
        nb::arg("phase_region"), nb::arg("func"), nb::arg("input_index"),
        nb::arg("auto_scale").none() = std::string("auto"),
        R"doc(Add an inequality constraint f<=0 over a region (state/time/control variables only).

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the constraint is evaluated.
func : VectorFunction
    Vector-valued constraint function; each output must be ≤ 0.
input_index : int, sequence of int, or str
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
    ``func``.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    obj.def("add_lu_var_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::add_lu_var_bound),
            nb::arg("phase_region"), nb::arg("var"), nb::arg("lowerbound"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound a variable from below and above (an inequality constraint).

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variable is read from, e.g. ``"Path"`` to bound it at every node.
var : int
    Variable index (into the packed ``[x, t, u, p]`` layout) to bound.
lowerbound : float
    Lower bound.
upperbound : float
    Upper bound.
scale : float, optional
    Constraint scaling applied to both the lower and upper bound. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.

Examples
--------
Bound the control variable along the whole path::

    phase.add_lu_var_bound("Path", 4, -0.1, 2.0)
)doc");
    obj.def("add_lower_var_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_var_bound),
            nb::arg("phase_region"), nb::arg("var"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound a variable from below (an inequality constraint).

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variable is read from.
var : int
    Variable index (into the packed ``[x, t, u, p]`` layout) to bound.
lowerbound : float
    Lower bound.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def("add_upper_var_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_var_bound),
            nb::arg("phase_region"), nb::arg("var"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound a variable from above (an inequality constraint).

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variable is read from.
var : int
    Variable index (into the packed ``[x, t, u, p]`` layout) to bound.
upperbound : float
    Upper bound.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def(
        "add_lu_func_bound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType,
                          double, double, double, ScaleType>(&ODEPhaseBase::add_lu_func_bound),
        nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
        nb::arg("sp_vars"), nb::arg("lowerbound"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
        nb::arg("auto_scale").none() = std::string("auto"),
        R"doc(Bound a scalar function of region variables from below and above.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the function is evaluated.
func : ScalarFunction
    Scalar-valued function to bound.
xt_u_vars : int, sequence of int, or str
    State/time/control variable indices passed to ``func``.
op_vars : int, sequence of int, or str
    ODE-parameter variable indices passed to ``func``.
sp_vars : int, sequence of int, or str
    Static-parameter variable indices passed to ``func``.
lowerbound : float
    Lower bound on the function value.
upperbound : float
    Upper bound on the function value.
scale : float, optional
    Shared constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def("add_lu_func_bound",
            nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, double, double, double,
                              ScaleType>(&ODEPhaseBase::add_lu_func_bound),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));
    //
    obj.def(
        "add_lower_func_bound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType,
                          double, double, ScaleType>(&ODEPhaseBase::add_lower_func_bound),
        nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
        nb::arg("sp_vars"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
        nb::arg("auto_scale").none() = std::string("auto"),
        R"doc(Bound a scalar function of region variables from below.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the function is evaluated.
func : ScalarFunction
    Scalar-valued function to bound.
xt_u_vars : int, sequence of int, or str
    State/time/control variable indices passed to ``func``.
op_vars : int, sequence of int, or str
    ODE-parameter variable indices passed to ``func``.
sp_vars : int, sequence of int, or str
    Static-parameter variable indices passed to ``func``.
lowerbound : float
    Lower bound on the function value.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def(
        "add_lower_func_bound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, double, double, ScaleType>(
            &ODEPhaseBase::add_lower_func_bound),
        nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
        nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"));

    obj.def(
        "add_upper_func_bound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType,
                          double, double, ScaleType>(&ODEPhaseBase::add_upper_func_bound),
        nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
        nb::arg("sp_vars"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
        nb::arg("auto_scale").none() = std::string("auto"),
        R"doc(Bound a scalar function of region variables from above.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region over which the function is evaluated.
func : ScalarFunction
    Scalar-valued function to bound.
xt_u_vars : int, sequence of int, or str
    State/time/control variable indices passed to ``func``.
op_vars : int, sequence of int, or str
    ODE-parameter variable indices passed to ``func``.
sp_vars : int, sequence of int, or str
    Static-parameter variable indices passed to ``func``.
upperbound : float
    Upper bound on the function value.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def(
        "add_upper_func_bound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, double, double, ScaleType>(
            &ODEPhaseBase::add_upper_func_bound),
        nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_up_vars"), nb::arg("upperbound"),
        nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_lu_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::add_lu_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the Euclidean norm of selected variables from below and above.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
xt_up_vars : int or sequence of int
    Variable indices whose norm is bounded.
lowerbound : float
    Lower bound on the norm.
upperbound : float
    Upper bound on the norm.
scale : float, optional
    Shared constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def("add_lu_squared_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::add_lu_squared_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the squared Euclidean norm of selected variables from below and above.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
xt_up_vars : int or sequence of int
    Variable indices whose squared norm is bounded.
lowerbound : float
    Lower bound on the squared norm.
upperbound : float
    Upper bound on the squared norm.
scale : float, optional
    Shared constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    //
    obj.def("add_lower_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the Euclidean norm of selected variables from below.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
xt_up_vars : int or sequence of int
    Variable indices whose norm is bounded.
lowerbound : float
    Lower bound on the norm.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def("add_lower_squared_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_squared_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the squared Euclidean norm of selected variables from below.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
xt_up_vars : int or sequence of int
    Variable indices whose squared norm is bounded.
lowerbound : float
    Lower bound on the squared norm.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");
    //
    obj.def("add_upper_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the Euclidean norm of selected variables from above.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
xt_up_vars : int or sequence of int
    Variable indices whose norm is bounded.
upperbound : float
    Upper bound on the norm.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");

    obj.def("add_upper_squared_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_squared_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the squared Euclidean norm of selected variables from above.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
xt_up_vars : int or sequence of int
    Variable indices whose squared norm is bounded.
upperbound : float
    Upper bound on the squared norm.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");
    //
    obj.def("add_lower_delta_var_bound",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_delta_var_bound),
            nb::arg("var"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the front-to-back change of a variable from below.

Parameters
----------
var : int or sequence of int
    Variable index (into the packed ``[x, t, u, p]`` layout).
lowerbound : float
    Lower bound on the change (back minus front).
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");
    obj.def("add_lower_delta_time_bound",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::add_lower_delta_time_bound),
            nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the elapsed time of the phase from below.

Parameters
----------
lowerbound : float
    Lower bound on ``t_back - t_front``.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");
    //
    obj.def("add_upper_delta_var_bound",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_delta_var_bound),
            nb::arg("var"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the front-to-back change of a variable from above.

Parameters
----------
var : int or sequence of int
    Variable index (into the packed ``[x, t, u, p]`` layout).
upperbound : float
    Upper bound on the change (back minus front).
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");
    obj.def("add_upper_delta_time_bound",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::add_upper_delta_time_bound),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Bound the elapsed time of the phase from above.

Parameters
----------
upperbound : float
    Upper bound on ``t_back - t_front``.
scale : float, optional
    Constraint scaling. Default 1.0.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new bound constraint.
)doc");
    //////////////////////////////////
    /////// StateObjectives /////////
    obj.def("add_state_objective",
            nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::add_state_objective),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
            nb::arg("sp_vars"), nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_state_objective",
            nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::add_state_objective),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("input_index"),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add a (terminal/boundary) objective evaluated at a region of the phase.

Adds the scalar value of ``func``, evaluated over the selected variables in
the given region, to the problem objective (a Mayer-type term).

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the function is evaluated over, e.g. ``"Back"`` for a terminal cost.
func : ScalarFunction
    Scalar-valued function of the selected variables.
input_index : int or sequence of int
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
    ``func``.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new objective.
)doc");

    obj.def("add_value_objective",
            nb::overload_cast<RegionType, VarIndexType, double, ScaleType>(
                &ODEPhaseBase::add_value_objective),
            nb::arg("phase_region"), nb::arg("var"), nb::arg("scale"),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add an objective equal to a scaled single variable at a region.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variable is read from.
var : int
    Variable index (into the packed ``[x, t, u, p]`` layout).
scale : float
    Multiplier applied to the variable's value in the objective. Use a
    negative value to maximize the variable.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new objective.
)doc");

    obj.def(
        "add_delta_var_objective",
        nb::overload_cast<VarIndexType, double, ScaleType>(&ODEPhaseBase::add_delta_var_objective),
        nb::arg("var"), nb::arg("scale"), nb::arg("auto_scale").none() = std::string("auto"),
        R"doc(Add an objective on the front-to-back change of a variable.

Parameters
----------
var : int
    Variable index (into the packed ``[x, t, u, p]`` layout).
scale : float
    Multiplier applied to the change (back value minus front value).
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new objective.
)doc");
    obj.def("add_delta_time_objective",
            nb::overload_cast<double, ScaleType>(&ODEPhaseBase::add_delta_time_objective),
            nb::arg("scale"), nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add an objective on the total elapsed time of the phase.

A scaled minimum-time objective: minimizes ``scale * (t_back - t_front)``.

Parameters
----------
scale : float
    Multiplier applied to the elapsed time. Use ``1.0`` for a minimum-time
    objective.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new objective.

Examples
--------
Minimize the phase duration::

    phase.add_delta_time_objective(1.0)
)doc");
    //////////////////////////////////
    /////// IntegralObjectives /////////
    obj.def(
        "add_integral_objective",
        nb::overload_cast<ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType, ScaleType>(
            &ODEPhaseBase::add_integral_objective),
        nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"), nb::arg("sp_vars"),
        nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_integral_objective",
            nb::overload_cast<ScalarFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::add_integral_objective),
            nb::arg("func"), nb::arg("input_index"),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add a running-cost (Lagrange) objective integrated over the phase.

Adds the time integral of ``func`` -- evaluated along the trajectory -- to the
problem objective.

Parameters
----------
func : ScalarFunction
    Scalar-valued integrand, a function of the selected variables.
input_index : int or sequence of int
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
    ``func`` at each node.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new objective.
)doc");
    //////////////////////////////////
    /////// IntegralParamFunction /////////
    obj.def("add_integral_param_function",
            nb::overload_cast<ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType, int,
                              ScaleType>(&ODEPhaseBase::add_integral_param_function),
            nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"), nb::arg("sp_vars"),
            nb::arg("int_param"), nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add an integral that accumulates into a parameter, with full variable bindings.

Integrates ``func`` along the path and adds the result to the ODE parameter
at index ``int_param``.

Parameters
----------
func : ScalarFunction
    Scalar-valued integrand.
xt_u_vars : int, sequence of int, or str
    State/time/control variable indices passed to ``func``.
op_vars : int, sequence of int, or str
    ODE-parameter variable indices passed to ``func``.
sp_vars : int, sequence of int, or str
    Static-parameter variable indices passed to ``func``.
int_param : int
    Index of the ODE parameter the integral accumulates into.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new integral-parameter function.
)doc");

    obj.def(
        "add_integral_param_function",
        nb::overload_cast<ScalarFunctionalX, VarIndexType, int, ScaleType>(
            &ODEPhaseBase::add_integral_param_function),
        nb::arg("func"), nb::arg("input_index"), nb::arg("int_param"),
        nb::arg("auto_scale").none() = std::string("auto"),
        R"doc(Add an integral that accumulates into a parameter (state/time/control variables only).

Parameters
----------
func : ScalarFunction
    Scalar-valued integrand.
input_index : int, sequence of int, or str
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) passed to
    ``func``.
int_param : int
    Index of the ODE parameter the integral accumulates into.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new integral-parameter function.
)doc");

    ///////////////////////////////////////////////////////////////////

    obj.def("add_equal_con", nb::overload_cast<StateConstraint>(&ODEPhaseBase::add_equal_con),
            R"doc(Add a pre-built equality constraint to the phase.

Parameters
----------
con : StateConstraint
    A region-bound equality constraint constructed externally.

Returns
-------
int
    The index assigned to the new constraint.
)doc");

    ///////////////////////////////////////////////////////////////////////////////

    obj.def("add_inequal_con", nb::overload_cast<StateConstraint>(&ODEPhaseBase::add_inequal_con),
            R"doc(Add a pre-built inequality constraint to the phase.

Parameters
----------
con : StateConstraint
    A region-bound inequality constraint constructed externally.

Returns
-------
int
    The index assigned to the new constraint.
)doc");
    ////////////////////////////////////////////////////////////////////////////
    obj.def("add_lu_var_bounds",
            nb::overload_cast<PhaseRegionFlags, Eigen::VectorXi, double, double, double>(
                &ODEPhaseBase::add_lu_var_bounds),
            R"doc(Bound several variables from below and above with a shared scale.

Parameters
----------
phase_region : PhaseRegionFlags or str
    Region the variables are read from.
vars : numpy.ndarray of int
    Variable indices (into the packed ``[x, t, u, p]`` layout) to bound.
lowerbound : float
    Lower bound applied to each variable.
upperbound : float
    Upper bound applied to each variable.
scale : float
    Shared constraint scaling.

Returns
-------
numpy.ndarray of int
    Indices assigned to the new bound constraints (one per variable).
)doc");
    obj.def("add_lu_var_bounds",
            nb::overload_cast<std::string, Eigen::VectorXi, double, double, double>(
                &ODEPhaseBase::add_lu_var_bounds),
            "");

    ////////////////////////////////////////////////////////////////////////////
    obj.def("add_state_objective",
            nb::overload_cast<StateObjective>(&ODEPhaseBase::add_state_objective),
            R"doc(Add a pre-built state (terminal) objective to the phase.

Parameters
----------
obj : StateObjective
    A region-bound scalar objective constructed externally.

Returns
-------
int
    The index assigned to the new objective.
)doc");

    ////////////////////////////////////////////////////////////////////////////

    obj.def("add_integral_objective",
            nb::overload_cast<StateObjective>(&ODEPhaseBase::add_integral_objective),
            R"doc(Add a pre-built integral (running-cost) objective to the phase.

Parameters
----------
obj : StateObjective
    A region-bound scalar integrand objective constructed externally.

Returns
-------
int
    The index assigned to the new objective.
)doc");

    ///////////////////////////////////////////////////////////////////////////////
    obj.def("add_integral_param_function",
            nb::overload_cast<StateObjective, int>(&ODEPhaseBase::add_integral_param_function),
            R"doc(Add a pre-built integral-parameter function that accumulates into a parameter.

Parameters
----------
obj : StateObjective
    A region-bound scalar integrand constructed externally.
int_param : int
    Index of the ODE parameter the integral value accumulates into.

Returns
-------
int
    The index assigned to the new integral-parameter function.
)doc");

    ////////////////////////////////////////////////////
    obj.def("get_mesh_info", &ODEPhaseBase::get_mesh_info,
            R"doc(Compute mesh error, equidistributed bin boundaries, and per-segment error.

Parameters
----------
integ : bool
    Use the reference integrator estimator (``True``) or the de Boor estimator
    (``False``).
n : int
    Number of equidistributing bins to produce.

Returns
-------
tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    ``(node_times, bin_boundaries, per_segment_error)`` — the non-dimensional
    node times, the ``n+1`` equidistributed bin boundary times, and the
    per-segment error vector.
)doc");
    obj.def("refine_traj_auto", &ODEPhaseBase::refine_traj_auto,
            R"doc(Run one automatic mesh-refinement step.

Estimates the per-interval discretization error and redistributes mesh nodes
to drive it below :attr:`mesh_tol`. This is invoked automatically when
:attr:`adaptive_mesh` is enabled; call it directly for manual control of the
refinement loop.
)doc");
    obj.def("calc_global_error", &ODEPhaseBase::calc_global_error,
            R"doc(Estimate the global (end-to-end) discretization error of the trajectory.

Returns
-------
numpy.ndarray
    Per-state global error estimate of the current trajectory.
)doc");
    obj.def("get_mesh_iters", &ODEPhaseBase::get_mesh_iters,
            R"doc(Return the per-iteration adaptive-mesh refinement history.

Returns
-------
list of MeshIterateInfo
    One :class:`MeshIterateInfo` record per mesh-refinement iteration that was
    performed during the last solve.
)doc");

    obj.def_rw("adaptive_mesh", &ODEPhaseBase::adaptive_mesh_,
               R"doc(Enable adaptive mesh refinement (bool, default ``False``).

When ``True``, :meth:`optimize`/:meth:`solve` runs the automatic mesh-refinement
loop until the per-interval error falls below :attr:`mesh_tol` or
:attr:`max_mesh_iters` is reached.
)doc");
    obj.def_rw("auto_scaling", &ODEPhaseBase::auto_scaling_,
               R"doc(Enable automatic variable/equation scaling (bool, default ``False``).

When ``True``, the transcription scales each decision variable and constraint
by values derived from the current trajectory. Writing this attribute directly
sets the flag only; use :meth:`set_auto_scaling` to update it and reset the
cached transcription in one step.
)doc");
    obj.def_rw("sync_objective_scales", &ODEPhaseBase::sync_objective_scales_,
               R"doc(Synchronize objective scales with auto-scaling (bool, default ``True``).

When ``True`` and :attr:`auto_scaling` is enabled, objective scale factors are
updated together with the variable/constraint scales.
)doc");

    obj.def("set_auto_scaling", &ODEPhaseBase::set_auto_scaling, nb::arg("auto_scaling") = true,
            R"doc(Enable or disable automatic variable/equation scaling.

Parameters
----------
auto_scaling : bool, optional
    ``True`` to enable automatic scaling (default), ``False`` to disable.
    Invalidates the cached transcription.
)doc");

    obj.def("set_adaptive_mesh", &ODEPhaseBase::set_adaptive_mesh, nb::arg("adaptive_mesh") = true,
            R"doc(Enable or disable adaptive mesh refinement.

Parameters
----------
adaptive_mesh : bool, optional
    ``True`` to enable adaptive refinement (default), ``False`` to disable.
)doc");

    obj.def(
        "set_units",
        [](ODEPhaseBase &self, nb::kwargs kwargs) {
            nb::module_ builtins = nb::module_::import_("builtins");
            nb::object py_int = builtins.attr("int");
            nb::object py_float = builtins.attr("float");
            nb::object py_list = builtins.attr("list");
            nb::object np_array = (nb::object)nb::module_::import_("numpy").attr("ndarray");
            nb::object np_float = (nb::object)nb::module_::import_("numpy").attr("float64");
            nb::object np_int = (nb::object)nb::module_::import_("numpy").attr("int32");

            Eigen::VectorXd Units(self.xtu_p_vars());
            Units.setOnes();

            for (const auto &kw : kwargs) {
                auto name = nb::cast<std::string>(kw.first);
                auto idxs = self.idx(name);
                Eigen::VectorXd units(idxs.size());
                units.setOnes();

                if (kw.second.type().is(py_int) || kw.second.type().is(py_float) ||
                    kw.second.type().is(np_float) || kw.second.type().is(np_int)) {
                    double unit = nb::cast<double>(kw.second);
                    units *= unit;
                } else if (kw.second.type().is(np_array) || kw.second.type().is(py_list)) {
                    int lenvec = nb::cast<int>(kw.second.attr("__len__")());
                    if (lenvec != idxs.size()) {
                        throw std::invalid_argument(fmt::format(
                            "Size of index group {0:} does not match units vector.", name));
                    }
                    for (int i = 0; i < lenvec; i++) {
                        auto elem = kw.second.attr("__getitem__")(nb::int_(i)).type();
                        if (!(elem.is(py_float) || elem.is(py_int) || elem.is(np_int) ||
                              elem.is(np_float))) {
                            nb::print(nb::str(elem));
                            throw std::invalid_argument(
                                "Vectors and lists must only contain doubles or floats");
                        }
                        units[i] = nb::cast<double>(kw.second.attr("__getitem__")(nb::int_(i)));
                    }
                } else {
                    throw std::invalid_argument("Invalid unit type");
                }

                for (int i = 0; i < idxs.size(); i++) {
                    Units[idxs[i]] = units[i];
                }
            }
            self.set_units(Units);
        },
        R"doc(Set per-variable scaling units for the state/time/control/ODE-param block.

Accepts keyword arguments whose names are variable-group names registered on the
phase (e.g. ``x``, ``t``, ``u``, ``p``); each value is a scalar or array of
units for that group. Alternatively, call the overload that accepts a raw
``numpy.ndarray`` of the full unit vector.

Parameters
----------
**kwargs
    Group name to unit value/array mapping. Scalar values are broadcast to the
    whole group; arrays must match the group size.
)doc");
    obj.def("set_units", nb::overload_cast<const Eigen::VectorXd &>(&ODEPhaseBase::set_units),
            R"doc(Set per-variable scaling units from a raw vector.

Parameters
----------
xtup_units : numpy.ndarray
    Per-variable scaling units for the full ``[x, t, u, p]`` layout, length
    equal to ``x_vars + 1 + u_vars + p_vars``.
)doc");

    obj.def("set_mesh_tol", &ODEPhaseBase::set_mesh_tol,
            R"doc(Set the target mesh-error tolerance for adaptive refinement.

Parameters
----------
t : float
    Tolerance (absolute value used); refinement stops when the estimated
    per-interval error falls below this threshold.
)doc");
    obj.def("set_mesh_red_factor", &ODEPhaseBase::set_mesh_red_factor,
            R"doc(Set the factor by which segment counts may be reduced during mesh refinement.

Parameters
----------
t : float
    Reduction factor (absolute value used); segments in low-error intervals
    are multiplied by this value.
)doc");
    obj.def("set_mesh_inc_factor", &ODEPhaseBase::set_mesh_inc_factor,
            R"doc(Set the factor by which segment counts may be increased during mesh refinement.

Parameters
----------
t : float
    Increase factor (absolute value used); segments in high-error intervals
    are multiplied by this value.
)doc");
    obj.def("set_mesh_err_factor", &ODEPhaseBase::set_mesh_err_factor,
            R"doc(Set the error-overshoot factor that triggers mesh refinement.

Parameters
----------
t : float
    Error factor (absolute value used); refinement is triggered when the
    estimated error exceeds ``mesh_tol * mesh_err_factor``.
)doc");
    obj.def("set_max_mesh_iters", &ODEPhaseBase::set_max_mesh_iters,
            R"doc(Set the maximum number of adaptive mesh-refinement iterations.

Parameters
----------
it : int
    Iteration cap (absolute value used).
)doc");
    obj.def("set_min_segments", &ODEPhaseBase::set_min_segments,
            R"doc(Set the minimum number of mesh segments allowed during adaptive refinement.

Parameters
----------
it : int
    Minimum segment count (absolute value used).
)doc");
    obj.def("set_max_segments", &ODEPhaseBase::set_max_segments,
            R"doc(Set the maximum number of mesh segments allowed during adaptive refinement.

Parameters
----------
it : int
    Maximum segment count (absolute value used).
)doc");
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<MeshErrorAggregation>(&ODEPhaseBase::set_mesh_error_criteria),
            R"doc(Set the aggregation function used to determine mesh convergence.

Parameters
----------
mode : MeshErrorAggregation or str
    Aggregation mode, e.g. ``"MAX"`` (default) or ``"AVG"``.  The mesh is
    considered converged when the aggregated per-interval error is below
    :attr:`mesh_tol`.
)doc");
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::set_mesh_error_criteria), "");
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<MeshErrorEstimators>(&ODEPhaseBase::set_mesh_error_estimator),
            R"doc(Set the method used to estimate per-interval discretization error.

Parameters
----------
mode : MeshErrorEstimators or str
    Estimator, e.g. ``"DEBOOR"`` (default) or ``"Integrator"``.
)doc");
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::set_mesh_error_estimator), "");
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<MeshErrorAggregation>(&ODEPhaseBase::set_mesh_error_distributor),
            R"doc(Set the aggregation used to distribute the mesh-error density across segments.

Parameters
----------
mode : MeshErrorAggregation or str
    Distribution aggregation mode, e.g. ``"AVG"`` (default) or ``"MAX"``.
)doc");
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::set_mesh_error_distributor), "");

    obj.def_rw(
        "print_mesh_info", &ODEPhaseBase::print_mesh_info_,
        "Print mesh-refinement diagnostics during the adaptive loop (bool, default ``True``).");
    obj.def_rw("max_mesh_iters", &ODEPhaseBase::max_mesh_iters_,
               "Maximum number of adaptive mesh-refinement iterations (int, default 10).");
    obj.def_rw(
        "mesh_tol", &ODEPhaseBase::mesh_tol_,
        "Target per-interval mesh-error tolerance for adaptive refinement (float, default 1e-6).");
    obj.def_prop_rw(
        "mesh_error_estimator", [](const ODEPhaseBase &self) { return self.mesh_error_estimator_; },
        [set_mesh_error_estimator](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_estimator(self, val);
        },
        R"doc(Method used to estimate per-interval discretization error (MeshErrorEstimators or str).

Accepts a :class:`MeshErrorEstimators` enum value or its string name (e.g.
``"DEBOOR"`` or ``"Integrator"``). Default: ``MeshErrorEstimators.DEBOOR``.
)doc");
    obj.def_prop_rw(
        "mesh_error_criteria", [](const ODEPhaseBase &self) { return self.mesh_error_criteria_; },
        [set_mesh_error_aggregation](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_aggregation(self.mesh_error_criteria_, val, "mesh_error_criteria_");
        },
        R"doc(Aggregation function used to determine mesh convergence (MeshErrorAggregation or str).

Accepts a :class:`MeshErrorAggregation` enum value or its string name (e.g.
``"MAX"`` or ``"AVG"``). The mesh is converged when the aggregated per-interval
error is below :attr:`mesh_tol`. Default: ``MeshErrorAggregation.MAX``.
)doc");
    obj.def_prop_rw(
        "mesh_error_distributor",
        [](const ODEPhaseBase &self) { return self.mesh_error_distributor_; },
        [set_mesh_error_aggregation](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_aggregation(self.mesh_error_distributor_, val,
                                       "mesh_error_distributor_");
        },
        R"doc(Aggregation used to distribute error density when redistributing mesh segments (MeshErrorAggregation or str).

Accepts a :class:`MeshErrorAggregation` enum value or its string name (e.g.
``"AVG"`` or ``"MAX"``). Default: ``MeshErrorAggregation.AVG``.
)doc");

    obj.def_rw("solve_only_first", &ODEPhaseBase::solve_only_first_,
               "Only solve (not optimize) on the first mesh-refinement iteration (bool, default "
               "``True``).");
    obj.def_rw("force_one_mesh_iter", &ODEPhaseBase::force_one_mesh_iter_,
               "Force at least one mesh-refinement iteration even if the initial mesh already "
               "meets tolerance (bool, default ``False``).");
    obj.def_rw("new_error", &ODEPhaseBase::new_error_,
               "Use the newer mesh-error formulation instead of the legacy one (bool, default "
               "``False``).");

    obj.def_rw("detect_control_switches", &ODEPhaseBase::detect_control_switches_,
               "Detect and refine the mesh around control switches (bool, default ``False``).");
    obj.def_rw("rel_switch_tol", &ODEPhaseBase::rel_switch_tol_,
               "Relative tolerance for control-switch detection (float, default 0.3).");
    obj.def_rw("abs_switch_tol", &ODEPhaseBase::abs_switch_tol_,
               "Absolute tolerance for control-switch detection (float, default 1e-6).");
    obj.def_rw("mesh_abort_flag", &ODEPhaseBase::mesh_abort_flag_,
               "Solver convergence flag that aborts the mesh-refinement loop early "
               "(ConvergenceFlags, default ``DIVERGING``).");

    obj.def_rw("num_extra_segs", &ODEPhaseBase::num_extra_segs_,
               "Extra segments added when refining a mesh interval (int, default 4).");
    obj.def_rw("mesh_red_factor", &ODEPhaseBase::mesh_red_factor_,
               "Factor by which segment counts may be reduced during adaptive refinement (float, "
               "default 0.5).");
    obj.def_rw("mesh_inc_factor", &ODEPhaseBase::mesh_inc_factor_,
               "Factor by which segment counts may be increased during adaptive refinement (float, "
               "default 5.0).");
    obj.def_rw("mesh_err_factor", &ODEPhaseBase::mesh_err_factor_,
               "Error-overshoot factor that triggers segment count increase during refinement "
               "(float, default 10.0).");
    obj.def_ro("mesh_converged", &ODEPhaseBase::mesh_converged_,
               "``True`` if the last adaptive mesh-refinement loop met the :attr:`mesh_tol` "
               "criterion (read-only bool).");
}
