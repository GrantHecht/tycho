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
bounds, constraints, and objectives that define a direct-collocation problem.
Phases are not constructed directly; build one from an ODE with
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

    obj.def("enable_vectorization", &ODEPhaseBase::enable_vectorization);

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
    order. This becomes the initial guess that collocation refines.

Notes
-----
Additional overloads accept a defect-segment count or an explicit
defects-binding/defects-per-block specification; see the C++ reference.
)doc");

    obj.def("switch_transcription_mode",
            nb::overload_cast<TranscriptionModes, VectorXd, VectorXi>(
                &ODEPhaseBase::switch_transcription_mode),
            "");
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

Builds the collocation NLP (defect constraints, user constraints, and
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
)doc");

    obj.def("set_static_params",
            nb::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::set_static_params), "");
    obj.def("set_static_params", nb::overload_cast<VectorXd>(&ODEPhaseBase::set_static_params), "");

    obj.def("add_static_params",
            nb::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::add_static_params));
    obj.def("add_static_params", nb::overload_cast<VectorXd>(&ODEPhaseBase::add_static_params));
    obj.def("add_static_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                            &ODEPhaseBase::add_static_param_vgroups));
    obj.def("set_static_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                            &ODEPhaseBase::set_static_param_vgroups));
    obj.def("add_static_param_vgroup", nb::overload_cast<Eigen::VectorXi, std::string>(
                                           &ODEPhaseBase::add_static_param_vgroup));
    obj.def("add_static_param_vgroup",
            nb::overload_cast<int, std::string>(&ODEPhaseBase::add_static_param_vgroup));

    obj.def("set_control_mode", nb::overload_cast<ControlModes>(&ODEPhaseBase::set_control_mode),
            "");
    obj.def("set_control_mode", nb::overload_cast<std::string>(&ODEPhaseBase::set_control_mode),
            "");

    obj.def("set_integral_mode", &ODEPhaseBase::set_integral_mode, "");

    obj.def("sub_static_params", &ODEPhaseBase::sub_static_params, "");

    obj.def("sub_variables",
            nb::overload_cast<PhaseRegionFlags, VectorXi, VectorXd>(&ODEPhaseBase::sub_variables),
            "");
    obj.def("sub_variable",
            nb::overload_cast<PhaseRegionFlags, int, double>(&ODEPhaseBase::sub_variable), "");

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
    obj.def("return_traj_range_nd", &ODEPhaseBase::return_traj_range_nd, "");
    obj.def("return_traj_table", &ODEPhaseBase::return_traj_table);

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
    obj.def("return_traj_error", &ODEPhaseBase::return_traj_error);

    obj.def("return_u_spline_con_lmults", &ODEPhaseBase::return_u_spline_con_lmults);
    obj.def("return_u_spline_con_vals", &ODEPhaseBase::return_u_spline_con_vals);

    obj.def("return_equal_con_lmults", &ODEPhaseBase::return_equal_con_lmults, "");
    obj.def("return_equal_con_vals", &ODEPhaseBase::return_equal_con_vals);
    obj.def("return_equal_con_scales", &ODEPhaseBase::return_equal_con_scales);

    obj.def("return_inequal_con_lmults", &ODEPhaseBase::return_inequal_con_lmults, "");
    obj.def("return_inequal_con_vals", &ODEPhaseBase::return_inequal_con_vals);
    obj.def("return_inequal_con_scales", &ODEPhaseBase::return_inequal_con_scales);

    obj.def("return_integral_objective_scales", &ODEPhaseBase::return_integral_objective_scales);
    obj.def("return_integral_param_function_scales",
            &ODEPhaseBase::return_integral_param_function_scales);
    obj.def("return_state_objective_scales", &ODEPhaseBase::return_state_objective_scales);
    obj.def("return_ode_output_scales", &ODEPhaseBase::return_ode_output_scales);

    obj.def("return_static_params", &ODEPhaseBase::return_static_params, "");

    obj.def("test_partitions", &ODEPhaseBase::test_partitions);

    obj.def("remove_equal_con", &ODEPhaseBase::remove_equal_con, "");
    obj.def("remove_inequal_con", &ODEPhaseBase::remove_inequal_con, "");
    obj.def("remove_state_objective", &ODEPhaseBase::remove_state_objective, "");
    obj.def("remove_integral_objective", &ODEPhaseBase::remove_integral_objective, "");
    obj.def("remove_integral_param_function", &ODEPhaseBase::remove_integral_param_function, "");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////
    ///// The New interface /////////////////////

    obj.def("add_equal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::add_equal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
            nb::arg("sp_vars"), nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_equal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::add_equal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("input_index"),
            nb::arg("auto_scale").none() = std::string("auto"));

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
            nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_delta_time_equal_con",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::add_delta_time_equal_con),
            nb::arg("value"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_value_lock",
            nb::overload_cast<RegionType, VarIndexType, ScaleType>(&ODEPhaseBase::add_value_lock),
            nb::arg("reg"), nb::arg("vars"), nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_periodicity_con",
            nb::overload_cast<VarIndexType, ScaleType>(&ODEPhaseBase::add_periodicity_con),
            nb::arg("vars"), nb::arg("auto_scale").none() = std::string("auto"));

    //////////////////////////////////
    /////// inequal_cons_
    obj.def("add_inequal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::add_inequal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("xt_u_vars"), nb::arg("op_vars"),
            nb::arg("sp_vars"), nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_inequal_con",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::add_inequal_con),
            nb::arg("phase_region"), nb::arg("func"), nb::arg("input_index"),
            nb::arg("auto_scale").none() = std::string("auto"));

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
        nb::arg("auto_scale").none() = std::string("auto"));

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
        nb::arg("auto_scale").none() = std::string("auto"));

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
        nb::arg("auto_scale").none() = std::string("auto"));

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
            nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_lu_squared_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::add_lu_squared_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));

    //
    obj.def("add_lower_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_lower_squared_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_squared_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("lowerbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"));
    //
    obj.def("add_upper_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_upper_squared_norm_bound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_squared_norm_bound),
            nb::arg("phase_region"), nb::arg("xt_up_vars"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("auto_scale").none() = std::string("auto"));
    //
    obj.def("add_lower_delta_var_bound",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_lower_delta_var_bound),
            nb::arg("var"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));
    obj.def("add_lower_delta_time_bound",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::add_lower_delta_time_bound),
            nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));
    //
    obj.def("add_upper_delta_var_bound",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::add_upper_delta_var_bound),
            nb::arg("var"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));
    obj.def("add_upper_delta_time_bound",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::add_upper_delta_time_bound),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("auto_scale").none() = std::string("auto"));
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
            nb::arg("int_param"), nb::arg("auto_scale").none() = std::string("auto"));

    obj.def("add_integral_param_function",
            nb::overload_cast<ScalarFunctionalX, VarIndexType, int, ScaleType>(
                &ODEPhaseBase::add_integral_param_function),
            nb::arg("func"), nb::arg("input_index"), nb::arg("int_param"),
            nb::arg("auto_scale").none() = std::string("auto"));

    ///////////////////////////////////////////////////////////////////

    obj.def("add_equal_con", nb::overload_cast<StateConstraint>(&ODEPhaseBase::add_equal_con), "");

    ///////////////////////////////////////////////////////////////////////////////

    obj.def("add_inequal_con", nb::overload_cast<StateConstraint>(&ODEPhaseBase::add_inequal_con),
            "");
    ////////////////////////////////////////////////////////////////////////////
    obj.def("add_lu_var_bounds",
            nb::overload_cast<PhaseRegionFlags, Eigen::VectorXi, double, double, double>(
                &ODEPhaseBase::add_lu_var_bounds),
            "");
    obj.def("add_lu_var_bounds",
            nb::overload_cast<std::string, Eigen::VectorXi, double, double, double>(
                &ODEPhaseBase::add_lu_var_bounds),
            "");

    ////////////////////////////////////////////////////////////////////////////
    obj.def("add_state_objective",
            nb::overload_cast<StateObjective>(&ODEPhaseBase::add_state_objective), "");

    ////////////////////////////////////////////////////////////////////////////

    obj.def("add_integral_objective",
            nb::overload_cast<StateObjective>(&ODEPhaseBase::add_integral_objective), "");

    ///////////////////////////////////////////////////////////////////////////////
    obj.def("add_integral_param_function",
            nb::overload_cast<StateObjective, int>(&ODEPhaseBase::add_integral_param_function), "");

    ////////////////////////////////////////////////////
    obj.def("get_mesh_info", &ODEPhaseBase::get_mesh_info);
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

    obj.def_rw("adaptive_mesh", &ODEPhaseBase::adaptive_mesh_);
    obj.def_rw("auto_scaling", &ODEPhaseBase::auto_scaling_);
    obj.def_rw("sync_objective_scales", &ODEPhaseBase::sync_objective_scales_);

    obj.def("set_auto_scaling", &ODEPhaseBase::set_auto_scaling, nb::arg("auto_scaling") = true);

    obj.def("set_adaptive_mesh", &ODEPhaseBase::set_adaptive_mesh, nb::arg("adaptive_mesh") = true);

    obj.def("set_units", [](ODEPhaseBase &self, nb::kwargs kwargs) {
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
                    throw std::invalid_argument(
                        fmt::format("Size of index group {0:} does not match units vector.", name));
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
    });
    obj.def("set_units", nb::overload_cast<const Eigen::VectorXd &>(&ODEPhaseBase::set_units));

    obj.def("set_mesh_tol", &ODEPhaseBase::set_mesh_tol);
    obj.def("set_mesh_red_factor", &ODEPhaseBase::set_mesh_red_factor);
    obj.def("set_mesh_inc_factor", &ODEPhaseBase::set_mesh_inc_factor);
    obj.def("set_mesh_err_factor", &ODEPhaseBase::set_mesh_err_factor);
    obj.def("set_max_mesh_iters", &ODEPhaseBase::set_max_mesh_iters);
    obj.def("set_min_segments", &ODEPhaseBase::set_min_segments);
    obj.def("set_max_segments", &ODEPhaseBase::set_max_segments);
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<MeshErrorAggregation>(&ODEPhaseBase::set_mesh_error_criteria));
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::set_mesh_error_criteria));
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<MeshErrorEstimators>(&ODEPhaseBase::set_mesh_error_estimator));
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::set_mesh_error_estimator));
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<MeshErrorAggregation>(&ODEPhaseBase::set_mesh_error_distributor));
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::set_mesh_error_distributor));

    obj.def_rw("print_mesh_info", &ODEPhaseBase::print_mesh_info_);
    obj.def_rw("max_mesh_iters", &ODEPhaseBase::max_mesh_iters_);
    obj.def_rw("mesh_tol", &ODEPhaseBase::mesh_tol_);
    obj.def_prop_rw(
        "mesh_error_estimator", [](const ODEPhaseBase &self) { return self.mesh_error_estimator_; },
        [set_mesh_error_estimator](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_estimator(self, val);
        });
    obj.def_prop_rw(
        "mesh_error_criteria", [](const ODEPhaseBase &self) { return self.mesh_error_criteria_; },
        [set_mesh_error_aggregation](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_aggregation(self.mesh_error_criteria_, val, "mesh_error_criteria_");
        });
    obj.def_prop_rw(
        "mesh_error_distributor",
        [](const ODEPhaseBase &self) { return self.mesh_error_distributor_; },
        [set_mesh_error_aggregation](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_aggregation(self.mesh_error_distributor_, val,
                                       "mesh_error_distributor_");
        });

    obj.def_rw("solve_only_first", &ODEPhaseBase::solve_only_first_);
    obj.def_rw("force_one_mesh_iter", &ODEPhaseBase::force_one_mesh_iter_);
    obj.def_rw("new_error", &ODEPhaseBase::new_error_);

    obj.def_rw("detect_control_switches", &ODEPhaseBase::detect_control_switches_);
    obj.def_rw("rel_switch_tol", &ODEPhaseBase::rel_switch_tol_);
    obj.def_rw("abs_switch_tol", &ODEPhaseBase::abs_switch_tol_);
    obj.def_rw("mesh_abort_flag", &ODEPhaseBase::mesh_abort_flag_);

    obj.def_rw("num_extra_segs", &ODEPhaseBase::num_extra_segs_);
    obj.def_rw("mesh_red_factor", &ODEPhaseBase::mesh_red_factor_);
    obj.def_rw("mesh_inc_factor", &ODEPhaseBase::mesh_inc_factor_);
    obj.def_rw("mesh_err_factor", &ODEPhaseBase::mesh_err_factor_);
    obj.def_ro("mesh_converged", &ODEPhaseBase::mesh_converged_);
}
