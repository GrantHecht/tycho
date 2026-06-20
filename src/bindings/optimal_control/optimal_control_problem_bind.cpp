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
using PhasePtr = std::shared_ptr<ODEPhaseBase>;
using PhaseRefType = std::variant<int, PhasePtr, std::string>;
using PhasePack = std::tuple<PhaseRefType, RegionType, VarIndexType, VarIndexType, VarIndexType>;

static void
build_link_interface(nb::class_<OptimalControlProblemBase, OptimizationProblemBase> &obj);

void TychoBind<OptimalControlProblemBase>::build(nb::module_ &m) {
    auto obj = nb::class_<OptimalControlProblemBase, OptimizationProblemBase>(
        m, "OptimalControlProblem",
        R"doc(A multi-phase optimal control problem.

Collects one or more :class:`PhaseInterface` phases and the *link* constraints
and objectives that couple them (e.g. continuity between the end of one phase
and the start of the next, or a cost spanning several phases). Each phase is
configured independently; the problem adds the cross-phase relationships and
solves all phases together.

Phases are referenced by an integer index (assigned in :meth:`add_phase`
order), by the phase object itself, or by a registered name. Solve the whole
problem with :meth:`optimize` (or :meth:`solve`), inherited from the shared
optimization-problem base.

Examples
--------
Assemble a multi-phase problem, link adjacent phases, and solve::

    ocp = OptimalControlProblem()
    ocp.add_phase(phase0)
    ocp.add_phase(phase1)
    ocp.add_forward_link_equal_con(0, 1, range(0, 5))
    ocp.optimize()
)doc");
    obj.def(nb::init<>(), R"doc(Construct an empty optimal control problem with no phases.)doc");

    build_link_interface(obj);

    //////////////////////////////////////////////////////////////////////////////

    obj.def("remove_link_equal_con", &OptimalControlProblemBase::remove_link_equal_con,
            R"doc(Remove a previously added link equality constraint by index.

Parameters
----------
index : int
    Index of the constraint to remove, or ``-1`` to remove the most recently
    added one.
)doc");
    obj.def("remove_link_inequal_con", &OptimalControlProblemBase::remove_link_inequal_con,
            R"doc(Remove a previously added link inequality constraint by index.

Parameters
----------
index : int
    Index of the constraint to remove, or ``-1`` to remove the most recently
    added one.
)doc");
    obj.def("remove_link_objective", &OptimalControlProblemBase::remove_link_objective,
            R"doc(Remove a previously added link objective by index.

Parameters
----------
index : int
    Index of the objective to remove, or ``-1`` to remove the most recently
    added one.
)doc");

    obj.def("add_phase", nb::overload_cast<PhasePtr>(&OptimalControlProblemBase::add_phase),
            R"doc(Add a phase to the problem.

Parameters
----------
phase : PhaseInterface
    The phase to append. Its problem index is assigned in call order
    (0, 1, ...).

Returns
-------
int
    The index assigned to the new phase.
)doc");

    obj.def("add_phases", &OptimalControlProblemBase::add_phases,
            R"doc(Add several phases at once.

Parameters
----------
phases : sequence of PhaseInterface
    Phases to append, in order.

Returns
-------
list of int
    The indices assigned to the new phases, in add order.
)doc");

    obj.def("get_phase_num", nb::overload_cast<PhasePtr>(&OptimalControlProblemBase::get_phase_num),
            R"doc(Return the problem index of a previously-added phase.

Parameters
----------
phase : PhaseInterface
    A phase that was added to this problem.

Returns
-------
int
    The phase's index within the problem.
)doc");

    obj.def("remove_phase", &OptimalControlProblemBase::remove_phase,
            R"doc(Remove a phase from the problem by index.

Parameters
----------
pnum : int
    Index of the phase to remove.
)doc");
    obj.def("phase", &OptimalControlProblemBase::phase,
            R"doc(Return a phase of the problem by index.

Parameters
----------
pnum : int
    Index of the phase to retrieve.

Returns
-------
PhaseInterface
    The requested phase.
)doc");

    obj.def("set_link_params",
            nb::overload_cast<VectorXd, VectorXd>(&OptimalControlProblemBase::set_link_params),
            R"doc(Set the shared link-parameter vector and its per-parameter scaling units.

Parameters
----------
parm : numpy.ndarray
    Link-parameter values.
units : numpy.ndarray
    Per-parameter scaling units; must be the same length as ``parm``.
)doc");
    obj.def("set_link_params",
            nb::overload_cast<VectorXd>(&OptimalControlProblemBase::set_link_params), "");

    obj.def("add_link_param_vgroups",
            nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                &OptimalControlProblemBase::add_link_param_vgroups),
            R"doc(Merge additional named link-parameter variable groups into the problem.

Parameters
----------
lpidxs : dict[str, numpy.ndarray of int]
    Map of group name to link-parameter index array. Existing groups with
    the same name are overwritten.
)doc");
    obj.def("set_link_param_vgroups",
            nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                &OptimalControlProblemBase::set_link_param_vgroups),
            R"doc(Replace all named link-parameter variable groups.

Parameters
----------
lpidxs : dict[str, numpy.ndarray of int]
    Map of group name to link-parameter index array. Any previously
    registered groups are discarded.
)doc");
    obj.def("add_link_param_vgroup",
            nb::overload_cast<Eigen::VectorXi, std::string>(
                &OptimalControlProblemBase::add_link_param_vgroup),
            R"doc(Register a named group of link-parameter indices.

Parameters
----------
idx : numpy.ndarray of int
    Link-parameter indices belonging to the group.
key : str
    Name for the group. Used to reference these indices by name in
    link-function variable selectors (the ``link_parms`` arguments of
    :meth:`add_link_equal_con`, :meth:`add_link_objective`, etc.).
)doc");
    obj.def("add_link_param_vgroup",
            nb::overload_cast<int, std::string>(&OptimalControlProblemBase::add_link_param_vgroup));

    obj.def("return_link_params", &OptimalControlProblemBase::return_link_params,
            R"doc(Return the current shared link-parameter vector.

Returns
-------
numpy.ndarray
    The link-parameter values set by :meth:`set_link_params`.
)doc");

    obj.def("transcribe", nb::overload_cast<bool, bool>(&OptimalControlProblemBase::transcribe),
            R"doc(Transcribe all phases and link functions into the shared NLP.

Parameters
----------
showstats : bool
    If ``True``, print problem-size statistics after transcription.
showfuns : bool
    If ``True``, print per-function details after transcription.
)doc");

    obj.def_ro("phases", &OptimalControlProblemBase::phases,
               R"doc(The list of phases registered with this problem, in add order.)doc");

    ///////////////////////
    obj.def("return_link_equal_con_vals", &OptimalControlProblemBase::return_link_equal_con_vals,
            R"doc(Return the residual values of a link equality constraint from the last solve.

Parameters
----------
index : int
    Index of the link equality constraint.

Returns
-------
list of numpy.ndarray
    Per-application residual vectors (one entry per application of the constraint).
)doc");
    obj.def("return_link_equal_con_lmults",
            &OptimalControlProblemBase::return_link_equal_con_lmults,
            R"doc(Return the Lagrange multipliers of a link equality constraint from the last solve.

Parameters
----------
index : int
    Index of the link equality constraint.

Returns
-------
list of numpy.ndarray
    Per-application Lagrange multiplier vectors.
)doc");

    obj.def("return_link_inequal_con_vals",
            &OptimalControlProblemBase::return_link_inequal_con_vals,
            R"doc(Return the residual values of a link inequality constraint from the last solve.

Parameters
----------
index : int
    Index of the link inequality constraint.

Returns
-------
list of numpy.ndarray
    Per-application residual vectors.
)doc");
    obj.def(
        "return_link_inequal_con_lmults",
        &OptimalControlProblemBase::return_link_inequal_con_lmults,
        R"doc(Return the Lagrange multipliers of a link inequality constraint from the last solve.

Parameters
----------
index : int
    Index of the link inequality constraint.

Returns
-------
list of numpy.ndarray
    Per-application Lagrange multiplier vectors.
)doc");

    obj.def("return_link_equal_con_scales",
            &OptimalControlProblemBase::return_link_equal_con_scales,
            R"doc(Return the per-output scale vector of a link equality constraint.

Parameters
----------
index : int
    Index of the link equality constraint.

Returns
-------
numpy.ndarray
    Per-output scale factors.
)doc");
    obj.def("return_link_inequal_con_scales",
            &OptimalControlProblemBase::return_link_inequal_con_scales,
            R"doc(Return the per-output scale vector of a link inequality constraint.

Parameters
----------
index : int
    Index of the link inequality constraint.

Returns
-------
numpy.ndarray
    Per-output scale factors.
)doc");
    obj.def("return_link_objective_scales",
            &OptimalControlProblemBase::return_link_objective_scales,
            R"doc(Return the per-output scale vector of a link objective.

Parameters
----------
index : int
    Index of the link objective.

Returns
-------
numpy.ndarray
    Per-output scale factors.
)doc");

    ///////////////////////

    obj.def_rw("auto_scaling", &OptimalControlProblemBase::auto_scaling_,
               R"doc(Enable automatic output scaling for link functions and phases.)doc");
    obj.def_rw("sync_objective_scales", &OptimalControlProblemBase::sync_objective_scales_,
               R"doc(Keep link-objective scales in sync when automatic scaling is updated.)doc");

    obj.def_rw("adaptive_mesh", &OptimalControlProblemBase::adaptive_mesh_,
               R"doc(Enable problem-wide adaptive mesh refinement across all phases.)doc");
    obj.def_rw("print_mesh_info", &OptimalControlProblemBase::print_mesh_info_,
               R"doc(Print mesh-refinement diagnostics after each mesh iteration.)doc");
    obj.def_rw("max_mesh_iters", &OptimalControlProblemBase::max_mesh_iters_,
               R"doc(Maximum number of adaptive mesh-refinement iterations.)doc");
    obj.def_ro(
        "mesh_converged", &OptimalControlProblemBase::mesh_converged_,
        R"doc(``True`` if all adaptive phases satisfied their mesh-error tolerance on the last run.)doc");
    obj.def_rw("solve_only_first", &OptimalControlProblemBase::solve_only_first_,
               R"doc(Run only the initial *solve* step on the first mesh-refinement iteration.

Only affects the compound ``SolveOptimize`` and ``SolveOptimizeSolve`` job
modes: when ``True``, subsequent mesh iterations drop the leading solve step
(running ``Optimize`` / ``OptimizeSolve`` instead). Has no effect on plain
``solve`` or ``optimize``.
)doc");

    obj.def_rw("mesh_abort_flag", &OptimalControlProblemBase::mesh_abort_flag_,
               R"doc(Severity threshold at which the adaptive mesh-refinement loop aborts early.

The loop aborts as soon as the solver's convergence flag is at or above this
severity (the check is ``flag >= mesh_abort_flag``). The flag ordering is
``CONVERGED < ACCEPTABLE < NOTCONVERGED < DIVERGING``. Default is
``DIVERGING`` (only divergence aborts); set it to ``NOTCONVERGED`` to also
abort when a mesh iteration fails to converge.
)doc");

    obj.def("set_adaptive_mesh", &OptimalControlProblemBase::set_adaptive_mesh,
            nb::arg("adaptive_mesh") = true, nb::arg("apply_to_phases") = true,
            R"doc(Enable or disable adaptive mesh refinement, optionally propagating to all phases.

Parameters
----------
adaptive_mesh : bool, optional
    Whether to enable adaptive mesh refinement. Default ``True``.
apply_to_phases : bool, optional
    If ``True``, also set the flag on every registered phase. Default ``True``.
)doc");
    obj.def("set_auto_scaling", &OptimalControlProblemBase::set_auto_scaling,
            nb::arg("auto_scaling") = true, nb::arg("apply_to_phases") = true,
            R"doc(Enable or disable automatic output scaling, optionally propagating to all phases.

Parameters
----------
auto_scaling : bool, optional
    Whether to enable automatic scaling. Default ``True``.
apply_to_phases : bool, optional
    If ``True``, also set the flag on every registered phase. Default ``True``.
)doc");

    obj.def("set_mesh_tol", &OptimalControlProblemBase::set_mesh_tol,
            R"doc(Set the mesh-error tolerance on every registered phase.

Parameters
----------
t : float
    Desired mesh-error tolerance.
)doc");
    obj.def("set_mesh_red_factor", &OptimalControlProblemBase::set_mesh_red_factor,
            R"doc(Set the mesh segment-count reduction factor on every registered phase.

Parameters
----------
t : float
    Reduction factor applied when the mesh is too fine.
)doc");
    obj.def("set_mesh_inc_factor", &OptimalControlProblemBase::set_mesh_inc_factor,
            R"doc(Set the mesh segment-count increase factor on every registered phase.

Parameters
----------
t : float
    Increase factor applied when the mesh is too coarse.
)doc");
    obj.def("set_mesh_err_factor", &OptimalControlProblemBase::set_mesh_err_factor,
            R"doc(Set the mesh error-overshoot factor on every registered phase.

Parameters
----------
t : float
    Overshoot factor used when estimating how many segments are needed.
)doc");
    obj.def("set_max_mesh_iters", &OptimalControlProblemBase::set_max_mesh_iters,
            R"doc(Set the maximum number of problem-level mesh-refinement iterations.

Parameters
----------
it : int
    Iteration cap.
)doc");
    obj.def("set_min_segments", &OptimalControlProblemBase::set_min_segments,
            R"doc(Set the minimum mesh segment count on every registered phase.

Parameters
----------
it : int
    Minimum number of collocation segments per phase.
)doc");
    obj.def("set_max_segments", &OptimalControlProblemBase::set_max_segments,
            R"doc(Set the maximum mesh segment count on every registered phase.

Parameters
----------
it : int
    Maximum number of collocation segments per phase.
)doc");
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<MeshErrorAggregation>(
                &OptimalControlProblemBase::set_mesh_error_criteria),
            R"doc(Set the mesh-error convergence aggregation on every registered phase.

Parameters
----------
m : MeshErrorAggregation
    Aggregation mode (e.g. ``Max``, ``Mean``) used to decide whether the mesh
    has converged.
)doc");
    obj.def("set_mesh_error_criteria", nb::overload_cast<const std::string &>(
                                           &OptimalControlProblemBase::set_mesh_error_criteria));
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<MeshErrorEstimators>(
                &OptimalControlProblemBase::set_mesh_error_estimator),
            R"doc(Set the mesh-error estimator on every registered phase.

Parameters
----------
m : MeshErrorEstimators
    Estimator used to compute the per-segment mesh error.
)doc");
    obj.def("set_mesh_error_estimator", nb::overload_cast<const std::string &>(
                                            &OptimalControlProblemBase::set_mesh_error_estimator));
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<MeshErrorAggregation>(
                &OptimalControlProblemBase::set_mesh_error_distributor),
            R"doc(Set the mesh-error distribution aggregation on every registered phase.

Parameters
----------
m : MeshErrorAggregation
    Aggregation mode used to distribute error across segments when refining.
)doc");
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<const std::string &>(
                &OptimalControlProblemBase::set_mesh_error_distributor));
}

static void
build_link_interface(nb::class_<OptimalControlProblemBase, OptimizationProblemBase> &obj) {

    //////////// equal_cons_////////////////////////////////////////
    {
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_equal_con),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add a link equality constraint ``func == 0`` over a list of phase targets.

Evaluates ``func`` on variables drawn from each entry of ``phasepack`` (and
optionally the shared link parameters) and enforces the result to be zero.

Parameters
----------
func : VectorFunction
    Vector-valued function; its output is constrained to zero.
phasepack : list of tuple
    Per-phase link targets, each specifying a phase reference, region, and
    variable selectors.
linkparams : numpy.ndarray of int, optional
    Indices into the shared link-parameter vector to also pass to ``func``.
    Default empty.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new link equality constraint.
)doc");

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_equal_con),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("xt_u_vars0"),
            nb::arg("op_vars0"), nb::arg("sp_vars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("xt_u_vars1"), nb::arg("op_vars1"), nb::arg("sp_vars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_link_equal_con),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_link_param_equal_con",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_equal_con),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Add an equality constraint ``func == 0`` on the shared link parameters.

Parameters
----------
func : VectorFunction
    Vector-valued function of the selected link parameters.
link_parms : list of numpy.ndarray of int
    Per-application link-parameter index vectors.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");
        obj.def("add_link_param_equal_con",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_equal_con),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_forward_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_forward_link_equal_con),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("vars"),
                nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Add a continuity (forward-link) constraint between two phases.

Constrains the selected variables at the *back* of ``phase0`` to equal the
same variables at the *front* of ``phase1`` -- the standard way to chain
phases into a continuous trajectory.

Parameters
----------
phase0 : int, PhaseInterface, or str
    The upstream phase (its back/final node is used).
phase1 : int, PhaseInterface, or str
    The downstream phase (its front/initial node is used).
vars : int or sequence of int
    Variable index/indices (into the packed ``[x, t, u, p]`` node layout, where
    ``p`` are the ODE parameters -- not the static parameters) to make
    continuous across the link.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
list of int
    The index/indices assigned to the added link constraint(s).

Examples
--------
Make the first five variables continuous across phases 0 and 1::

    ocp.add_forward_link_equal_con(0, 1, range(0, 5))
)doc");

        obj.def("add_param_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, RegionType, VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_param_link_equal_con),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("reg0"), nb::arg("vars"),
                nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Chain consecutive phases with equal-parameter continuity constraints.

For each consecutive pair of phases from ``phase0`` to ``phase1``, adds a
link equality constraint enforcing that the selected parameter variables
are identical across the join.

Parameters
----------
phase0 : int, PhaseInterface, or str
    Starting phase reference.
phase1 : int, PhaseInterface, or str
    Ending phase reference (must have a higher index than ``phase0``).
reg0 : PhaseRegionFlags or str
    Parameter region (must be ``ODEParams`` or ``StaticParams``).
vars : int or sequence of int
    Parameter variable index/indices to make continuous.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
list of int
    The indices assigned to the added constraints (one per adjacent phase pair).
)doc");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType,
                                  VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_direct_link_equal_con),
                nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"), nb::arg("phase1"),
                nb::arg("reg1"), nb::arg("v1"), nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Constrain selected variables from two phase regions to be exactly equal.

Adds a link equality constraint enforcing ``v0 - v1 == 0`` between the
selected variables drawn from ``phase0`` and ``phase1``.

Parameters
----------
phase0 : int, PhaseInterface, or str
    First phase reference.
reg0 : PhaseRegionFlags or str
    Region of ``phase0`` to read ``v0`` from.
v0 : int or sequence of int
    Variable index/indices from ``phase0``.
phase1 : int, PhaseInterface, or str
    Second phase reference.
reg1 : PhaseRegionFlags or str
    Region of ``phase1`` to read ``v1`` from.
v1 : int or sequence of int
    Variable index/indices from ``phase1``.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");
    }

    //////////// inequal_cons_////////////////////////////////////////
    {
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_inequal_con),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("auto_scale").none() = std::string("auto"),
            R"doc(Add a link inequality constraint ``func <= 0`` over a list of phase targets.

Evaluates ``func`` on variables drawn from each entry of ``phasepack`` (and
optionally the shared link parameters) and enforces the result to be
non-positive.

Parameters
----------
func : VectorFunction
    Vector-valued function; each output is constrained to be ``<= 0``.
phasepack : list of tuple
    Per-phase link targets, each specifying a phase reference, region, and
    variable selectors.
linkparams : numpy.ndarray of int, optional
    Indices into the shared link-parameter vector to also pass to ``func``.
    Default empty.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new link inequality constraint.
)doc");

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_inequal_con),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("xt_u_vars0"),
            nb::arg("op_vars0"), nb::arg("sp_vars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("xt_u_vars1"), nb::arg("op_vars1"), nb::arg("sp_vars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_link_inequal_con),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_link_param_inequal_con",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_inequal_con),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Add an inequality constraint ``func <= 0`` on the shared link parameters.

Parameters
----------
func : VectorFunction
    Vector-valued function of the selected link parameters.
link_parms : list of numpy.ndarray of int
    Per-application link-parameter index vectors.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new constraint.
)doc");
        obj.def("add_link_param_inequal_con",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_inequal_con),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"));
    }
    //////////// objectives_ ////////////////////////////////////////
    {
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_objective),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("auto_scale").none() = std::string("auto"));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_objective),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("xt_u_vars0"),
            nb::arg("op_vars0"), nb::arg("sp_vars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("xt_u_vars1"), nb::arg("op_vars1"), nb::arg("sp_vars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_link_objective),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Add an objective coupling variables of two phases.

Adds the scalar value of ``func`` -- evaluated on selected variables drawn
from a region of ``phase0`` and a region of ``phase1`` (and optionally the
shared link parameters) -- to the problem objective.

Parameters
----------
func : ScalarFunction
    Scalar-valued function of the linked variables.
phase0 : int, PhaseInterface, or str
    First phase.
reg0 : PhaseRegionFlags or str
    Region of ``phase0`` the variables ``v0`` are read from.
v0 : int or sequence of int
    Variable index/indices from ``phase0``.
phase1 : int, PhaseInterface, or str
    Second phase.
reg1 : PhaseRegionFlags or str
    Region of ``phase1`` the variables ``v1`` are read from.
v1 : int or sequence of int
    Variable index/indices from ``phase1``.
linkparams : numpy.ndarray of int, optional
    Indices into the shared link-parameter vector to also pass to ``func``.
    Default empty.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Returns
-------
int
    The index assigned to the new link objective.
)doc");

        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_objective),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"),
                R"doc(Add a scalar objective on the shared link parameters.

Parameters
----------
func : ScalarFunction
    Scalar-valued function of the selected link parameters.
link_parms : list of numpy.ndarray of int
    Per-application link-parameter index vectors.
auto_scale : ScaleModes or str, optional
    Output-scaling mode. Defaults to ``"auto"``.

Returns
-------
int
    The index assigned to the new link objective.
)doc");
        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_objective),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"));
    }
}
