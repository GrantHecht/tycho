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

    obj.def("remove_link_equal_con", &OptimalControlProblemBase::remove_link_equal_con, "");
    obj.def("remove_link_inequal_con", &OptimalControlProblemBase::remove_link_inequal_con, "");
    obj.def("remove_link_objective", &OptimalControlProblemBase::remove_link_objective, "");

    obj.def("add_phase", nb::overload_cast<PhasePtr>(&OptimalControlProblemBase::add_phase),
            R"doc(Add a phase to the problem.

Parameters
----------
phase : PhaseInterface
    The phase to append. Its problem index is assigned in call order
    (0, 1, ...).
)doc");

    obj.def("add_phases", &OptimalControlProblemBase::add_phases,
            R"doc(Add several phases at once.

Parameters
----------
phases : sequence of PhaseInterface
    Phases to append, in order.
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
            nb::overload_cast<VectorXd, VectorXd>(&OptimalControlProblemBase::set_link_params));
    obj.def("set_link_params",
            nb::overload_cast<VectorXd>(&OptimalControlProblemBase::set_link_params), "");

    obj.def("add_link_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                          &OptimalControlProblemBase::add_link_param_vgroups));
    obj.def("set_link_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                          &OptimalControlProblemBase::set_link_param_vgroups));
    obj.def("add_link_param_vgroup", nb::overload_cast<Eigen::VectorXi, std::string>(
                                         &OptimalControlProblemBase::add_link_param_vgroup));
    obj.def("add_link_param_vgroup",
            nb::overload_cast<int, std::string>(&OptimalControlProblemBase::add_link_param_vgroup));

    obj.def("return_link_params", &OptimalControlProblemBase::return_link_params, "");

    obj.def("transcribe", nb::overload_cast<bool, bool>(&OptimalControlProblemBase::transcribe),
            "");

    obj.def_ro("phases", &OptimalControlProblemBase::phases, "");

    ///////////////////////
    obj.def("return_link_equal_con_vals", &OptimalControlProblemBase::return_link_equal_con_vals);
    obj.def("return_link_equal_con_lmults",
            &OptimalControlProblemBase::return_link_equal_con_lmults);

    obj.def("return_link_inequal_con_vals",
            &OptimalControlProblemBase::return_link_inequal_con_vals);
    obj.def("return_link_inequal_con_lmults",
            &OptimalControlProblemBase::return_link_inequal_con_lmults);

    obj.def("return_link_equal_con_scales",
            &OptimalControlProblemBase::return_link_equal_con_scales);
    obj.def("return_link_inequal_con_scales",
            &OptimalControlProblemBase::return_link_inequal_con_scales);
    obj.def("return_link_objective_scales",
            &OptimalControlProblemBase::return_link_objective_scales);

    ///////////////////////

    obj.def_rw("auto_scaling", &OptimalControlProblemBase::auto_scaling_);
    obj.def_rw("sync_objective_scales", &OptimalControlProblemBase::sync_objective_scales_);

    obj.def_rw("adaptive_mesh", &OptimalControlProblemBase::adaptive_mesh_);
    obj.def_rw("print_mesh_info", &OptimalControlProblemBase::print_mesh_info_);
    obj.def_rw("max_mesh_iters", &OptimalControlProblemBase::max_mesh_iters_);
    obj.def_ro("mesh_converged", &OptimalControlProblemBase::mesh_converged_);
    obj.def_rw("solve_only_first", &OptimalControlProblemBase::solve_only_first_);

    obj.def_rw("mesh_abort_flag", &OptimalControlProblemBase::mesh_abort_flag_);

    obj.def("set_adaptive_mesh", &OptimalControlProblemBase::set_adaptive_mesh,
            nb::arg("adaptive_mesh") = true, nb::arg("apply_to_phases") = true);
    obj.def("set_auto_scaling", &OptimalControlProblemBase::set_auto_scaling,
            nb::arg("auto_scaling") = true, nb::arg("apply_to_phases") = true);

    obj.def("set_mesh_tol", &OptimalControlProblemBase::set_mesh_tol);
    obj.def("set_mesh_red_factor", &OptimalControlProblemBase::set_mesh_red_factor);
    obj.def("set_mesh_inc_factor", &OptimalControlProblemBase::set_mesh_inc_factor);
    obj.def("set_mesh_err_factor", &OptimalControlProblemBase::set_mesh_err_factor);
    obj.def("set_max_mesh_iters", &OptimalControlProblemBase::set_max_mesh_iters);
    obj.def("set_min_segments", &OptimalControlProblemBase::set_min_segments);
    obj.def("set_max_segments", &OptimalControlProblemBase::set_max_segments);
    obj.def("set_mesh_error_criteria", nb::overload_cast<MeshErrorAggregation>(
                                           &OptimalControlProblemBase::set_mesh_error_criteria));
    obj.def("set_mesh_error_criteria", nb::overload_cast<const std::string &>(
                                           &OptimalControlProblemBase::set_mesh_error_criteria));
    obj.def("set_mesh_error_estimator", nb::overload_cast<MeshErrorEstimators>(
                                            &OptimalControlProblemBase::set_mesh_error_estimator));
    obj.def("set_mesh_error_estimator", nb::overload_cast<const std::string &>(
                                            &OptimalControlProblemBase::set_mesh_error_estimator));
    obj.def("set_mesh_error_distributor",
            nb::overload_cast<MeshErrorAggregation>(
                &OptimalControlProblemBase::set_mesh_error_distributor));
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
            nb::arg("auto_scale").none() = std::string("auto"));

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
                nb::arg("auto_scale").none() = std::string("auto"));
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
    Variable index/indices (into the packed ``[x, t, u, p]`` layout) to make
    continuous across the link.
auto_scale : ScaleModes or str, optional
    Output-scaling mode (``"auto"``, ``"custom"``, or ``"none"``). Defaults to
    ``"auto"``.

Examples
--------
Make the first five variables continuous across phases 0 and 1::

    ocp.add_forward_link_equal_con(0, 1, range(0, 5))
)doc");

        obj.def("add_param_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, RegionType, VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_param_link_equal_con),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("reg0"), nb::arg("vars"),
                nb::arg("auto_scale").none() = std::string("auto"));

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType,
                                  VarIndexType, ScaleType>(
                    &OptimalControlProblemBase::add_direct_link_equal_con),
                nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"), nb::arg("phase1"),
                nb::arg("reg1"), nb::arg("v1"), nb::arg("auto_scale").none() = std::string("auto"));
    }

    //////////// inequal_cons_////////////////////////////////////////
    {
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblemBase::add_link_inequal_con),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("auto_scale").none() = std::string("auto"));

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
                nb::arg("auto_scale").none() = std::string("auto"));
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
                nb::arg("auto_scale").none() = std::string("auto"));
        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblemBase::add_link_param_objective),
                nb::arg("func"), nb::arg("link_parms"),
                nb::arg("auto_scale").none() = std::string("auto"));
    }
}
