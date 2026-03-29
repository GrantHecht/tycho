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
using RegVec = Eigen::Matrix<PhaseRegionFlags, -1, 1>;
using LinkConstraint = LinkFunction<VectorFunctionalX>;
using LinkObjective = LinkFunction<ScalarFunctionalX>;
using PhaseRefType = std::variant<int, PhasePtr, std::string>;
using PhasePack = std::tuple<PhaseRefType, RegionType, VarIndexType, VarIndexType, VarIndexType>;

static void BuildNewLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj);
static void BuildOldLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj);

void TychoBind<OptimalControlProblem>::Build(nb::module_ &m) {
    auto obj =
        nb::class_<OptimalControlProblem, OptimizationProblemBase>(m, "OptimalControlProblem");
    obj.def(nb::init<>());

    BuildNewLinkIterface(obj);
    BuildOldLinkIterface(obj);

    //////////////////////////////////////////////////////////////////////////////

    obj.def("add_link_param_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::add_link_param_equal_con),
            "");
    obj.def("add_link_param_equal_con",
            nb::overload_cast<VectorFunctionalX, VectorXi>(
                &OptimalControlProblem::add_link_param_equal_con),
            "");
    obj.def("add_link_param_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::add_link_param_inequal_con),
            "");
    obj.def("add_link_param_inequal_con",
            nb::overload_cast<VectorFunctionalX, VectorXi>(
                &OptimalControlProblem::add_link_param_inequal_con),
            "");
    obj.def("add_link_param_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::add_link_param_objective),
            "");
    obj.def("add_link_param_objective",
            nb::overload_cast<ScalarFunctionalX, VectorXi>(
                &OptimalControlProblem::add_link_param_objective),
            "");

    //////////////////////////////////////////////////////////////////////////////

    obj.def("remove_link_equal_con", &OptimalControlProblem::remove_link_equal_con, "");
    obj.def("remove_link_inequal_con", &OptimalControlProblem::remove_link_inequal_con, "");
    obj.def("remove_link_objective", &OptimalControlProblem::remove_link_objective, "");

    obj.def("add_phase", nb::overload_cast<PhasePtr>(&OptimalControlProblem::add_phase), "");

    obj.def("add_phases", &OptimalControlProblem::add_phases);

    obj.def("get_phase_num", nb::overload_cast<PhasePtr>(&OptimalControlProblem::get_phase_num));

    obj.def("remove_phase", &OptimalControlProblem::remove_phase, "");
    obj.def("phase", &OptimalControlProblem::Phase, "");

    obj.def("set_link_params",
            nb::overload_cast<VectorXd, VectorXd>(&OptimalControlProblem::set_link_params));
    obj.def("set_link_params", nb::overload_cast<VectorXd>(&OptimalControlProblem::set_link_params),
            "");

    obj.def("add_link_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                          &OptimalControlProblem::add_link_param_vgroups));
    obj.def("set_link_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                          &OptimalControlProblem::set_link_param_vgroups));
    obj.def("add_link_param_vgroup", nb::overload_cast<Eigen::VectorXi, std::string>(
                                         &OptimalControlProblem::add_link_param_vgroup));
    obj.def("add_link_param_vgroup",
            nb::overload_cast<int, std::string>(&OptimalControlProblem::add_link_param_vgroup));

    obj.def("return_link_params", &OptimalControlProblem::return_link_params, "");

    obj.def("transcribe", nb::overload_cast<bool, bool>(&OptimalControlProblem::transcribe), "");

    obj.def_ro("phases", &OptimalControlProblem::phases, "");

    ///////////////////////
    obj.def("return_link_equal_con_vals", &OptimalControlProblem::return_link_equal_con_vals);
    obj.def("return_link_equal_con_lmults", &OptimalControlProblem::return_link_equal_con_lmults);

    obj.def("return_link_inequal_con_vals", &OptimalControlProblem::return_link_inequal_con_vals);
    obj.def("return_link_inequal_con_lmults",
            &OptimalControlProblem::return_link_inequal_con_lmults);

    obj.def("return_link_equal_con_scales", &OptimalControlProblem::return_link_equal_con_scales);
    obj.def("return_link_inequal_con_scales",
            &OptimalControlProblem::return_link_inequal_con_scales);
    obj.def("return_link_objective_scales", &OptimalControlProblem::return_link_objective_scales);

    ///////////////////////

    obj.def_rw("auto_scaling", &OptimalControlProblem::auto_scaling_);
    obj.def_rw("sync_objective_scales", &OptimalControlProblem::sync_objective_scales_);

    obj.def_rw("adaptive_mesh", &OptimalControlProblem::adaptive_mesh_);
    obj.def_rw("print_mesh_info", &OptimalControlProblem::print_mesh_info_);
    obj.def_rw("max_mesh_iters", &OptimalControlProblem::max_mesh_iters_);
    obj.def_ro("mesh_converged", &OptimalControlProblem::mesh_converged_);
    obj.def_rw("solve_only_first", &OptimalControlProblem::solve_only_first_);

    obj.def_rw("mesh_abort_flag", &OptimalControlProblem::mesh_abort_flag_);

    obj.def("set_adaptive_mesh", &OptimalControlProblem::set_adaptive_mesh,
            nb::arg("adaptive_mesh_") = true, nb::arg("ApplyToPhases") = true);
    obj.def("set_auto_scaling", &OptimalControlProblem::set_auto_scaling,
            nb::arg("auto_scaling_") = true, nb::arg("ApplyToPhases") = true);

    obj.def("set_mesh_tol", &OptimalControlProblem::set_mesh_tol);
    obj.def("set_mesh_red_factor", &OptimalControlProblem::set_mesh_red_factor);
    obj.def("set_mesh_inc_factor", &OptimalControlProblem::set_mesh_inc_factor);
    obj.def("set_mesh_err_factor", &OptimalControlProblem::set_mesh_err_factor);
    obj.def("set_max_mesh_iters", &OptimalControlProblem::set_max_mesh_iters);
    obj.def("set_min_segments", &OptimalControlProblem::set_min_segments);
    obj.def("set_max_segments", &OptimalControlProblem::set_max_segments);
    obj.def("set_mesh_error_criteria", nb::overload_cast<MeshErrorAggregation>(
                                           &OptimalControlProblem::set_mesh_error_criteria));
    obj.def("set_mesh_error_criteria", nb::overload_cast<const std::string &>(
                                           &OptimalControlProblem::set_mesh_error_criteria));
    obj.def("set_mesh_error_estimator", nb::overload_cast<MeshErrorEstimators>(
                                            &OptimalControlProblem::set_mesh_error_estimator));
    obj.def("set_mesh_error_estimator", nb::overload_cast<const std::string &>(
                                            &OptimalControlProblem::set_mesh_error_estimator));
    obj.def("set_mesh_error_distributor", nb::overload_cast<MeshErrorAggregation>(
                                              &OptimalControlProblem::set_mesh_error_distributor));
    obj.def("set_mesh_error_distributor", nb::overload_cast<const std::string &>(
                                              &OptimalControlProblem::set_mesh_error_distributor));
}

static void BuildNewLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj) {

    //////////// EqualCons////////////////////////////////////////
    {
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::add_link_equal_con),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::add_link_equal_con),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::add_link_equal_con),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_param_equal_con",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::add_link_param_equal_con),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("add_link_param_equal_con",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::add_link_param_equal_con),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_forward_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::add_forward_link_equal_con),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("vars"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_param_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, RegionType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::add_param_link_equal_con),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("reg0"), nb::arg("vars"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType,
                                  VarIndexType, ScaleType>(
                    &OptimalControlProblem::add_direct_link_equal_con),
                nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"), nb::arg("phase1"),
                nb::arg("reg1"), nb::arg("v1"), nb::arg("AutoScale").none() = std::string("auto"));
    }

    //////////// InequalCons////////////////////////////////////////
    {
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::add_link_inequal_con),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::add_link_inequal_con),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::add_link_inequal_con),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_param_inequal_con",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::add_link_param_inequal_con),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("add_link_param_inequal_con",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::add_link_param_inequal_con),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
    }
    //////////// Objectives ////////////////////////////////////////
    {
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::add_link_objective),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::add_link_objective),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::add_link_objective),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::add_link_param_objective),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::add_link_param_objective),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
    }
}

static void BuildOldLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj) {
    {
        ////////////////// Legacy EqualCons//////////
        obj.def("add_link_equal_con",
                nb::overload_cast<LinkConstraint>(&OptimalControlProblem::add_link_equal_con), "");

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));

        /// ///
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::add_link_equal_con));

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_equal_con));

        ////////////////////////////

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::add_link_equal_con),
                "");

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, RegVec,
                                  std::vector<std::vector<std::shared_ptr<ODEPhaseBase>>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::add_link_equal_con),
                "");

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::add_link_equal_con),
                "");
        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::add_link_equal_con),
                "");

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::add_link_equal_con),
                "");
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::add_link_equal_con),
            "");

        obj.def("add_forward_link_equal_con",
                nb::overload_cast<int, int, VectorXi>(
                    &OptimalControlProblem::add_forward_link_equal_con),
                "");

        obj.def("add_forward_link_equal_con",
                nb::overload_cast<PhasePtr, PhasePtr, VectorXi>(
                    &OptimalControlProblem::add_forward_link_equal_con),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<LinkFlags, int, VectorXi, int, VectorXi>(
                    &OptimalControlProblem::add_direct_link_equal_con),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, int, PhaseRegionFlags, VectorXi, int,
                                  PhaseRegionFlags, VectorXi>(
                    &OptimalControlProblem::add_direct_link_equal_con),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, PhasePtr, PhaseRegionFlags, VectorXi, PhasePtr,
                                  PhaseRegionFlags, VectorXi>(
                    &OptimalControlProblem::add_direct_link_equal_con),
                "");

        //
        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, int, std::string, VectorXi, int, std::string,
                                  VectorXi>(&OptimalControlProblem::add_direct_link_equal_con),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, PhasePtr, std::string, VectorXi, PhasePtr,
                                  std::string, VectorXi>(
                    &OptimalControlProblem::add_direct_link_equal_con),
                "");
    }

    //////////////////Legacy  InequalCons//////////
    {

        obj.def("add_link_inequal_con",
                nb::overload_cast<LinkConstraint>(&OptimalControlProblem::add_link_inequal_con),
                "");

        ////////////////////////////

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));

        /// ///
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::add_link_inequal_con));

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_inequal_con));

        ////////////////////////////

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::add_link_inequal_con),
                "");

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::add_link_inequal_con),
                "");

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::add_link_inequal_con),
                "");

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::add_link_inequal_con),
            "");

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::add_link_inequal_con));

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::add_link_inequal_con));

        ////////////////////////////////////////
        obj.def("add_link_objective",
                nb::overload_cast<LinkObjective>(&OptimalControlProblem::add_link_objective), "");
    }

    {
        //////////////////Legacy LinkObjectives//////////

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));

        /// ///
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::add_link_objective));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::add_link_objective));

        ////////////////////////////

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::add_link_objective),
                "");
        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::add_link_objective),
                "");

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::add_link_objective),
                "");
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::add_link_objective),
            "");

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::add_link_objective),
                "");

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::add_link_objective),
                "");
    }
}
