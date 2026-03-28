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
//   - Namespace: Tycho
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
                &OptimalControlProblem::addLinkParamEqualCon),
            "");
    obj.def("add_link_param_equal_con",
            nb::overload_cast<VectorFunctionalX, VectorXi>(
                &OptimalControlProblem::addLinkParamEqualCon),
            "");
    obj.def("add_link_param_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkParamInequalCon),
            "");
    obj.def("add_link_param_inequal_con",
            nb::overload_cast<VectorFunctionalX, VectorXi>(
                &OptimalControlProblem::addLinkParamInequalCon),
            "");
    obj.def("add_link_param_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkParamObjective),
            "");
    obj.def("add_link_param_objective",
            nb::overload_cast<ScalarFunctionalX, VectorXi>(
                &OptimalControlProblem::addLinkParamObjective),
            "");

    //////////////////////////////////////////////////////////////////////////////

    obj.def("remove_link_equal_con", &OptimalControlProblem::removeLinkEqualCon, "");
    obj.def("remove_link_inequal_con", &OptimalControlProblem::removeLinkInequalCon, "");
    obj.def("remove_link_objective", &OptimalControlProblem::removeLinkObjective, "");

    obj.def("add_phase", nb::overload_cast<PhasePtr>(&OptimalControlProblem::addPhase), "");

    obj.def("add_phases", &OptimalControlProblem::addPhases);

    obj.def("get_phase_num", nb::overload_cast<PhasePtr>(&OptimalControlProblem::getPhaseNum));

    obj.def("remove_phase", &OptimalControlProblem::removePhase, "");
    obj.def("phase", &OptimalControlProblem::Phase, "");

    obj.def("set_link_params",
            nb::overload_cast<VectorXd, VectorXd>(&OptimalControlProblem::setLinkParams));
    obj.def("set_link_params", nb::overload_cast<VectorXd>(&OptimalControlProblem::setLinkParams),
            "");

    obj.def("add_link_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                          &OptimalControlProblem::addLinkParamVgroups));
    obj.def("set_link_param_vgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                          &OptimalControlProblem::setLinkParamVgroups));
    obj.def("add_link_param_vgroup", nb::overload_cast<Eigen::VectorXi, std::string>(
                                         &OptimalControlProblem::addLinkParamVgroup));
    obj.def("add_link_param_vgroup",
            nb::overload_cast<int, std::string>(&OptimalControlProblem::addLinkParamVgroup));

    obj.def("return_link_params", &OptimalControlProblem::returnLinkParams, "");

    obj.def("transcribe", nb::overload_cast<bool, bool>(&OptimalControlProblem::transcribe), "");

    obj.def_ro("phases", &OptimalControlProblem::phases, "");

    ///////////////////////
    obj.def("return_link_equal_con_vals", &OptimalControlProblem::returnLinkEqualConVals);
    obj.def("return_link_equal_con_lmults", &OptimalControlProblem::returnLinkEqualConLmults);

    obj.def("return_link_inequal_con_vals", &OptimalControlProblem::returnLinkInequalConVals);
    obj.def("return_link_inequal_con_lmults", &OptimalControlProblem::returnLinkInequalConLmults);

    obj.def("return_link_equal_con_scales", &OptimalControlProblem::returnLinkEqualConScales);
    obj.def("return_link_inequal_con_scales", &OptimalControlProblem::returnLinkInequalConScales);
    obj.def("return_link_objective_scales", &OptimalControlProblem::returnLinkObjectiveScales);

    ///////////////////////

    obj.def_rw("auto_scaling", &OptimalControlProblem::AutoScaling);
    obj.def_rw("sync_objective_scales", &OptimalControlProblem::SyncObjectiveScales);

    obj.def_rw("adaptive_mesh", &OptimalControlProblem::AdaptiveMesh);
    obj.def_rw("print_mesh_info", &OptimalControlProblem::PrintMeshInfo);
    obj.def_rw("max_mesh_iters", &OptimalControlProblem::MaxMeshIters);
    obj.def_ro("mesh_converged", &OptimalControlProblem::MeshConverged);
    obj.def_rw("solve_only_first", &OptimalControlProblem::SolveOnlyFirst);

    obj.def_rw("mesh_abort_flag", &OptimalControlProblem::MeshAbortFlag);

    obj.def("set_adaptive_mesh", &OptimalControlProblem::setAdaptiveMesh,
            nb::arg("AdaptiveMesh") = true, nb::arg("ApplyToPhases") = true);
    obj.def("set_auto_scaling", &OptimalControlProblem::setAutoScaling,
            nb::arg("AutoScaling") = true, nb::arg("ApplyToPhases") = true);

    obj.def("set_mesh_tol", &OptimalControlProblem::setMeshTol);
    obj.def("set_mesh_red_factor", &OptimalControlProblem::setMeshRedFactor);
    obj.def("set_mesh_inc_factor", &OptimalControlProblem::setMeshIncFactor);
    obj.def("set_mesh_err_factor", &OptimalControlProblem::setMeshErrFactor);
    obj.def("set_max_mesh_iters", &OptimalControlProblem::setMaxMeshIters);
    obj.def("set_min_segments", &OptimalControlProblem::setMinSegments);
    obj.def("set_max_segments", &OptimalControlProblem::setMaxSegments);
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<MeshErrorAggregation>(&OptimalControlProblem::setMeshErrorCriteria));
    obj.def("set_mesh_error_criteria",
            nb::overload_cast<const std::string &>(&OptimalControlProblem::setMeshErrorCriteria));
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<MeshErrorEstimators>(&OptimalControlProblem::setMeshErrorEstimator));
    obj.def("set_mesh_error_estimator",
            nb::overload_cast<const std::string &>(&OptimalControlProblem::setMeshErrorEstimator));
    obj.def("set_mesh_error_distributor", nb::overload_cast<MeshErrorAggregation>(
                                              &OptimalControlProblem::setMeshErrorDistributor));
    obj.def("set_mesh_error_distributor", nb::overload_cast<const std::string &>(
                                              &OptimalControlProblem::setMeshErrorDistributor));
}

static void BuildNewLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj) {

    //////////// EqualCons////////////////////////////////////////
    {
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkEqualCon),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkEqualCon),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addLinkEqualCon),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_param_equal_con",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::addLinkParamEqualCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("add_link_param_equal_con",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::addLinkParamEqualCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_forward_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addForwardLinkEqualCon),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("vars"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_param_link_equal_con",
                nb::overload_cast<PhaseRefType, PhaseRefType, RegionType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addParamLinkEqualCon),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("reg0"), nb::arg("vars"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType,
                                  VarIndexType, ScaleType>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"), nb::arg("phase1"),
                nb::arg("reg1"), nb::arg("v1"), nb::arg("AutoScale").none() = std::string("auto"));
    }

    //////////// InequalCons////////////////////////////////////////
    {
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkInequalCon),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkInequalCon),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addLinkInequalCon),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_param_inequal_con",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::addLinkParamInequalCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("add_link_param_inequal_con",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::addLinkParamInequalCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
    }
    //////////// Objectives ////////////////////////////////////////
    {
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkObjective),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkObjective),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addLinkObjective),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::addLinkParamObjective),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("add_link_param_objective",
                nb::overload_cast<ScalarFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::addLinkParamObjective),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
    }
}

static void BuildOldLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj) {
    {
        ////////////////// Legacy EqualCons//////////
        obj.def("add_link_equal_con",
                nb::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkEqualCon), "");

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        /// ///
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkEqualCon));

        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        ////////////////////////////

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkEqualCon),
                "");

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, RegVec,
                                  std::vector<std::vector<std::shared_ptr<ODEPhaseBase>>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkEqualCon),
                "");

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkEqualCon),
                "");
        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::addLinkEqualCon),
                "");

        obj.def("add_link_equal_con",
                nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkEqualCon),
                "");
        obj.def(
            "add_link_equal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::addLinkEqualCon),
            "");

        obj.def(
            "add_forward_link_equal_con",
            nb::overload_cast<int, int, VectorXi>(&OptimalControlProblem::addForwardLinkEqualCon),
            "");

        obj.def("add_forward_link_equal_con",
                nb::overload_cast<PhasePtr, PhasePtr, VectorXi>(
                    &OptimalControlProblem::addForwardLinkEqualCon),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<LinkFlags, int, VectorXi, int, VectorXi>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, int, PhaseRegionFlags, VectorXi, int,
                                  PhaseRegionFlags, VectorXi>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                "");

        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, PhasePtr, PhaseRegionFlags, VectorXi, PhasePtr,
                                  PhaseRegionFlags, VectorXi>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                "");

        //
        obj.def("add_direct_link_equal_con",
                nb::overload_cast<VectorFunctionalX, int, std::string, VectorXi, int, std::string,
                                  VectorXi>(&OptimalControlProblem::addDirectLinkEqualCon),
                "");

        obj.def(
            "add_direct_link_equal_con",
            nb::overload_cast<VectorFunctionalX, PhasePtr, std::string, VectorXi, PhasePtr,
                              std::string, VectorXi>(&OptimalControlProblem::addDirectLinkEqualCon),
            "");
    }

    //////////////////Legacy  InequalCons//////////
    {

        obj.def("add_link_inequal_con",
                nb::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkInequalCon), "");

        ////////////////////////////

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        /// ///
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkInequalCon));

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        ////////////////////////////

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkInequalCon),
                "");

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::addLinkInequalCon),
                "");

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkInequalCon),
                "");

        obj.def(
            "add_link_inequal_con",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::addLinkInequalCon),
            "");

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkInequalCon));

        obj.def("add_link_inequal_con",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkInequalCon));

        ////////////////////////////////////////
        obj.def("add_link_objective",
                nb::overload_cast<LinkObjective>(&OptimalControlProblem::addLinkObjective), "");
    }

    {
        //////////////////Legacy LinkObjectives//////////

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        /// ///
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkObjective));

        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        ////////////////////////////

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkObjective),
                "");
        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::addLinkObjective),
                "");

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkObjective),
                "");
        obj.def(
            "add_link_objective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::addLinkObjective),
            "");

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkObjective),
                "");

        obj.def("add_link_objective",
                nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkObjective),
                "");
    }
}
