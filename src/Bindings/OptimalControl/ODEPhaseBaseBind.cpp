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

#include "ODEPhaseBind.h"
#include "PyDocString/OptimalControl/ODEPhaseBase_doc.h"

using namespace Tycho;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using VectorFunctionalX = GenericFunction<-1, -1>;
using ScalarFunctionalX = GenericFunction<-1, 1>;
using StateConstraint = StateFunction<VectorFunctionalX>;
using StateObjective = StateFunction<ScalarFunctionalX>;

void TychoBind<ODEPhaseBase>::Build(nb::module_ &m) {
    using namespace doc;
    auto obj = nb::class_<ODEPhaseBase, OptimizationProblemBase>(m, "PhaseInterface");
    obj.doc() = "Base Class for All Optimal Control Phases";

    auto set_mesh_error_estimator = [](ODEPhaseBase &self, nb::object val) {
        if (nb::isinstance<MeshErrorEstimators>(val))
            self.MeshErrorEstimator = nb::cast<MeshErrorEstimators>(val);
        else if (nb::isinstance<nb::str>(val))
            self.MeshErrorEstimator = strto_MeshErrorEstimator(nb::cast<std::string>(val));
        else
            throw nb::type_error("expected MeshErrorEstimators enum or str");
    };
    auto set_mesh_error_aggregation = [](MeshErrorAggregation &target, nb::object val,
                                         const char *name) {
        if (nb::isinstance<MeshErrorAggregation>(val))
            target = nb::cast<MeshErrorAggregation>(val);
        else if (nb::isinstance<nb::str>(val))
            target = strto_MeshErrorAggregation(nb::cast<std::string>(val));
        else if (std::string(name) == "MeshErrorCriteria")
            throw nb::type_error("expected MeshErrorAggregation enum or str for "
                                 "MeshErrorCriteria");
        else
            throw nb::type_error("expected MeshErrorAggregation enum or str for "
                                 "MeshErrorDistributor");
    };

    obj.def("enable_vectorization", &ODEPhaseBase::enable_vectorization);

    obj.def(
        "setTraj",
        nb::overload_cast<const std::vector<Eigen::VectorXd> &, Eigen::VectorXd, Eigen::VectorXi>(
            &ODEPhaseBase::setTraj),
        ODEPhaseBase_setTraj1);

    obj.def("setTraj", nb::overload_cast<const std::vector<Eigen::VectorXd> &, Eigen::VectorXd,
                                         Eigen::VectorXi, bool>(&ODEPhaseBase::setTraj));

    obj.def("setTraj",
            nb::overload_cast<const std::vector<Eigen::VectorXd> &, int>(&ODEPhaseBase::setTraj),
            ODEPhaseBase_setTraj2);

    obj.def("setTraj", nb::overload_cast<const std::vector<Eigen::VectorXd> &, int, bool>(
                           &ODEPhaseBase::setTraj));

    obj.def("setTraj",
            nb::overload_cast<const std::vector<Eigen::VectorXd> &>(&ODEPhaseBase::setTraj));

    obj.def("switchTranscriptionMode",
            nb::overload_cast<TranscriptionModes, VectorXd, VectorXi>(
                &ODEPhaseBase::switchTranscriptionMode),
            ODEPhaseBase_switchTranscriptionMode1);
    obj.def("switchTranscriptionMode",
            nb::overload_cast<TranscriptionModes>(&ODEPhaseBase::switchTranscriptionMode),
            ODEPhaseBase_switchTranscriptionMode2);

    obj.def(
        "switchTranscriptionMode",
        nb::overload_cast<std::string, VectorXd, VectorXi>(&ODEPhaseBase::switchTranscriptionMode),
        ODEPhaseBase_switchTranscriptionMode1);
    obj.def("switchTranscriptionMode",
            nb::overload_cast<std::string>(&ODEPhaseBase::switchTranscriptionMode),
            ODEPhaseBase_switchTranscriptionMode2);

    obj.def("transcribe", nb::overload_cast<bool, bool>(&ODEPhaseBase::transcribe),
            ODEPhaseBase_transcribe);

    obj.def("refineTrajManual", nb::overload_cast<int>(&ODEPhaseBase::refineTrajManual),
            ODEPhaseBase_refineTrajManual1);
    obj.def("refineTrajManual",
            nb::overload_cast<VectorXd, VectorXi>(&ODEPhaseBase::refineTrajManual),
            ODEPhaseBase_refineTrajManual2);
    obj.def("refineTrajEqual", &ODEPhaseBase::refineTrajEqual, ODEPhaseBase_refineTrajEqual);

    obj.def("setStaticParams",
            nb::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::setStaticParams),
            ODEPhaseBase_setStaticParams);
    obj.def("setStaticParams", nb::overload_cast<VectorXd>(&ODEPhaseBase::setStaticParams),
            ODEPhaseBase_setStaticParams);

    obj.def("addStaticParams",
            nb::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::addStaticParams));
    obj.def("addStaticParams", nb::overload_cast<VectorXd>(&ODEPhaseBase::addStaticParams));
    obj.def("addStaticParamVgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                         &ODEPhaseBase::addStaticParamVgroups));
    obj.def("setStaticParamVgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                         &ODEPhaseBase::setStaticParamVgroups));
    obj.def("addStaticParamVgroup",
            nb::overload_cast<Eigen::VectorXi, std::string>(&ODEPhaseBase::addStaticParamVgroup));
    obj.def("addStaticParamVgroup",
            nb::overload_cast<int, std::string>(&ODEPhaseBase::addStaticParamVgroup));

    obj.def("setControlMode", nb::overload_cast<ControlModes>(&ODEPhaseBase::setControlMode),
            ODEPhaseBase_setControlMode);
    obj.def("setControlMode", nb::overload_cast<std::string>(&ODEPhaseBase::setControlMode),
            ODEPhaseBase_setControlMode);

    obj.def("setIntegralMode", &ODEPhaseBase::setIntegralMode, ODEPhaseBase_setIntegralMode);

    obj.def("subStaticParams", &ODEPhaseBase::subStaticParams, ODEPhaseBase_subStaticParams);

    obj.def("subVariables",
            nb::overload_cast<PhaseRegionFlags, VectorXi, VectorXd>(&ODEPhaseBase::subVariables),
            ODEPhaseBase_subVariables);
    obj.def("subVariable",
            nb::overload_cast<PhaseRegionFlags, int, double>(&ODEPhaseBase::subVariable),
            ODEPhaseBase_subVariable);

    obj.def("subVariables",
            nb::overload_cast<std::string, VectorXi, VectorXd>(&ODEPhaseBase::subVariables),
            ODEPhaseBase_subVariables);
    obj.def("subVariable", nb::overload_cast<std::string, int, double>(&ODEPhaseBase::subVariable),
            ODEPhaseBase_subVariable);

    obj.def("returnTraj", &ODEPhaseBase::returnTraj, ODEPhaseBase_returnTraj);
    obj.def("returnTrajRange", &ODEPhaseBase::returnTrajRange, ODEPhaseBase_returnTrajRange);
    obj.def("returnTrajRangeND", &ODEPhaseBase::returnTrajRangeND, ODEPhaseBase_returnTrajRangeND);
    obj.def("returnTrajTable", &ODEPhaseBase::returnTrajTable);

    obj.def("returnCostateTraj", &ODEPhaseBase::returnCostateTraj, ODEPhaseBase_returnCostateTraj);
    obj.def("returnTrajError", &ODEPhaseBase::returnTrajError);

    obj.def("returnUSplineConLmults", &ODEPhaseBase::returnUSplineConLmults);
    obj.def("returnUSplineConVals", &ODEPhaseBase::returnUSplineConVals);

    obj.def("returnEqualConLmults", &ODEPhaseBase::returnEqualConLmults,
            ODEPhaseBase_returnEqualConLmults);
    obj.def("returnEqualConVals", &ODEPhaseBase::returnEqualConVals);
    obj.def("returnEqualConScales", &ODEPhaseBase::returnEqualConScales);

    obj.def("returnInequalConLmults", &ODEPhaseBase::returnInequalConLmults,
            ODEPhaseBase_returnInequalConLmults);
    obj.def("returnInequalConVals", &ODEPhaseBase::returnInequalConVals);
    obj.def("returnInequalConScales", &ODEPhaseBase::returnInequalConScales);

    obj.def("returnIntegralObjectiveScales", &ODEPhaseBase::returnIntegralObjectiveScales);
    obj.def("returnIntegralParamFunctionScales", &ODEPhaseBase::returnIntegralParamFunctionScales);
    obj.def("returnStateObjectiveScales", &ODEPhaseBase::returnStateObjectiveScales);
    obj.def("returnODEOutputScales", &ODEPhaseBase::returnODEOutputScales);

    obj.def("returnStaticParams", &ODEPhaseBase::returnStaticParams,
            ODEPhaseBase_returnStaticParam);

    obj.def("test_threads", &ODEPhaseBase::test_threads);

    obj.def("removeEqualCon", &ODEPhaseBase::removeEqualCon, ODEPhaseBase_removeEqualCon);
    obj.def("removeInequalCon", &ODEPhaseBase::removeInequalCon, ODEPhaseBase_removeInequalCon);
    obj.def("removeStateObjective", &ODEPhaseBase::removeStateObjective,
            ODEPhaseBase_removeStateObjective);
    obj.def("removeIntegralObjective", &ODEPhaseBase::removeIntegralObjective,
            ODEPhaseBase_removeIntegralObjective);
    obj.def("removeIntegralParamFunction", &ODEPhaseBase::removeIntegralParamFunction,
            ODEPhaseBase_removeIntegralParamFunction);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////
    ///// The New interface /////////////////////

    obj.def("addEqualCon",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::addEqualCon),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"),
            nb::arg("SPVars"), nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addEqualCon",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::addEqualCon),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("InputIndex"),
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addBoundaryValue",
            nb::overload_cast<RegionType, VarIndexType, const std::variant<double, VectorXd> &,
                              ScaleType>(&ODEPhaseBase::addBoundaryValue),
            nb::arg("PhaseRegion"), nb::arg("Index"), nb::arg("Value"),
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addDeltaVarEqualCon",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addDeltaVarEqualCon),
            nb::arg("var"), nb::arg("value"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addDeltaTimeEqualCon",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::addDeltaTimeEqualCon),
            nb::arg("value"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addValueLock",
            nb::overload_cast<RegionType, VarIndexType, ScaleType>(&ODEPhaseBase::addValueLock),
            nb::arg("reg"), nb::arg("vars"), nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addPeriodicityCon",
            nb::overload_cast<VarIndexType, ScaleType>(&ODEPhaseBase::addPeriodicityCon),
            nb::arg("vars"), nb::arg("AutoScale").none() = std::string("auto"));

    //////////////////////////////////
    /////// InequalCons
    obj.def("addInequalCon",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::addInequalCon),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"),
            nb::arg("SPVars"), nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addInequalCon",
            nb::overload_cast<RegionType, VectorFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::addInequalCon),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("InputIndex"),
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addLUVarBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::addLUVarBound),
            nb::arg("PhaseRegion"), nb::arg("var"), nb::arg("lowerbound"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));
    obj.def("addLowerVarBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addLowerVarBound),
            nb::arg("PhaseRegion"), nb::arg("var"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addUpperVarBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addUpperVarBound),
            nb::arg("PhaseRegion"), nb::arg("var"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def(
        "addLUFuncBound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType,
                          double, double, double, ScaleType>(&ODEPhaseBase::addLUFuncBound),
        nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"),
        nb::arg("SPVars"), nb::arg("lowerbound"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
        nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addLUFuncBound",
            nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, double, double, double,
                              ScaleType>(&ODEPhaseBase::addLUFuncBound),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUPVars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));
    //
    obj.def(
        "addLowerFuncBound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType,
                          double, double, ScaleType>(&ODEPhaseBase::addLowerFuncBound),
        nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"),
        nb::arg("SPVars"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
        nb::arg("AutoScale").none() = std::string("auto"));

    obj.def(
        "addLowerFuncBound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, double, double, ScaleType>(
            &ODEPhaseBase::addLowerFuncBound),
        nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUPVars"), nb::arg("lowerbound"),
        nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));

    obj.def(
        "addUpperFuncBound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType,
                          double, double, ScaleType>(&ODEPhaseBase::addUpperFuncBound),
        nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"),
        nb::arg("SPVars"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
        nb::arg("AutoScale").none() = std::string("auto"));

    obj.def(
        "addUpperFuncBound",
        nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, double, double, ScaleType>(
            &ODEPhaseBase::addUpperFuncBound),
        nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUPVars"), nb::arg("upperbound"),
        nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addLUNormBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::addLUNormBound),
            nb::arg("PhaseRegion"), nb::arg("XtUPVars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addLUSquaredNormBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, double, ScaleType>(
                &ODEPhaseBase::addLUSquaredNormBound),
            nb::arg("PhaseRegion"), nb::arg("XtUPVars"), nb::arg("lowerbound"),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));

    //
    obj.def("addLowerNormBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addLowerNormBound),
            nb::arg("PhaseRegion"), nb::arg("XtUPVars"), nb::arg("lowerbound"),
            nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addLowerSquaredNormBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addLowerSquaredNormBound),
            nb::arg("PhaseRegion"), nb::arg("XtUPVars"), nb::arg("lowerbound"),
            nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));
    //
    obj.def("addUpperNormBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addUpperNormBound),
            nb::arg("PhaseRegion"), nb::arg("XtUPVars"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addUpperSquaredNormBound",
            nb::overload_cast<RegionType, VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addUpperSquaredNormBound),
            nb::arg("PhaseRegion"), nb::arg("XtUPVars"), nb::arg("upperbound"),
            nb::arg("scale") = 1.0, nb::arg("AutoScale").none() = std::string("auto"));
    //
    obj.def("addLowerDeltaVarBound",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addLowerDeltaVarBound),
            nb::arg("Var"), nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));
    obj.def("addLowerDeltaTimeBound",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::addLowerDeltaTimeBound),
            nb::arg("lowerbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));
    //
    obj.def("addUpperDeltaVarBound",
            nb::overload_cast<VarIndexType, double, double, ScaleType>(
                &ODEPhaseBase::addUpperDeltaVarBound),
            nb::arg("Var"), nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));
    obj.def("addUpperDeltaTimeBound",
            nb::overload_cast<double, double, ScaleType>(&ODEPhaseBase::addUpperDeltaTimeBound),
            nb::arg("upperbound"), nb::arg("scale") = 1.0,
            nb::arg("AutoScale").none() = std::string("auto"));
    //////////////////////////////////
    /////// StateObjectives /////////
    obj.def("addStateObjective",
            nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, VarIndexType,
                              VarIndexType, ScaleType>(&ODEPhaseBase::addStateObjective),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"),
            nb::arg("SPVars"), nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addStateObjective",
            nb::overload_cast<RegionType, ScalarFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::addStateObjective),
            nb::arg("PhaseRegion"), nb::arg("Func"), nb::arg("InputIndex"),
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addValueObjective",
            nb::overload_cast<RegionType, VarIndexType, double, ScaleType>(
                &ODEPhaseBase::addValueObjective),
            nb::arg("PhaseRegion"), nb::arg("Var"), nb::arg("scale"),
            nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addDeltaVarObjective",
            nb::overload_cast<VarIndexType, double, ScaleType>(&ODEPhaseBase::addDeltaVarObjective),
            nb::arg("Var"), nb::arg("scale"), nb::arg("AutoScale").none() = std::string("auto"));
    obj.def("addDeltaTimeObjective",
            nb::overload_cast<double, ScaleType>(&ODEPhaseBase::addDeltaTimeObjective),
            nb::arg("Var"), nb::arg("AutoScale").none() = std::string("auto"));
    //////////////////////////////////
    /////// IntegralObjectives /////////
    obj.def(
        "addIntegralObjective",
        nb::overload_cast<ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType, ScaleType>(
            &ODEPhaseBase::addIntegralObjective),
        nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"), nb::arg("SPVars"),
        nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addIntegralObjective",
            nb::overload_cast<ScalarFunctionalX, VarIndexType, ScaleType>(
                &ODEPhaseBase::addIntegralObjective),
            nb::arg("Func"), nb::arg("InputIndex"),
            nb::arg("AutoScale").none() = std::string("auto"));
    //////////////////////////////////
    /////// IntegralParamFunction /////////
    obj.def("addIntegralParamFunction",
            nb::overload_cast<ScalarFunctionalX, VarIndexType, VarIndexType, VarIndexType, int,
                              ScaleType>(&ODEPhaseBase::addIntegralParamFunction),
            nb::arg("Func"), nb::arg("XtUVars"), nb::arg("OPVars"), nb::arg("SPVars"),
            nb::arg("IntParam"), nb::arg("AutoScale").none() = std::string("auto"));

    obj.def("addIntegralParamFunction",
            nb::overload_cast<ScalarFunctionalX, VarIndexType, int, ScaleType>(
                &ODEPhaseBase::addIntegralParamFunction),
            nb::arg("Func"), nb::arg("InputIndex"), nb::arg("IntParam"),
            nb::arg("AutoScale").none() = std::string("auto"));

    ///////////////////////////////////////////////////////////////////

    obj.def("addEqualCon", nb::overload_cast<StateConstraint>(&ODEPhaseBase::addEqualCon),
            ODEPhaseBase_addEqualCon1);

    ///////////////////////////////////////////////////////////////////////////////

    obj.def("addInequalCon", nb::overload_cast<StateConstraint>(&ODEPhaseBase::addInequalCon),
            ODEPhaseBase_addInequalCon1);
    ////////////////////////////////////////////////////////////////////////////
    obj.def("addLUVarBounds",
            nb::overload_cast<PhaseRegionFlags, Eigen::VectorXi, double, double, double>(
                &ODEPhaseBase::addLUVarBounds),
            ODEPhaseBase_addLUVarBounds);
    obj.def("addLUVarBounds",
            nb::overload_cast<std::string, Eigen::VectorXi, double, double, double>(
                &ODEPhaseBase::addLUVarBounds),
            ODEPhaseBase_addLUVarBounds);

    ////////////////////////////////////////////////////////////////////////////
    obj.def("addStateObjective",
            nb::overload_cast<StateObjective>(&ODEPhaseBase::addStateObjective),
            ODEPhaseBase_addStateObjective);

    ////////////////////////////////////////////////////////////////////////////

    obj.def("addIntegralObjective",
            nb::overload_cast<StateObjective>(&ODEPhaseBase::addIntegralObjective),
            ODEPhaseBase_addIntegralObjective1);

    ///////////////////////////////////////////////////////////////////////////////
    obj.def("addIntegralParamFunction",
            nb::overload_cast<StateObjective, int>(&ODEPhaseBase::addIntegralParamFunction),
            ODEPhaseBase_addIntegralParamFunction1);

    ////////////////////////////////////////////////////
    obj.def("getMeshInfo", &ODEPhaseBase::getMeshInfo);
    obj.def("refineTrajAuto", &ODEPhaseBase::refineTrajAuto);
    obj.def("calc_global_error", &ODEPhaseBase::calc_global_error);
    obj.def("getMeshIters", &ODEPhaseBase::getMeshIters);

    obj.def_rw("AdaptiveMesh", &ODEPhaseBase::AdaptiveMesh);
    obj.def_rw("AutoScaling", &ODEPhaseBase::AutoScaling);
    obj.def_rw("SyncObjectiveScales", &ODEPhaseBase::SyncObjectiveScales);

    obj.def("setAutoScaling", &ODEPhaseBase::setAutoScaling, nb::arg("AutoScaling") = true);

    obj.def("setAdaptiveMesh", &ODEPhaseBase::setAdaptiveMesh, nb::arg("AdaptiveMesh") = true);

    obj.def("setUnits", [](ODEPhaseBase &self, nb::kwargs kwargs) {
        nb::module_ builtins = nb::module_::import_("builtins");
        nb::object py_int = builtins.attr("int");
        nb::object py_float = builtins.attr("float");
        nb::object py_list = builtins.attr("list");
        nb::object np_array = (nb::object)nb::module_::import_("numpy").attr("ndarray");
        nb::object np_float = (nb::object)nb::module_::import_("numpy").attr("float64");
        nb::object np_int = (nb::object)nb::module_::import_("numpy").attr("int32");

        Eigen::VectorXd Units(self.XtUPVars());
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
        self.setUnits(Units);
    });
    obj.def("setUnits", nb::overload_cast<const Eigen::VectorXd &>(&ODEPhaseBase::setUnits));

    obj.def("setMeshTol", &ODEPhaseBase::setMeshTol);
    obj.def("setMeshRedFactor", &ODEPhaseBase::setMeshRedFactor);
    obj.def("setMeshIncFactor", &ODEPhaseBase::setMeshIncFactor);
    obj.def("setMeshErrFactor", &ODEPhaseBase::setMeshErrFactor);
    obj.def("setMaxMeshIters", &ODEPhaseBase::setMaxMeshIters);
    obj.def("setMinSegments", &ODEPhaseBase::setMinSegments);
    obj.def("setMaxSegments", &ODEPhaseBase::setMaxSegments);
    obj.def("setMeshErrorCriteria",
            nb::overload_cast<MeshErrorAggregation>(&ODEPhaseBase::setMeshErrorCriteria));
    obj.def("setMeshErrorCriteria",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::setMeshErrorCriteria));
    obj.def("setMeshErrorEstimator",
            nb::overload_cast<MeshErrorEstimators>(&ODEPhaseBase::setMeshErrorEstimator));
    obj.def("setMeshErrorEstimator",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::setMeshErrorEstimator));
    obj.def("setMeshErrorDistributor",
            nb::overload_cast<MeshErrorAggregation>(&ODEPhaseBase::setMeshErrorDistributor));
    obj.def("setMeshErrorDistributor",
            nb::overload_cast<const std::string &>(&ODEPhaseBase::setMeshErrorDistributor));

    obj.def_rw("PrintMeshInfo", &ODEPhaseBase::PrintMeshInfo);
    obj.def_rw("MaxMeshIters", &ODEPhaseBase::MaxMeshIters);
    obj.def_rw("MeshTol", &ODEPhaseBase::MeshTol);
    obj.def_prop_rw(
        "MeshErrorEstimator", [](const ODEPhaseBase &self) { return self.MeshErrorEstimator; },
        [set_mesh_error_estimator](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_estimator(self, val);
        });
    obj.def_prop_rw(
        "MeshErrorCriteria", [](const ODEPhaseBase &self) { return self.MeshErrorCriteria; },
        [set_mesh_error_aggregation](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_aggregation(self.MeshErrorCriteria, val, "MeshErrorCriteria");
        });
    obj.def_prop_rw(
        "MeshErrorDistributor", [](const ODEPhaseBase &self) { return self.MeshErrorDistributor; },
        [set_mesh_error_aggregation](ODEPhaseBase &self, nb::object val) {
            set_mesh_error_aggregation(self.MeshErrorDistributor, val, "MeshErrorDistributor");
        });

    obj.def_rw("SolveOnlyFirst", &ODEPhaseBase::SolveOnlyFirst);
    obj.def_rw("ForceOneMeshIter", &ODEPhaseBase::ForceOneMeshIter);
    obj.def_rw("NewError", &ODEPhaseBase::NewError);

    obj.def_rw("DetectControlSwitches", &ODEPhaseBase::DetectControlSwitches);
    obj.def_rw("RelSwitchTol", &ODEPhaseBase::RelSwitchTol);
    obj.def_rw("AbsSwitchTol", &ODEPhaseBase::AbsSwitchTol);
    obj.def_rw("MeshAbortFlag", &ODEPhaseBase::MeshAbortFlag);

    obj.def_rw("NumExtraSegs", &ODEPhaseBase::NumExtraSegs);
    obj.def_rw("MeshRedFactor", &ODEPhaseBase::MeshRedFactor);
    obj.def_rw("MeshIncFactor", &ODEPhaseBase::MeshIncFactor);
    obj.def_rw("MeshErrFactor", &ODEPhaseBase::MeshErrFactor);
    obj.def_ro("MeshConverged", &ODEPhaseBase::MeshConverged);
}
