#include "ODEPhaseBind.h"
#include "PyDocString/OptimalControl/OptimalControlProblem_doc.h"

using namespace Tycho;
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
    using namespace doc;
    auto obj =
        nb::class_<OptimalControlProblem, OptimizationProblemBase>(m, "OptimalControlProblem");
    obj.def(nb::init<>());

    BuildNewLinkIterface(obj);
    BuildOldLinkIterface(obj);

    //////////////////////////////////////////////////////////////////////////////

    obj.def("addLinkParamEqualCon",
            nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkParamEqualCon),
            OptimalControlProblem_addLinkParamEqualCon1);
    obj.def("addLinkParamEqualCon",
            nb::overload_cast<VectorFunctionalX, VectorXi>(
                &OptimalControlProblem::addLinkParamEqualCon),
            OptimalControlProblem_addLinkParamEqualCon2);
    obj.def("addLinkParamInequalCon",
            nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkParamInequalCon),
            OptimalControlProblem_addLinkParamInequalCon1);
    obj.def("addLinkParamInequalCon",
            nb::overload_cast<VectorFunctionalX, VectorXi>(
                &OptimalControlProblem::addLinkParamInequalCon),
            OptimalControlProblem_addLinkParamInequalCon2);
    obj.def("addLinkParamObjective",
            nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkParamObjective),
            OptimalControlProblem_addLinkParamObjective1);
    obj.def("addLinkParamObjective",
            nb::overload_cast<ScalarFunctionalX, VectorXi>(
                &OptimalControlProblem::addLinkParamObjective),
            OptimalControlProblem_addLinkParamObjective2);

    //////////////////////////////////////////////////////////////////////////////

    obj.def("removeLinkEqualCon", &OptimalControlProblem::removeLinkEqualCon,
            OptimalControlProblem_removeLinkEqualCon);
    obj.def("removeLinkInequalCon", &OptimalControlProblem::removeLinkInequalCon,
            OptimalControlProblem_removeLinkEqualCon);
    obj.def("removeLinkObjective", &OptimalControlProblem::removeLinkObjective,
            OptimalControlProblem_removeLinkObjective);

    obj.def("addPhase", nb::overload_cast<PhasePtr>(&OptimalControlProblem::addPhase),
            OptimalControlProblem_addPhase);

    obj.def("addPhases", &OptimalControlProblem::addPhases);

    obj.def("getPhaseNum", nb::overload_cast<PhasePtr>(&OptimalControlProblem::getPhaseNum));

    obj.def("removePhase", &OptimalControlProblem::removePhase, OptimalControlProblem_removePhase);
    obj.def("Phase", &OptimalControlProblem::Phase, OptimalControlProblem_Phase);

    obj.def("setLinkParams",
            nb::overload_cast<VectorXd, VectorXd>(&OptimalControlProblem::setLinkParams));
    obj.def("setLinkParams", nb::overload_cast<VectorXd>(&OptimalControlProblem::setLinkParams),
            OptimalControlProblem_setLinkParams);

    obj.def("addLinkParamVgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                       &OptimalControlProblem::addLinkParamVgroups));
    obj.def("setLinkParamVgroups", nb::overload_cast<std::map<std::string, Eigen::VectorXi>>(
                                       &OptimalControlProblem::setLinkParamVgroups));
    obj.def("addLinkParamVgroup", nb::overload_cast<Eigen::VectorXi, std::string>(
                                      &OptimalControlProblem::addLinkParamVgroup));
    obj.def("addLinkParamVgroup",
            nb::overload_cast<int, std::string>(&OptimalControlProblem::addLinkParamVgroup));

    obj.def("returnLinkParams", &OptimalControlProblem::returnLinkParams,
            OptimalControlProblem_returnLinkParams);

    obj.def("transcribe", nb::overload_cast<bool, bool>(&OptimalControlProblem::transcribe),
            OptimalControlProblem_transcribe);

    obj.def_ro("Phases", &OptimalControlProblem::phases, OptimalControlProblem_Phases);

    ///////////////////////
    obj.def("returnLinkEqualConVals", &OptimalControlProblem::returnLinkEqualConVals);
    obj.def("returnLinkEqualConLmults", &OptimalControlProblem::returnLinkEqualConLmults);

    obj.def("returnLinkInequalConVals", &OptimalControlProblem::returnLinkInequalConVals);
    obj.def("returnLinkInequalConLmults", &OptimalControlProblem::returnLinkInequalConLmults);

    obj.def("returnLinkEqualConScales", &OptimalControlProblem::returnLinkEqualConScales);
    obj.def("returnLinkInequalConScales", &OptimalControlProblem::returnLinkInequalConScales);
    obj.def("returnLinkObjectiveScales", &OptimalControlProblem::returnLinkObjectiveScales);

    ///////////////////////

    obj.def_rw("AutoScaling", &OptimalControlProblem::AutoScaling);
    obj.def_rw("SyncObjectiveScales", &OptimalControlProblem::SyncObjectiveScales);

    obj.def_rw("AdaptiveMesh", &OptimalControlProblem::AdaptiveMesh);
    obj.def_rw("PrintMeshInfo", &OptimalControlProblem::PrintMeshInfo);
    obj.def_rw("MaxMeshIters", &OptimalControlProblem::MaxMeshIters);
    obj.def_ro("MeshConverged", &OptimalControlProblem::MeshConverged);
    obj.def_rw("SolveOnlyFirst", &OptimalControlProblem::SolveOnlyFirst);

    obj.def_rw("MeshAbortFlag", &OptimalControlProblem::MeshAbortFlag);

    obj.def("setAdaptiveMesh", &OptimalControlProblem::setAdaptiveMesh,
            nb::arg("AdaptiveMesh") = true, nb::arg("ApplyToPhases") = true);
    obj.def("setAutoScaling", &OptimalControlProblem::setAutoScaling, nb::arg("AutoScaling") = true,
            nb::arg("ApplyToPhases") = true);

    obj.def("setMeshTol", &OptimalControlProblem::setMeshTol);
    obj.def("setMeshRedFactor", &OptimalControlProblem::setMeshRedFactor);
    obj.def("setMeshIncFactor", &OptimalControlProblem::setMeshIncFactor);
    obj.def("setMeshErrFactor", &OptimalControlProblem::setMeshErrFactor);
    obj.def("setMaxMeshIters", &OptimalControlProblem::setMaxMeshIters);
    obj.def("setMinSegments", &OptimalControlProblem::setMinSegments);
    obj.def("setMaxSegments", &OptimalControlProblem::setMaxSegments);
    obj.def("setMeshErrorCriteria", &OptimalControlProblem::setMeshErrorCriteria);
    obj.def("setMeshErrorEstimator", &OptimalControlProblem::setMeshErrorEstimator);
}

static void BuildNewLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj) {

    //////////// EqualCons////////////////////////////////////////
    {
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkEqualCon),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkEqualCon),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addLinkEqualCon),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addLinkParamEqualCon",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::addLinkParamEqualCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("addLinkParamEqualCon",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::addLinkParamEqualCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addForwardLinkEqualCon",
                nb::overload_cast<PhaseRefType, PhaseRefType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addForwardLinkEqualCon),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("vars"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addParamLinkEqualCon",
                nb::overload_cast<PhaseRefType, PhaseRefType, RegionType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addParamLinkEqualCon),
                nb::arg("phase0"), nb::arg("phase1"), nb::arg("reg0"), nb::arg("vars"),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addDirectLinkEqualCon",
                nb::overload_cast<PhaseRefType, RegionType, VarIndexType, PhaseRefType, RegionType,
                                  VarIndexType, ScaleType>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"), nb::arg("phase1"),
                nb::arg("reg1"), nb::arg("v1"), nb::arg("AutoScale").none() = std::string("auto"));
    }

    //////////// InequalCons////////////////////////////////////////
    {
        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkInequalCon),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkInequalCon),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addLinkInequalCon",
                nb::overload_cast<VectorFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addLinkInequalCon),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addLinkParamInequalCon",
                nb::overload_cast<VectorFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::addLinkParamInequalCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("addLinkParamInequalCon",
                nb::overload_cast<VectorFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::addLinkParamInequalCon),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
    }
    //////////// Objectives ////////////////////////////////////////
    {
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, std::vector<PhasePack>, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkObjective),
            nb::arg("func"), nb::arg("phasepack"), nb::arg("linkparams") = VectorXi(),
            nb::arg("AutoScale").none() = std::string("auto"));

        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, PhaseRefType, RegionType, VarIndexType,
                              VarIndexType, VarIndexType, VarIndexType, ScaleType>(
                &OptimalControlProblem::addLinkObjective),
            nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("XtUVars0"),
            nb::arg("OPVars0"), nb::arg("SPVars0"), nb::arg("phase1"), nb::arg("reg1"),
            nb::arg("XtUVars1"), nb::arg("OPVars1"), nb::arg("SPVars1"),
            nb::arg("linkparams") = VectorXi(), nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addLinkObjective",
                nb::overload_cast<ScalarFunctionalX, PhaseRefType, RegionType, VarIndexType,
                                  PhaseRefType, RegionType, VarIndexType, VarIndexType, ScaleType>(
                    &OptimalControlProblem::addLinkObjective),
                nb::arg("func"), nb::arg("phase0"), nb::arg("reg0"), nb::arg("v0"),
                nb::arg("phase1"), nb::arg("reg1"), nb::arg("v1"),
                nb::arg("linkparams") = VectorXi(),
                nb::arg("AutoScale").none() = std::string("auto"));

        obj.def("addLinkParamObjective",
                nb::overload_cast<ScalarFunctionalX, std::vector<VectorXi>, ScaleType>(
                    &OptimalControlProblem::addLinkParamObjective),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
        obj.def("addLinkParamObjective",
                nb::overload_cast<ScalarFunctionalX, VectorXi, ScaleType>(
                    &OptimalControlProblem::addLinkParamObjective),
                nb::arg("func"), nb::arg("LinkParms"),
                nb::arg("AutoScale").none() = std::string("auto"));
    }
}

static void BuildOldLinkIterface(nb::class_<OptimalControlProblem, OptimizationProblemBase> &obj) {
    using namespace doc;

    {
        ////////////////// Legacy EqualCons//////////
        obj.def("addLinkEqualCon",
                nb::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkEqualCon),
                OptimalControlProblem_addLinkEqualCon1);

        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        /// ///
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkEqualCon));

        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        ////////////////////////////

        obj.def("addLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkEqualCon),
                OptimalControlProblem_addLinkEqualCon2);

        obj.def("addLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, RegVec,
                                  std::vector<std::vector<std::shared_ptr<ODEPhaseBase>>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkEqualCon),
                OptimalControlProblem_addLinkEqualCon2);

        obj.def("addLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkEqualCon),
                OptimalControlProblem_addLinkEqualCon2);
        obj.def("addLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::addLinkEqualCon),
                OptimalControlProblem_addLinkEqualCon2);

        obj.def("addLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkEqualCon),
                OptimalControlProblem_addLinkEqualCon2);
        obj.def(
            "addLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);

        obj.def(
            "addForwardLinkEqualCon",
            nb::overload_cast<int, int, VectorXi>(&OptimalControlProblem::addForwardLinkEqualCon),
            OptimalControlProblem_addForwardLinkEqualCon);

        obj.def("addForwardLinkEqualCon",
                nb::overload_cast<PhasePtr, PhasePtr, VectorXi>(
                    &OptimalControlProblem::addForwardLinkEqualCon),
                OptimalControlProblem_addForwardLinkEqualCon);

        obj.def("addDirectLinkEqualCon",
                nb::overload_cast<LinkFlags, int, VectorXi, int, VectorXi>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                OptimalControlProblem_addDirectLinkEqualCon);

        obj.def("addDirectLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, int, PhaseRegionFlags, VectorXi, int,
                                  PhaseRegionFlags, VectorXi>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                OptimalControlProblem_addDirectLinkEqualCon);

        obj.def("addDirectLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, PhasePtr, PhaseRegionFlags, VectorXi, PhasePtr,
                                  PhaseRegionFlags, VectorXi>(
                    &OptimalControlProblem::addDirectLinkEqualCon),
                OptimalControlProblem_addDirectLinkEqualCon);

        //
        obj.def("addDirectLinkEqualCon",
                nb::overload_cast<VectorFunctionalX, int, std::string, VectorXi, int, std::string,
                                  VectorXi>(&OptimalControlProblem::addDirectLinkEqualCon),
                OptimalControlProblem_addDirectLinkEqualCon);

        obj.def(
            "addDirectLinkEqualCon",
            nb::overload_cast<VectorFunctionalX, PhasePtr, std::string, VectorXi, PhasePtr,
                              std::string, VectorXi>(&OptimalControlProblem::addDirectLinkEqualCon),
            OptimalControlProblem_addDirectLinkEqualCon);
    }

    //////////////////Legacy  InequalCons//////////
    {

        obj.def("addLinkInequalCon",
                nb::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkInequalCon),
                OptimalControlProblem_addLinkInequalCon);

        ////////////////////////////

        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        /// ///
        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkInequalCon));

        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        ////////////////////////////

        obj.def("addLinkInequalCon",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkInequalCon),
                OptimalControlProblem_addLinkInequalCon);

        obj.def("addLinkInequalCon",
                nb::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::addLinkInequalCon),
                OptimalControlProblem_addLinkInequalCon);

        obj.def("addLinkInequalCon",
                nb::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkInequalCon),
                OptimalControlProblem_addLinkInequalCon);

        obj.def(
            "addLinkInequalCon",
            nb::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::addLinkInequalCon),
            OptimalControlProblem_addLinkInequalCon);

        obj.def("addLinkInequalCon",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkInequalCon));

        obj.def("addLinkInequalCon",
                nb::overload_cast<VectorFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkInequalCon));

        ////////////////////////////////////////
        obj.def("addLinkObjective",
                nb::overload_cast<LinkObjective>(&OptimalControlProblem::addLinkObjective),
                OptimalControlProblem_addLinkObjective);
    }

    {
        //////////////////Legacy LinkObjectives//////////

        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, std::vector<std::string>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        /// ///
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, std::vector<std::string>,
                              std::vector<std::vector<PhasePtr>>, std::vector<VectorXi>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>>(
                &OptimalControlProblem::addLinkObjective));

        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              std::vector<VectorXi>, std::vector<VectorXi>, std::vector<VectorXi>,
                              std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        ////////////////////////////

        obj.def("addLinkObjective",
                nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkObjective),
                OptimalControlProblem_addLinkObjective);
        obj.def("addLinkObjective",
                nb::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>,
                                  VectorXi>(&OptimalControlProblem::addLinkObjective),
                OptimalControlProblem_addLinkObjective);

        obj.def("addLinkObjective",
                nb::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                    &OptimalControlProblem::addLinkObjective),
                OptimalControlProblem_addLinkObjective);
        obj.def(
            "addLinkObjective",
            nb::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>,
                              VectorXi>(&OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);

        obj.def("addLinkObjective",
                nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<VectorXi>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkObjective),
                OptimalControlProblem_addLinkObjective);

        obj.def("addLinkObjective",
                nb::overload_cast<ScalarFunctionalX, RegVec, std::vector<std::vector<PhasePtr>>,
                                  std::vector<VectorXi>, std::vector<VectorXi>>(
                    &OptimalControlProblem::addLinkObjective),
                OptimalControlProblem_addLinkObjective);
    }
}
