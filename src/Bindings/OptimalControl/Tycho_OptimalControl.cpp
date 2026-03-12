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

#include "Tycho_OptimalControl.h"
#include "MeshIterateInfoBind.h"
#include "ODEArgumentsBind.h"

#include "Integrators/Integrator.h"

namespace Tycho {

// Forward declarations for builders defined in separate .cpp files.
void GenericODESBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart4(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart5(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart6(FunctionRegistry &reg, nb::module_ &m);

void RKFlagsBuild(nb::module_ &m);

static void OCPFlagsBuild(nb::module_ &m) {
    nb::enum_<TranscriptionModes>(m, "TranscriptionModes")
        .value("LGL3", TranscriptionModes::LGL3)
        .value("LGL5", TranscriptionModes::LGL5)
        .value("LGL7", TranscriptionModes::LGL7)
        .value("Trapezoidal", TranscriptionModes::Trapezoidal)
        .value("CentralShooting", TranscriptionModes::CentralShooting);

    nb::enum_<IntegralModes>(m, "IntegralModes")
        .value("BaseIntegral", IntegralModes::BaseIntegral)
        .value("TrapIntegral", IntegralModes::TrapIntegral);

    nb::enum_<ControlModes>(m, "ControlModes")
        .value("HighestOrderSpline", ControlModes::HighestOrderSpline)
        .value("FirstOrderSpline", ControlModes::FirstOrderSpline)
        .value("NoSpline", ControlModes::NoSpline)
        .value("BlockConstant", ControlModes::BlockConstant);

    nb::enum_<PhaseRegionFlags>(m, "PhaseRegionFlags")
        .value("Front", PhaseRegionFlags::Front)
        .value("Back", PhaseRegionFlags::Back)
        .value("Path", PhaseRegionFlags::Path)
        .value("NodalPath", PhaseRegionFlags::NodalPath)
        .value("FrontandBack", PhaseRegionFlags::FrontandBack)
        .value("BackandFront", PhaseRegionFlags::BackandFront)
        .value("Params", PhaseRegionFlags::Params)
        .value("InnerPath", PhaseRegionFlags::InnerPath)
        .value("ODEParams", PhaseRegionFlags::ODEParams)
        .value("StaticParams", PhaseRegionFlags::StaticParams)
        .value("PairWisePath", PhaseRegionFlags::PairWisePath);

    nb::enum_<LinkFlags>(m, "LinkFlags")
        .value("BackToFront", LinkFlags::BackToFront)
        .value("BackToBack", LinkFlags::BackToBack)
        .value("FrontToBack", LinkFlags::FrontToBack)
        .value("ParamsToParams", LinkFlags::ParamsToParams)
        .value("LinkParams", LinkFlags::LinkParams)
        .value("FrontToFront", LinkFlags::FrontToFront)
        .value("PathToPath", LinkFlags::PathToPath)
        .value("BackTwoToTwoFront", LinkFlags::BackTwoToTwoFront)
        .value("FrontTwoToTwoBack", LinkFlags::FrontTwoToTwoBack);

    nb::enum_<ScaleModes>(m, "ScaleModes")
        .value("AUTO", ScaleModes::AUTO)
        .value("CUSTOM", ScaleModes::CUSTOM)
        .value("NONE", ScaleModes::NONE);

    nb::enum_<MeshErrorEstimators>(m, "MeshErrorEstimators")
        .value("DEBOOR", MeshErrorEstimators::DEBOOR)
        .value("INTEGRATOR", MeshErrorEstimators::INTEGRATOR);

    nb::enum_<MeshErrorAggregation>(m, "MeshErrorAggregation")
        .value("MAX", MeshErrorAggregation::MAX)
        .value("AVG", MeshErrorAggregation::AVG)
        .value("GEOMETRIC", MeshErrorAggregation::GEOMETRIC)
        .value("ENDTOEND", MeshErrorAggregation::ENDTOEND);

    m.def("strto_PhaseRegionFlag",
          nb::overload_cast<const std::string &>(&Tycho::strto_PhaseRegionFlag));
}

void OptimalControlBuild(FunctionRegistry &reg, nb::module_ &m) {
    auto &oc = reg.getOptimalControlModule();
    RKFlagsBuild(oc);
    OCPFlagsBuild(oc);

    TychoBind<StateFunction<GenericFunction<-1, -1>>>::Build(oc, "StateConstraint");
    TychoBind<StateFunction<GenericFunction<-1, 1>>>::Build(oc, "StateObjective");
    TychoBind<LinkFunction<GenericFunction<-1, -1>>>::Build(oc, "LinkConstraint");
    TychoBind<LinkFunction<GenericFunction<-1, 1>>>::Build(oc, "LinkObjective");

    TychoBind<MeshIterateInfo>::Build(oc);

    TychoBind<ODEPhaseBase>::Build(oc);
    TychoBind<OptimalControlProblem>::Build(oc);
    TychoBind<LGLInterpTable>::Build(oc);

    reg.Build_Register<InterpFunction<-1>>(oc, "InterpFunction");
    reg.Build_Register<InterpFunction<1>>(oc, "InterpFunction_1");
    reg.Build_Register<InterpFunction<3>>(oc, "InterpFunction_3");
    reg.Build_Register<InterpFunction<6>>(oc, "InterpFunction_6");

    TychoBind<FDDerivArbitrary<Eigen::VectorXd>>::Build(oc, "FiniteDiffTable");
    TychoBind<ODEArguments<-1, -1, -1>>::Build(oc, "ODEArguments");

    GenericODESBuildPart1(reg, oc);
    GenericODESBuildPart2(reg, oc);
    GenericODESBuildPart3(reg, oc);
    GenericODESBuildPart4(reg, oc);
    GenericODESBuildPart5(reg, oc);
    GenericODESBuildPart6(reg, oc);
}

} // namespace Tycho
