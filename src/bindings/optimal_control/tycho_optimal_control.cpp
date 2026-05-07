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

#include "tycho_optimal_control.h"
#include "mesh_iterate_info_bind.h"
#include "ode_arguments_bind.h"

#include "tycho/detail/integrators/integrator.h"

namespace tycho {

// Forward declarations for builders defined in separate .cpp files.
void GenericODESBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart4(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart6(FunctionRegistry &reg, nb::module_ &m);

void GenericODESIntegratorPart1(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart2(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart3(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart4(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart6(FunctionRegistry &reg, nb::module_ &m);

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

    m.def("strto_phase_region_flag",
          nb::overload_cast<const std::string &>(&strto_PhaseRegionFlag));
}

void optimal_control_build(FunctionRegistry &reg, nb::module_ &m) {
    auto &oc = reg.getOptimalControlModule();
    RKFlagsBuild(oc);
    OCPFlagsBuild(oc);

    TychoBind<StateFunction<GenericFunction<-1, -1>>>::build(oc, "StateConstraint");
    TychoBind<StateFunction<GenericFunction<-1, 1>>>::build(oc, "StateObjective");

    TychoBind<MeshIterateInfo>::build(oc);

    // EventPack: 3-arg ctor for explicit C++ construction, tuple-accepting
    // init so Python callers can pass (vf, direction, stop_count) tuples
    // directly.
    nb::class_<tycho::integrators::EventPack>(oc, "EventPack")
        .def(nb::init<tycho::vf::GenericFunction<-1, 1>, int, int>(), nb::arg("vf"),
             nb::arg("direction") = 0, nb::arg("stop_count") = 0)
        .def(
            "__init__",
            [](tycho::integrators::EventPack *self, const nb::tuple &t) {
                if (t.size() != 3)
                    throw nb::type_error(
                        "EventPack(tuple) expects exactly 3 elements: (vf, direction, stop_count)");
                new (self)
                    tycho::integrators::EventPack(nb::cast<tycho::vf::GenericFunction<-1, 1>>(t[0]),
                                                  nb::cast<int>(t[1]), nb::cast<int>(t[2]));
            },
            nb::arg("t"))
        .def_rw("vf", &tycho::integrators::EventPack::vf)
        .def_prop_rw("direction", &tycho::integrators::EventPack::direction,
                     &tycho::integrators::EventPack::set_direction)
        .def_prop_rw("stop_count", &tycho::integrators::EventPack::stop_count,
                     &tycho::integrators::EventPack::set_stop_count);
    nb::implicitly_convertible<nb::tuple, tycho::integrators::EventPack>();

    TychoBind<ODEPhaseBase>::build(oc);
    TychoBind<OptimalControlProblemBase>::build(oc);
    TychoBind<LGLInterpTable>::build(oc);

    reg.build_register<InterpFunction<-1>>(oc, "InterpFunction");
    reg.build_register<InterpFunction<1>>(oc, "InterpFunction_1");
    reg.build_register<InterpFunction<3>>(oc, "InterpFunction_3");
    reg.build_register<InterpFunction<6>>(oc, "InterpFunction_6");

    TychoBind<FDDerivArbitrary<Eigen::VectorXd>>::build(oc, "FiniteDiffTable");
    TychoBind<ODEArguments<-1, -1, -1>>::build(oc, "ODEArguments");

    GenericODESBuildPart1(reg, oc);
    GenericODESBuildPart2(reg, oc);
    GenericODESBuildPart3(reg, oc);
    GenericODESBuildPart4(reg, oc);
    GenericODESBuildPart6(reg, oc);

    // Integrator bindings — must be registered after the core ODE parts above
    // so that the submodules already exist.
    GenericODESIntegratorPart1(reg, oc);
    GenericODESIntegratorPart2(reg, oc);
    GenericODESIntegratorPart3(reg, oc);
    GenericODESIntegratorPart4(reg, oc);
    GenericODESIntegratorPart6(reg, oc);
}

} // namespace tycho
