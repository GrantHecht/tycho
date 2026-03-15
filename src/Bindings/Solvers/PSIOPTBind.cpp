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

#include "PSIOPTBind.h"
#include "PSIOPT.h"
#include "PyDocString/Solvers/PSIOPT_doc.h"

using namespace Tycho;

void TychoBind<PSIOPT>::Build(nb::module_ &m) {
    using namespace doc;
    using BarrierModes = PSIOPT::BarrierModes;
    using LineSearchModes = PSIOPT::LineSearchModes;
    using QPPivotModes = PSIOPT::QPPivotModes;
    using PDStepStrategies = PSIOPT::PDStepStrategies;
    using ConvergenceFlags = PSIOPT::ConvergenceFlags;
    using AlgorithmModes = PSIOPT::AlgorithmModes;
    using QPOrderingModes = PSIOPT::QPOrderingModes;
    using BestCriteriaModes = PSIOPT::BestCriteriaModes;
    auto obj = nb::class_<PSIOPT>(m, "PSIOPT");
    obj.def(nb::init<std::shared_ptr<NonLinearProgram>>());
    obj.def(nb::init<>());

    obj.def("optimize", &PSIOPT::optimize, PSIOPT_optimize);
    obj.def("solve_optimize", &PSIOPT::solve_optimize, PSIOPT_solve_optimize);
    obj.def("solve", &PSIOPT::solve, PSIOPT_solve);
    obj.def("setQPParams", &PSIOPT::setQPParams);

    obj.def_rw("MaxIters", &PSIOPT::MaxIters, PSIOPT_MaxIters);
    obj.def_rw("MaxAccIters", &PSIOPT::MaxAccIters, PSIOPT_MaxAccIters);
    obj.def_rw("MaxLSIters", &PSIOPT::MaxLSIters, PSIOPT_MaxLSIters);

    obj.def("set_MaxIters", &PSIOPT::set_MaxIters);
    obj.def("set_MaxAccIters", &PSIOPT::set_MaxAccIters);
    obj.def("set_MaxLSIters", &PSIOPT::set_MaxLSIters);

    obj.def_rw("alphaRed", &PSIOPT::alphaRed, PSIOPT_alphaRed);
    obj.def("set_alphaRed", &PSIOPT::set_alphaRed);

    obj.def_rw("WideConsole", &PSIOPT::WideConsole);

    obj.def_rw("FastFactorAlg", &PSIOPT::FastFactorAlg, PSIOPT_FastFactorAlg);

    obj.def_rw("LastTotalTime", &PSIOPT::LastTotalTime, PSIOPT_LastUserTime);
    obj.def_rw("LastPreTime", &PSIOPT::LastPreTime, PSIOPT_LastUserTime);
    obj.def_rw("LastFuncTime", &PSIOPT::LastFuncTime, PSIOPT_LastUserTime);
    obj.def_rw("LastKKTTime", &PSIOPT::LastKKTTime, PSIOPT_LastQPTime);
    obj.def_rw("LastMiscTime", &PSIOPT::LastMiscTime, PSIOPT_LastQPTime);
    obj.def_rw("LastPrintTime", &PSIOPT::LastPrintTime, PSIOPT_LastQPTime);
    obj.def_rw("LastMKLInitTime", &PSIOPT::LastMKLInitTime, PSIOPT_LastMKLInitTime);
    obj.def_rw("LastIterNum", &PSIOPT::LastIterNum, PSIOPT_LastIterNum);
    obj.def_rw("LastObjVal", &PSIOPT::LastObjVal);

    obj.def_rw("ObjScale", &PSIOPT::ObjScale, PSIOPT_ObjScale);
    obj.def_rw("PrintLevel", &PSIOPT::PrintLevel, PSIOPT_PrintLevel);
    obj.def("set_PrintLevel", &PSIOPT::set_PrintLevel);

    obj.def_rw("ConvergeFlag", &PSIOPT::ConvergeFlag);

    obj.def("get_ConvergenceFlag", &PSIOPT::get_ConvergenceFlag);

    obj.def_rw("KKTtol", &PSIOPT::KKTtol, PSIOPT_KKTtol);
    obj.def_rw("Bartol", &PSIOPT::Bartol, PSIOPT_Bartol);
    obj.def_rw("EContol", &PSIOPT::EContol, PSIOPT_EContol);
    obj.def_rw("IContol", &PSIOPT::IContol, PSIOPT_IContol);

    obj.def("set_KKTtol", &PSIOPT::set_KKTtol);
    obj.def("set_Bartol", &PSIOPT::set_Bartol);
    obj.def("set_EContol", &PSIOPT::set_EContol);
    obj.def("set_IContol", &PSIOPT::set_IContol);

    obj.def("set_tols", &PSIOPT::set_tols, nb::arg("KKTtol") = 1.0e-6, nb::arg("EContol") = 1.0e-6,
            nb::arg("IContol") = 1.0e-6, nb::arg("Bartol") = 1.0e-6);

    obj.def_rw("AccKKTtol", &PSIOPT::AccKKTtol, PSIOPT_AccKKTtol);
    obj.def_rw("AccBartol", &PSIOPT::AccBartol, PSIOPT_AccBartol);
    obj.def_rw("AccEContol", &PSIOPT::AccEContol, PSIOPT_AccEContol);
    obj.def_rw("AccIContol", &PSIOPT::AccIContol, PSIOPT_AccIContol);

    obj.def("set_AccKKTtol", &PSIOPT::set_AccKKTtol);
    obj.def("set_AccBartol", &PSIOPT::set_AccBartol);
    obj.def("set_AccEContol", &PSIOPT::set_AccEContol);
    obj.def("set_AccIContol", &PSIOPT::set_AccIContol);

    obj.def("set_Acctols", &PSIOPT::set_Acctols, nb::arg("AccKKTtol") = 1.0e-2,
            nb::arg("AccEContol") = 1.0e-3, nb::arg("AccIContol") = 1.0e-3,
            nb::arg("AccBartol") = 1.0e-3);

    obj.def_rw("DivKKTtol", &PSIOPT::DivKKTtol, PSIOPT_DivKKTtol);
    obj.def_rw("DivBartol", &PSIOPT::DivBartol, PSIOPT_DivBartol);
    obj.def_rw("DivEContol", &PSIOPT::DivEContol, PSIOPT_DivEContol);
    obj.def_rw("DivIContol", &PSIOPT::DivIContol, PSIOPT_DivIContol);

    obj.def("set_DivKKTtol", &PSIOPT::set_DivKKTtol);
    obj.def("set_DivBartol", &PSIOPT::set_DivBartol);
    obj.def("set_DivEContol", &PSIOPT::set_DivEContol);
    obj.def("set_DivIContol", &PSIOPT::set_DivIContol);

    obj.def_rw("NegSlackReset", &PSIOPT::NegSlackReset, PSIOPT_NegSlackReset);

    obj.def_rw("BoundFraction", &PSIOPT::BoundFraction, PSIOPT_BoundFraction);
    obj.def("set_BoundFraction", &PSIOPT::set_BoundFraction);

    obj.def_rw("BoundPush", &PSIOPT::BoundPush, PSIOPT_BoundPush);

    /////////////////////////////////////////////////////////////

    obj.def_rw("deltaH", &PSIOPT::deltaH, PSIOPT_deltaH);
    obj.def_rw("incrH", &PSIOPT::incrH, PSIOPT_incrH);
    obj.def_rw("decrH", &PSIOPT::decrH, PSIOPT_decrH);

    obj.def("set_deltaH", &PSIOPT::set_deltaH);
    obj.def("set_incrH", &PSIOPT::set_incrH);
    obj.def("set_decrH", &PSIOPT::set_decrH);

    obj.def("set_HpertParams", &PSIOPT::set_HpertParams, nb::arg("deltaH"), nb::arg("incrH"),
            nb::arg("decrH"));

    /////////////////////////////////////////////////////////////
    obj.def_rw("initMu", &PSIOPT::initMu, PSIOPT_initMu);
    obj.def_rw("MinMu", &PSIOPT::MinMu, PSIOPT_MinMu);
    obj.def_rw("MaxMu", &PSIOPT::MaxMu, PSIOPT_MaxMu);

    obj.def_rw("MaxSOC", &PSIOPT::MaxSOC, PSIOPT_MaxSOC);

    obj.def_rw("PDStepStrategy", &PSIOPT::PDStepStrategy, PSIOPT_PDStepStrategy);
    obj.def_rw("SOEboundRelax", &PSIOPT::SOEboundRelax, PSIOPT_SOEboundRelax);
    obj.def_rw("QPParSolve", &PSIOPT::QPParSolve, PSIOPT_QPParSolve);

    obj.def_rw("SoeMode", &PSIOPT::SoeMode, PSIOPT_SoeMode);

    //////////////////////////////////////////////////////////////////////////////////////////////////

    obj.def_rw("OptBarMode", &PSIOPT::OptBarMode, PSIOPT_OptBarMode);
    obj.def_rw("SoeBarMode", &PSIOPT::SoeBarMode, PSIOPT_SoeBarMode);

    obj.def("set_OptBarMode", nb::overload_cast<BarrierModes>(&PSIOPT::set_OptBarMode));
    obj.def("set_OptBarMode", nb::overload_cast<const std::string &>(&PSIOPT::set_OptBarMode));
    obj.def("set_SoeBarMode", nb::overload_cast<BarrierModes>(&PSIOPT::set_SoeBarMode));
    obj.def("set_SoeBarMode", nb::overload_cast<const std::string &>(&PSIOPT::set_SoeBarMode));

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("OptLSMode", &PSIOPT::OptLSMode, PSIOPT_OptLSMode);
    obj.def_rw("SoeLSMode", &PSIOPT::SoeLSMode, PSIOPT_SoeLSMode);

    obj.def("set_OptLSMode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_OptLSMode));
    obj.def("set_OptLSMode", nb::overload_cast<const std::string &>(&PSIOPT::set_OptLSMode));
    obj.def("set_SoeLSMode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_SoeLSMode));
    obj.def("set_SoeLSMode", nb::overload_cast<const std::string &>(&PSIOPT::set_SoeLSMode));

    //////////////////////////////////////////////////////////////////////////////////////////////////

    obj.def_rw("ForceQPanalysis", &PSIOPT::ForceQPanalysis, PSIOPT_ForceQPanalysis);
    obj.def_rw("QPRefSteps", &PSIOPT::QPRefSteps, PSIOPT_QPRefSteps);

    obj.def_rw("QPPivotPerturb", &PSIOPT::QPPivotPerturb, PSIOPT_QPPivotPerturb);
    obj.def_rw("QPThreads", &PSIOPT::QPThreads, PSIOPT_QPThreads);
    obj.def_rw("QPPivotStrategy", &PSIOPT::QPPivotStrategy, PSIOPT_QPPivotStrategy);

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("QPOrderingMode", &PSIOPT::QPOrd, PSIOPT_QPOrd);

    obj.def("set_QPOrderingMode", nb::overload_cast<QPOrderingModes>(&PSIOPT::set_QPOrderingMode));
    obj.def("set_QPOrderingMode",
            nb::overload_cast<const std::string &>(&PSIOPT::set_QPOrderingMode));

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("QPPrint", &PSIOPT::QPPrint);

    obj.def_rw("Diagnostic", &PSIOPT::Diagnostic);

    obj.def_rw("ReturnBest", &PSIOPT::ReturnBest);
    obj.def_prop_rw(
        "BestCriteria", [](const PSIOPT &self) { return self.BestCriteria; },
        [](PSIOPT &self, nb::object val) {
            if (nb::isinstance<BestCriteriaModes>(val))
                self.BestCriteria = nb::cast<BestCriteriaModes>(val);
            else if (nb::isinstance<nb::str>(val))
                self.BestCriteria = PSIOPT::strto_BestCriteriaMode(nb::cast<std::string>(val));
            else
                throw nb::type_error("expected BestCriteriaModes enum or str");
        });
    obj.def("set_BestCriteria", nb::overload_cast<BestCriteriaModes>(&PSIOPT::set_BestCriteria));
    obj.def("set_BestCriteria", nb::overload_cast<const std::string &>(&PSIOPT::set_BestCriteria));

    obj.def_rw("storespmat", &PSIOPT::storespmat, PSIOPT_storespmat);
    obj.def("getSPmat", &PSIOPT::getSPmat, PSIOPT_getSPmat);
    obj.def("getSPmat2", &PSIOPT::getSPmat2, PSIOPT_getSPmat2);

    obj.def_rw("CNRMode", &PSIOPT::CNRMode, PSIOPT_CNRMode);

    nb::enum_<BarrierModes>(m, "BarrierModes")
        .value("PROBE", BarrierModes::PROBE)
        .value("LOQO", BarrierModes::LOQO);
    nb::enum_<LineSearchModes>(m, "LineSearchModes")
        .value("AUGLANG", LineSearchModes::AUGLANG)
        .value("LANG", LineSearchModes::LANG)
        .value("L1", LineSearchModes::L1)
        .value("NOLS", LineSearchModes::NOLS);
    nb::enum_<QPPivotModes>(m, "QPPivotModes")
        .value("OneByOne", QPPivotModes::OneByOne)
        .value("TwoByTwo", QPPivotModes::TwoByTwo);
    nb::enum_<PDStepStrategies>(m, "PDStepStrategies")
        .value("PrimSlackEq_Iq", PDStepStrategies::PrimSlackEq_Iq)
        .value("AllMinimum", PDStepStrategies::AllMinimum)
        .value("PrimSlack_EqIq", PDStepStrategies::PrimSlack_EqIq)
        .value("MaxEq", PDStepStrategies::MaxEq);
    nb::enum_<ConvergenceFlags>(m, "ConvergenceFlags", nb::is_arithmetic())
        .value("CONVERGED", ConvergenceFlags::CONVERGED)
        .value("ACCEPTABLE", ConvergenceFlags::ACCEPTABLE)
        .value("NOTCONVERGED", ConvergenceFlags::NOTCONVERGED)
        .value("DIVERGING", ConvergenceFlags::DIVERGING);
    nb::enum_<AlgorithmModes>(m, "AlgorithmModes")
        .value("OPT", AlgorithmModes::OPT)
        .value("OPTNO", AlgorithmModes::OPTNO)
        .value("SOE", AlgorithmModes::SOE)
        .value("INIT", AlgorithmModes::INIT);

    nb::enum_<QPOrderingModes>(m, "QPOrderingModes")
        .value("MINDEG", QPOrderingModes::MINDEG)
        .value("METIS", QPOrderingModes::METIS)
        .value("PARMETIS", QPOrderingModes::PARMETIS);
    nb::enum_<BestCriteriaModes>(m, "BestCriteriaModes")
        .value("ECONS", BestCriteriaModes::ECONS)
        .value("ICONS", BestCriteriaModes::ICONS)
        .value("KKT", BestCriteriaModes::KKT)
        .value("OBJ", BestCriteriaModes::OBJ);
}
