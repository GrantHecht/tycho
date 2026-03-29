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

#include "psiopt_bind.h"
#include "tycho/detail/solvers/psiopt.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

void TychoBind<PSIOPT>::Build(nb::module_ &m) {
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

    obj.def("optimize", &PSIOPT::optimize, "");
    obj.def("solve_optimize", &PSIOPT::solve_optimize, "");
    obj.def("solve", &PSIOPT::solve, "");
    obj.def("set_qp_params", &PSIOPT::setQPParams);

    obj.def_rw("max_iters", &PSIOPT::MaxIters, "");
    obj.def_rw("max_acc_iters", &PSIOPT::MaxAccIters, "");
    obj.def_rw("max_ls_iters", &PSIOPT::MaxLSIters, "");

    obj.def("set_max_iters", &PSIOPT::set_MaxIters);
    obj.def("set_max_acc_iters", &PSIOPT::set_MaxAccIters);
    obj.def("set_max_ls_iters", &PSIOPT::set_MaxLSIters);

    obj.def_rw("alpha_red", &PSIOPT::alphaRed, "");
    obj.def("set_alpha_red", &PSIOPT::set_alphaRed);

    obj.def_rw("wide_console", &PSIOPT::WideConsole);

    obj.def_rw("fast_factor_alg", &PSIOPT::FastFactorAlg, "");

    obj.def_rw("last_total_time", &PSIOPT::LastTotalTime, "");
    obj.def_rw("last_pre_time", &PSIOPT::LastPreTime, "");
    obj.def_rw("last_func_time", &PSIOPT::LastFuncTime, "");
    obj.def_rw("last_kkt_time", &PSIOPT::LastKKTTime, "");
    obj.def_rw("last_misc_time", &PSIOPT::LastMiscTime, "");
    obj.def_rw("last_print_time", &PSIOPT::LastPrintTime, "");
    obj.def_rw("last_solver_init_time", &PSIOPT::LastSolverInitTime, "");
    obj.def_rw("last_iter_num", &PSIOPT::LastIterNum, "");
    obj.def_rw("last_obj_val", &PSIOPT::LastObjVal);

    obj.def_rw("obj_scale", &PSIOPT::ObjScale, "");
    obj.def_rw("print_level", &PSIOPT::PrintLevel, "");
    obj.def("set_print_level", &PSIOPT::set_PrintLevel);

    obj.def_rw("converge_flag", &PSIOPT::ConvergeFlag);

    obj.def("get_convergence_flag", &PSIOPT::get_ConvergenceFlag);

    obj.def_rw("kkt_tol", &PSIOPT::KKTtol, "");
    obj.def_rw("bar_tol", &PSIOPT::Bartol, "");
    obj.def_rw("eq_con_tol", &PSIOPT::EContol, "");
    obj.def_rw("ineq_con_tol", &PSIOPT::IContol, "");

    obj.def("set_kkt_tol", &PSIOPT::set_KKTtol);
    obj.def("set_bar_tol", &PSIOPT::set_Bartol);
    obj.def("set_eq_con_tol", &PSIOPT::set_EContol);
    obj.def("set_ineq_con_tol", &PSIOPT::set_IContol);

    obj.def("set_tols", &PSIOPT::set_tols, nb::arg("kkt_tol") = 1.0e-6,
            nb::arg("eq_con_tol") = 1.0e-6, nb::arg("ineq_con_tol") = 1.0e-6,
            nb::arg("bar_tol") = 1.0e-6);

    obj.def_rw("acc_kkt_tol", &PSIOPT::AccKKTtol, "");
    obj.def_rw("acc_bar_tol", &PSIOPT::AccBartol, "");
    obj.def_rw("acc_eq_con_tol", &PSIOPT::AccEContol, "");
    obj.def_rw("acc_ineq_con_tol", &PSIOPT::AccIContol, "");

    obj.def("set_acc_kkt_tol", &PSIOPT::set_AccKKTtol);
    obj.def("set_acc_bar_tol", &PSIOPT::set_AccBartol);
    obj.def("set_acc_eq_con_tol", &PSIOPT::set_AccEContol);
    obj.def("set_acc_ineq_con_tol", &PSIOPT::set_AccIContol);

    obj.def("set_acc_tols", &PSIOPT::set_Acctols, nb::arg("acc_kkt_tol") = 1.0e-2,
            nb::arg("acc_eq_con_tol") = 1.0e-3, nb::arg("acc_ineq_con_tol") = 1.0e-3,
            nb::arg("acc_bar_tol") = 1.0e-3);

    obj.def_rw("div_kkt_tol", &PSIOPT::DivKKTtol, "");
    obj.def_rw("div_bar_tol", &PSIOPT::DivBartol, "");
    obj.def_rw("div_eq_con_tol", &PSIOPT::DivEContol, "");
    obj.def_rw("div_ineq_con_tol", &PSIOPT::DivIContol, "");

    obj.def("set_div_kkt_tol", &PSIOPT::set_DivKKTtol);
    obj.def("set_div_bar_tol", &PSIOPT::set_DivBartol);
    obj.def("set_div_eq_con_tol", &PSIOPT::set_DivEContol);
    obj.def("set_div_ineq_con_tol", &PSIOPT::set_DivIContol);

    obj.def_rw("neg_slack_reset", &PSIOPT::NegSlackReset, "");

    obj.def_rw("bound_fraction", &PSIOPT::BoundFraction, "");
    obj.def("set_bound_fraction", &PSIOPT::set_BoundFraction);

    obj.def_rw("bound_push", &PSIOPT::BoundPush, "");

    /////////////////////////////////////////////////////////////

    obj.def_rw("delta_h", &PSIOPT::deltaH, "");
    obj.def_rw("incr_h", &PSIOPT::incrH, "");
    obj.def_rw("decr_h", &PSIOPT::decrH, "");

    obj.def("set_delta_h", &PSIOPT::set_deltaH);
    obj.def("set_incr_h", &PSIOPT::set_incrH);
    obj.def("set_decr_h", &PSIOPT::set_decrH);

    obj.def("set_hpert_params", &PSIOPT::set_HpertParams, nb::arg("delta_h"), nb::arg("incr_h"),
            nb::arg("decr_h"));

    /////////////////////////////////////////////////////////////
    obj.def_rw("init_mu", &PSIOPT::initMu, "");
    obj.def_rw("min_mu", &PSIOPT::MinMu, "");
    obj.def_rw("max_mu", &PSIOPT::MaxMu, "");

    obj.def_rw("max_soc", &PSIOPT::MaxSOC, "");

    obj.def_rw("pd_step_strategy", &PSIOPT::PDStepStrategy, "");
    obj.def_rw("so_ebound_relax", &PSIOPT::SOEboundRelax, "");
    obj.def_rw("qp_par_solve", &PSIOPT::QPParSolve, "");

    obj.def_rw("soe_mode", &PSIOPT::SoeMode, "");

    //////////////////////////////////////////////////////////////////////////////////////////////////

    obj.def_rw("opt_bar_mode", &PSIOPT::OptBarMode, "");
    obj.def_rw("soe_bar_mode", &PSIOPT::SoeBarMode, "");

    obj.def("set_opt_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_OptBarMode));
    obj.def("set_opt_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_OptBarMode));
    obj.def("set_soe_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_SoeBarMode));
    obj.def("set_soe_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_SoeBarMode));

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("opt_ls_mode", &PSIOPT::OptLSMode, "");
    obj.def_rw("soe_ls_mode", &PSIOPT::SoeLSMode, "");

    obj.def("set_opt_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_OptLSMode));
    obj.def("set_opt_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_OptLSMode));
    obj.def("set_soe_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_SoeLSMode));
    obj.def("set_soe_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_SoeLSMode));

    //////////////////////////////////////////////////////////////////////////////////////////////////

    obj.def_rw("force_q_panalysis", &PSIOPT::ForceQPanalysis, "");
    obj.def_rw("qp_ref_steps", &PSIOPT::QPRefSteps, "");

    obj.def_rw("qp_pivot_perturb", &PSIOPT::QPPivotPerturb, "");
    obj.def_rw("qp_threads", &PSIOPT::QPThreads, "");
    obj.def_rw("qp_pivot_strategy", &PSIOPT::QPPivotStrategy, "");

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("qp_ordering_mode", &PSIOPT::QPOrd, "");

    obj.def("set_qp_ordering_mode",
            nb::overload_cast<QPOrderingModes>(&PSIOPT::set_QPOrderingMode));
    obj.def("set_qp_ordering_mode",
            nb::overload_cast<const std::string &>(&PSIOPT::set_QPOrderingMode));

#ifdef USE_ACCELERATE_SPARSE
    obj.def_rw("accel_pivot_tolerance", &PSIOPT::AccelPivotTolerance);
    obj.def_rw("accel_zero_tolerance", &PSIOPT::AccelZeroTolerance);
    obj.def("set_accel_pivot_tolerance", &PSIOPT::set_AccelPivotTolerance);
    obj.def("set_accel_zero_tolerance", &PSIOPT::set_AccelZeroTolerance);
#endif

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("qp_print", &PSIOPT::QPPrint);

    obj.def_rw("diagnostic", &PSIOPT::Diagnostic);

    obj.def_rw("return_best", &PSIOPT::ReturnBest);
    obj.def_prop_rw(
        "best_criteria", [](const PSIOPT &self) { return self.BestCriteria; },
        [](PSIOPT &self, nb::object val) {
            if (nb::isinstance<BestCriteriaModes>(val))
                self.BestCriteria = nb::cast<BestCriteriaModes>(val);
            else if (nb::isinstance<nb::str>(val))
                self.BestCriteria = PSIOPT::strto_BestCriteriaMode(nb::cast<std::string>(val));
            else
                throw nb::type_error("expected BestCriteriaModes enum or str");
        });
    obj.def("set_best_criteria", nb::overload_cast<BestCriteriaModes>(&PSIOPT::set_BestCriteria));
    obj.def("set_best_criteria", nb::overload_cast<const std::string &>(&PSIOPT::set_BestCriteria));

    obj.def_rw("storespmat", &PSIOPT::storespmat, "");
    obj.def("get_sp_mat", &PSIOPT::getSPmat, "");
    obj.def("get_sp_mat2", &PSIOPT::getSPmat2, "");

    obj.def_rw("cnr_mode", &PSIOPT::CNRMode, "");

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
