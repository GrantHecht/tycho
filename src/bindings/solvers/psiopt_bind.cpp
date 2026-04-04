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
    obj.def("set_qp_params", &PSIOPT::set_qp_params);

    obj.def_rw("max_iters", &PSIOPT::max_iters_, "");
    obj.def_rw("max_acc_iters", &PSIOPT::max_acc_iters_, "");
    obj.def_rw("max_ls_iters", &PSIOPT::max_ls_iters_, "");

    obj.def("set_max_iters", &PSIOPT::set_max_iters);
    obj.def("set_max_acc_iters", &PSIOPT::set_max_acc_iters);
    obj.def("set_max_ls_iters", &PSIOPT::set_max_ls_iters);

    obj.def_rw("alpha_red", &PSIOPT::alpha_red_, "");
    obj.def("set_alpha_red", &PSIOPT::set_alpha_red);

    obj.def_rw("wide_console", &PSIOPT::wide_console_);

    obj.def_rw("fast_factor_alg", &PSIOPT::fast_factor_alg_, "");

    obj.def_rw("last_total_time", &PSIOPT::last_total_time_, "");
    obj.def_rw("last_pre_time", &PSIOPT::last_pre_time_, "");
    obj.def_rw("last_func_time", &PSIOPT::last_func_time_, "");
    obj.def_rw("last_kkt_time", &PSIOPT::last_kkt_time_, "");
    obj.def_rw("last_misc_time", &PSIOPT::last_misc_time_, "");
    obj.def_rw("last_print_time", &PSIOPT::last_print_time_, "");
    obj.def_rw("last_solver_init_time", &PSIOPT::last_solver_init_time_, "");
    obj.def_rw("last_iter_num", &PSIOPT::last_iter_num_, "");
    obj.def_rw("last_obj_val", &PSIOPT::last_obj_val_);

    obj.def_rw("obj_scale", &PSIOPT::obj_scale_, "");
    obj.def_rw("print_level", &PSIOPT::print_level_, "");
    obj.def("set_print_level", &PSIOPT::set_print_level);

    obj.def_rw("converge_flag", &PSIOPT::converge_flag_);

    obj.def("get_convergence_flag", &PSIOPT::get_convergence_flag);

    obj.def_rw("kkt_tol", &PSIOPT::kkt_tol_, "");
    obj.def_rw("bar_tol", &PSIOPT::bar_tol_, "");
    obj.def_rw("eq_con_tol", &PSIOPT::econ_tol_, "");
    obj.def_rw("ineq_con_tol", &PSIOPT::icon_tol_, "");

    obj.def("set_kkt_tol", &PSIOPT::set_kkt_tol);
    obj.def("set_bar_tol", &PSIOPT::set_bar_tol);
    obj.def("set_eq_con_tol", &PSIOPT::set_econ_tol);
    obj.def("set_ineq_con_tol", &PSIOPT::set_icon_tol);

    obj.def("set_tols", &PSIOPT::set_tols, nb::arg("kkt_tol") = 1.0e-6,
            nb::arg("eq_con_tol") = 1.0e-6, nb::arg("ineq_con_tol") = 1.0e-6,
            nb::arg("bar_tol") = 1.0e-6);

    obj.def_rw("acc_kkt_tol", &PSIOPT::acc_kkt_tol_, "");
    obj.def_rw("acc_bar_tol", &PSIOPT::acc_bar_tol_, "");
    obj.def_rw("acc_eq_con_tol", &PSIOPT::acc_econ_tol_, "");
    obj.def_rw("acc_ineq_con_tol", &PSIOPT::acc_icon_tol_, "");

    obj.def("set_acc_kkt_tol", &PSIOPT::set_acc_kkt_tol);
    obj.def("set_acc_bar_tol", &PSIOPT::set_acc_bar_tol);
    obj.def("set_acc_eq_con_tol", &PSIOPT::set_acc_econ_tol);
    obj.def("set_acc_ineq_con_tol", &PSIOPT::set_acc_icon_tol);

    obj.def("set_acc_tols", &PSIOPT::set_acc_tols, nb::arg("acc_kkt_tol") = 1.0e-2,
            nb::arg("acc_eq_con_tol") = 1.0e-3, nb::arg("acc_ineq_con_tol") = 1.0e-3,
            nb::arg("acc_bar_tol") = 1.0e-3);

    obj.def_rw("div_kkt_tol", &PSIOPT::div_kkt_tol_, "");
    obj.def_rw("div_bar_tol", &PSIOPT::div_bar_tol_, "");
    obj.def_rw("div_eq_con_tol", &PSIOPT::div_econ_tol_, "");
    obj.def_rw("div_ineq_con_tol", &PSIOPT::div_icon_tol_, "");

    obj.def("set_div_kkt_tol", &PSIOPT::set_div_kkt_tol);
    obj.def("set_div_bar_tol", &PSIOPT::set_div_bar_tol);
    obj.def("set_div_eq_con_tol", &PSIOPT::set_div_econ_tol);
    obj.def("set_div_ineq_con_tol", &PSIOPT::set_div_icon_tol);

    obj.def_rw("neg_slack_reset", &PSIOPT::neg_slack_reset_, "");

    obj.def_rw("bound_fraction", &PSIOPT::bound_fraction_, "");
    obj.def("set_bound_fraction", &PSIOPT::set_bound_fraction);

    obj.def_rw("bound_push", &PSIOPT::bound_push_, "");

    /////////////////////////////////////////////////////////////

    obj.def_rw("delta_h", &PSIOPT::delta_h_, "");
    obj.def_rw("incr_h", &PSIOPT::incr_h_, "");
    obj.def_rw("decr_h", &PSIOPT::decr_h_, "");

    obj.def("set_delta_h", &PSIOPT::set_delta_h);
    obj.def("set_incr_h", &PSIOPT::set_incr_h);
    obj.def("set_decr_h", &PSIOPT::set_decr_h);

    obj.def("set_hpert_params", &PSIOPT::set_hpert_params, nb::arg("delta_h"), nb::arg("incr_h"),
            nb::arg("decr_h"));

    /////////////////////////////////////////////////////////////
    obj.def_rw("init_mu", &PSIOPT::init_mu_, "");
    obj.def_rw("min_mu", &PSIOPT::min_mu_, "");
    obj.def_rw("max_mu", &PSIOPT::max_mu_, "");

    obj.def_rw("max_soc", &PSIOPT::max_soc_, "");

    obj.def_rw("pd_step_strategy", &PSIOPT::pd_step_strategy_, "");
    obj.def_rw("soe_bound_relax", &PSIOPT::soe_bound_relax_, "");
    obj.def_rw("qp_par_solve", &PSIOPT::qp_par_solve_, "");

    obj.def_rw("soe_mode", &PSIOPT::soe_mode_, "");

    //////////////////////////////////////////////////////////////////////////////////////////////////

    obj.def_rw("opt_bar_mode", &PSIOPT::opt_bar_mode_, "");
    obj.def_rw("soe_bar_mode", &PSIOPT::soe_bar_mode_, "");

    obj.def("set_opt_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_opt_bar_mode));
    obj.def("set_opt_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_opt_bar_mode));
    obj.def("set_soe_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_soe_bar_mode));
    obj.def("set_soe_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_soe_bar_mode));

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("opt_ls_mode", &PSIOPT::opt_ls_mode_, "");
    obj.def_rw("soe_ls_mode", &PSIOPT::soe_ls_mode_, "");

    obj.def("set_opt_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_opt_ls_mode));
    obj.def("set_opt_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_opt_ls_mode));
    obj.def("set_soe_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_soe_ls_mode));
    obj.def("set_soe_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_soe_ls_mode));

    //////////////////////////////////////////////////////////////////////////////////////////////////

    obj.def_rw("force_qp_analysis", &PSIOPT::force_qp_analysis_, "");
    obj.def_rw("qp_ref_steps", &PSIOPT::qp_ref_steps_, "");

    obj.def_rw("qp_pivot_perturb", &PSIOPT::qp_pivot_perturb_, "");
    obj.def_rw("qp_threads", &PSIOPT::qp_threads_, "");
    obj.def_rw("qp_pivot_strategy", &PSIOPT::qp_pivot_strategy_, "");

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("qp_ordering_mode", &PSIOPT::qp_ord_, "");

    obj.def("set_qp_ordering_mode",
            nb::overload_cast<QPOrderingModes>(&PSIOPT::set_qp_ordering_mode));
    obj.def("set_qp_ordering_mode",
            nb::overload_cast<const std::string &>(&PSIOPT::set_qp_ordering_mode));

#ifdef USE_ACCELERATE_SPARSE
    obj.def_rw("accel_pivot_tolerance", &PSIOPT::accel_pivot_tolerance_);
    obj.def_rw("accel_zero_tolerance", &PSIOPT::accel_zero_tolerance_);
    obj.def("set_accel_pivot_tolerance", &PSIOPT::set_accel_pivot_tolerance);
    obj.def("set_accel_zero_tolerance", &PSIOPT::set_accel_zero_tolerance);
#endif

    //////////////////////////////////////////////////////////////////////////////////////////////////
    obj.def_rw("qp_print", &PSIOPT::qp_print_);

    obj.def_rw("return_best", &PSIOPT::return_best_);
    obj.def_prop_rw(
        "best_criteria", [](const PSIOPT &self) { return self.best_criteria_; },
        [](PSIOPT &self, nb::object val) {
            if (nb::isinstance<BestCriteriaModes>(val))
                self.best_criteria_ = nb::cast<BestCriteriaModes>(val);
            else if (nb::isinstance<nb::str>(val))
                self.best_criteria_ = PSIOPT::strto_BestCriteriaMode(nb::cast<std::string>(val));
            else
                throw nb::type_error("expected BestCriteriaModes enum or str");
        });
    obj.def("set_best_criteria", nb::overload_cast<BestCriteriaModes>(&PSIOPT::set_best_criteria));
    obj.def("set_best_criteria",
            nb::overload_cast<const std::string &>(&PSIOPT::set_best_criteria));

    obj.def_rw("cnr_mode", &PSIOPT::cnr_mode_, "");

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
