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
//   - Settings struct: def_rw -> def_prop_rw for fields moved into PSIOPT::Settings
// =============================================================================

#include "psiopt_bind.h"
#include "tycho/detail/solvers/psiopt.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

// Helper macros for binding settings fields as read-write properties on PSIOPT.
// These produce lambda-based def_prop_rw that forward to settings_.field.
#define BIND_SETTINGS_RW(obj, pyname, field, ...)                                                  \
    obj.def_prop_rw(                                                                               \
        pyname, [](const PSIOPT &self) { return self.settings_.field; },                           \
        [](PSIOPT &self, decltype(self.settings_.field) v) { self.settings_.field = v; }           \
        __VA_OPT__(, ) __VA_ARGS__)

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

    BIND_SETTINGS_RW(obj, "max_iters", max_iters_, "");
    BIND_SETTINGS_RW(obj, "max_acc_iters", max_acc_iters_, "");
    BIND_SETTINGS_RW(obj, "max_ls_iters", max_ls_iters_, "");

    obj.def("set_max_iters", &PSIOPT::set_max_iters);
    obj.def("set_max_acc_iters", &PSIOPT::set_max_acc_iters);
    obj.def("set_max_ls_iters", &PSIOPT::set_max_ls_iters);

    BIND_SETTINGS_RW(obj, "alpha_red", alpha_red_, "");
    obj.def("set_alpha_red", &PSIOPT::set_alpha_red);

    BIND_SETTINGS_RW(obj, "wide_console", wide_console_);

    BIND_SETTINGS_RW(obj, "fast_factor_alg", fast_factor_alg_, "");

    obj.def_rw("last_total_time", &PSIOPT::last_total_time_, "");
    obj.def_rw("last_pre_time", &PSIOPT::last_pre_time_, "");
    obj.def_rw("last_func_time", &PSIOPT::last_func_time_, "");
    obj.def_rw("last_kkt_time", &PSIOPT::last_kkt_time_, "");
    obj.def_rw("last_misc_time", &PSIOPT::last_misc_time_, "");
    obj.def_rw("last_print_time", &PSIOPT::last_print_time_, "");
    obj.def_rw("last_solver_init_time", &PSIOPT::last_solver_init_time_, "");
    obj.def_rw("last_iter_num", &PSIOPT::last_iter_num_, "");
    obj.def_rw("last_obj_val", &PSIOPT::last_obj_val_);

    BIND_SETTINGS_RW(obj, "obj_scale", obj_scale_, "");
    BIND_SETTINGS_RW(obj, "print_level", print_level_, "");
    obj.def("set_print_level", &PSIOPT::set_print_level);

    obj.def_rw("converge_flag", &PSIOPT::converge_flag_);

    obj.def("get_convergence_flag", &PSIOPT::get_convergence_flag);

    BIND_SETTINGS_RW(obj, "kkt_tol", kkt_tol_, "");
    BIND_SETTINGS_RW(obj, "bar_tol", bar_tol_, "");
    BIND_SETTINGS_RW(obj, "eq_con_tol", econ_tol_, "");
    BIND_SETTINGS_RW(obj, "ineq_con_tol", icon_tol_, "");

    obj.def("set_kkt_tol", &PSIOPT::set_kkt_tol);
    obj.def("set_bar_tol", &PSIOPT::set_bar_tol);
    obj.def("set_eq_con_tol", &PSIOPT::set_econ_tol);
    obj.def("set_ineq_con_tol", &PSIOPT::set_icon_tol);

    obj.def("set_tols", &PSIOPT::set_tols, nb::arg("kkt_tol") = 1.0e-6,
            nb::arg("eq_con_tol") = 1.0e-6, nb::arg("ineq_con_tol") = 1.0e-6,
            nb::arg("bar_tol") = 1.0e-6);

    BIND_SETTINGS_RW(obj, "acc_kkt_tol", acc_kkt_tol_, "");
    BIND_SETTINGS_RW(obj, "acc_bar_tol", acc_bar_tol_, "");
    BIND_SETTINGS_RW(obj, "acc_eq_con_tol", acc_econ_tol_, "");
    BIND_SETTINGS_RW(obj, "acc_ineq_con_tol", acc_icon_tol_, "");

    obj.def("set_acc_kkt_tol", &PSIOPT::set_acc_kkt_tol);
    obj.def("set_acc_bar_tol", &PSIOPT::set_acc_bar_tol);
    obj.def("set_acc_eq_con_tol", &PSIOPT::set_acc_econ_tol);
    obj.def("set_acc_ineq_con_tol", &PSIOPT::set_acc_icon_tol);

    obj.def("set_acc_tols", &PSIOPT::set_acc_tols, nb::arg("acc_kkt_tol") = 1.0e-2,
            nb::arg("acc_eq_con_tol") = 1.0e-3, nb::arg("acc_ineq_con_tol") = 1.0e-3,
            nb::arg("acc_bar_tol") = 1.0e-3);

    BIND_SETTINGS_RW(obj, "div_kkt_tol", div_kkt_tol_, "");
    BIND_SETTINGS_RW(obj, "div_bar_tol", div_bar_tol_, "");
    BIND_SETTINGS_RW(obj, "div_eq_con_tol", div_econ_tol_, "");
    BIND_SETTINGS_RW(obj, "div_ineq_con_tol", div_icon_tol_, "");

    obj.def("set_div_kkt_tol", &PSIOPT::set_div_kkt_tol);
    obj.def("set_div_bar_tol", &PSIOPT::set_div_bar_tol);
    obj.def("set_div_eq_con_tol", &PSIOPT::set_div_econ_tol);
    obj.def("set_div_ineq_con_tol", &PSIOPT::set_div_icon_tol);

    BIND_SETTINGS_RW(obj, "neg_slack_reset", neg_slack_reset_, "");

    BIND_SETTINGS_RW(obj, "bound_fraction", bound_fraction_, "");
    obj.def("set_bound_fraction", &PSIOPT::set_bound_fraction);

    BIND_SETTINGS_RW(obj, "bound_push", bound_push_, "");

    /////////////////////////////////////////////////////////////

    BIND_SETTINGS_RW(obj, "delta_h", delta_h_, "");
    BIND_SETTINGS_RW(obj, "incr_h", incr_h_, "");
    BIND_SETTINGS_RW(obj, "decr_h", decr_h_, "");

    obj.def("set_delta_h", &PSIOPT::set_delta_h);
    obj.def("set_incr_h", &PSIOPT::set_incr_h);
    obj.def("set_decr_h", &PSIOPT::set_decr_h);

    obj.def("set_hpert_params", &PSIOPT::set_hpert_params, nb::arg("delta_h"), nb::arg("incr_h"),
            nb::arg("decr_h"));

    /////////////////////////////////////////////////////////////
    BIND_SETTINGS_RW(obj, "init_mu", init_mu_, "");
    BIND_SETTINGS_RW(obj, "min_mu", min_mu_, "");
    BIND_SETTINGS_RW(obj, "max_mu", max_mu_, "");

    BIND_SETTINGS_RW(obj, "max_soc", max_soc_, "");

    BIND_SETTINGS_RW(obj, "pd_step_strategy", pd_step_strategy_, "");
    BIND_SETTINGS_RW(obj, "soe_bound_relax", soe_bound_relax_, "");
    BIND_SETTINGS_RW(obj, "qp_par_solve", qp_par_solve_, "");

    BIND_SETTINGS_RW(obj, "soe_mode", soe_mode_, "");

    //////////////////////////////////////////////////////////////////////////////////////////////////

    BIND_SETTINGS_RW(obj, "opt_bar_mode", opt_bar_mode_, "");
    BIND_SETTINGS_RW(obj, "soe_bar_mode", soe_bar_mode_, "");

    obj.def("set_opt_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_opt_bar_mode));
    obj.def("set_opt_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_opt_bar_mode));
    obj.def("set_soe_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_soe_bar_mode));
    obj.def("set_soe_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_soe_bar_mode));

    //////////////////////////////////////////////////////////////////////////////////////////////////
    BIND_SETTINGS_RW(obj, "opt_ls_mode", opt_ls_mode_, "");
    BIND_SETTINGS_RW(obj, "soe_ls_mode", soe_ls_mode_, "");

    obj.def("set_opt_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_opt_ls_mode));
    obj.def("set_opt_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_opt_ls_mode));
    obj.def("set_soe_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_soe_ls_mode));
    obj.def("set_soe_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_soe_ls_mode));

    //////////////////////////////////////////////////////////////////////////////////////////////////

    BIND_SETTINGS_RW(obj, "force_qp_analysis", force_qp_analysis_, "");
    BIND_SETTINGS_RW(obj, "qp_ref_steps", qp_ref_steps_, "");

    BIND_SETTINGS_RW(obj, "qp_pivot_perturb", qp_pivot_perturb_, "");
    BIND_SETTINGS_RW(obj, "qp_threads", qp_threads_, "");
    BIND_SETTINGS_RW(obj, "qp_pivot_strategy", qp_pivot_strategy_, "");

    //////////////////////////////////////////////////////////////////////////////////////////////////
    BIND_SETTINGS_RW(obj, "qp_ordering_mode", qp_ord_, "");

    obj.def("set_qp_ordering_mode",
            nb::overload_cast<QPOrderingModes>(&PSIOPT::set_qp_ordering_mode));
    obj.def("set_qp_ordering_mode",
            nb::overload_cast<const std::string &>(&PSIOPT::set_qp_ordering_mode));

#ifdef USE_ACCELERATE_SPARSE
    BIND_SETTINGS_RW(obj, "accel_pivot_tolerance", accel_pivot_tolerance_);
    BIND_SETTINGS_RW(obj, "accel_zero_tolerance", accel_zero_tolerance_);
    obj.def("set_accel_pivot_tolerance", &PSIOPT::set_accel_pivot_tolerance);
    obj.def("set_accel_zero_tolerance", &PSIOPT::set_accel_zero_tolerance);
#endif

    //////////////////////////////////////////////////////////////////////////////////////////////////
    BIND_SETTINGS_RW(obj, "qp_print", qp_print_);

    BIND_SETTINGS_RW(obj, "return_best", return_best_);
    obj.def_prop_rw(
        "best_criteria", [](const PSIOPT &self) { return self.settings_.best_criteria_; },
        [](PSIOPT &self, nb::object val) {
            if (nb::isinstance<BestCriteriaModes>(val))
                self.settings_.best_criteria_ = nb::cast<BestCriteriaModes>(val);
            else if (nb::isinstance<nb::str>(val))
                self.settings_.best_criteria_ =
                    PSIOPT::strto_BestCriteriaMode(nb::cast<std::string>(val));
            else
                throw nb::type_error("expected BestCriteriaModes enum or str");
        });
    obj.def("set_best_criteria", nb::overload_cast<BestCriteriaModes>(&PSIOPT::set_best_criteria));
    obj.def("set_best_criteria",
            nb::overload_cast<const std::string &>(&PSIOPT::set_best_criteria));

    BIND_SETTINGS_RW(obj, "cnr_mode", cnr_mode_, "");

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
