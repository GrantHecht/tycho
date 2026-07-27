// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
//   - PSIOPT refactor (PR #39): Settings/SolveResult structs with def_prop_rw/def_prop_ro,
//     validated setters, result read-only bindings, dead binding removal
// =============================================================================

#include "psiopt_bind.h"
#include "tycho/detail/solvers/ipopt_backend.h"
#include "tycho/detail/solvers/psiopt.h"

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

// Helper macros for binding settings fields as read-write properties on PSIOPT.
// These produce lambda-based def_prop_rw that forward through the settings() accessor.
#define BIND_SETTINGS_RW(obj, pyname, field, ...)                                                  \
    obj.def_prop_rw(                                                                               \
        pyname, [](const PSIOPT &self) { return self.settings().field; },                          \
        [](PSIOPT &self, decltype(self.settings().field) v) {                                      \
            self.settings().field = v;                                                             \
        } __VA_OPT__(, ) __VA_ARGS__)

// Like BIND_SETTINGS_RW, but routes the setter through a validated method.
// Use for fields that have a corresponding set_* method with validation logic.
#define BIND_SETTINGS_VALIDATED(obj, pyname, field, setter, ...)                                   \
    obj.def_prop_rw(                                                                               \
        pyname, [](const PSIOPT &self) { return self.settings().field; },                          \
        [](PSIOPT &self, decltype(self.settings().field) v) { self.setter(v); } __VA_OPT__(, )     \
            __VA_ARGS__)

// Helper macro for binding result fields as read-only properties on PSIOPT.
// These produce lambda-based def_prop_ro that forward through the result() accessor.
#define BIND_RESULT_RO(obj, pyname, field, ...)                                                    \
    obj.def_prop_ro(pyname, [](const PSIOPT &self) { return self.result().field; } __VA_OPT__(, )  \
                                __VA_ARGS__)

void TychoBind<PSIOPT>::build(nb::module_ &m) {
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

    obj.def("optimize", &PSIOPT::optimize, nb::call_guard<nb::gil_scoped_release>(), "");
    obj.def("solve_optimize", &PSIOPT::solve_optimize, nb::call_guard<nb::gil_scoped_release>(),
            "");
    obj.def("solve", &PSIOPT::solve, nb::call_guard<nb::gil_scoped_release>(), "");

    BIND_SETTINGS_VALIDATED(obj, "max_iters", max_iters_, set_max_iters, "");
    BIND_SETTINGS_VALIDATED(obj, "max_acc_iters", max_acc_iters_, set_max_acc_iters, "");
    BIND_SETTINGS_VALIDATED(obj, "max_ls_iters", max_ls_iters_, set_max_ls_iters, "");

    obj.def("set_max_iters", &PSIOPT::set_max_iters);
    obj.def("set_max_acc_iters", &PSIOPT::set_max_acc_iters);
    obj.def("set_max_ls_iters", &PSIOPT::set_max_ls_iters);

    BIND_SETTINGS_VALIDATED(
        obj, "max_soc", max_soc_, set_max_soc,
        "Maximum number of second-order correction steps attempted after a rejected trial "
        "step. 0 (default) disables second-order correction entirely, so the solver behaves "
        "exactly as it did before this feature existed; the recommended enable value is 4 "
        "(Wachter & Biegler 2006).");
    BIND_SETTINGS_VALIDATED(
        obj, "ls_extended_iters", ls_extended_iters_, set_ls_extended_iters,
        "Extra backtracking trials allowed on the classic line-search ladder once the normal "
        "cap and second-order correction (if enabled) are exhausted. 0 (default) disables "
        "extended backtracking entirely.");

    BIND_SETTINGS_VALIDATED(obj, "alpha_red", alpha_red_, set_alpha_red, "");
    obj.def("set_alpha_red", &PSIOPT::set_alpha_red);

    BIND_SETTINGS_RW(obj, "wide_console", wide_console_);

    BIND_SETTINGS_RW(obj, "fast_factor_alg", fast_factor_alg_, "");

    BIND_RESULT_RO(obj, "last_total_time", total_time_, "");
    BIND_RESULT_RO(obj, "last_pre_time", pre_time_, "");
    BIND_RESULT_RO(obj, "last_func_time", func_time_, "");
    BIND_RESULT_RO(obj, "last_kkt_time", kkt_time_, "");
    obj.def_prop_ro(
        "last_misc_time", [](const PSIOPT &self) { return self.result().misc_time(); }, "");
    BIND_RESULT_RO(obj, "last_print_time", print_time_, "");
    BIND_RESULT_RO(obj, "last_solver_init_time", solver_init_time_, "");
    BIND_RESULT_RO(obj, "last_iter_num", iter_num_, "");
    BIND_RESULT_RO(obj, "last_obj_val", obj_val_);
    BIND_RESULT_RO(obj, "last_primals", primals_, "");

    BIND_RESULT_RO(obj, "last_soc_steps", soc_steps_taken_,
                   "Number of second-order correction back-substitutions performed during the most "
                   "recent solve. Always 0 unless max_soc is set > 0.");
    BIND_RESULT_RO(obj, "last_watchdog_activations", watchdog_activations_,
                   "Number of times the watchdog recovery heuristic armed during the most recent "
                   "solve. Always 0 unless watchdog is enabled.");
    obj.def_prop_ro(
        "last_recovery_depth_histogram",
        [](const PSIOPT &self) {
            const auto &h = self.result().recovery_depth_histogram_;
            return std::vector<int>(h.begin(), h.end());
        },
        "Counts of how each rejected step's recovery was resolved during the most recent "
        "solve, as a 5-element list: [second-order correction, extended backtracking, "
        "watchdog, unresolved, restoration]. The final bucket only increments when "
        "restoration_mode is proximal_switch or l1_nested.");

    BIND_RESULT_RO(obj, "last_funnel_width", last_funnel_width_,
                   "Final funnel width (tau) at the end of the most recent solve's last phase. "
                   "-1.0 unless acceptance_strategy is funnel, or if no acceptance test ran.");
    BIND_RESULT_RO(obj, "last_filter_size", last_filter_size_,
                   "Final number of stored filter (theta, phi) pairs at the end of the most "
                   "recent solve's last phase. -1 unless acceptance_strategy is filter.");
    BIND_RESULT_RO(obj, "last_filter_resets", last_filter_resets_,
                   "Number of filter-reset-heuristic clears during the most recent solve's last "
                   "phase. -1 unless acceptance_strategy is filter.");

    BIND_RESULT_RO(obj, "last_monotone_switches", last_monotone_switches_,
                   "Number of free -> monotone handoffs during the most recent solve's last "
                   "phase. -1 unless barrier_governor is monitored.");
    BIND_RESULT_RO(obj, "last_monotone_iters", last_monotone_iters_,
                   "Number of iterations spent in monotone mode during the most recent solve's "
                   "last phase. -1 unless barrier_governor is monitored.");

    BIND_RESULT_RO(obj, "last_feas_rest_entries", last_feas_rest_entries_,
                   "Number of times feasibility restoration was entered during the most recent "
                   "solve's last phase. -1 unless restoration_mode is proximal_switch or "
                   "l1_nested (no restoration strategy is constructed when restoration_mode is "
                   "off). Counts identically under both modes -- l1_nested has no separate "
                   "inner/outer iteration split, so this and last_feas_rest_iters mean the same "
                   "thing regardless of which mode is selected.");
    BIND_RESULT_RO(obj, "last_feas_rest_iters", last_feas_rest_iters_,
                   "Number of iterations spent in the feasibility-restoration phase during the "
                   "most recent solve's last phase. -1 unless restoration_mode is "
                   "proximal_switch or l1_nested.");

    BIND_RESULT_RO(obj, "last_prox_reg_primal", last_prox_reg_primal_,
                   "Persistent primal base shift (rho_k) applied to the Hessian diagonal at the "
                   "last factorized iteration of the most recent solve's last phase. -1.0 unless "
                   "inertia_mode is proximal_regularization (or if that phase converged before "
                   "its first factorization, so no shift was ever applied).");
    BIND_RESULT_RO(obj, "last_prox_reg_dual", last_prox_reg_dual_,
                   "Barrier-scaled dual shift (delta_c) subtracted from the constraint-row "
                   "diagonals at the last factorized iteration of the most recent solve's last "
                   "phase. -1.0 unless inertia_mode is proximal_regularization (or if that phase "
                   "converged before its first factorization); 0.0 if that iteration fell inside "
                   "a nested l1 restoration phase, where the shift is suppressed.");

    BIND_RESULT_RO(obj, "last_eval_exception", last_eval_exception_,
                   "Message of the most recent trial-point evaluation exception absorbed during "
                   "the most recent solve call, or the empty string when every evaluation "
                   "succeeded. A populated value means the acceptance machinery rejected one or "
                   "more un-evaluable trial steps (for example an iterate that stepped outside "
                   "an interpolation table's domain) and the solve continued -- to full "
                   "recovery, to a graceful ACCEPTABLE-level exit at an already-acceptable "
                   "iterate, or into feasibility restoration; a solve with none of those paths "
                   "available raises RuntimeError instead. In a multi-phase solve, an earlier "
                   "phase's message persists on this property even when a later phase aborts, "
                   "since the diagnostic is written at each phase's close.");

    BIND_SETTINGS_VALIDATED(obj, "obj_scale", obj_scale_, set_obj_scale, "");
    BIND_SETTINGS_VALIDATED(obj, "print_level", print_level_, set_print_level, "");
    obj.def("set_print_level", &PSIOPT::set_print_level);

    BIND_RESULT_RO(obj, "converge_flag", converge_flag_);

    BIND_SETTINGS_VALIDATED(obj, "kkt_tol", kkt_tol_, set_kkt_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "bar_tol", bar_tol_, set_bar_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "eq_con_tol", econ_tol_, set_econ_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "ineq_con_tol", icon_tol_, set_icon_tol, "");

    obj.def("set_kkt_tol", &PSIOPT::set_kkt_tol);
    obj.def("set_bar_tol", &PSIOPT::set_bar_tol);
    obj.def("set_eq_con_tol", &PSIOPT::set_econ_tol);
    obj.def("set_ineq_con_tol", &PSIOPT::set_icon_tol);

    obj.def("set_tols", &PSIOPT::set_tols, nb::arg("kkt_tol") = 1.0e-6,
            nb::arg("eq_con_tol") = 1.0e-6, nb::arg("ineq_con_tol") = 1.0e-6,
            nb::arg("bar_tol") = 1.0e-6);

    BIND_SETTINGS_VALIDATED(obj, "acc_kkt_tol", acc_kkt_tol_, set_acc_kkt_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "acc_bar_tol", acc_bar_tol_, set_acc_bar_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "acc_eq_con_tol", acc_econ_tol_, set_acc_econ_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "acc_ineq_con_tol", acc_icon_tol_, set_acc_icon_tol, "");

    obj.def("set_acc_kkt_tol", &PSIOPT::set_acc_kkt_tol);
    obj.def("set_acc_bar_tol", &PSIOPT::set_acc_bar_tol);
    obj.def("set_acc_eq_con_tol", &PSIOPT::set_acc_econ_tol);
    obj.def("set_acc_ineq_con_tol", &PSIOPT::set_acc_icon_tol);

    obj.def("set_acc_tols", &PSIOPT::set_acc_tols, nb::arg("acc_kkt_tol") = 1.0e-2,
            nb::arg("acc_eq_con_tol") = 1.0e-3, nb::arg("acc_ineq_con_tol") = 1.0e-3,
            nb::arg("acc_bar_tol") = 1.0e-3);

    BIND_SETTINGS_VALIDATED(obj, "div_kkt_tol", div_kkt_tol_, set_div_kkt_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "div_bar_tol", div_bar_tol_, set_div_bar_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "div_eq_con_tol", div_econ_tol_, set_div_econ_tol, "");
    BIND_SETTINGS_VALIDATED(obj, "div_ineq_con_tol", div_icon_tol_, set_div_icon_tol, "");

    obj.def("set_div_kkt_tol", &PSIOPT::set_div_kkt_tol);
    obj.def("set_div_bar_tol", &PSIOPT::set_div_bar_tol);
    obj.def("set_div_eq_con_tol", &PSIOPT::set_div_econ_tol);
    obj.def("set_div_ineq_con_tol", &PSIOPT::set_div_icon_tol);

    BIND_SETTINGS_VALIDATED(obj, "neg_slack_reset", neg_slack_reset_, set_neg_slack_reset, "");

    BIND_SETTINGS_VALIDATED(obj, "bound_fraction", bound_fraction_, set_bound_fraction, "");
    obj.def("set_bound_fraction", &PSIOPT::set_bound_fraction);

    BIND_SETTINGS_VALIDATED(obj, "bound_push", bound_push_, set_bound_push, "");

    // --- Hessian perturbation ---
    BIND_SETTINGS_VALIDATED(obj, "delta_h", delta_h_, set_delta_h, "");
    BIND_SETTINGS_VALIDATED(obj, "incr_h", incr_h_, set_incr_h, "");
    BIND_SETTINGS_VALIDATED(obj, "decr_h", decr_h_, set_decr_h, "");

    obj.def("set_delta_h", &PSIOPT::set_delta_h);
    obj.def("set_incr_h", &PSIOPT::set_incr_h);
    obj.def("set_decr_h", &PSIOPT::set_decr_h);

    obj.def("set_hpert_params", &PSIOPT::set_hpert_params, nb::arg("delta_h"), nb::arg("incr_h"),
            nb::arg("decr_h"));

    BIND_SETTINGS_RW(
        obj, "inertia_mode", inertia_mode_,
        "KKT inertia-correction / regularization mode: classic (default) reproduces the "
        "on-demand inertia ladder bit-for-bit -- each iteration first attempts an unperturbed "
        "factorization and only shifts the Hessian diagonal (by increasing amounts) when the "
        "factorization reports wrong inertia, with no constraint-block shift. "
        "proximal_regularization bakes two shifts into the base matrix every iteration instead "
        "of the classic zero-perturbation first attempt: a small persistent, decaying primal "
        "base shift on the Hessian diagonal, and an always-on barrier-scaled dual shift on the "
        "constraint-row diagonals (suppressed while a nested l1 restoration phase is active, "
        "since the elastic pivots already regularize those rows). The same incr_h/decr_h "
        "escalation ladder still fires on top when the base attempt has wrong inertia or is "
        "singular; a singular base attempt is treated as wrong inertia under this mode (classic "
        "only warns and proceeds). See InertiaModes for the full mechanism and literature "
        "citations, and last_prox_reg_primal/last_prox_reg_dual for the resulting diagnostics.");

    // --- Barrier parameters ---
    BIND_SETTINGS_VALIDATED(obj, "init_mu", init_mu_, set_init_mu, "");
    BIND_SETTINGS_VALIDATED(obj, "min_mu", min_mu_, set_min_mu, "");
    BIND_SETTINGS_VALIDATED(obj, "max_mu", max_mu_, set_max_mu, "");

    // --- Algorithm modes ---
    BIND_SETTINGS_RW(obj, "pd_step_strategy", pd_step_strategy_, "");
    BIND_SETTINGS_VALIDATED(obj, "qp_par_solve", qp_par_solve_, set_qp_par_solve, "");

    BIND_SETTINGS_RW(obj, "soe_mode", soe_mode_, "");

    BIND_SETTINGS_RW(obj, "opt_bar_mode", opt_bar_mode_, "");
    BIND_SETTINGS_RW(obj, "soe_bar_mode", soe_bar_mode_, "");

    obj.def("set_opt_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_opt_bar_mode));
    obj.def("set_opt_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_opt_bar_mode));
    obj.def("set_soe_bar_mode", nb::overload_cast<BarrierModes>(&PSIOPT::set_soe_bar_mode));
    obj.def("set_soe_bar_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_soe_bar_mode));

    // --- Line search modes ---
    BIND_SETTINGS_RW(obj, "opt_ls_mode", opt_ls_mode_, "");
    BIND_SETTINGS_RW(obj, "soe_ls_mode", soe_ls_mode_, "");

    obj.def("set_opt_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_opt_ls_mode));
    obj.def("set_opt_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_opt_ls_mode));
    obj.def("set_soe_ls_mode", nb::overload_cast<LineSearchModes>(&PSIOPT::set_soe_ls_mode));
    obj.def("set_soe_ls_mode", nb::overload_cast<const std::string &>(&PSIOPT::set_soe_ls_mode));

    // --- Step-acceptance / recovery strategy ---
    BIND_SETTINGS_RW(
        obj, "acceptance_strategy", acceptance_strategy_,
        "Step-acceptance strategy: classic_merit (default) reproduces the original fused "
        "backtracking merit line search bit-for-bit; merit switches to the modernized "
        "penalty-based acceptance test selected by merit_penalty_rule; funnel switches to a "
        "single-scalar bound on constraint violation, tightened while accepted iterates stay "
        "within it; filter switches to a (violation, objective) Wachter-Biegler-style filter. "
        "funnel and filter are designed to operate above a monotone barrier safeguard, so each "
        "requires either barrier_governor=monitored or never_monotone=True (ValueError at "
        "validate() time otherwise); watchdog is compatible with all four strategies. These are "
        "heuristically-motivated acceptance alternatives, not one another's strict "
        "improvement -- compare against classic_merit on your own problem before adopting "
        "one.");
    BIND_SETTINGS_RW(
        obj, "merit_penalty_rule", merit_penalty_rule_,
        "Penalty-parameter update rule for the modernized merit strategy; only read when "
        "acceptance_strategy is merit. wmno (default) updates a single penalty value from "
        "the directional-derivative condition; flexible tracks a penalty interval and "
        "accepts a step that improves the merit for at least one value in that interval.");
    BIND_SETTINGS_RW(
        obj, "watchdog", watchdog_,
        "Enables the watchdog recovery heuristic, which tolerates a temporarily worse step "
        "after repeated rejections instead of immediately shrinking the step further. false "
        "(default) preserves the original behavior.");
    BIND_SETTINGS_RW(
        obj, "barrier_governor", barrier_governor_,
        "Barrier-parameter governor: classic_adaptive (default) reproduces the original "
        "PROBE/LOQO free-mode barrier update bit-for-bit; monitored composes a "
        "classic_adaptive delegate with a KKT-error monitor that watches a sliding reference "
        "window of recent iterations and, when free-mode progress is no longer a sufficient "
        "decrease relative to that window, hands off to a monotone (Fiacco-McCormick) mode "
        "with the barrier parameter initialized to 0.8 times the average complementarity and "
        "held fixed until the barrier subproblem converges, then decreased; the monitor "
        "re-enters free mode once progress against the (frozen) reference window resumes. "
        "Each free->monotone handoff and each monotone barrier-parameter decrease resets the "
        "acceptance strategy's per-barrier-subproblem state — the filter set is cleared and "
        "the violation thresholds (and funnel width) are RE-DERIVED from the current "
        "iterate's violation on the next acceptance test, exactly as a new barrier "
        "subproblem re-bases them in Ipopt. The funnel/filter "
        "acceptance strategies are designed to operate above a monotone barrier safeguard, "
        "which classic_adaptive does not provide; validate() raises ValueError if they are "
        "combined with classic_adaptive unless never_monotone is set. Any acceptance_strategy "
        "may pair with monitored.");
    BIND_SETTINGS_RW(
        obj, "never_monotone", never_monotone_,
        "Expert escape hatch, mirroring Ipopt's never-monotone-mode: explicitly accepts "
        "running funnel/filter above barrier_governor=classic_adaptive without its monotone "
        "safeguard, forfeiting that guard rather than switching to barrier_governor=monitored. "
        "false (default). Contradictory with barrier_governor=monitored (which already "
        "supplies the monotone fallback this knob opts out of) -- validate() raises ValueError "
        "on that combination.");
    BIND_SETTINGS_RW(
        obj, "restoration_mode", restoration_mode_,
        "Feasibility-restoration mode selector: off (default) reproduces today's behavior "
        "bit-identically -- no restoration strategy is constructed, so every restoration branch "
        "in the solver is provably dead. proximal_switch enables the proximal feasibility "
        "mode-switch: when the recovery chain exhausts on a rejected step, the solver switches "
        "to a pure feasibility phase -- the objective is replaced by a proximal term centered on "
        "the switch point (coefficient sqrt(mu) at entry) while all constraints and barrier "
        "machinery keep running -- and returns to the true objective once the acceptance "
        "strategy's infeasibility-reduction test passes (per-strategy: classic_merit uses a "
        "relative infeasibility-reduction test against the entry point, Ipopt "
        "restoration-convergence style; merit reduces against the smallest-known infeasibility "
        "held from the optimality phase -- frozen at restoration entry and unchanged by "
        "feasibility-phase iterates; funnel/filter use their own reference-solver tests). "
        "l1_nested enables the nested l1 elastic feasibility restoration instead: the same "
        "trigger and the same acceptance-strategy exit test, but the elastic reformulation runs "
        "as a condensed in-place phase reusing the outer barrier algorithm's own KKT system, "
        "rather than swapping the outer objective for a proximal term -- see RestorationModes "
        "for the mechanism and Ipopt-lineage citations. Unlike proximal_switch, l1_nested first "
        "tries a soft feasibility pre-stage (full fraction-to-boundary steps tested under a "
        "primal-dual-error reduction rule) and only escalates to the full elastic switch after "
        "several soft steps in a row fail to recover; proximal_switch has no pre-stage and "
        "switches directly. Both modes refuse entry at a near-feasible point or once the "
        "per-phase budget max_feas_rest is exhausted. Composes with every acceptance_strategy "
        "and barrier_governor (no matrix restrictions -- every shipped acceptance strategy "
        "implements the exit test either mode relies on). Mode-switch lineage: Knitro's "
        "bar_switchobj=scalarprox for proximal_switch, with entry/exit semantics derived from "
        "Ipopt's restoration phase and Uno's phase switching for both modes.");
    BIND_SETTINGS_VALIDATED(
        obj, "max_feas_rest", max_feas_rest_, set_max_feas_rest,
        "Per-phase cap on the number of times feasibility restoration may be entered. 0 "
        "disables restoration entry entirely (the budget is exhausted before the first entry); "
        "2 (default). Ignored when restoration_mode is off. Negative values raise ValueError "
        "immediately on assignment; validate() re-checks non-negativity as a backstop.");

    // --- QP solver ---
    BIND_SETTINGS_RW(obj, "force_qp_analysis", force_qp_analysis_, "");
    BIND_SETTINGS_VALIDATED(obj, "qp_ref_steps", qp_ref_steps_, set_qp_ref_steps, "");
    BIND_SETTINGS_VALIDATED(obj, "qp_pivot_perturb", qp_pivot_perturb_, set_qp_pivot_perturb, "");
    BIND_SETTINGS_VALIDATED(obj, "qp_matching", qp_matching_, set_qp_matching, "");
    BIND_SETTINGS_VALIDATED(obj, "qp_scaling", qp_scaling_, set_qp_scaling, "");
    BIND_SETTINGS_VALIDATED(obj, "qp_threads", qp_threads_, set_qp_threads, "");
    BIND_SETTINGS_RW(obj, "qp_pivot_strategy", qp_pivot_strategy_, "");

    // --- QP ordering ---
    BIND_SETTINGS_RW(obj, "qp_ordering_mode", qp_ord_, "");

    obj.def("set_qp_ordering_mode",
            nb::overload_cast<QPOrderingModes>(&PSIOPT::set_qp_ordering_mode));
    obj.def("set_qp_ordering_mode",
            nb::overload_cast<const std::string &>(&PSIOPT::set_qp_ordering_mode));

#ifdef USE_ACCELERATE_SPARSE
    BIND_SETTINGS_VALIDATED(obj, "accel_pivot_tolerance", accel_pivot_tolerance_,
                            set_accel_pivot_tolerance);
    BIND_SETTINGS_VALIDATED(obj, "accel_zero_tolerance", accel_zero_tolerance_,
                            set_accel_zero_tolerance);
    obj.def("set_accel_pivot_tolerance", &PSIOPT::set_accel_pivot_tolerance);
    obj.def("set_accel_zero_tolerance", &PSIOPT::set_accel_zero_tolerance);
#endif

    // --- Output/result ---
    BIND_SETTINGS_RW(obj, "qp_print", qp_print_);

    BIND_SETTINGS_RW(obj, "return_best", return_best_);
    obj.def_prop_rw(
        "best_criteria", [](const PSIOPT &self) { return self.settings().best_criteria_; },
        [](PSIOPT &self, nb::object val) {
            if (nb::isinstance<BestCriteriaModes>(val))
                self.settings().best_criteria_ = nb::cast<BestCriteriaModes>(val);
            else if (nb::isinstance<nb::str>(val))
                self.settings().best_criteria_ =
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
    nb::enum_<AcceptanceStrategies>(m, "AcceptanceStrategies")
        .value("classic_merit", AcceptanceStrategies::classic_merit)
        .value("merit", AcceptanceStrategies::merit)
        .value("funnel", AcceptanceStrategies::funnel,
               "Single-scalar upper bound on constraint violation (the funnel width), "
               "tightened while accepted iterates remain within it (Kiessling, "
               "Leyffer & Vanaret funnel formulation, implemented after Uno's funnel). "
               "Requires a monotone barrier safeguard, so it rejects combination with the "
               "default barrier_governor=classic_adaptive unless never_monotone=True "
               "(ValueError at validate() time); composes with watchdog. Heuristically motivated -- no "
               "convergence guarantee is implied; compare against classic_merit and filter "
               "on your own problem.")
        .value("filter", AcceptanceStrategies::filter,
               "(Constraint violation, objective) pair filter with margined dominance "
               "(Wachter-Biegler filter line search, Ipopt lineage). Requires a monotone "
               "barrier safeguard, so it rejects combination with the default "
               "barrier_governor=classic_adaptive unless never_monotone=True (ValueError at "
               "validate() time); "
               "composes with watchdog. Heuristically motivated -- no convergence guarantee "
               "is implied; compare against classic_merit and funnel on your own problem.");
    nb::enum_<MeritPenaltyRules>(m, "MeritPenaltyRules")
        .value("wmno", MeritPenaltyRules::wmno)
        .value("flexible", MeritPenaltyRules::flexible);
    nb::enum_<BarrierGovernors>(m, "BarrierGovernors")
        .value("classic_adaptive", BarrierGovernors::classic_adaptive,
               "The classic PROBE/LOQO free-mode barrier update, unchanged -- the "
               "bit-identical default.")
        .value("monitored", BarrierGovernors::monitored,
               "Free<->monotone monitored barrier governor: a KKT-error monitor hands off "
               "to a Fiacco-McCormick monotone mode when free-mode progress stalls, then "
               "re-enters free mode once progress resumes -- see the barrier_governor "
               "property docstring for the full mechanism.");
    nb::enum_<RestorationModes>(m, "RestorationModes")
        .value("off", RestorationModes::off,
               "No feasibility restoration -- the bit-identical default. No restoration "
               "strategy is constructed, so every restoration branch in the solver is provably "
               "dead.")
        .value("proximal_switch", RestorationModes::proximal_switch,
               "Proximal feasibility mode-switch: on a ladder-exhausted step rejection, keep "
               "the same barrier algorithm running but swap the true objective for a proximal "
               "term pulling the primals back toward the switch point, until the acceptance "
               "strategy's infeasibility-reduction test passes -- see the restoration_mode "
               "property docstring for the full mechanism. Composes with every "
               "acceptance_strategy and barrier_governor.")
        .value("l1_nested", RestorationModes::l1_nested,
               "Nested l1 elastic feasibility restoration: on a ladder-exhausted step "
               "rejection, solve the l1 elastic reformulation of the current KKT system as a "
               "condensed in-place phase -- each row gets a pair of nonnegative elastic slacks "
               "(n, p) absorbing the residual, penalized at rho=1e3 plus a proximity term "
               "pulling the primals back toward the switch point with weight sqrt(mu) -- rather "
               "than switching the outer objective the way proximal_switch does; the phase "
               "reuses the outer barrier algorithm's own KKT system instead of spinning up a "
               "separate nested solver. Constants (the penalty rho, the proximity weight "
               "factor, the entry/re-entry rules) are pinned at Ipopt's restoration-phase "
               "literature defaults (coin-or/Ipopt's IpRestoIpoptNLP / "
               "IpRestoIterateInitializer / IpRestoMinC_1Nrm). Before committing to the full "
               "elastic switch, a soft feasibility pre-stage first tries ordinary "
               "fraction-to-boundary steps under a primal-dual-error reduction rule for a "
               "bounded number of consecutive iterations (adapted from Ipopt's soft "
               "restoration phase) and only escalates once that budget is exhausted; "
               "proximal_switch has no such pre-stage. Prefer l1_nested over proximal_switch "
               "when a stall is a genuinely constraint-infeasible point the elastic "
               "reformulation can relax productively (the pre-stage also gives it a cheaper "
               "recovery attempt before the full switch); prefer proximal_switch for a simpler, "
               "cheaper mode-switch with no elastic-slack bookkeeping. Returns to the true "
               "objective on the same acceptance-strategy infeasibility-reduction test "
               "proximal_switch uses -- see the restoration_mode property docstring. Composes "
               "with every acceptance_strategy and barrier_governor; the diagnostics "
               "last_feas_rest_entries/last_feas_rest_iters count identically for both "
               "modes.");
    nb::enum_<InertiaModes>(m, "InertiaModes")
        .value("classic", InertiaModes::classic,
               "The on-demand inertia ladder inline in factor_impl -- the bit-identical "
               "default. Each iteration first attempts an unperturbed factorization and only "
               "shifts the Hessian diagonal (by increasing amounts) when the factorization "
               "reports wrong inertia. No constraint-block shift.")
        .value("proximal_regularization", InertiaModes::proximal_regularization,
               "Proximal primal-dual regularization: a small persistent, decaying primal base "
               "shift (rho_k, floored at 1e-10, the Cipolla-Gondzio floor) on the Hessian "
               "diagonal, plus an always-on barrier-scaled dual shift (delta_c = 1e-8 * "
               "mu^0.25, Ipopt's jacobian_regularization_value/exponent constants, matching its "
               "perturb_always_cd semantics) on the constraint-row diagonals, are baked into "
               "the base matrix every iteration in place of the classic zero-perturbation first "
               "attempt -- the same escalation ladder still fires on top when the base attempt "
               "has wrong inertia or is singular (a singular base attempt is itself treated as "
               "wrong inertia under this mode, unlike classic's warn-and-proceed). rho_k decays "
               "toward its floor by decr_h each iteration the base attempt sufficed, or "
               "persists at the decayed total shift (rho_k plus the ladder's last delta) when "
               "the ladder fired. The dual shift is suppressed while a nested l1 restoration "
               "phase is active -- the elastic pivots already regularize those constraint rows "
               "at a magnitude the dual shift would be negligible against, and stacking it would "
               "make the elastic step-recovery algebra inconsistent with the solved system; the "
               "proximal mode-switch restoration touches only the primal diagonal, so the dual "
               "shift stays on under it. No new tunable constants -- rho_k's floor and the dual "
               "shift's scale/exponent are fixed. See last_prox_reg_primal/last_prox_reg_dual "
               "for the per-solve diagnostics this mode reports.");

    nb::class_<IpoptRunInfo>(m, "IpoptRunInfo")
        .def_ro("ran", &IpoptRunInfo::ran_)
        .def_ro("status", &IpoptRunInfo::status_)
        .def_ro("normalized", &IpoptRunInfo::normalized_)
        .def_ro("converge_flag", &IpoptRunInfo::converge_flag_)
        .def_ro("iterations", &IpoptRunInfo::iterations_)
        .def_ro("objective", &IpoptRunInfo::objective_)
        .def_ro("constraint_violation", &IpoptRunInfo::constraint_violation_)
        .def_ro("wall_time_s", &IpoptRunInfo::wall_time_s_);

    nb::enum_<NLPSolvers>(m, "NLPSolvers",
                          "NLP solver backend selector for the solve/optimize entry points.")
        .value("psiopt", NLPSolvers::psiopt, "Built-in interior-point solver (default).")
        .value("ipopt", NLPSolvers::ipopt,
               "Linked Ipopt on the identical transcribed NLP (requires ENABLE_IPOPT build).");

    m.def("ipopt_available", &ipopt_backend::available,
          "True when this build was configured with ENABLE_IPOPT.");

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
