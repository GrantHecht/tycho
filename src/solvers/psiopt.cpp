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
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - Configuration fields grouped into Settings struct
//   - Phase dispatch refactored into run_phase_sequence
//   - Line search methods extracted (ls_lang, ls_l1, ls_auglang)
//   - Validated setter methods added
//   - Printing extracted to psiopt_print.cpp
// =============================================================================

#include "tycho/detail/solvers/psiopt.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "tycho/detail/solvers/solver_init.h"
#include "tycho/detail/utils/timer.h"

// Globalization component interfaces. Included here (rather than from
// psiopt.h) so this, the actual TU that builds PSIOPT, exercises them on
// every build without psiopt.h having to include a directory of headers
// that themselves need the complete PSIOPT class (a circular-include
// arrangement that is fragile for the "middle" headers below — see the
// include-discipline note in solver_context.h). Dependency-ordered.
#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/backtracking_line_search.h"
#include "tycho/detail/solvers/globalization/barrier_governor.h"
#include "tycho/detail/solvers/globalization/classic_adaptive_governor.h"
#include "tycho/detail/solvers/globalization/monitored_governor.h"
#include "tycho/detail/solvers/globalization/merit_acceptance.h"
#include "tycho/detail/solvers/globalization/modern_merit.h"
#include "tycho/detail/solvers/globalization/funnel_acceptance.h"
#include "tycho/detail/solvers/globalization/filter_acceptance.h"
#include "tycho/detail/solvers/globalization/inertia_regularization.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/noop_recovery.h"
#include "tycho/detail/solvers/globalization/soc.h"
#include "tycho/detail/solvers/globalization/watchdog.h"
#include "tycho/detail/solvers/globalization/restoration.h"
#include "tycho/detail/solvers/globalization/proximal_restoration.h"
#include "tycho/detail/solvers/globalization/l1_restoration.h"
#include "tycho/detail/solvers/globalization/feasibility_stall.h"
#include "tycho/detail/solvers/globalization/feasibility_switch_recovery.h"

#ifndef USE_ACCELERATE_SPARSE
#include <mkl.h>
#endif

namespace {

// Name of a non-classic acceptance strategy, for the validate() error message
// below. classic_merit never reaches this helper (the guard that calls it is
// gated on acceptance_strategy_ != classic_merit).
const char *acceptance_strategy_name(tycho::solvers::AcceptanceStrategies strategy) {
    using tycho::solvers::AcceptanceStrategies;
    switch (strategy) {
    case AcceptanceStrategies::merit:
        return "merit";
    case AcceptanceStrategies::funnel:
        return "funnel";
    case AcceptanceStrategies::filter:
        return "filter";
    case AcceptanceStrategies::classic_merit:
        return "classic_merit";
    }
    return "unknown";
}

// Per-iterate acceptable tier: all four monitored residuals strictly inside
// their acceptable tolerances. This is the single definition of "this iterate
// is acceptable" -- converge_check() applies it over a trailing window of
// max_acc_iters_ iterates to declare ConvergenceFlags::ACCEPTABLE, and
// alg_impl's un-evaluable-step bypass applies it to the current iterate to
// decide whether a failed line search can exit at the acceptable level instead
// of aborting. Both call sites must agree, so neither open-codes the four
// comparisons.
bool psiopt_iterate_acceptable(const tycho::solvers::IterateInfo &it,
                               const tycho::solvers::PSIOPT::Settings &settings) {
    return (it.kkt_inf_ < settings.acc_kkt_tol_) && (it.econ_inf_ < settings.acc_econ_tol_) &&
           (it.icon_inf_ < settings.acc_icon_tol_) && (it.barr_inf_ < settings.acc_bar_tol_);
}

} // namespace

// =============================================================================
// Static string-to-enum converters
// =============================================================================

auto tycho::solvers::PSIOPT::strto_OrderingMode(const std::string &str) -> QPOrderingModes {
    if (str == "MINDEG")
        return QPOrderingModes::MINDEG;
    else if (str == "METIS")
        return QPOrderingModes::METIS;
    else if (str == "PARMETIS" || str == "MTMETIS")
        return QPOrderingModes::PARMETIS;
    else {
        throw std::invalid_argument(
            fmt::format("Unrecognized QPOrderingMode: {0}\n"
                        "Valid Options Are: MINDEG, METIS, PARMETIS (alias: MTMETIS)",
                        str));
    }
}

auto tycho::solvers::PSIOPT::strto_LineSearchMode(const std::string &str) -> LineSearchModes {
    if (str == "L1")
        return LineSearchModes::L1;
    else if (str == "NOLS")
        return LineSearchModes::NOLS;
    else if (str == "LANG")
        return LineSearchModes::LANG;
    else if (str == "AUGLANG")
        return LineSearchModes::AUGLANG;
    else {
        throw std::invalid_argument(
            fmt::format("Unrecognized LineSearchMode: {0}\n"
                        "Valid Options Are: AUGLANG, LANG, L1, NOLS\n"
                        "Note: L2 was removed. Use L1, LANG, AUGLANG, or NOLS.",
                        str));
    }
}

auto tycho::solvers::PSIOPT::strto_BarrierMode(const std::string &str) -> BarrierModes {
    if (str == "PROBE")
        return BarrierModes::PROBE;
    else if (str == "LOQO")
        return BarrierModes::LOQO;
    else {
        throw std::invalid_argument(
            fmt::format("Unrecognized BarrierMode: {0}\n"
                        "Valid Options Are: LOQO, PROBE\n"
                        "Note: FIACCO and BARDISABLED were removed. Use LOQO or PROBE.",
                        str));
    }
}

auto tycho::solvers::PSIOPT::strto_BestCriteriaMode(const std::string &str) -> BestCriteriaModes {
    if (str == "ECons" || str == "ECon")
        return BestCriteriaModes::ECONS;
    else if (str == "ICons" || str == "ICon")
        return BestCriteriaModes::ICONS;
    else if (str == "KKT")
        return BestCriteriaModes::KKT;
    else if (str == "Obj" || str == "Prim Obj")
        return BestCriteriaModes::OBJ;
    else {
        throw std::invalid_argument(fmt::format("Unrecognized BestCriteriaMode: {0}", str));
    }
}

// =============================================================================
// Validated setter methods
// =============================================================================

void tycho::solvers::PSIOPT::set_max_iters(int max_iters) {
    if (max_iters < 1) {
        throw std::invalid_argument("max_iters must be greater than 0.");
    }
    settings_.max_iters_ = max_iters;
}

void tycho::solvers::PSIOPT::set_max_acc_iters(int max_acc_iters) {
    if (max_acc_iters < 1) {
        throw std::invalid_argument("max_acc_iters must be greater than 0.");
    }
    settings_.max_acc_iters_ = max_acc_iters;
}

void tycho::solvers::PSIOPT::set_max_ls_iters(int max_ls_iters) {
    if (max_ls_iters < 0) {
        throw std::invalid_argument("max_ls_iters must be non-negative (>= 0).");
    }
    settings_.max_ls_iters_ = max_ls_iters;
}

void tycho::solvers::PSIOPT::set_all_max_iters(int m1, int m2) {
    set_max_iters(m1);
    set_max_acc_iters(m2);
}

void tycho::solvers::PSIOPT::set_max_soc(int max_soc) {
    if (max_soc < 0)
        throw std::invalid_argument(fmt::format("max_soc must be non-negative, got {}", max_soc));
    settings_.max_soc_ = max_soc;
}

void tycho::solvers::PSIOPT::set_ls_extended_iters(int ls_extended_iters) {
    if (ls_extended_iters < 0)
        throw std::invalid_argument(
            fmt::format("ls_extended_iters must be non-negative, got {}", ls_extended_iters));
    settings_.ls_extended_iters_ = ls_extended_iters;
}

void tycho::solvers::PSIOPT::set_max_feas_rest(int max_feas_rest) {
    if (max_feas_rest < 0)
        throw std::invalid_argument(
            fmt::format("max_feas_rest must be non-negative, got {}", max_feas_rest));
    settings_.max_feas_rest_ = max_feas_rest;
}

void tycho::solvers::PSIOPT::set_kkt_tol(double kkt_tol) {
    if (!std::isfinite(kkt_tol) || kkt_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("kkt_tol must be finite and positive, got {}", kkt_tol));
    settings_.kkt_tol_ = kkt_tol;
}

void tycho::solvers::PSIOPT::set_bar_tol(double bar_tol) {
    if (!std::isfinite(bar_tol) || bar_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("bar_tol must be finite and positive, got {}", bar_tol));
    settings_.bar_tol_ = bar_tol;
}

void tycho::solvers::PSIOPT::set_econ_tol(double econ_tol) {
    if (!std::isfinite(econ_tol) || econ_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("econ_tol must be finite and positive, got {}", econ_tol));
    settings_.econ_tol_ = econ_tol;
}

void tycho::solvers::PSIOPT::set_icon_tol(double icon_tol) {
    if (!std::isfinite(icon_tol) || icon_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("icon_tol must be finite and positive, got {}", icon_tol));
    settings_.icon_tol_ = icon_tol;
}

void tycho::solvers::PSIOPT::set_tols(double kkt_tol, double econ_tol, double icon_tol,
                                      double bar_tol) {
    this->set_kkt_tol(kkt_tol);
    this->set_econ_tol(econ_tol);
    this->set_icon_tol(icon_tol);
    this->set_bar_tol(bar_tol);
}

void tycho::solvers::PSIOPT::set_acc_kkt_tol(double acc_kkt_tol) {
    if (!std::isfinite(acc_kkt_tol) || acc_kkt_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("acc_kkt_tol must be finite and positive, got {}", acc_kkt_tol));
    settings_.acc_kkt_tol_ = acc_kkt_tol;
}

void tycho::solvers::PSIOPT::set_acc_bar_tol(double acc_bar_tol) {
    if (!std::isfinite(acc_bar_tol) || acc_bar_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("acc_bar_tol must be finite and positive, got {}", acc_bar_tol));
    settings_.acc_bar_tol_ = acc_bar_tol;
}

void tycho::solvers::PSIOPT::set_acc_econ_tol(double acc_econ_tol) {
    if (!std::isfinite(acc_econ_tol) || acc_econ_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("acc_econ_tol must be finite and positive, got {}", acc_econ_tol));
    settings_.acc_econ_tol_ = acc_econ_tol;
}

void tycho::solvers::PSIOPT::set_acc_icon_tol(double acc_icon_tol) {
    if (!std::isfinite(acc_icon_tol) || acc_icon_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("acc_icon_tol must be finite and positive, got {}", acc_icon_tol));
    settings_.acc_icon_tol_ = acc_icon_tol;
}

void tycho::solvers::PSIOPT::set_acc_tols(double acc_kkt_tol, double acc_econ_tol,
                                          double acc_icon_tol, double acc_bar_tol) {
    this->set_acc_kkt_tol(acc_kkt_tol);
    this->set_acc_econ_tol(acc_econ_tol);
    this->set_acc_icon_tol(acc_icon_tol);
    this->set_acc_bar_tol(acc_bar_tol);
}

void tycho::solvers::PSIOPT::set_unacc_tols(double kktol, double etol, double itol, double bartol) {
    auto validate = [](double v, const char *name) {
        if (!std::isfinite(v) || v <= 0.0)
            throw std::invalid_argument(
                fmt::format("{} must be finite and positive, got {}", name, v));
    };
    validate(kktol, "unacc_kkt_tol");
    validate(etol, "unacc_econ_tol");
    validate(itol, "unacc_icon_tol");
    validate(bartol, "unacc_bar_tol");
    settings_.unacc_kkt_tol_ = kktol;
    settings_.unacc_bar_tol_ = bartol;
    settings_.unacc_econ_tol_ = etol;
    settings_.unacc_icon_tol_ = itol;
}

void tycho::solvers::PSIOPT::set_div_kkt_tol(double div_kkt_tol) {
    if (!std::isfinite(div_kkt_tol) || div_kkt_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("div_kkt_tol must be finite and positive, got {}", div_kkt_tol));
    settings_.div_kkt_tol_ = div_kkt_tol;
}

void tycho::solvers::PSIOPT::set_div_bar_tol(double div_bar_tol) {
    if (!std::isfinite(div_bar_tol) || div_bar_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("div_bar_tol must be finite and positive, got {}", div_bar_tol));
    settings_.div_bar_tol_ = div_bar_tol;
}

void tycho::solvers::PSIOPT::set_div_econ_tol(double div_econ_tol) {
    if (!std::isfinite(div_econ_tol) || div_econ_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("div_econ_tol must be finite and positive, got {}", div_econ_tol));
    settings_.div_econ_tol_ = div_econ_tol;
}

void tycho::solvers::PSIOPT::set_div_icon_tol(double div_icon_tol) {
    if (!std::isfinite(div_icon_tol) || div_icon_tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("div_icon_tol must be finite and positive, got {}", div_icon_tol));
    settings_.div_icon_tol_ = div_icon_tol;
}

void tycho::solvers::PSIOPT::set_div_tols(double div_kkt_tol, double div_econ_tol,
                                          double div_icon_tol, double div_bar_tol) {
    this->set_div_kkt_tol(div_kkt_tol);
    this->set_div_econ_tol(div_econ_tol);
    this->set_div_icon_tol(div_icon_tol);
    this->set_div_bar_tol(div_bar_tol);
}

void tycho::solvers::PSIOPT::set_bound_fraction(double bound_fraction) {
    if (bound_fraction >= 1.0 || bound_fraction <= 0.0) {
        throw std::invalid_argument("bound_fraction must be between 0 and 1.");
    }
    settings_.bound_fraction_ = bound_fraction;
}

void tycho::solvers::PSIOPT::set_bound_push(double bound_push) {
    if (bound_push <= 0.0) {
        throw std::invalid_argument("bound_push must be greater than 0.");
    }
    settings_.bound_push_ = bound_push;
}

void tycho::solvers::PSIOPT::set_alpha_red(double ared) {
    if (ared <= 1.0) {
        throw std::invalid_argument("alpha_red must be greater than 1.0");
    }
    settings_.alpha_red_ = ared;
}

void tycho::solvers::PSIOPT::set_delta_h(double delta_h) {
    if (delta_h <= 0.0) {
        throw std::invalid_argument("delta_h must be greater than 0.");
    }
    settings_.delta_h_ = delta_h;
}

void tycho::solvers::PSIOPT::set_incr_h(double incr_h) {
    if (incr_h <= 1.0) {
        throw std::invalid_argument("incr_h must be greater than 1.0.");
    }
    settings_.incr_h_ = incr_h;
}

void tycho::solvers::PSIOPT::set_decr_h(double decr_h) {
    if (decr_h >= 1.0 || decr_h <= 0) {
        throw std::invalid_argument("decr_h must be between 0 and 1.");
    }
    settings_.decr_h_ = decr_h;
}

void tycho::solvers::PSIOPT::set_hpert_params(double delta_h, double incr_h, double decr_h) {
    this->set_delta_h(delta_h);
    this->set_incr_h(incr_h);
    this->set_decr_h(decr_h);
}

void tycho::solvers::PSIOPT::set_print_level(int plevel) {
    if (plevel < 0)
        throw std::invalid_argument(
            fmt::format("print_level must be non-negative, got {}", plevel));
    settings_.print_level_ = plevel;
}

void tycho::solvers::PSIOPT::set_qp_ordering_mode(QPOrderingModes mode) {
    settings_.qp_ord_ = mode;
}

void tycho::solvers::PSIOPT::set_qp_ordering_mode(const std::string &str) {
    settings_.qp_ord_ = strto_OrderingMode(str);
}

void tycho::solvers::PSIOPT::set_opt_bar_mode(BarrierModes mode) { settings_.opt_bar_mode_ = mode; }

void tycho::solvers::PSIOPT::set_opt_bar_mode(const std::string &str) {
    settings_.opt_bar_mode_ = strto_BarrierMode(str);
}

void tycho::solvers::PSIOPT::set_soe_bar_mode(BarrierModes mode) { settings_.soe_bar_mode_ = mode; }

void tycho::solvers::PSIOPT::set_soe_bar_mode(const std::string &str) {
    settings_.soe_bar_mode_ = strto_BarrierMode(str);
}

void tycho::solvers::PSIOPT::set_opt_ls_mode(LineSearchModes mode) {
    settings_.opt_ls_mode_ = mode;
}

void tycho::solvers::PSIOPT::set_opt_ls_mode(const std::string &str) {
    settings_.opt_ls_mode_ = strto_LineSearchMode(str);
}

void tycho::solvers::PSIOPT::set_soe_ls_mode(LineSearchModes mode) {
    settings_.soe_ls_mode_ = mode;
}

void tycho::solvers::PSIOPT::set_soe_ls_mode(const std::string &str) {
    settings_.soe_ls_mode_ = strto_LineSearchMode(str);
}

void tycho::solvers::PSIOPT::set_best_criteria(BestCriteriaModes mode) {
    settings_.best_criteria_ = mode;
}

void tycho::solvers::PSIOPT::set_best_criteria(const std::string &str) {
    settings_.best_criteria_ = strto_BestCriteriaMode(str);
}

#ifdef USE_ACCELERATE_SPARSE
void tycho::solvers::PSIOPT::set_accel_pivot_tolerance(double tol) {
    if (!std::isfinite(tol) || tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("accel_pivot_tolerance must be finite and positive, got {}", tol));
    settings_.accel_pivot_tolerance_ = tol;
}

void tycho::solvers::PSIOPT::set_accel_zero_tolerance(double tol) {
    if (!std::isfinite(tol) || tol <= 0.0)
        throw std::invalid_argument(
            fmt::format("accel_zero_tolerance must be finite and positive, got {}", tol));
    settings_.accel_zero_tolerance_ = tol;
}
#endif

void tycho::solvers::PSIOPT::set_init_mu(double mu) {
    if (!std::isfinite(mu) || mu <= 0.0)
        throw std::invalid_argument(fmt::format("init_mu must be finite and positive, got {}", mu));
    settings_.init_mu_ = mu;
}

void tycho::solvers::PSIOPT::set_min_mu(double mu) {
    if (!std::isfinite(mu) || mu <= 0.0)
        throw std::invalid_argument(fmt::format("min_mu must be finite and positive, got {}", mu));
    settings_.min_mu_ = mu;
}

void tycho::solvers::PSIOPT::set_max_mu(double mu) {
    if (!std::isfinite(mu) || mu <= 0.0)
        throw std::invalid_argument(fmt::format("max_mu must be finite and positive, got {}", mu));
    settings_.max_mu_ = mu;
}

void tycho::solvers::PSIOPT::set_neg_slack_reset(double val) {
    if (!std::isfinite(val) || val <= 0.0)
        throw std::invalid_argument(
            fmt::format("neg_slack_reset must be finite and positive, got {}", val));
    settings_.neg_slack_reset_ = val;
}

void tycho::solvers::PSIOPT::set_qp_threads(int n) {
    if (n < 1)
        throw std::invalid_argument(fmt::format("qp_threads must be >= 1, got {}", n));
    settings_.qp_threads_ = n;
}

void tycho::solvers::PSIOPT::set_qp_pivot_perturb(int v) {
    if (v < 0)
        throw std::invalid_argument(
            fmt::format("qp_pivot_perturb must be non-negative, got {}", v));
    settings_.qp_pivot_perturb_ = v;
}

void tycho::solvers::PSIOPT::set_qp_ref_steps(int v) {
    if (v < 0)
        throw std::invalid_argument(fmt::format("qp_ref_steps must be non-negative, got {}", v));
    settings_.qp_ref_steps_ = v;
}

void tycho::solvers::PSIOPT::set_qp_par_solve(int v) {
    if (v != 0 && v != 1)
        throw std::invalid_argument(fmt::format("qp_par_solve must be 0 or 1, got {}", v));
    settings_.qp_par_solve_ = v;
}

void tycho::solvers::PSIOPT::set_qp_matching(int v) {
    if (v != 0 && v != 1)
        throw std::invalid_argument(fmt::format("qp_matching must be 0 or 1, got {}", v));
    settings_.qp_matching_ = v;
}

void tycho::solvers::PSIOPT::set_qp_scaling(int v) {
    if (v != 0 && v != 1)
        throw std::invalid_argument(fmt::format("qp_scaling must be 0 or 1, got {}", v));
    settings_.qp_scaling_ = v;
}

void tycho::solvers::PSIOPT::set_obj_scale(double scale) {
    if (!std::isfinite(scale) || scale == 0.0)
        throw std::invalid_argument(
            fmt::format("obj_scale must be finite and non-zero, got {}", scale));
    settings_.obj_scale_ = scale;
}

// =============================================================================
// Settings validation
// =============================================================================

void tycho::solvers::PSIOPT::Settings::validate() const {
    auto pos_finite = [](double v, const char *name) {
        if (!std::isfinite(v) || v <= 0.0)
            throw std::invalid_argument(
                fmt::format("{} must be finite and positive, got {}", name, v));
    };
    auto pos_int = [](int v, const char *name) {
        if (v < 1)
            throw std::invalid_argument(fmt::format("{} must be >= 1, got {}", name, v));
    };

    // --- Iteration limits ---
    pos_int(max_iters_, "max_iters");
    pos_int(max_acc_iters_, "max_acc_iters");
    pos_int(max_refac_, "max_refac");
    if (max_ls_iters_ < 0)
        throw std::invalid_argument(
            fmt::format("max_ls_iters must be non-negative, got {}", max_ls_iters_));
    if (max_soc_ < 0)
        throw std::invalid_argument(
            fmt::format("max_soc must be non-negative, got {}", max_soc_));
    if (ls_extended_iters_ < 0)
        throw std::invalid_argument(fmt::format(
            "ls_extended_iters must be non-negative, got {}", ls_extended_iters_));
    // Per-phase feasibility-restoration entry budget. 0 is valid (disables
    // restoration entry even when restoration_mode_ != off); negative is not.
    if (max_feas_rest_ < 0)
        throw std::invalid_argument(
            fmt::format("max_feas_rest must be non-negative, got {}", max_feas_rest_));

    // --- Strategy-combination guards ---
    // The SOC and extended-backtracking recovery links re-drive the acceptance
    // backtrack through the mechanism (GlobalizationMechanism::
    // run_acceptance_backtrack), which dispatches to the classic merit test or
    // to the generic AcceptanceStrategy::is_iterate_acceptable surface
    // (merit / funnel / filter) as appropriate — so a corrected or extended
    // step is tested against the SAME acceptance criteria the ordinary step
    // faced. Both links therefore compose with every acceptance strategy; there
    // is no classic-merit-only restriction. (The watchdog was always compatible
    // with every strategy.)

    // funnel/filter are designed to operate above a monotone barrier safeguard
    // (this is a factual dependency statement, not a convergence-guarantee
    // claim — neither strategy is proven to converge with or without it).
    // classic_adaptive (the default governor) is free-mode only, so pairing it
    // with funnel/filter silently drops that safeguard unless the user
    // explicitly opts in via never_monotone. classic_merit/merit are
    // unaffected by this guard in every combination (bit-identity for the
    // default path; merit + monitored is allowed opt-in, same as merit +
    // classic_adaptive).
    if ((acceptance_strategy_ == AcceptanceStrategies::funnel ||
         acceptance_strategy_ == AcceptanceStrategies::filter) &&
        barrier_governor_ == BarrierGovernors::classic_adaptive && !never_monotone_)
        throw std::invalid_argument(fmt::format(
            "acceptance_strategy={} is designed to operate above the monotone barrier "
            "safeguard, which barrier_governor=classic_adaptive (the default) does not "
            "provide: set barrier_governor=monitored, or set never_monotone=True to "
            "explicitly accept adaptive-only operation",
            acceptance_strategy_name(acceptance_strategy_)));

    // never_monotone is an expert escape that explicitly accepts running
    // WITHOUT a monotone safeguard; barrier_governor=monitored already
    // provides one, so the combination is a direct contradiction.
    if (never_monotone_ && barrier_governor_ == BarrierGovernors::monitored)
        throw std::invalid_argument(
            "never_monotone=True is contradictory with barrier_governor=monitored: the "
            "monitored governor already provides the monotone safeguard never_monotone "
            "opts out of; set barrier_governor=classic_adaptive or never_monotone=False");

    // --- Convergence tolerances ---
    pos_finite(kkt_tol_, "kkt_tol");
    pos_finite(econ_tol_, "econ_tol");
    pos_finite(icon_tol_, "icon_tol");
    pos_finite(bar_tol_, "bar_tol");

    // --- Acceptable tolerances ---
    pos_finite(acc_kkt_tol_, "acc_kkt_tol");
    pos_finite(acc_econ_tol_, "acc_econ_tol");
    pos_finite(acc_icon_tol_, "acc_icon_tol");
    pos_finite(acc_bar_tol_, "acc_bar_tol");

    // --- Unacceptable tolerances ---
    pos_finite(unacc_kkt_tol_, "unacc_kkt_tol");
    pos_finite(unacc_econ_tol_, "unacc_econ_tol");
    pos_finite(unacc_icon_tol_, "unacc_icon_tol");
    pos_finite(unacc_bar_tol_, "unacc_bar_tol");

    // --- Divergence tolerances ---
    pos_finite(div_kkt_tol_, "div_kkt_tol");
    pos_finite(div_econ_tol_, "div_econ_tol");
    pos_finite(div_icon_tol_, "div_icon_tol");
    pos_finite(div_bar_tol_, "div_bar_tol");

    // --- Cross-field: convergence tols <= acceptable tols <= divergence tols ---
    if (kkt_tol_ > acc_kkt_tol_)
        throw std::invalid_argument(
            fmt::format("kkt_tol ({}) must be <= acc_kkt_tol ({})", kkt_tol_, acc_kkt_tol_));
    if (econ_tol_ > acc_econ_tol_)
        throw std::invalid_argument(
            fmt::format("econ_tol ({}) must be <= acc_econ_tol ({})", econ_tol_, acc_econ_tol_));
    if (icon_tol_ > acc_icon_tol_)
        throw std::invalid_argument(
            fmt::format("icon_tol ({}) must be <= acc_icon_tol ({})", icon_tol_, acc_icon_tol_));
    if (bar_tol_ > acc_bar_tol_)
        throw std::invalid_argument(
            fmt::format("bar_tol ({}) must be <= acc_bar_tol ({})", bar_tol_, acc_bar_tol_));
    if (acc_kkt_tol_ > div_kkt_tol_)
        throw std::invalid_argument(fmt::format("acc_kkt_tol ({}) must be <= div_kkt_tol ({})",
                                                acc_kkt_tol_, div_kkt_tol_));
    if (acc_econ_tol_ > div_econ_tol_)
        throw std::invalid_argument(fmt::format("acc_econ_tol ({}) must be <= div_econ_tol ({})",
                                                acc_econ_tol_, div_econ_tol_));
    if (acc_icon_tol_ > div_icon_tol_)
        throw std::invalid_argument(fmt::format("acc_icon_tol ({}) must be <= div_icon_tol ({})",
                                                acc_icon_tol_, div_icon_tol_));
    if (acc_bar_tol_ > div_bar_tol_)
        throw std::invalid_argument(fmt::format("acc_bar_tol ({}) must be <= div_bar_tol ({})",
                                                acc_bar_tol_, div_bar_tol_));

    // --- Barrier parameters ---
    pos_finite(init_mu_, "init_mu");
    pos_finite(min_mu_, "min_mu");
    pos_finite(max_mu_, "max_mu");
    if (min_mu_ > max_mu_)
        throw std::invalid_argument(
            fmt::format("min_mu ({}) must be <= max_mu ({})", min_mu_, max_mu_));
    if (init_mu_ < min_mu_ || init_mu_ > max_mu_)
        throw std::invalid_argument(fmt::format(
            "init_mu ({}) must be within [min_mu ({}), max_mu ({})]", init_mu_, min_mu_, max_mu_));

    // --- Step parameters ---
    if (bound_fraction_ <= 0.0 || bound_fraction_ >= 1.0)
        throw std::invalid_argument("bound_fraction must be in (0, 1)");
    if (bound_push_ <= 0.0)
        throw std::invalid_argument("bound_push must be > 0");
    pos_finite(neg_slack_reset_, "neg_slack_reset");
    pos_finite(soe_bound_relax_, "soe_bound_relax");
    if (alpha_red_ <= 1.0)
        throw std::invalid_argument("alpha_red must be > 1.0");

    // --- Hessian perturbation ---
    if (delta_h_ <= 0.0)
        throw std::invalid_argument("delta_h must be > 0");
    if (incr_h_ <= 1.0)
        throw std::invalid_argument("incr_h must be > 1.0");
    if (decr_h_ <= 0.0 || decr_h_ >= 1.0)
        throw std::invalid_argument("decr_h must be in (0, 1)");

    // --- QP solver ---
    pos_int(qp_threads_, "qp_threads");
    if (qp_pivot_perturb_ < 0)
        throw std::invalid_argument(
            fmt::format("qp_pivot_perturb must be non-negative, got {}", qp_pivot_perturb_));
    if (qp_ref_steps_ < 0)
        throw std::invalid_argument(
            fmt::format("qp_ref_steps must be non-negative, got {}", qp_ref_steps_));
    if (qp_par_solve_ != 0 && qp_par_solve_ != 1)
        throw std::invalid_argument(
            fmt::format("qp_par_solve must be 0 or 1, got {}", qp_par_solve_));
    if (qp_matching_ != 0 && qp_matching_ != 1)
        throw std::invalid_argument(
            fmt::format("qp_matching must be 0 or 1, got {}", qp_matching_));
    if (qp_scaling_ != 0 && qp_scaling_ != 1)
        throw std::invalid_argument(
            fmt::format("qp_scaling must be 0 or 1, got {}", qp_scaling_));

    // --- Objective ---
    if (!std::isfinite(obj_scale_) || obj_scale_ == 0.0)
        throw std::invalid_argument("obj_scale must be finite and non-zero");

    // --- Output ---
    if (print_level_ < 0)
        throw std::invalid_argument(
            fmt::format("print_level must be non-negative, got {}", print_level_));

#ifdef USE_ACCELERATE_SPARSE
    // --- Accelerate sparse solver ---
    pos_finite(accel_pivot_tolerance_, "accel_pivot_tolerance");
    pos_finite(accel_zero_tolerance_, "accel_zero_tolerance");
#endif
}

// =============================================================================
// QP parameter setup
// =============================================================================

void tycho::solvers::PSIOPT::set_qp_params() {
#ifdef USE_ACCELERATE_SPARSE
    // Accelerate interface uses different configuration methods
    switch (settings_.qp_ord_) {
    case QPOrderingModes::MINDEG:
        this->kkt_sol_.set_order(SparseOrderAMD);
        break;
    case QPOrderingModes::METIS:
        // Serial METIS: faster than MT-METIS at all tested scales (up to
        // ~5400 primal variables) due to per-call thread coordination overhead.
        this->kkt_sol_.set_order(SparseOrderMetis);
        break;
    case QPOrderingModes::PARMETIS:
        // MT-METIS (macOS 26+): currently slower than serial METIS at tested
        // scales. Retained for tracking Apple's improvements across releases.
#ifdef TYCHO_HAS_MTMETIS
        this->kkt_sol_.set_order(SparseOrderMTMetis);
#else
        this->kkt_sol_.set_order(SparseOrderMetis);
#endif
        break;
    default:
        throw std::invalid_argument("Unknown QPOrderingMode");
    }
    this->kkt_sol_.set_num_threads(settings_.qp_threads_);
    this->kkt_sol_.set_iterative_refinement(settings_.qp_ref_steps_ > 0);
    this->kkt_sol_.set_iterative_refinement_iterations(settings_.qp_ref_steps_);
    this->kkt_sol_.set_pivot_tolerance(settings_.accel_pivot_tolerance_);
    this->kkt_sol_.set_zero_tolerance(settings_.accel_zero_tolerance_);
#else
    this->kkt_sol_.ord_ = static_cast<int>(settings_.qp_ord_);
    this->kkt_sol_.pivotstrat_ = static_cast<int>(settings_.qp_pivot_strategy_);
    this->kkt_sol_.pivotpert_ = settings_.qp_pivot_perturb_;
    this->kkt_sol_.matching_ = settings_.qp_matching_;
    this->kkt_sol_.scaling_ = settings_.qp_scaling_;
    this->kkt_sol_.iterref_ = settings_.qp_ref_steps_;
    this->kkt_sol_.alg_ = static_cast<int>(settings_.qp_alg_);
    this->kkt_sol_.msglvl_ = settings_.qp_print_;

    if (settings_.cnr_mode_)
        this->kkt_sol_.threads_ = settings_.qp_threads_;
    this->kkt_sol_.parsolve_ = settings_.qp_par_solve_;
    this->kkt_sol_.set_params();
#endif
}

// =============================================================================
// KKT matrix analysis
// =============================================================================

bool tycho::solvers::PSIOPT::analyze_kkt_matrix() {
    bool docompute = true;
    if (this->qp_analyzed_ && !(settings_.force_qp_analysis_)) {
        docompute = false;
    } else {
        this->qp_analyzed_ = true;
        docompute = true;
    }
    return docompute;
}

// =============================================================================
// Release
// =============================================================================

void tycho::solvers::PSIOPT::release() {
    this->kkt_sol_.release();
    this->qp_analyzed_ = false;
    this->nlp_.reset();
    result_.primals_.resize(0);
    result_.eq_lmults_.resize(0);
    result_.iq_lmults_.resize(0);
    result_.eq_cons_.resize(0);
    result_.iq_cons_.resize(0);
}

// =============================================================================
// Barrier math helpers
// =============================================================================

void tycho::solvers::PSIOPT::apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S,
                                                Eigen::Ref<Eigen::VectorXd> FXI) const {
    for (int i = 0; i < this->slack_vars_; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < settings_.neg_slack_reset_) {
            si = settings_.neg_slack_reset_;
        }

        if (fxi < 0.0) {
            FXI[i] = 0.0;
            S[i] = std::max(std::abs(fxi), settings_.neg_slack_reset_);
        } else {
            FXI[i] += si;
        }
    }
}

// max_step_to_boundary was extracted verbatim into BacktrackingLineSearch
// (src/solvers/psiopt_globalization.cpp).

void tycho::solvers::PSIOPT::complementarity(Eigen::Ref<Eigen::VectorXd> S,
                                             Eigen::Ref<Eigen::VectorXd> LI, double &avgcomp,
                                             double &mincomp, double &maxcomp) const {
    // Buffer-hoist ONLY: keep the exact Eigen .sum()/minCoeff()/maxCoeff()
    // reduction expressions unchanged. avgcomp feeds mu (see mpc_mu/loqo_mu
    // call sites), so a hand-fused loop that reorders the sum could perturb
    // the reduction by a ULP under fast-math and change iterates -- forbidden.
    // This change only removes the per-call heap allocation of StLI.
    this->stli_scratch_.resize(S.size());
    this->stli_scratch_ = S.cwiseProduct(LI);
    mincomp = this->stli_scratch_.minCoeff();
    maxcomp = this->stli_scratch_.maxCoeff();
    avgcomp = this->stli_scratch_.sum() / double(this->stli_scratch_.size());
}

void tycho::solvers::PSIOPT::augment_complementarity_nested(double &avgcomp, double &mincomp,
                                                            double &maxcomp, int base_count) const {
    // Off the nested restoration path this is a pure no-op: the aggregates keep
    // the exact values complementarity() produced, so the default/proximal
    // barrier machinery is byte-identical (the CBWR gate depends on it).
    if (!(this->restoration_ && this->restoration_->is_active() && this->restoration_->is_nested()))
        return;

    // The elastic (n,p,z) bound pairs of the restoration barrier subproblem are
    // complementary at restoration scale even after the ORIGINAL slack/multiplier
    // pairs collapse to solve-tolerance; feeding only the original pairs to the
    // barrier-parameter oracle would drive mu to its floor and freeze the phase.
    // Aggregate the elastic pairs separately, then combine WITHOUT re-reducing
    // the original pairs: union min is the min of the two mins, union max the max
    // of the two maxes, and the union average is the count-weighted average
    // (original sum reconstructed as avgcomp*base_count).
    double esum = 0.0;
    double emin = 0.0;
    double emax = 0.0;
    int ecount = 0;
    this->restoration_->nested_complementarity(esum, emin, emax, ecount);
    if (ecount == 0)
        return;

    if (base_count > 0) {
        mincomp = std::min(mincomp, emin);
        maxcomp = std::max(maxcomp, emax);
        avgcomp = (avgcomp * double(base_count) + esum) / double(base_count + ecount);
    } else {
        mincomp = emin;
        maxcomp = emax;
        avgcomp = esum / double(ecount);
    }
}

void tycho::solvers::PSIOPT::barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                             Eigen::Ref<Eigen::VectorXd> S,
                                             Eigen::Ref<Eigen::VectorXd> LI, double mu) {
    this->hp_scratch_.resize(S.size());
    this->hp_scratch_ = LI.cwiseQuotient(S);
    for (int i = 0; i < this->inequal_cons_; i++) {
        if (this->hp_scratch_[i] < 0.0) {
            this->hp_scratch_[i] = mu / (S[i] * S[i]);
        }
    }
    this->nlp_->assign_kkt_slack_hessian(this->hp_scratch_, KKTmat);
}

// loqo_mu / mpc_mu were extracted verbatim into ClassicAdaptiveGovernor
// (src/solvers/psiopt_globalization.cpp); the barrier-parameter
// update now runs through governor_->update_barrier().

// =============================================================================
// NLP eval dispatch methods
// =============================================================================

void tycho::solvers::PSIOPT::eval_kkt(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_kkt(
        obj_scale, XSL.head(primal_vars_), XSL.segment(primal_vars_ + slack_vars_, equal_cons_),
        XSL.tail(inequal_cons_), val, GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_), AGXS_FX.tail(inequal_cons_),
        KKTmat);
}

void tycho::solvers::PSIOPT::eval_kkt_no(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                         EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                         Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_kkt_no(
        obj_scale, XSL.head(primal_vars_), XSL.segment(primal_vars_ + slack_vars_, equal_cons_),
        XSL.tail(inequal_cons_), val, GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_), AGXS_FX.tail(inequal_cons_),
        KKTmat);
}

void tycho::solvers::PSIOPT::eval_aug(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_aug(
        obj_scale, XSL.head(primal_vars_), XSL.segment(primal_vars_ + slack_vars_, equal_cons_),
        XSL.tail(inequal_cons_), val, GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_), AGXS_FX.tail(inequal_cons_),
        KKTmat);
}

void tycho::solvers::PSIOPT::eval_soe(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->nlp_->eval_soe(
        obj_scale, XSL.head(primal_vars_), XSL.segment(primal_vars_ + slack_vars_, equal_cons_),
        XSL.tail(inequal_cons_), val, GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_), AGXS_FX.tail(inequal_cons_),
        KKTmat);
}

// =============================================================================
// Solver initialization and NLP setup
// =============================================================================

void tycho::solvers::PSIOPT::ensure_solver_initialized() {
    double initMs = ::tycho::solvers::ensure_solver_initialized();
    if (initMs > 0.0) {
        this->result_.solver_init_time_ = initMs / 1000.0;
        // Suppress the init line when init was trivially fast (< 0.5 ms).
        constexpr double kSolverInitPrintThresholdMs = 0.5;
        if (initMs > kSolverInitPrintThresholdMs && settings_.print_level_ < 2) {
            fmt::print(" Solver Initialization : ");
            fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", initMs);
        }
    }
}

// Constructors and destructor are defined here (not inline in the
// header) because the std::unique_ptr<AcceptanceStrategy>,
// std::unique_ptr<GlobalizationMechanism>, and std::unique_ptr<BarrierGovernor>
// members need their complete concrete types for their destructors — reached
// through both the destructor and the constructors' exception-cleanup paths.
// Bodies are the former header-inline bodies, unchanged.
tycho::solvers::PSIOPT::PSIOPT() {
    settings_.qp_threads_ = std::min(TYCHO_DEFAULT_QP_THREADS, tycho::utils::get_core_count());
}

tycho::solvers::PSIOPT::PSIOPT(std::shared_ptr<NonLinearProgram> np) {
    settings_.qp_threads_ = std::min(TYCHO_DEFAULT_QP_THREADS, tycho::utils::get_core_count());
    this->set_nlp(np);
}

tycho::solvers::PSIOPT::~PSIOPT() = default;

void tycho::solvers::PSIOPT::set_nlp(std::shared_ptr<NonLinearProgram> np) {
    if (!np)
        throw std::invalid_argument("PSIOPT::set_nlp: NonLinearProgram pointer must not be null");
    this->nlp_ = np;
    this->primal_vars_ = this->nlp_->primal_vars_;
    this->equal_cons_ = this->nlp_->equal_cons_;
    this->inequal_cons_ = this->nlp_->inequal_cons_;
    this->slack_vars_ = this->nlp_->slack_vars_;
    this->kkt_dim_ = this->nlp_->kkt_dim_;
    if (kkt_dim_ != primal_vars_ + slack_vars_ + equal_cons_ + inequal_cons_)
        throw std::logic_error(
            fmt::format("PSIOPT::set_nlp: NLP kkt_dim ({}) != primal_vars ({}) + slack_vars ({}) "
                        "+ equal_cons ({}) + inequal_cons ({})",
                        kkt_dim_, primal_vars_, slack_vars_, equal_cons_, inequal_cons_));

    // acceptance_/mechanism_/governor_/recovery_ are rebuilt from Settings by
    // rebuild_globalization_components(), NOT here: that construction runs
    // once per solve invocation (at every run_phase_sequence() entry) rather
    // than only on (re)transcription, so construction-time knobs
    // (acceptance_strategy, max_soc, ls_extended_iters, watchdog,
    // merit_penalty_rule) take effect on the very next solve even without an
    // intervening set_nlp() call. See rebuild_globalization_components()'s
    // definition below for the neutrality argument on the default path.
    this->set_qp_params();
#ifdef USE_ACCELERATE_SPARSE
    accelerate_set_num_threads(settings_.qp_threads_);
#else
    // Thread-local, not global: only the calling thread's Pardiso thread
    // count is affected, so concurrent PSIOPT drivers in the same process
    // (or on other threads) cannot clobber each other's setting. Return
    // value (previous local count) is intentionally discarded — this is a
    // fire-and-forget set, matching the prior global-setter semantics.
    mkl_set_num_threads_local(settings_.qp_threads_);
#endif

    this->nlp_->analyze_sparsity(this->kkt_sol_.get_matrix());
#ifdef USE_ACCELERATE_SPARSE
    // we need to call this to update the internal AccelSparseMatrix since
    // we changed the sparsity pattern via the reference returned from get_matrix.
    this->kkt_sol_.reinitialize_internal_matrix_representation();
#endif
    this->qp_analyzed_ = false;
}

// (Re)builds the four globalization components from the CURRENT Settings.
// Called once at the top of every run_phase_sequence() — i.e. once per solve
// invocation (optimize()/solve()/solve_optimize()/etc. all route through
// run_phase_sequence()) — rather than only from set_nlp() (which runs only on
// (re)transcription). This makes construction-time knobs (acceptance_strategy,
// max_soc, ls_extended_iters, watchdog, merit_penalty_rule) live at the next
// solve even when no set_nlp() call intervenes, matching every other Settings
// field (which alg_impl/governor_/etc. already read live off settings_ each
// iteration). Before this fix, these four knobs were snapshotted at whichever
// (re)transcription last ran set_nlp() and silently ignored by a later
// solve() call that changed them without retranscribing — see
// tychopy/test/test_Solvers/test_psiopt_globalization_settings.py's
// test_ComponentRebuildTakesEffectWithoutRetranscription for the reachable-
// from-Python repro (the two acceptance strategies produce different
// iteration counts from the same cold start, so a stale acceptance_ is
// directly observable).
//
// Neutrality on the default (all-off) path: this call constructs the exact
// same four concrete types (ClassicMeritAcceptance, BacktrackingLineSearch,
// ClassicAdaptiveGovernor, NoopRecovery) that set_nlp() used to construct —
// only the MOMENT of construction moves (every solve entry vs. every
// (re)transcription). No consumer can observe the difference: nothing reads
// acceptance_/mechanism_/governor_/recovery_ between set_nlp() returning and
// run_phase_sequence() reaching this call (verified by grep — the only
// consumers are alg_impl's dispatch and the per-phase reset() calls, both
// inside run_phase_sequence()'s own call graph), and ClassicMeritAcceptance's
// SolverContext captures (this->nlp_.get(), and primal_vars_/slack_vars_/
// equal_cons_/inequal_cons_/kkt_dim_ by const reference) are already their
// final values at run_phase_sequence() entry: set_nlp() always runs first
// (run_phase_sequence() throws if nlp_ is unset) and is the only place those
// members are written, so re-snapshotting them here at solve time reproduces
// bit-identical captures to the old per-transcription construction.
void tycho::solvers::PSIOPT::rebuild_globalization_components() {
    // (Re)build the optional feasibility-restoration mode-switch FIRST, so the
    // ClassicMeritAcceptance SolverContext below captures a valid (or null)
    // restoration_ pointer. Default (off) leaves it null — no RestorationStrategy
    // is constructed and every restoration branch in the solver stays dead.
    // proximal_switch builds a ProximalSwitchRestoration; l1_nested builds a
    // NestedL1Restoration instead (the condensed l1 elastic reformulation,
    // globalization/l1_restoration.h) — either way the matching outermost
    // FeasibilitySwitchRecovery link is wrapped around the recovery chain
    // below. No strategy-compatibility validation is needed for either mode:
    // every shipped acceptance strategy (classic_merit, merit, funnel, filter)
    // implements RestorationStrategy's exit test (is_infeasibility_
    // sufficiently_reduced / the strategy-specific equivalent), which is what
    // FeasibilitySwitchRecovery and alg_impl's restoration-active branches
    // rely on to end a restoration episode — so restoration_mode_ composes
    // with all four unconditionally, by construction.
    if (this->settings_.restoration_mode_ == RestorationModes::proximal_switch) {
        this->restoration_ = std::make_unique<ProximalSwitchRestoration>();
    } else if (this->settings_.restoration_mode_ == RestorationModes::l1_nested) {
        this->restoration_ = std::make_unique<NestedL1Restoration>();
    } else {
        this->restoration_.reset();
    }

    // (Re)build the step-acceptance strategy. Default (classic_merit) builds
    // ClassicMeritAcceptance wired to a SolverContext view of this solver.
    // Opting in (acceptance_strategy_ == merit/funnel/filter) builds the
    // corresponding strategy instead — all three drive through the GENERIC
    // compute_step path and carry no SolverContext (the mechanism owns
    // trial-point eval); merit's penalty state is seeded from
    // merit_penalty_rule_, while funnel/filter are default-constructed and
    // derive their bounds lazily from the first iterate they see each phase.
    if (this->settings_.acceptance_strategy_ == AcceptanceStrategies::merit) {
        this->acceptance_ =
            std::make_unique<ModernMeritAcceptance>(this->settings_.merit_penalty_rule_);
    } else if (this->settings_.acceptance_strategy_ == AcceptanceStrategies::funnel) {
        this->acceptance_ = std::make_unique<FunnelAcceptance>();
    } else if (this->settings_.acceptance_strategy_ == AcceptanceStrategies::filter) {
        // HARD CONTRACT: seed the restoration-exit constraint-tolerance floor
        // from the live settings tolerance. FilterAcceptance's exit test
        // (is_infeasibility_sufficiently_reduced) floors the relative
        // θ-reduction with restoration_constraint_tol_; leaving it at the
        // header default would silently decouple the filter exit floor from the
        // user-configured econ_tol_. Set unconditionally when a FilterAcceptance
        // is built (independent of restoration_mode_) — it is inert unless a
        // restoration episode actually drives the exit test.
        auto filter = std::make_unique<FilterAcceptance>();
        filter->set_restoration_constraint_tol(this->settings_.econ_tol_);
        this->acceptance_ = std::move(filter);
    } else {
        this->acceptance_ = std::make_unique<ClassicMeritAcceptance>(
            SolverContext{this->nlp_.get(), this->kkt_sol_, this->settings_, this->primal_vars_,
                          this->slack_vars_, this->equal_cons_, this->inequal_cons_, this->kkt_dim_,
                          this->stli_scratch_, this->hp_scratch_, this->best_xsl_scratch_,
                          this->best_rhs_scratch_, this->restoration_.get(),
                          &this->eval_error_log_});
    }

    // The step-length globalization mechanism. Stateless (holds
    // NO solver state per GlobalizationMechanism's ownership rule) — every call
    // receives the live SolverContext as an explicit parameter — so it is
    // constructed with no context here; alg_impl builds the SolverContext view
    // it passes to compute_step / max_primal_dual_step.
    this->mechanism_ = std::make_unique<BacktrackingLineSearch>();

    // The barrier-parameter governor. Default (classic_adaptive) builds
    // ClassicAdaptiveGovernor, which is stateless (holds NO solver state per
    // BarrierGovernor's ownership rule) — every update_barrier() call receives
    // the live SolverContext and the GlobalizationMechanism as explicit
    // parameters — so it is constructed with no context here; alg_impl builds
    // the SolverContext view and passes *mechanism_ to update_barrier. Opting
    // in (barrier_governor_ == monitored) builds MonitoredBarrierGovernor
    // instead, default-constructed (it composes its own ClassicAdaptiveGovernor
    // free-mode delegate internally — see monitored_governor.h); it carries its
    // own monotone-mode bookkeeping (the KKT-error reference window, current
    // mode, current monotone mu) as private state, cleared by reset() at each
    // phase boundary like every other governor's phase-change hook.
    if (this->settings_.barrier_governor_ == BarrierGovernors::monitored) {
        this->governor_ = std::make_unique<MonitoredBarrierGovernor>();
    } else {
        this->governor_ = std::make_unique<ClassicAdaptiveGovernor>();
    }

    // The post-rejection recovery chain. Every concrete implementation
    // (NoopRecovery, SocRecovery, ExtendedBacktrackRecovery, ChainedRecovery)
    // except WatchdogRecovery is stateless (holds no solver state, per
    // RecoveryChain's ownership rule) and needs no context at construction;
    // alg_impl builds the SolverContext view and passes the live working set
    // to on_step_rejected. Default (max_soc_ == 0, ls_extended_iters_ == 0,
    // watchdog_ == false — all off) installs plain NoopRecovery, which always
    // returns kAcceptAsIs so the solve path is bit-identical to its
    // pre-recovery-chain behavior.
    //
    // Opting in to SOC and/or extended backtracking composes them (in that
    // fixed order — see ChainedRecovery's class doc, globalization/
    // watchdog.h) into a ChainedRecovery; either link may be individually
    // enabled. The watchdog, if enabled, then wraps whatever chain resulted
    // (even plain NoopRecovery) as an outer decorator, per WatchdogRecovery's
    // class doc.
    if (this->settings_.max_soc_ > 0 || this->settings_.ls_extended_iters_ > 0) {
        std::unique_ptr<RecoveryChain> soc_link =
            this->settings_.max_soc_ > 0 ? std::make_unique<SocRecovery>() : nullptr;
        std::unique_ptr<RecoveryChain> extended_link =
            this->settings_.ls_extended_iters_ > 0 ? std::make_unique<ExtendedBacktrackRecovery>()
                                                    : nullptr;
        this->recovery_ =
            std::make_unique<ChainedRecovery>(std::move(soc_link), std::move(extended_link));
    } else {
        this->recovery_ = std::make_unique<NoopRecovery>();
    }
    if (this->settings_.watchdog_) {
        this->recovery_ = std::make_unique<WatchdogRecovery>(std::move(this->recovery_));
    }

    // Feasibility restoration wraps the OUTERMOST recovery link (built only when
    // restoration_mode_ != off, i.e. exactly when restoration_ above is
    // non-null — proximal_switch or l1_nested). It delegates to the whole
    // inner chain (Noop/Chained/Watchdog) and intercepts only its
    // ladder-exhausted kAcceptAsIs to hand off to restoration — see
    // feasibility_switch_recovery.h (the nested-vs-non-nested soft-pre-stage
    // branch inside it is driven by restoration_->is_nested(), so this wrap
    // condition itself does not need to distinguish the two modes). Off by
    // default, so the recovery chain is unchanged on the default path.
    if (this->settings_.restoration_mode_ != RestorationModes::off) {
        this->recovery_ = std::make_unique<FeasibilitySwitchRecovery>(std::move(this->recovery_));
    }
}

// max_primal_dual_step was extracted verbatim into BacktrackingLineSearch
// (src/solvers/psiopt_globalization.cpp). alg_impl drives it
// through mechanism_ (fused into compute_step on the main path; via the public
// method at the PROBE predictor call site).

// fill_residual_info() deliberately excludes barr_obj_/mu_/p_pivots_. barr_obj_ is only
// evaluated by the caller AFTER the barrier-parameter update block (barrier_objective()
// runs on the just-updated `mu`); for BarrierModes::PROBE that update itself needs the
// KKT solve (mpc_mu() consumes the predictor DXSL). p_pivots_ similarly only reflects a
// real value once this iteration's factor_impl() has run. All three are therefore NOT
// "residuals" in the sense converge_check()/return_best_/print_exit_stats consume them
// (none of those three read mu_/barr_obj_/p_pivots_ -- verified: converge_check() reads
// only kkt_inf_/econ_inf_/icon_inf_/barr_inf_; return_best_ reads econ_inf_/icon_inf_/
// kkt_inf_/prim_obj_; print_exit_stats reads prim_obj_/kkt_inf_/barr_inf_/econ_inf_/
// icon_inf_). Every field this function DOES set is fully determined by rhs/xsl alone,
// unconditionally available right after eval_nlp + the barrier/complementarity block --
// before any factorization -- and provably unchanged for the remainder of the
// iteration (RHS's prim_grad()/eq_cons()/iq_cons()/all_cons() blocks and XSL's
// slacks()/iq_lmults()/eq_lmults() are not written again until `XSL += alpha*DXSL`,
// which only executes strictly after this iteration's earliest possible break).
void tycho::solvers::PSIOPT::fill_residual_info(KKTVector &xsl, KKTVector &rhs, double pobj,
                                                IterateInfo &iter) const {

    iter.prim_obj_ = pobj;
    iter.kkt_inf_ = rhs.prim_grad().lpNorm<Eigen::Infinity>();

    double avgcomp = 0;
    double mincomp = 0;
    double maxcomp = 0;
    if (inequal_cons_ > 0) {
        iter.icon_inf_ = rhs.iq_cons().lpNorm<Eigen::Infinity>();
        iter.icon_norm_err_ = rhs.iq_cons().norm();
        iter.max_i_mult_ = xsl.iq_lmults().lpNorm<Eigen::Infinity>();
        this->complementarity(xsl.slacks(), xsl.iq_lmults(), avgcomp, mincomp, maxcomp);
        // While a nested restoration phase is active, the barrier error the
        // convergence check and the monitored governor's KKT-error monitor read
        // must reflect the elastic pairs too (dead no-op otherwise).
        this->augment_complementarity_nested(avgcomp, mincomp, maxcomp,
                                             static_cast<int>(xsl.slacks().size()));

        iter.barr_inf_ = maxcomp;
        iter.barr_norm_err_ = avgcomp;
    }
    if (equal_cons_ > 0) {
        iter.econ_inf_ = rhs.eq_cons().lpNorm<Eigen::Infinity>();
        iter.econ_norm_err_ = rhs.eq_cons().norm();
        iter.max_e_mult_ = xsl.eq_lmults().lpNorm<Eigen::Infinity>();
    }

    iter.kkt_norm_err_ = rhs.prim_grad().norm();

    if (equal_cons_ > 0 || inequal_cons_ > 0)
        iter.all_con_norm_err_ = rhs.all_cons().norm();
}

void tycho::solvers::PSIOPT::fill_iter_info(KKTVector &xsl, KKTVector &rhs, double pobj,
                                            double bobj, double mu, IterateInfo &iter) const {
    this->fill_residual_info(xsl, rhs, pobj, iter);
    iter.barr_obj_ = bobj;
    iter.mu_ = mu;
    iter.p_pivots_ = this->kkt_sol_.ppivs();
}

void tycho::solvers::PSIOPT::eval_nlp(AlgorithmModes algmode, double obj_scale,
                                      ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                      double mu) {
    std::fill_n(KKTmat.valuePtr(), KKTmat.nonZeros(), 0.0);

    // Feasibility-restoration evaluation seam. Dead on the default path
    // (restoration_ is null unless a restoration mode is selected). While
    // restoration is active the true objective is uniformly replaced by a
    // solver-internal restoration objective: route through the objective-free
    // KKT (constraints + their Hessians, exactly the OPTNO/SOE shape), then
    // inject the restoration objective value, its gradient into the (now-zero)
    // objective-gradient block, and its diagonal Hessian via the solver
    // primal-diagonal slot the SOE/INIT modes already use. The auxiliary/barrier
    // terms are untouched, and obj_scale never multiplies the restoration
    // objective. The convergence check needs no mode code: it reads whatever
    // lands in prim_grad() downstream (the objective-free-mode precedent).
    if (this->restoration_ && this->restoration_->is_active()) {
        // Nested condensed l1 restoration. The elastic pair (n,p) and their bound
        // multipliers are eliminated analytically: their curvature lands on the
        // constraint-row pivot slots — NEGATED, because the (y,y) diagonal of the
        // condensed system is −pivot while the solver scatters the pivot slot as a
        // +coefficient onto that diagonal (finalize_data / fill_solver_coeffs) —
        // their proximal term substitutes the objective exactly as the proximal
        // switch does, and each constraint-row RHS carries the condensed residual
        // r̃ in place of the raw residual. μ is the live phase barrier parameter
        // (η(μ) is recomputed on every evaluation). Pivots and the primal diagonal
        // must be set BEFORE eval_kkt_no (they scatter inside fill_solver_coeffs)
        // and reset to 0.0 after, mirroring the set_primal_diags discipline.
        if (this->restoration_->is_nested()) {
            const int ec = this->equal_cons_;
            const int ic = this->inequal_cons_;
            this->resto_pdiag_scratch_.resize(this->primal_vars_);
            this->restoration_->nested_primal_diagonal(mu, this->resto_pdiag_scratch_);
            this->nlp_->set_primal_diags(this->resto_pdiag_scratch_);
            this->resto_epiv_scratch_ = -this->restoration_->e_pivots();
            this->resto_ipiv_scratch_ = -this->restoration_->i_pivots();
            this->nlp_->set_e_pivots(this->resto_epiv_scratch_);
            this->nlp_->set_i_pivots(this->resto_ipiv_scratch_);
            eval_kkt_no(0.0, XSL, val, GX, AGXS_FX, KKTmat);
            this->nlp_->set_primal_diags(0.0);
            this->nlp_->set_e_pivots(0.0);
            this->nlp_->set_i_pivots(0.0);
            val = this->restoration_->nested_objective(mu, XSL.head(primal_vars_));
            this->restoration_->add_nested_gradient(mu, XSL.head(primal_vars_),
                                                    GX.head(primal_vars_));
            // Replace the raw constraint residuals with the condensed r̃. Copy the
            // raw residual c out first: condensed_residuals reads c and writes r̃,
            // and the target segments alias the raw-c source in the RHS vector. The
            // inequality residual for the elastic row is the slack-completed g(x)+s
            // (the same residual the ordinary path forms via apply_reset_slacks,
            // which is suppressed for the nested phase so it cannot clobber r̃);
            // eval_kkt_no leaves only the raw g(x) in the RHS, so add the slacks
            // here. This is also the true original-problem inequality infeasibility
            // the exit ratchet/classification reads back from resto_ic_scratch_.
            this->resto_ec_scratch_ = AGXS_FX.segment(primal_vars_ + slack_vars_, ec);
            this->resto_ic_scratch_ = AGXS_FX.tail(ic) + XSL.segment(primal_vars_, slack_vars_);
            this->restoration_->condensed_residuals(
                mu, this->resto_ec_scratch_, this->resto_ic_scratch_,
                XSL.segment(primal_vars_ + slack_vars_, ec), XSL.tail(ic),
                AGXS_FX.segment(primal_vars_ + slack_vars_, ec), AGXS_FX.tail(ic));
            return;
        }

        // Proximal mode-switch: uniform proximal-objective substitution, no
        // constraint-row modification (the constraints are unchanged in this
        // mode). Byte-identical to the pre-nested seam.
        this->nlp_->set_primal_diags(this->restoration_->proximal_diagonal());
        eval_kkt_no(0.0, XSL, val, GX, AGXS_FX, KKTmat);
        this->nlp_->set_primal_diags(0.0);
        val = this->restoration_->proximal_objective(XSL.head(primal_vars_));
        this->restoration_->add_proximal_gradient(XSL.head(primal_vars_), GX.head(primal_vars_));
        return;
    }

    switch (algmode) {
    case AlgorithmModes::OPT:
        eval_kkt(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::OPTNO:
        eval_kkt_no(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);

        break;
    case AlgorithmModes::INIT:
        eval_aug(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::SOE:
        this->nlp_->set_primal_diags(1.0);
        eval_soe(0.0, XSL, val, GX, AGXS_FX, KKTmat);
        this->nlp_->set_primal_diags(0.0);
        GX.head(primal_vars_).setZero();
        AGXS_FX.head(primal_vars_).setZero();
        break;
    default:
        throw std::invalid_argument("Unknown AlgorithmMode");
    }
}

// Feasibility-restoration exit measures. Dead on the default path (only
// called from restoration-active branches, all guarded on
// `restoration_ && restoration_->is_active()`). The current iterate's
// prim_obj_/cur.objective is φ_prox (the proximal objective substituted by the
// eval seam in eval_nlp above) while restoration is active — it must never be
// handed to notify_switch_to_optimality or reported as obj_val_, since both
// consumers expect true-objective scale. This re-evaluates the true objective
// once at the live primals, matching the non-OPT obj_val_ eval pattern below
// (zero the accumulator, then let eval_obj accumulate into it).
tycho::solvers::ProgressMeasures tycho::solvers::PSIOPT::build_restoration_exit_measures(
    double obj_scale, double infeasibility, ConstEigenRef<VectorXd> primals, double barr_obj) {
    ProgressMeasures measures;
    measures.infeasibility = infeasibility;
    measures.objective = 0.0;
    this->nlp_->eval_obj(obj_scale, primals, measures.objective);
    measures.auxiliary = barr_obj;
    return measures;
}

// Shared feasibility-restoration entry orchestration for the
// kSwitchToFeasibility case. Dead on the default path (only reached with
// restoration_ non-null). Factors the entry sub-steps once so the proximal and
// nested families do not duplicate the notify/recovery-reset scaffolding.
void tycho::solvers::PSIOPT::enter_feasibility_restoration(Eigen::VectorXd &XSL,
                                                           Eigen::VectorXd &RHS, double prim_obj,
                                                           double barr_obj, double &mu) {
    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_rhs = kkt_view(RHS);

    // The entry measures are the TRUE-objective (θ, f) at the current iterate —
    // this point was evaluated in optimality mode; restoration begins next
    // iteration. θ is the L1 norm of the current KKT constraint block, matching
    // what FeasibilitySwitchRecovery's entry_permitted guard consulted.
    ProgressMeasures entry;
    entry.infeasibility = v_rhs.all_cons().template lpNorm<1>();
    entry.objective = prim_obj;
    entry.auxiliary = barr_obj;

    if (this->restoration_->is_nested()) {
        // Nested l1 restoration: full elastic initialization from the current
        // residual vectors (equality h(x); inequality g(x)+s), then the verified
        // entry init (Ipopt RestoIterateInitializer::SetInitialIterates). Only
        // enter_nested is called — NOT also enter_restoration — since each
        // increments the per-phase entry counter and the nested path owns its
        // own snapshot.
        this->restoration_->enter_nested(entry, v_xsl.primals(), v_rhs.eq_cons(), v_rhs.iq_cons(),
                                         mu);

        // Stash the outer barrier parameter (restored by the multiplier re-entry
        // on exit), then set μ to the restoration barrier parameter and reset the
        // governor so the phase gets a fresh in-phase barrier schedule. Seed the
        // κ_resto ratchet with the entry-point original-problem infeasibility and
        // arm the first-iteration guard.
        this->stashed_mu_ = mu;
        mu = this->restoration_->entry_mu();
        this->governor_->reset();
        this->resto_first_iter_ = true;
        // Re-arm the one-shot second-level re-center budget for this episode.
        this->resto_recentered_ = false;
        // Seed the raw-residual copies the exit tests and the max-iterations
        // teardown read. The eval seam refreshes them every active iteration,
        // but if the outer loop exhausts its iteration budget on the very
        // iteration that entered the phase, no nested evaluation runs — without
        // this seed the teardown would read empty (first phase) or stale
        // (later phases) buffers.
        this->resto_ec_scratch_ = v_rhs.eq_cons();
        this->resto_ic_scratch_ = v_rhs.iq_cons();
        double theta_orig = 0.0;
        if (this->equal_cons_ > 0)
            theta_orig =
                std::max(theta_orig, v_rhs.eq_cons().template lpNorm<Eigen::Infinity>());
        if (this->inequal_cons_ > 0)
            theta_orig =
                std::max(theta_orig, v_rhs.iq_cons().template lpNorm<Eigen::Infinity>());
        this->resto_theta_orig_prev_ = theta_orig;

        // Verified entry multiplier init. In tycho's slack-complementarity
        // formulation the inequality multipliers ARE the slack/bound multipliers
        // (s·λ = μ, strictly positive), so they take Ipopt's min(ρ, current)
        // clamp on the bound multipliers (keeps them positive); the free-sign
        // equality constraint multipliers are the y that Ipopt starts at zero
        // (least_square_mults at the shipped default reset threshold 0).
        if (this->equal_cons_ > 0)
            v_xsl.eq_lmults().setZero();
        if (this->inequal_cons_ > 0)
            v_xsl.iq_lmults() = v_xsl.iq_lmults().cwiseMin(kRestoPenaltyParameter);
    } else {
        // Proximal mode-switch: snapshot the center, freeze ζ from the live μ.
        this->restoration_->enter_restoration(entry, v_xsl.primals(), mu);
    }

    this->acceptance_->notify_switch_to_feasibility(entry);
    // Reset the recovery chain across the mode switch (see the entry rationale at
    // the kSwitchToFeasibility case). Once per transition.
    this->recovery_->reset();
}

// The nested phase's multiplier re-entry sequence (Ipopt
// MinC_1NrmRestorationPhase::PerformRestoration, strict order). Dead on the
// default path. Shared byte-for-byte by the κ_resto ratchet exit and the
// near-feasible stall exit.
void tycho::solvers::PSIOPT::exit_feasibility_restoration_nested(Eigen::VectorXd &XSL,
                                                                 double obj_scale,
                                                                 double theta_orig, double barr_obj,
                                                                 double &mu) {
    KKTVector v_xsl = kkt_view(XSL);

    // (1) The phase's final x/s are already in XSL — kept as both current and
    // trial. (2) Slack-multiplier Newton complementarity step under the STASHED
    // outer μ (Ipopt ComputeBoundMultiplierStep): the general step
    // Δz = [(s_curr − s_trial)·z + μ_outer]/s_curr − z reduces to
    // Δz = μ_outer/s − z here, because this in-place solver keeps the phase's
    // final slacks as both current and trial (s_curr == s_trial). Damped by the
    // dual fraction-to-boundary rule (τ = bound_fraction_, the same τ the
    // solver's slack/bound steps use) so every multiplier stays strictly
    // positive. (3) If the largest updated multiplier exceeds
    // kBoundMultResetThreshold, reset ALL inequality multipliers to 1 (Ipopt
    // resets every bound multiplier, not just the offenders).
    if (this->inequal_cons_ > 0) {
        auto s = v_xsl.slacks();
        auto z = v_xsl.iq_lmults();
        this->resto_dz_scratch_.resize(this->inequal_cons_);
        this->resto_dz_scratch_ = this->stashed_mu_ * s.cwiseInverse() - z;
        const double tau = settings_.bound_fraction_;
        double alpha_dual = 1.0;
        for (int i = 0; i < this->inequal_cons_; ++i) {
            const double dz = this->resto_dz_scratch_[i];
            if (dz < -tau * z[i]) {
                const double an = -tau * z[i] / dz;
                if (an < alpha_dual)
                    alpha_dual = an;
            }
        }
        z += alpha_dual * this->resto_dz_scratch_;
        if (z.cwiseAbs().maxCoeff() > kBoundMultResetThreshold)
            z.setConstant(1.0);
    }

    // (4) Equality constraint multipliers ← 0 (Ipopt least_square_mults at the
    // shipped-default reset threshold 0 sets y_c/y_d to zero; verified against
    // IpDefaultIterateInitializer).
    if (this->equal_cons_ > 0)
        v_xsl.eq_lmults().setZero();

    // (5) Restore the stashed outer μ, reset the governor, exit restoration, and
    // notify the acceptance strategy of the switch back to optimality with
    // true-objective exit measures (the loop's prim_obj is φ_l1 while active —
    // never valid for the optimality filter/funnel; re-evaluated here). Reset the
    // recovery chain across the transition. BestXSL tracking resumes implicitly
    // once is_active() flips false.
    mu = this->stashed_mu_;
    this->governor_->reset();
    this->restoration_->exit_restoration();
    this->acceptance_->notify_switch_to_optimality(
        this->build_restoration_exit_measures(obj_scale, theta_orig, v_xsl.primals(), barr_obj));
    this->recovery_->reset();
}

// Per-iteration κ_resto ratchet: the original-problem infeasibility must fall to
// at most max(kKappaResto · previous-iteration value, econ_tol_) (Ipopt
// RestoConvCheck::CheckConvergence's orig_inf_pr_max, single-tolerance floor).
bool tycho::solvers::PSIOPT::resto_ratchet_passes(double theta_orig) const {
    return theta_orig <= std::max(kKappaResto * this->resto_theta_orig_prev_, settings_.econ_tol_);
}

// Second-level elastic re-centering fallback (nested l1 phase, disclosure (f) in
// l1_restoration.h). Dead on the default path. One-shot per consecutive-failure
// run: the resto_recentered_ flag blocks a re-center loop (a re-center takes a
// zero primal/dual step, so an unbounded retry could stall). Reads the raw
// residuals the eval seam saved this iteration (resto_ec_/ic_scratch_) and
// re-solves the elastic pairs in closed form at the current phase μ.
bool tycho::solvers::PSIOPT::try_recenter_elastics(double mu) {
    if (this->resto_recentered_)
        return false; // budget already consumed this failure run — fall through.
    this->restoration_->recenter_elastics(mu, this->resto_ec_scratch_, this->resto_ic_scratch_);
    this->resto_recentered_ = true;
    return true;
}

// Primal-dual system error at μ: the ∞-norm of the full KKT residual. See the
// declaration in psiopt.h for the Ipopt mapping. Dead on the default path.
double tycho::solvers::PSIOPT::primal_dual_error(KKTVector &xsl, KKTVector &rhs, double mu) const {
    double err = 0.0;
    if (this->primal_vars_ > 0)
        err = std::max(err, rhs.prim_grad().template lpNorm<Eigen::Infinity>());
    if (this->equal_cons_ > 0)
        err = std::max(err, rhs.eq_cons().template lpNorm<Eigen::Infinity>());
    if (this->inequal_cons_ > 0) {
        err = std::max(err, rhs.iq_cons().template lpNorm<Eigen::Infinity>());
        // Complementarity deviation max|s_i·z_i − μ| (Ipopt's primal_dual_system_
        // error uses the shifted complementarity residual, not the raw product).
        const double comp =
            (xsl.slacks().cwiseProduct(xsl.iq_lmults()).array() - mu).abs().maxCoeff();
        err = std::max(err, comp);
    }
    return err;
}

// Nested soft feasibility pre-stage trial. See the declaration in psiopt.h. Dead
// on the default path (only reached with a nested restoration strategy, via the
// kSoftFeasibilityStep recovery action; restoration is NOT yet active here).
bool tycho::solvers::PSIOPT::try_soft_feasibility_step(AlgorithmModes algmode, double obj_scale,
                                                       double mu, Eigen::VectorXd &XSL,
                                                       Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                                       Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                                                       Eigen::VectorXd &GX) {
    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_rhs = kkt_view(RHS);
    // Current point: the live RHS already carries the full stationarity (the
    // main loop added the objective/barrier gradient before this point) and the
    // constraint residuals, so no re-evaluation is needed.
    const double curr_pd = this->primal_dual_error(v_xsl, v_rhs, mu);
    if (curr_pd == 0.0)
        return true; // already at the KKT point for this μ — trivially reduced.

    // Full fraction-to-boundary trial step: DXSL already carries compute_step's
    // fraction-to-boundary scaling, so the full step is the whole of DXSL. The
    // step keeps the slacks and bound multipliers strictly positive, so the
    // trial complementarity term is well defined.
    XSL2 = XSL + DXSL;
    KKTVector v_xsl2 = kkt_view(XSL2);
    KKTVector v_rhs2 = kkt_view(RHS2);
    RHS2.setZero();
    GX.setZero();
    double trial_obj = 0.0;
    // Evaluate exactly as the main loop populates the current iterate's RHS: the
    // objective/constraint eval, then fold the objective gradient into the
    // primal stationarity block, then slack-complete the inequality residual.
    // The throwaway KKT matrix is re-zeroed and re-filled at the next iteration's
    // eval, so reusing the assembly buffer here is safe.
    // An un-evaluable trial is not a reduced one: report no reduction so the
    // caller escalates to the full restoration entry.
    try {
        this->eval_nlp(algmode, obj_scale, XSL2, trial_obj, GX, RHS2, this->kkt_sol_.get_matrix(),
                       mu);
    } catch (const std::exception &e) {
        this->eval_error_log_.record(e.what());
        return false;
    } catch (...) {
        this->eval_error_log_.record_unknown();
        return false;
    }
    v_rhs2.prim_grad() += GX;
    if (this->inequal_cons_ > 0)
        this->apply_reset_slacks(v_xsl2.slacks(), v_rhs2.iq_cons());
    const double trial_pd = this->primal_dual_error(v_xsl2, v_rhs2, mu);

    return trial_pd <= kSoftRestoPdErrorReductionFactor * curr_pd;
}

tycho::ConvergenceFlags tycho::solvers::PSIOPT::converge_check(std::vector<IterateInfo> &iters) {
    assert(!iters.empty() && "converge_check called with empty iteration history");
    ConvergenceFlags Flag = ConvergenceFlags::CONVERGED;
    IterateInfo last = iters.back();
    bool KKTFeas = (last.kkt_inf_ < settings_.kkt_tol_);
    bool EConFeas = (last.econ_inf_ < settings_.econ_tol_);
    bool IConFeas = (last.icon_inf_ < settings_.icon_tol_);
    bool BarFeas = (last.barr_inf_ < settings_.bar_tol_);

    // Per-iterate divergent predicate: any monitored residual either non-finite
    // or past its divergence threshold. This is the exact condition that used to
    // abort the solve outright; it now splits into two disjoint verdicts. A
    // non-finite residual (nan_inf) is an immediate, unrecoverable abort. A
    // finite residual merely past threshold (beyond_thresholds) is treated as a
    // possibly-recoverable transient and only aborts once it has persisted for
    // kDivergencePersistIters iterations in a row (see the constant's rationale
    // in psiopt.h). Splitting the two keeps the hard-error path immediate while
    // giving genuinely recoverable overshoots (Maratos-class) room to recover.
    auto iterate_divergent = [this](const IterateInfo &it) {
        return (it.kkt_inf_ > settings_.div_kkt_tol_) || !std::isfinite(it.kkt_inf_) ||
               (it.econ_inf_ > settings_.div_econ_tol_) || !std::isfinite(it.econ_inf_) ||
               (it.icon_inf_ > settings_.div_icon_tol_) || !std::isfinite(it.icon_inf_) ||
               (it.barr_inf_ > settings_.div_bar_tol_) || !std::isfinite(it.barr_inf_);
    };

    bool nan_inf = !std::isfinite(last.kkt_inf_) || !std::isfinite(last.econ_inf_) ||
                   !std::isfinite(last.icon_inf_) || !std::isfinite(last.barr_inf_);
    bool beyond_thresholds =
        (last.kkt_inf_ > settings_.div_kkt_tol_) || (last.econ_inf_ > settings_.div_econ_tol_) ||
        (last.icon_inf_ > settings_.div_icon_tol_) || (last.barr_inf_ > settings_.div_bar_tol_);

    if (nan_inf) {
        // Non-finite residual: unrecoverable, abort immediately regardless of
        // history length. Preserves the original hard-error semantics.
        Flag = ConvergenceFlags::DIVERGING;
        return Flag;
    }
    if (beyond_thresholds) {
        // Finite overshoot. Declare DIVERGING only once the trailing window of
        // kDivergencePersistIters iterates is ALL divergent (this iterate is the
        // newest of that window, and is divergent by construction). Histories
        // shorter than the window cannot declare DIVERGING. Scans trailing
        // history exactly as the acceptable-classification loop below does. Only
        // runs when a threshold has tripped, so the common (non-diverging) path
        // pays nothing beyond the comparisons above.
        if (int(iters.size()) >= kDivergencePersistIters) {
            bool window_all_divergent = true;
            for (int i = 0; i < kDivergencePersistIters; i++) {
                if (!iterate_divergent(iters[int(iters.size()) - i - 1])) {
                    window_all_divergent = false;
                    break;
                }
            }
            if (window_all_divergent) {
                Flag = ConvergenceFlags::DIVERGING;
                return Flag;
            }
        }
        // Window not yet full of divergent iterates: fall through to the ordinary
        // convergence classification (below) and keep iterating. A residual past
        // the divergence threshold cannot satisfy the convergence or acceptable
        // tolerances, so this necessarily yields NOTCONVERGED.
    }

    if (KKTFeas && EConFeas && IConFeas && BarFeas) {
        Flag = ConvergenceFlags::CONVERGED;
        return Flag;
    } else if (int(iters.size()) > settings_.max_acc_iters_) {
        int nfeas = 0;
        for (int i = 0; i < settings_.max_acc_iters_; i++) {
            if (psiopt_iterate_acceptable(iters[int(iters.size()) - i - 1], settings_))
                nfeas++;
            else
                break;
        }
        if (nfeas == settings_.max_acc_iters_) {
            Flag = ConvergenceFlags::ACCEPTABLE;
            return Flag;
        }
    }
    Flag = ConvergenceFlags::NOTCONVERGED;
    return Flag;
}

// Inertia-correcting factorization: if the LDLT factorization has incorrect
// inertia (more negative eigenvalues than expected from the constraint block),
// perturb the primal diagonal by increasing amounts until correct inertia is
// achieved or max_refac_ attempts are exhausted.
int tycho::solvers::PSIOPT::factor_impl(bool docompute, bool Zfac, double ipurt, double incpurt0,
                                        double incpurt, double &finalpert, double &cumpert,
                                        double base_prox, double dual_shift) {
    auto Inertia = [&]() {
        return this->kkt_sol_.neigs() - (this->equal_cons_ + this->inequal_cons_);
    };
    auto RankDef = [&]() {
        if ((this->kkt_sol_.neigs() + this->kkt_sol_.peigs() - this->kkt_dim_) != 0) {
            if (settings_.print_level_ < 3)
                fmt::print(fmt::fg(fmt::color::yellow),
                           "Warning: Potential Rank Deficiency Detected\n");
        }
    };
    // T6 (dead-status fix): kkt_sol_.info() was computed by every Compute()/
    // Refactor() call below and never read anywhere -- a dead status. This records
    // the last non-Success status into result_.last_kkt_info_ (surfaced only by
    // print_exit_stats(), see psiopt_print.cpp) and, for hard failures only, emits
    // an immediate diagnostic gated the same as the sibling RankDef()/perturbation-
    // exhausted warnings in this function. NumericalIssue (Pardiso info -4/-7:
    // zero/near-zero pivot; Accelerate factorization-failed/singular) is a NORMAL,
    // expected condition while probing perturbations during inertia correction
    // below -- printing on every occurrence would spam the console for any problem
    // with an indefinite KKT system, so it is recorded but not printed by default.
    // Purely observational: no return value or branch below is touched by this
    // check.
    auto CheckInfo = [&]() {
        Eigen::ComputationInfo info = this->kkt_sol_.info();
        if (info != Eigen::Success) {
            this->result_.last_kkt_info_ = info;
            if (info != Eigen::NumericalIssue && settings_.print_level_ < 3) {
                fmt::print(fmt::fg(fmt::color::yellow),
                           "Warning: KKT factorization reported a hard error (info={})\n",
                           static_cast<int>(info));
            }
        }
    };
    auto Perturb = [&](double p) {
        this->nlp_->perturb_kkt_p_diags(p, this->kkt_sol_.get_matrix());
    };
    auto Refactor = [&]() { this->kkt_sol_.refactorize_internal(); };
    auto Compute = [&]() { this->kkt_sol_.compute_internal(); };
    int IncEigs;
    cumpert = 0.0;

    if (settings_.inertia_mode_ == InertiaModes::proximal_regularization) {
        // Proximal primal-dual base shift. The persistent primal shift base_prox
        // (ρ_k, decayed across iterations by alg_impl) is added to the (1,1)
        // Hessian-block diagonal, and the barrier-scaled dual shift dual_shift
        // (δ_c) is subtracted from every constraint-row diagonal, ONCE per
        // iteration before the first factorization -- so both are part of the
        // base matrix for the whole ladder and stable across every
        // back-substitution on the live factorization (SOC, elastic recovery).
        // This informed base attempt replaces the classic zero-perturbation
        // attempt; the Zfac cycling heuristic is not consulted (the base attempt
        // IS the informed attempt, so there is no wasted unperturbed
        // factorization for it to skip). The ladder's finalpert/cumpert
        // accounting below is unchanged -- it counts only the ladder increments,
        // never the base shifts, which alg_impl tracks separately.
        auto PerturbC = [&](double p) {
            this->nlp_->perturb_kkt_c_diags(p, this->kkt_sol_.get_matrix());
        };
        Perturb(base_prox);
        if (dual_shift != 0.0)
            PerturbC(-dual_shift);
        if (!docompute)
            Refactor();
        else
            Compute();
        CheckInfo();
        RankDef();
        IncEigs = Inertia();
        finalpert = 0.0;
        // A singular base factorization is treated as wrong inertia (enter the
        // ladder) -- the reference fallback once the dual shift is already
        // applied and the matrix is still singular. Classic treats the same
        // condition as warn-and-proceed (RankDef only warns); that behavior is
        // untouched on the classic branch below.
        bool singular = (this->kkt_sol_.neigs() + this->kkt_sol_.peigs() - this->kkt_dim_) != 0;
        if (IncEigs <= 0 && !singular)
            return 0;
    } else if (Zfac || docompute) {
        if (!docompute)
            Refactor();
        else
            Compute();
        CheckInfo();
        RankDef();
        IncEigs = Inertia();
        finalpert = 0.0;
        if (IncEigs <= 0)
            return 0;
    }
    double p = ipurt;

    for (int i = 0; i < settings_.max_refac_; i++) {
        Perturb(p);
        // Display-only accumulator: the running sum of every
        // Perturb() delta applied so far this call -- i.e. the actual total added
        // to the KKT diagonal. Tracked purely for the HPert column; `finalpert`
        // below (the last delta, consumed by the Hpert0 warm-start) is untouched.
        cumpert += p;
        Refactor();
        CheckInfo();
        RankDef();
        IncEigs = Inertia();
        finalpert = p;

        if (IncEigs <= 0)
            return i + 1;
        if (i == 0)
            p *= incpurt0;
        else
            p *= incpurt;
        p -= finalpert;
    }
    if (settings_.print_level_ < 3)
        fmt::print(fmt::fg(fmt::color::yellow),
                   "Warning: Inertia correction exhausted ({} perturbation attempts, "
                   "{} excess eigenvalue(s))\n",
                   settings_.max_refac_, IncEigs);
    return settings_.max_refac_;
}

Eigen::VectorXd tycho::solvers::PSIOPT::alg_impl(AlgorithmModes algmode, BarrierModes barmode,
                                                 LineSearchModes lsmode, double obj_scale,
                                                 double MuI, Eigen::Ref<Eigen::VectorXd> xsl) {
    Eigen::VectorXd XSL = xsl;
    Eigen::VectorXd RHS(this->kkt_dim_);
    Eigen::VectorXd DXSL(this->kkt_dim_);
    Eigen::VectorXd RHS2(this->kkt_dim_);
    Eigen::VectorXd PGX(this->primal_vars_);

    // Per-phase: print_exit_stats reports this phase's factorization status, so
    // a status left over from an earlier phase in the sequence must not leak in.
    this->result_.last_kkt_info_ = Eigen::Success;

    Eigen::VectorXd Temp(this->kkt_dim_);

    // Reserve-once: bind to persistent member scratch instead of a fresh empty
    // local, so repeated alg_impl calls (one per phase) don't re-allocate from
    // scratch when settings_.return_best_ is enabled.
    Eigen::VectorXd &BestXSL = this->best_xsl_scratch_;
    Eigen::VectorXd &BestRHS = this->best_rhs_scratch_;
    double BestCriteriaVal = 1.0e10;
    int BestIter = 0;

    double mu = MuI;

    // Create KKTVector views over the working vectors
    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_rhs = kkt_view(RHS);
    // v_dxsl / v_temp: the former view over DXSL and the PROBE-predictor view
    // (Temp = XSL + DXSL -> mpc_mu) moved into ClassicAdaptiveGovernor, which
    // rebuilds them internally from the raw DXSL/Temp blocks; no alg_impl
    // caller remains.

    // References-only view of this solver, passed to the
    // step-length mechanism (mechanism_) at its call sites below. Built once
    // here (dims/settings/scratch are stable for the solve); it must not
    // outlive this alg_impl frame or the PSIOPT members it references.
    SolverContext ctx{this->nlp_.get(),         this->kkt_sol_,          this->settings_,
                      this->primal_vars_,       this->slack_vars_,       this->equal_cons_,
                      this->inequal_cons_,      this->kkt_dim_,          this->stli_scratch_,
                      this->hp_scratch_,        this->best_xsl_scratch_, this->best_rhs_scratch_,
                      this->restoration_.get(), &this->eval_error_log_};

    // Windowed sustained-worsening detector for the feasibility-only stage (see
    // feasibility_stall.h). Consulted only when a restoration strategy is
    // configured and inactive; the default path never observes it. Per-phase
    // lifetime, like every other alg_impl-local mode state.
    FeasibilityStallDetector feas_stall;

    tycho::utils::Timer Runtimer;
    tycho::utils::Timer Funtimer;
    tycho::utils::Timer QPtimer;
    tycho::utils::Timer CBtimer; // Callback time falls into misc_time_ implicitly (misc = total -
                                 // pre - kkt - func - print)
    tycho::utils::Timer Printtimer;

    double Hpert0 = settings_.delta_h_;
    // Persistent primal base shift ρ_k for the proximal_regularization inertia
    // mode. Per-phase lifetime, exactly like the ladder's Hpert0/FirstPert
    // warm-start memory (one alg_impl call = one phase; NOT reset at restoration
    // episode entry/exit, which run in-phase). Initialized at the
    // Cipolla–Gondzio floor; dead (never read) on the classic path.
    double rho_k = tycho::solvers::kProxRegFloor;
    // Last shifts actually applied at a factorization this phase (sentinel -1
    // until the first factorized iteration). The convergence probe appended to
    // `iters` on a converged exit never factorizes, so the trailing history
    // entry does NOT carry the final applied shifts -- these locals do.
    double last_prox_primal = -1.0;
    double last_prox_dual = -1.0;
    std::vector<IterateInfo> iters;
    iters.reserve(settings_.max_iters_);
    ConvergenceFlags ExitCode = ConvergenceFlags::NOTCONVERGED;
    bool FirstPert = true;

    // Feasibility-restoration obj_val_ override. Dead on the default path
    // (never set true unless restoration_ is non-null). Two terminal restoration
    // exits leave the loop with the last-filled iters.back().prim_obj_ still at
    // φ_prox (the proximal objective substituted while restoration was active):
    // the in-loop "converged to a locally infeasible point" break, and the
    // post-loop max-iters/DIVERGING-while-active catch-all. Both record the true
    // objective (re-evaluated via build_restoration_exit_measures) here so the
    // unconditional algmode==OPT obj_val_ assignment below the main loop can be
    // corrected afterward, once, in a single place.
    bool restoration_was_active = false;
    double restoration_true_obj = 0.0;

    Runtimer.start();
    for (int i = 0; i < settings_.max_iters_; i++) {
        IterateInfo Citer;
        Citer.iter = i;

        double avgcomp = 0;
        double mincomp = 0;
        double maxcomp = 0;
        double alpha = 1.0;
        double alphap = 1.0;
        double alphad = 1.0;

        RHS.setZero();
        PGX.setZero();
        double prim_obj = 0;
        double barr_obj = 0;

        // Evaluate NLP and build barrier terms
        Funtimer.start();

        this->eval_nlp(algmode, obj_scale, XSL, prim_obj, PGX, RHS, this->kkt_sol_.get_matrix(),
                       mu);

        if (this->inequal_cons_ > 0) {
            // apply_reset_slacks completes the raw inequality residual g(x) into the
            // slack form g(x)+s (and resets negative slacks). The nested restoration
            // eval seam has ALREADY formed the condensed residual r̃ from the
            // slack-completed residual and written it into the iq RHS; running the
            // completion again here would add the slack a second time or (when r̃ is
            // negative) zero the row outright, destroying the elastic Newton
            // direction. Skip it while a nested restoration phase is active — the
            // slacks stay strictly positive through the elastic fraction-to-boundary
            // caps, so barrier_hessian/complementarity still consume valid slacks.
            const bool nested_resto_active = this->restoration_ &&
                                             this->restoration_->is_active() &&
                                             this->restoration_->is_nested();
            if (!nested_resto_active)
                this->apply_reset_slacks(v_xsl.slacks(), v_rhs.iq_cons());
            this->barrier_hessian(this->kkt_sol_.get_matrix(), v_xsl.slacks(), v_xsl.iq_lmults(),
                                  mu);
            this->complementarity(v_xsl.slacks(), v_xsl.iq_lmults(), avgcomp, mincomp, maxcomp);
            // avgcomp/mincomp feed the free-mode barrier-parameter oracle
            // (update_barrier below). This augmentation is load-bearing on every
            // path that reaches that oracle while a nested phase is active: the
            // monitored governor still consults it in its own guarded free mode,
            // where omitting the elastic pairs would let the oracle collapse mu.
            // Under a free (classic_adaptive) governor the in-phase update is
            // instead routed to update_barrier_monotone (which ignores these
            // aggregates); there the elastic complementarity that matters is the
            // copy folded into barr_inf_ via fill_residual_info, read by the
            // Fiacco-McCormick subproblem-convergence gate and converge_check. Dead
            // no-op off the nested path (original aggregates returned untouched).
            this->augment_complementarity_nested(avgcomp, mincomp, maxcomp,
                                                 static_cast<int>(v_xsl.slacks().size()));
        }

        Funtimer.stop();
        if (this->early_callback_enabled_) {
            CBtimer.start();
            this->early_callback_(i, obj_scale, XSL, prim_obj, PGX, RHS,
                                  this->kkt_sol_.get_matrix());
            CBtimer.stop();
        }

        // Assemble KKT gradient and factorize with inertia correction
        QPtimer.start();
        v_rhs.prim_grad() += PGX;

        // Check convergence before factorizing the converged iterate: every
        // residual converge_check() consumes is now
        // fully determined -- kkt_inf_ reads prim_grad() (just updated above; the
        // barrier writes later this iteration target the *disjoint* dual_grad()
        // block, see psiopt.h:472-473 for prim_grad()/dual_grad()'s segment
        // boundaries), econ_inf_/icon_inf_ read eq_cons()/iq_cons() (set by
        // eval_nlp + apply_reset_slacks above, before this point), and barr_inf_ is
        // the complementarity(slacks, iq_lmults) computed above from XSL, which
        // itself is not written again until `XSL += alpha*DXSL` below this loop's
        // exit check -- i.e. strictly after this iteration's earliest possible
        // break. So converge_check() sees byte-identical residual inputs whether
        // it runs here or at its original post-line-search position: firing here
        // cannot change WHICH verdict is reached, it only skips the
        // factor+solve+line-search that a CONVERGED/ACCEPTABLE/DIVERGING iterate
        // never needed (that work only ever feeds `XSL += alpha*DXSL`, which those
        // three exit codes never reach).
        this->fill_residual_info(v_xsl, v_rhs, prim_obj, Citer);
        iters.push_back(Citer);
        ConvergenceFlags PreExitCode = this->converge_check(iters);

        // Feasibility-restoration mode handling. Dead on the default path
        // (restoration_ is null). While active, the KKT gradient/objective this
        // iterate was evaluated under is the PROXIMAL one (the eval seam swapped
        // it), so a converge_check "converged/acceptable" verdict here means the
        // PROXIMAL subproblem converged, NOT that the true NLP is solved — it
        // must never be reported as a solve. Intercept before the early-exit
        // block below.
        if (this->restoration_ && this->restoration_->is_active()) {
            if (this->restoration_->is_nested()) {
                // Nested l1 restoration exit tests (Ipopt RestoConvCheck structure):
                // first-iteration guard, then the per-iteration κ_resto ratchet, then
                // the acceptance-strategy exit test; all three pass → the full
                // multiplier re-entry sequence. θ_orig is the ORIGINAL-problem
                // infeasibility at the current point. The eval seam replaced the RHS
                // constraint rows with the condensed r̃, so the raw residuals come
                // from the seam's saved copies (resto_ec_/ic_scratch_, populated this
                // same iteration before the r̃ overwrite) — no extra NLP evaluation.
                double theta_orig = 0.0;
                if (this->equal_cons_ > 0)
                    theta_orig = std::max(
                        theta_orig, this->resto_ec_scratch_.template lpNorm<Eigen::Infinity>());
                if (this->inequal_cons_ > 0)
                    theta_orig = std::max(
                        theta_orig, this->resto_ic_scratch_.template lpNorm<Eigen::Infinity>());

                ProgressMeasures cur;
                cur.infeasibility = theta_orig; // TRUE original-problem infeasibility.
                cur.objective =
                    prim_obj; // = φ_l1 while active; the exit test reads only infeasibility.
                cur.auxiliary = barr_obj;
                const double resto_failure_threshold =
                    kRestoFailureFeasibilityFactor * settings_.econ_tol_;

                if (this->resto_first_iter_) {
                    // First phase iteration: take at least one step before any exit
                    // test (Ipopt first_resto_iter_). Re-seed the ratchet baseline.
                    this->resto_first_iter_ = false;
                    this->resto_theta_orig_prev_ = theta_orig;
                    this->restoration_->note_iteration();
                } else if (PreExitCode == ConvergenceFlags::CONVERGED ||
                           PreExitCode == ConvergenceFlags::ACCEPTABLE) {
                    // Condensed subproblem converged/stalled: near-feasible → exit via
                    // the full multiplier re-entry sequence; still infeasible → the
                    // problem is locally infeasible (same classification and failure
                    // threshold as the proximal mode).
                    this->resto_theta_orig_prev_ = theta_orig;
                    if (theta_orig <= resto_failure_threshold) {
                        this->restoration_->note_iteration();
                        this->exit_feasibility_restoration_nested(XSL, obj_scale, theta_orig,
                                                                  barr_obj, mu);
                        iters.pop_back();
                        QPtimer.stop();
                        continue;
                    }
                    // Locally infeasible: tear down (restore μ, reset governor) and
                    // stop NOT-converged. No multiplier re-entry — the phase failed.
                    {
                        ProgressMeasures exit_measures = this->build_restoration_exit_measures(
                            obj_scale, theta_orig, v_xsl.primals(), barr_obj);
                        restoration_was_active = true;
                        this->restoration_->note_iteration();
                        mu = this->stashed_mu_;
                        this->governor_->reset();
                        this->restoration_->exit_restoration();
                        this->acceptance_->notify_switch_to_optimality(exit_measures);
                        this->recovery_->reset();
                    }
                    iters.back().mu_ = mu;
                    QPtimer.stop();
                    ExitCode = ConvergenceFlags::NOTCONVERGED;
                    if (settings_.return_best_) {
                        XSL = BestXSL;
                        RHS = BestRHS;
                    }
                    restoration_true_obj = 0.0;
                    this->nlp_->eval_obj(obj_scale, v_xsl.primals(), restoration_true_obj);
                    this->result_.converge_flag_ = ExitCode;
                    if (settings_.print_level_ == 0) {
                        Printtimer.start();
                        this->print_last_iterate(iters);
                        Printtimer.stop();
                    }
                    if (settings_.print_level_ < 3)
                        fmt::print(fmt::fg(fmt::color::yellow),
                                   "Feasibility restoration converged to a locally infeasible "
                                   "point (infeasibility {:.3e} > {:.3e}); stopping "
                                   "(not converged).\n",
                                   theta_orig, resto_failure_threshold);
                    break;
                } else if (PreExitCode == ConvergenceFlags::NOTCONVERGED) {
                    // κ_resto ratchet (per-iteration, vs the previous iteration's
                    // value) AND the acceptance-strategy exit test (vs the frozen
                    // entry reference): both must pass to leave the phase. The ratchet
                    // reads resto_theta_orig_prev_ BEFORE it is updated this iteration.
                    const bool ratchet = this->resto_ratchet_passes(theta_orig);
                    this->resto_theta_orig_prev_ = theta_orig;
                    if (ratchet && this->acceptance_->is_infeasibility_sufficiently_reduced(
                                       this->restoration_->reference(), cur)) {
                        this->restoration_->note_iteration();
                        this->exit_feasibility_restoration_nested(XSL, obj_scale, theta_orig,
                                                                  barr_obj, mu);
                        iters.pop_back();
                        QPtimer.stop();
                        continue;
                    }
                    // Staying in restoration mode this iteration.
                    this->restoration_->note_iteration();
                }
                // DIVERGING while active falls through to the early-exit block below;
                // the post-loop teardown clears restoration before alg_impl returns.
            } else {
                ProgressMeasures cur;
                cur.infeasibility = v_rhs.all_cons().template lpNorm<1>();
                cur.objective = prim_obj; // = φ_prox while active (set by the eval seam).
                cur.auxiliary = barr_obj;
                // Stall classification uses the FAILURE threshold (1e2 · tol), not
                // the far-stricter entry guard: a proximal-subproblem stall at
                // violation <= 1e2 · tol is the recoverable "reached a
                // (near-)feasible point" outcome, and only beyond it is local
                // infeasibility declared (see kRestoFailureFeasibilityFactor's
                // citation block in proximal_restoration.h).
                const double resto_failure_threshold =
                    kRestoFailureFeasibilityFactor * settings_.econ_tol_;

                if (PreExitCode == ConvergenceFlags::CONVERGED ||
                    PreExitCode == ConvergenceFlags::ACCEPTABLE) {
                    if (cur.infeasibility <= resto_failure_threshold) {
                        // Proximal subproblem converged AND the true constraints are
                        // near-feasible: leave restoration and resume the true
                        // objective. The same iterate is re-evaluated in optimality
                        // mode next iteration (the per-phase budget prevents cycling).
                        // notify_switch_to_optimality augments this pair into the
                        // restored OPTIMALITY filter/funnel, whose accumulated pairs
                        // are all true-objective-scale -- cur.objective (= φ_prox) is
                        // never valid there, so the true objective is re-evaluated at
                        // the live primals via the shared exit-measures helper.
                        // Count this exit iteration in the in-mode total (it was a
                        // feasibility-mode iterate; the stay-in-mode note_iteration
                        // below only counts iterations that keep going).
                        this->restoration_->note_iteration();
                        this->restoration_->exit_restoration();
                        this->acceptance_->notify_switch_to_optimality(
                            this->build_restoration_exit_measures(obj_scale, cur.infeasibility,
                                                                  v_xsl.primals(), barr_obj));
                        // Reset the recovery chain across the mode switch (see the
                        // entry-side rationale at the kSwitchToFeasibility case): the
                        // watchdog's objective-scale-bound snapshot/counters must not
                        // survive back into the optimality phase. Once per transition.
                        this->recovery_->reset();
                        iters.pop_back();
                        QPtimer.stop();
                        continue;
                    }
                    // Proximal subproblem converged/stalled at a still-infeasible
                    // point: the problem is locally infeasible (Ipopt's
                    // restoration-convergence failure classification). Tear down
                    // restoration BEFORE returning so the phase-boundary reset() sees
                    // optimality mode, then stop with the NOT-converged verdict.
                    {
                        // The notify measures record the point restoration exited
                        // (the live iterate) — the filter augment describes the
                        // exit point itself, independent of the return_best_
                        // reporting substitution below.
                        ProgressMeasures exit_measures = this->build_restoration_exit_measures(
                            obj_scale, cur.infeasibility, v_xsl.primals(), barr_obj);
                        restoration_was_active = true;
                        // Count this exit iteration in the in-mode total (see the
                        // near-feasible exit above).
                        this->restoration_->note_iteration();
                        this->restoration_->exit_restoration();
                        this->acceptance_->notify_switch_to_optimality(exit_measures);
                        // Reset the recovery chain across the mode switch (see the
                        // kSwitchToFeasibility entry rationale). Once per transition.
                        this->recovery_->reset();
                    }
                    iters.back().mu_ = mu;
                    QPtimer.stop();
                    ExitCode = ConvergenceFlags::NOTCONVERGED;
                    if (settings_.return_best_) {
                        XSL = BestXSL;
                        RHS = BestRHS;
                    }
                    // obj_val_ must describe the RETURNED primals: evaluate after
                    // the return_best_ substitution above (which may have replaced
                    // XSL), unlike the notify measures, which record the exit
                    // point. With return_best_ off the two evaluations coincide.
                    restoration_true_obj = 0.0;
                    this->nlp_->eval_obj(obj_scale, v_xsl.primals(), restoration_true_obj);
                    this->result_.converge_flag_ = ExitCode;
                    if (settings_.print_level_ == 0) {
                        Printtimer.start();
                        this->print_last_iterate(iters);
                        Printtimer.stop();
                    }
                    if (settings_.print_level_ < 3)
                        fmt::print(fmt::fg(fmt::color::yellow),
                                   "Feasibility restoration converged to a locally infeasible "
                                   "point (infeasibility {:.3e} > {:.3e}); stopping "
                                   "(not converged).\n",
                                   cur.infeasibility, resto_failure_threshold);
                    break;
                }

                if (PreExitCode == ConvergenceFlags::NOTCONVERGED) {
                    // Subproblem not converged: has infeasibility fallen enough,
                    // relative to the restoration entry point, to leave restoration?
                    // Runs from an accepted feasibility-mode iterate; at the entry
                    // point cur == reference so this cannot fire (θ_trial == θ_ref).
                    if (this->acceptance_->is_infeasibility_sufficiently_reduced(
                            this->restoration_->reference(), cur)) {
                        // Count this exit iteration in the in-mode total (the
                        // stay-in-mode note_iteration below is skipped on exit).
                        this->restoration_->note_iteration();
                        this->restoration_->exit_restoration();
                        this->acceptance_->notify_switch_to_optimality(
                            this->build_restoration_exit_measures(obj_scale, cur.infeasibility,
                                                                  v_xsl.primals(), barr_obj));
                        // Reset the recovery chain across the mode switch (see the
                        // kSwitchToFeasibility entry rationale). Once per transition.
                        this->recovery_->reset();
                        iters.pop_back();
                        QPtimer.stop();
                        continue;
                    }
                    // Staying in restoration mode this iteration.
                    this->restoration_->note_iteration();
                }
                // DIVERGING while active falls through to the early-exit block below;
                // the post-loop teardown clears restoration before alg_impl returns.
            }
        }

        if (PreExitCode == ConvergenceFlags::CONVERGED ||
            PreExitCode == ConvergenceFlags::ACCEPTABLE ||
            PreExitCode == ConvergenceFlags::DIVERGING) {
            // Converged/acceptable/(residual-)diverging before ever factorizing
            // this iterate. mu_ is set below (the loop's current barrier
            // parameter -- the value this iterate was evaluated under -- is
            // meaningful and display-only). The remaining un-set fields --
            // barr_obj_, alpha_p/d/t_,
            // h_facs_, h_pert_/_cum_, p_pivots_, ls_iters_, merit_val_ -- stay at
            // IterateInfo's fresh per-iteration defaults (0, or 1.0 for the
            // alphas): there is no factorization, barrier update, or line search to
            // report. This is the one accepted, cosmetic change versus today: the
            // final printed row's Mu/Bar Obj/AlphaP/AlphaD/AlphaT/LS/PPS/HF/HPert
            // columns read as defaults rather than a wasted factorization's real
            // values. Nothing that feeds the verdict, the returned primals, or
            // print_exit_stats reads any of those fields (converge_check() and
            // print_exit_stats read only prim_obj_/kkt_inf_/barr_inf_/econ_inf_/
            // icon_inf_, and return_best_'s criteria switch below reads only
            // econ_inf_/icon_inf_/kkt_inf_/prim_obj_ -- all of which
            // fill_residual_info() already set correctly above).
            iters.back().mu_ = mu;
            QPtimer.stop();
            ExitCode = PreExitCode;

            // Suspend best-iterate tracking while restoration is active (dead on
            // the default path: restoration_ null). A feasibility-mode iterate's
            // prim_obj/kkt_inf are proximal-scale and must not compete with
            // true-objective iterates for "best" — otherwise return_best_ could
            // report a mixed-scale winner. Only DIVERGING-while-active reaches
            // this early-exit block (CONVERGED/ACCEPTABLE are intercepted by the
            // restoration handling above).
            if (settings_.return_best_ && !(this->restoration_ && this->restoration_->is_active())) {
                double critval;
                switch (settings_.best_criteria_) {
                case BestCriteriaModes::ECONS:
                    critval = iters.back().econ_inf_;
                    break;
                case BestCriteriaModes::ICONS:
                    critval = iters.back().icon_inf_;
                    break;
                case BestCriteriaModes::KKT:
                    critval = iters.back().kkt_inf_;
                    break;
                case BestCriteriaModes::OBJ:
                    critval = iters.back().prim_obj_;
                    break;
                default:
                    throw std::invalid_argument("Unknown BestCriteriaModes");
                }
                if (critval <= BestCriteriaVal || i == 0) {
                    BestCriteriaVal = critval;
                    BestXSL = XSL;
                    BestRHS = RHS;
                    BestIter = i;
                }
            }

            if (this->late_callback_enabled_) {
                CBtimer.start();
                this->late_callback_(iters.back(), XSL, RHS);
                CBtimer.stop();
            }

            if (settings_.print_level_ == 0) {
                Printtimer.start();
                this->print_last_iterate(iters);
                Printtimer.stop();
            }

            if (ExitCode != ConvergenceFlags::CONVERGED && settings_.return_best_) {
                XSL = BestXSL;
                RHS = BestRHS;
            }

            this->result_.converge_flag_ = ExitCode;
            break;
        }

        // Set only by the feasibility-stage stall dispatch below, when the stage
        // is stalled, its restoration budget is spent, and recovery bought it
        // nothing: it forces this iteration to be the last one of the phase,
        // without touching the convergence verdict. Read once, next to the
        // exit_at_acceptable upgrade at the bottom of the loop. Provably false
        // whenever restoration is off — the only write is inside the guarded
        // block.
        bool exit_stage_stalled = false;

        // Feasibility-stage stall detection. The zero-objective stage accepts
        // every step under the default no-line-search stage configuration,
        // so the rejected-trial recovery gate does not dispatch
        // restoration from here; this is the missing signal (see
        // feasibility_stall.h). Guarded so the default path (restoration off)
        // performs no work at all, and an active restoration episode is left
        // to run its own course — the detector neither observes nor exits
        // while an episode is running. The detector is consulted exactly once
        // per iteration, and a stall only ends the phase once recovery has
        // been given its chance and has bought nothing.
        //
        // What the detector certifies is SUSTAINED WORSENING: a violation
        // sitting at least 25% above the stage's own best for a full window of
        // consecutive iterations. Nothing below dispatches into a plateaued or
        // an improving stage — those burn their iteration budget exactly as
        // they did before this seam existed. That is deliberate: worsening is
        // the only class in which a dispatched episode has measured value, and
        // episodes injected into quietly succeeding stages measurably cost
        // verdicts. The outcomes:
        //
        //   1. Not worsening: nothing happens.
        //   2. Worsening and entry permitted: enter restoration exactly as the
        //      optimize path's switch does, discard this iteration's
        //      residual-only history entry, and re-enter the loop so the next
        //      evaluation runs the restoration subproblem; the window re-arms
        //      so the resumed stage restarts it from the post-restoration
        //      point, while the violation at THIS dispatch is recorded as the
        //      yardstick for outcomes 3b/3c below.
        //   3. Worsening and entry refused. What that means depends on where the
        //      stage is:
        //      a. Already near-feasible: the constraints are at their floor and
        //         the barrier residual is still grinding down with mu, so the
        //         violation cannot improve and, once it has drifted up off that
        //         floor, the detector will keep firing every iteration.
        //         This is an endgame, not a stall — do nothing and let the
        //         stage finish. The repeated no-op is deliberate; per iteration
        //         it costs the L1 norm, the detector's observe(), the virtual
        //         entry_permitted() and the near_feasible() test, and a couple
        //         of compares — negligible beside the factorization.
        //      b. Still below the violation at its LAST restoration dispatch:
        //         the stage has gained ground since recovery last handed it
        //         back and is still consuming those gains, so it is winning
        //         slowly even though the per-phase entry budget is spent. Do
        //         nothing and let it run.
        //      c. Otherwise: the budget is spent AND the stage is no better off
        //         than where recovery last left it, so recovery has proven it
        //         cannot help — no mechanism left to consult and no progress to
        //         protect. End the phase instead of burning the rest of the
        //         iteration budget. The iteration finishes its normal
        //         bookkeeping and the loop exits through the standard teardown,
        //         so converge_check reports the honest verdict and (with
        //         return_best_ on) the exit hands back the best-seen iterate,
        //         exactly like every other non-CONVERGED exit. In a multi-phase
        //         sequence the next phase resumes from there. The detector is
        //         deliberately NOT re-armed: the phase is ending.
        //      Measuring against the LAST dispatch rather than the first is
        //      what makes 3b/3c ask the right question: has the stage gained
        //      anything since recovery last handed it back? The rule composes
        //      with the detector's worsening test. After an episode the window
        //      re-arms against the post-restoration point, so the only stage
        //      that can reach 3b/3c at all is one that went on worsening from
        //      there: a stage that levelled off after its episode, or that is
        //      crawling down from it, never fires again and runs on untouched.
        //      Of the stages that do fire again, one still under the violation
        //      recorded at that dispatch is consuming ground the episode bought
        //      and keeps running (3b), while one that has climbed back to or
        //      above where recovery found it has nothing left to show for the
        //      episode and ends (3c). The graceful end therefore reaches
        //      exactly the class the dispatch does — a stage that keeps getting
        //      worse — and no other.
        //      A phase whose entry was refused from the very start never
        //      recorded a dispatch at all, so 3b's comparison against infinity
        //      holds and the phase never ends here. For a near-feasible stage
        //      that is 3a anyway; for max_feas_rest_ == 0 it means a user who
        //      turned restoration episodes off keeps the pre-existing
        //      burn-the-budget behaviour, which is the right default — this
        //      exit exists to stop a stage that recovery could not save, and
        //      recovery was never allowed to try.
        if ((algmode == AlgorithmModes::SOE || algmode == AlgorithmModes::OPTNO) &&
            this->restoration_ && !this->restoration_->is_active()) {
            const int ncons_fs = this->equal_cons_ + this->inequal_cons_;
            const double theta_fs = RHS.tail(ncons_fs).template lpNorm<1>();
            if (feas_stall.observe(theta_fs)) {
                if (this->restoration_->entry_permitted(theta_fs, ctx)) {
                    // Entry measures are (theta, 0, 0) here: prim_obj and barr_obj are
                    // still their pre-factorization 0.0 initializers (eval_soe/eval_kkt_no
                    // never write the objective, and the barrier objective is computed
                    // further down), matching the other pre-factorization restoration
                    // seams above and unlike the post-line-search seams, which pass a
                    // live barrier objective.
                    feas_stall.note_dispatch(theta_fs);
                    this->enter_feasibility_restoration(XSL, RHS, prim_obj, barr_obj, mu);
                    feas_stall.reset_window();
                    iters.pop_back();
                    QPtimer.stop();
                    continue;
                }
                // Measured against the same rounding-noise floor the detector
                // itself uses, so an ulp of drift at a plateau does not read as
                // ground gained. Vacuously true (infinity reference) when the
                // phase never dispatched.
                const bool net_progress = theta_fs < (1.0 - kFeasStallMinRelImprovement) *
                                                         feas_stall.theta_at_last_dispatch_;
                if (!this->restoration_->near_feasible(theta_fs, ctx) && !net_progress) {
                    exit_stage_stalled = true;
                    if (settings_.print_level_ < 3)
                        fmt::print(fmt::fg(fmt::color::yellow),
                                   "Feasibility phase stalled with its restoration budget "
                                   "exhausted and no relative improvement over the violation "
                                   "at its last restoration entry (infeasibility {:.3e}, "
                                   "{:.3e} at that entry); ending the phase — the convergence "
                                   "check still reports the final verdict, which may be "
                                   "acceptable.\n",
                                   theta_fs, feas_stall.theta_at_last_dispatch_);
                }
            }
        }

        iters.pop_back();

        double nhpert = 0;
        // Display-only accumulator: the cumulative inertia-perturbation
        // total for this iteration's factor_impl() call, for the HPert table column.
        // Kept fully separate from nhpert (the last delta), which alone feeds the
        // Hpert0 warm-start below -- see the comment at that read site.
        double nhpert_cum = 0;
        double Incr = settings_.incr_h_;
        double Incr2 = settings_.incr_h_;
        if (FirstPert)
            Incr2 *= settings_.incr_h_;
        // Cycling heuristic: if the last 4 consecutive iterations all required
        // Hessian perturbation (h_facs_ > 0), skip the zero-perturbation attempt
        // to avoid wasted factorizations when the problem is persistently
        // near-singular. The (i*3)%4 != 0 condition samples 3/4 of iterations,
        // periodically re-probing for recovered inertia.
        bool Zfac = true;
        if (settings_.fast_factor_alg_ && i > 6 && ((i * 3) % 4) != 0) {
            bool cycling = true;
            for (int j = 0; j < 4; j++) {
                int ns = iters[iters.size() - 1 - j].h_facs_;
                if (ns == 0) {
                    cycling = false;
                    break;
                }
            }
            Zfac = !cycling;
        }

        // Proximal primal-dual regularization base shifts for this iteration
        // (0.0 on the classic path, which leaves factor_impl byte-identical).
        // base_prox = ρ_k, the persistent primal shift. dual_shift = δ_c, the
        // barrier-scaled dual shift, computed against the same μ the KKT assembly
        // used above -- SUPPRESSED to 0.0 while a nested l1 restoration phase is
        // active, because the elastic pivots already own the constraint-row
        // diagonals (~1/μ) and the condensed elastic step recovery assumes the
        // (y,y) diagonal equals the elastic pivot exactly (see
        // globalization/inertia_regularization.h). The proximal mode-switch
        // restoration touches only the primal diagonal, so δ_c stays on under it.
        double base_prox = 0.0;
        double dual_shift = 0.0;
        if (settings_.inertia_mode_ == InertiaModes::proximal_regularization) {
            base_prox = rho_k;
            bool nested_active = this->restoration_ && this->restoration_->is_active() &&
                                 this->restoration_->is_nested();
            dual_shift = nested_active ? 0.0 : tycho::solvers::dual_regularization(mu);
        }

        Citer.h_facs_ = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert, nhpert_cum,
                                          base_prox, dual_shift);
        // Note: if factor_impl exhausted all perturbation attempts (h_facs_ == max_refac_),
        // we proceed rather than aborting. The line search evaluates actual function values
        // and will reject truly bad steps by reducing alpha. Forcing GoodStep=false here
        // would be an algorithmic change that could break existing convergence behavior.

        if (Citer.h_facs_ > 0) {
            // Hpert0 warm-start MUST keep consuming nhpert (the last perturbation
            // DELTA) byte-identically -- do not substitute nhpert_cum here (see
            // the display-only-accumulator comment above nhpert_cum's declaration).
            Hpert0 = std::max(settings_.delta_h_, nhpert * settings_.decr_h_);
            FirstPert = false;
        }
        Citer.h_pert_ = nhpert;
        Citer.h_pert_cum_ = nhpert_cum;

        if (settings_.inertia_mode_ == InertiaModes::proximal_regularization) {
            // Record the shifts applied this iteration (sentinel -1 on the
            // classic path), then decay the persistent primal base shift toward
            // its floor. If the base attempt sufficed (h_facs_ == 0) the implicit
            // trust-region radius grows while curvature stays good; if the ladder
            // fired, the decayed total successful shift (ρ_k plus the last ladder
            // delta nhpert) persists into the next iteration. The Hpert0/FirstPert
            // warm-start above is unchanged -- ρ_k is a separate, additional state.
            Citer.prox_reg_primal_ = base_prox;
            Citer.prox_reg_dual_ = dual_shift;
            last_prox_primal = base_prox;
            last_prox_dual = dual_shift;
            double applied_total = (Citer.h_facs_ > 0) ? (rho_k + nhpert) : rho_k;
            rho_k = tycho::solvers::prox_reg_decay(applied_total, settings_.decr_h_);
        }

        // Update barrier parameter and compute search direction. The whole
        // PROBE/LOQO switch + common clamp/objective/gradient tail is now
        // ClassicAdaptiveGovernor::update_barrier; the
        // `if (inequal_cons_ > 0)` guard stays here, exactly as the block was
        // guarded before extraction, so the governor is invoked only when there
        // are inequality constraints (barrier terms). The PROBE predictor's KKT
        // solve moves INTO the governor; the REAL step solve below (a distinct
        // second solve) stays here. avgcomp/mincomp feed the mu oracles;
        // *mechanism_ lets the PROBE predictor reuse the step-scaling. Citer
        // (this iterate's residuals, filled by the convergence check above) was
        // popped back off `iters` at the continuing-path pop above -- it is not
        // re-appended until after the line search below -- so it is passed
        // explicitly rather than through `iters` (whose back(), here, would be
        // the PREVIOUS iterate, and is empty outright on iteration 0); a
        // monitored free<->monotone governor reads Citer's residuals to decide
        // the free<->monotone switch. `mu_event` is its out-signal, initialized
        // false each iteration so the classic governor (which never writes it)
        // leaves the reset below dead — bit-identical on the default path.
        bool mu_event = false;
        // The barrier parameter this iteration's KKT system and the condensed
        // elastic system (r̃, the resto gradient, the primal-diagonal Hessian
        // piece) were all built under is `mu` at eval time. The governor may
        // advance `mu` below (only when inequal_cons_ > 0). While a NESTED
        // restoration phase is active, the recovered elastic steps and the resto
        // trial objective MUST use that same eval-time barrier parameter, or the
        // back-substituted (n,p,z) steps are inconsistent with the KKT solve
        // (the eliminated rows' RHS r̃ was formed at eval_mu). We therefore hold
        // the resto algebra at eval_mu this iteration and let the governor's
        // advanced mu take effect at the NEXT iteration's eval — mirroring the
        // default path's predictor-corrector μ discipline (the barrier Hessian is
        // assembled at eval_mu, the barrier gradient/objective the governor
        // writes are at the advanced μ). On the default and proximal paths
        // step_mu == mu (post-update), so every downstream FP op is byte-identical
        // when restoration is off or non-nested.
        const double eval_mu = mu;
        // nested_active is computed BEFORE the barrier update so the update can
        // route to the monotone schedule. Provably false on the default path
        // (restoration_ null → short-circuit); the value is the same one the
        // step_mu select below already needed, only hoisted, so no FP changes.
        const bool nested_active = this->restoration_ && this->restoration_->is_active() &&
                                   this->restoration_->is_nested();
        // Force the monotone in-phase barrier schedule only for a governor that
        // does not supply its own safeguard (the free-mode classic_adaptive
        // governor). The monitored governor already forces a safeguarded
        // Fiacco-McCormick decrease in its own update_barrier, so it drives the
        // in-phase update itself — overlaying a second, differently anchored
        // monotone schedule would perturb its established convergence.
        const bool force_monotone_barrier =
            nested_active && !governor_->provides_restoration_barrier_safeguard();
        if (this->inequal_cons_ > 0) {
            if (force_monotone_barrier) {
                // Ipopt's default restoration mu_strategy is MONOTONE: while the
                // nested l1 phase is active the barrier parameter must follow the
                // safeguarded Fiacco-McCormick ladder anchored at the entry
                // resto_mu, NOT the free-mode oracle. Under a free oracle every
                // complementarity product (including the elastic bound pairs) chases
                // whatever mu the oracle proposes, so any mu is self-consistent and
                // mu collapses to its floor before the elastics shrink — the
                // condensed elastic pivot then explodes and the phase freezes on a
                // wrong-basin l1 minimizer. Routing here makes the free-mode oracles
                // (LOQO and PROBE's predictor) UNREACHABLE for the duration of the
                // phase under a free governor; the configured mode resumes at exit
                // (which restores the stashed mu and resets the governor). See
                // BarrierGovernor::update_barrier_monotone.
                mu = governor_->update_barrier_monotone(mu, XSL, RHS, ctx, barr_obj, Citer,
                                                        mu_event);
            } else {
                mu = governor_->update_barrier(barmode, mu, avgcomp, mincomp, XSL, RHS, DXSL, Temp,
                                               *mechanism_, ctx, barr_obj, Citer, mu_event);
            }
        }
        const double step_mu = nested_active ? eval_mu : mu;

        // Per-barrier-subproblem acceptance reset: when the governor's monotone
        // mode begins a new barrier subproblem (fresh mu), the acceptance
        // strategy's filter/funnel must clear. Placed here — after update_barrier
        // (complementarity -> factor -> update_barrier) and BEFORE the real step
        // solve and line search below — so the reset lands before this
        // iteration's line search consumes the acceptance strategy. Dead on the
        // classic path (mu_event stays false there).
        if (mu_event) {
            this->acceptance_->reset();
        }

        // The REAL step solve (distinct from the PROBE predictor solve, which
        // moved into ClassicAdaptiveGovernor::update_barrier — see its solve-into
        // comment): direct assignment + in-place negate avoids the extra
        // temporary that `-kkt_sol_.solve(RHS)` forces.
        DXSL = this->kkt_sol_.solve(RHS);
        DXSL = -DXSL;
        bool GoodStep = std::isfinite(DXSL.squaredNorm());

        // Nested-restoration elastic step recovery. Dead unless a nested
        // restoration strategy is active. The condensed KKT solved for the
        // constraint-multiplier steps Δy in the DXSL eq/iq blocks; recover the
        // eliminated elastic slack/bound-multiplier steps from them BEFORE the
        // fraction-to-boundary machinery scales those blocks in compute_step. The
        // recovered steps feed the elastic caps consulted inside
        // max_primal_dual_step and are committed by apply_elastic_step below.
        if (nested_active) {
            KKTVector v_dxsl = kkt_view(DXSL);
            this->restoration_->recover_elastic_steps(step_mu, v_xsl.eq_lmults(),
                                                      v_xsl.iq_lmults(), v_dxsl.eq_lmults(),
                                                      v_dxsl.iq_lmults());
        }
        QPtimer.stop();

        // Line search. lsobjscale is hoisted out of the GoodStep block below so
        // the recovery-chain hook can forward the identical merit objective
        // scale (obj_scale * lsobjscale) to its acceptance re-test; its value is
        // a pure select on algmode (0.0 for SOE/OPTNO, else 1.0), so hoisting the
        // declaration does not change the value passed to compute_step.
        //
        // While feasibility restoration is active the user objective must
        // contribute exactly 0.0 to the trial merit (the proximal objective is
        // added instead by the trial seams), so lsobjscale is 0.0 there too. The
        // added disjunct is provably false on the default path (restoration_ is
        // null → short-circuit), so the selected value — and every FP operation
        // downstream — is byte-identical when restoration is off.
        double lsobjscale = (algmode == AlgorithmModes::SOE || algmode == AlgorithmModes::OPTNO ||
                             (this->restoration_ && this->restoration_->is_active()))
                                ? 0.0
                                : 1.0;

        // Trial evaluations from here to the end of the recovery hook may
        // absorb NLP evaluation exceptions; the delta against this snapshot
        // is this iteration's count and drives the un-evaluable-step bypass
        // below.
        const int eval_errs_before = this->eval_error_log_.count_;

        // Set only by the un-evaluable-step bypass below, when the committed
        // iterate already satisfies the acceptable tier: it forces this
        // iteration to be the last one, so the solve reports the acceptable
        // level instead of aborting. Read once, next to the !GoodStep
        // divergence override at the bottom of the loop.
        bool exit_at_acceptable = false;

        Funtimer.start();
        if (GoodStep) {
            // compute_step fuses the fraction-to-boundary scaling (former
            // `if (inequal_cons_ > 0) max_primal_dual_step(...)`, now guarded
            // identically inside compute_step and MUTATING DXSL in place) and
            // the acceptance backtrack on the scaled DXSL. This is the riskiest
            // FP-order seam: negate -> block-scale by
            // alphap/alphad -> `xsl + alpha*dxsl` trial -> `XSL += alpha*DXSL`.
            alpha = mechanism_->compute_step(lsmode, obj_scale * lsobjscale, step_mu, prim_obj,
                                             barr_obj, XSL, DXSL, Temp, RHS, RHS2, *acceptance_,
                                             alphap, alphad, Citer, iters, ctx);

        } else {
            Citer.h_facs_ = -1;
        }

        Funtimer.stop();

        // Recovery-chain hook. This is where a rejected step's
        // recovery gets a say -- SOC -> extended-backtrack -> watchdog-revert
        // dispatch from this point (see rebuild_globalization_components's
        // wiring comment); the feasibility switch remains a future link. The
        // inertia/perturbation ladder above (factor_impl's Zfac cycling +
        // escalation) is a SEPARATE mechanism and stays out of this chain
        // (a future inertia-dispatch stage may wire it in) -- it is NOT
        // invoked or bypassed here.
        //
        // The call is GATED on an actual rejection: should_dispatch_recovery
        // fires the hook only when the line search reported the trial step
        // not-accepted (Citer.accepted_ == false, set by the merit test) AND the
        // KKT step direction was usable (GoodStep). An accepted step -- full or
        // backtracked -- never reaches the hook, and the !GoodStep path (which
        // runs no line search) is excluded too. On the default solve path every
        // step is accepted, so the hook is never invoked at all.
        //
        // By default recovery_ is a NoopRecovery
        // (rebuild_globalization_components() installs it whenever
        // max_soc_ == 0, ls_extended_iters_ == 0, and watchdog_ == false),
        // whose on_step_rejected() unconditionally returns kAcceptAsIs and
        // touches no state, so the kAcceptAsIs branch below is exactly
        // today's control flow (take whatever alpha/DXSL compute_step
        // produced). When any of SOC/extended-backtrack/watchdog are enabled,
        // recovery_ may instead return kRetry after replacing DXSL/alpha (and
        // alphap/alphad) in place with an accepted or reverted step — applied
        // by the XSL += alpha*DXSL commit below.
        //
        // resolved_depth is seeded to the unresolved sentinel and only
        // written by a link that actually resolves the rejection (see
        // recovery_chain.h's kRecoveryDepth* constants); it always ends up
        // valid by the time the histogram below reads it, since every link
        // on every path either writes it or leaves the seeded default.
        if (should_dispatch_recovery(GoodStep, Citer)) {
            int resolved_depth = kRecoveryDepthUnresolved;
            const RecoveryChain::Action recovery_action = this->recovery_->on_step_rejected(
                Citer, iters, ctx, *acceptance_, *mechanism_, lsmode, obj_scale * lsobjscale,
                step_mu, prim_obj, barr_obj, XSL, DXSL, Temp, RHS, RHS2, alpha, alphap, alphad,
                this->result_.soc_steps_taken_, resolved_depth,
                this->result_.watchdog_activations_);
            switch (recovery_action) {
            case RecoveryChain::Action::kAcceptAsIs:
                // Classic ladder-exhaustion fallback: take the step compute_step
                // produced (DXSL/alpha unchanged). EXCEPTION (dead off the nested
                // path): while a nested l1 restoration phase is active and no
                // recovery link resolved the rejection (resolved_depth still the
                // unresolved sentinel — the discriminator FeasibilitySwitchRecovery
                // uses, since a watchdog-resolved kAcceptAsIs stamps its own depth),
                // the second-level fallback re-centers the elastic pairs in closed
                // form at the current phase μ INSTEAD of taking the failed step
                // (disclosure (f) in l1_restoration.h). try_recenter_elastics
                // enforces the one-shot budget: on the first exhaustion of a
                // consecutive-failure run it re-centers, discards the failed step
                // (alpha = 0 no-ops the XSL += alpha*DXSL commit — the re-centered
                // elastics change the NEXT iteration's condensed system), and marks
                // the iterate into the restoration recovery bucket; a second
                // consecutive exhaustion falls through here to accept-as-is. The
                // iteration is already counted in the in-mode total by the
                // top-of-loop stay-in-mode note_iteration(), so it is not
                // re-counted here.
                if (nested_active && resolved_depth == kRecoveryDepthUnresolved &&
                    this->try_recenter_elastics(step_mu)) {
                    alpha = 0.0;
                    resolved_depth = kRecoveryDepthRestoration;
                } else if (this->eval_error_log_.count_ > eval_errs_before) {
                    // Un-evaluable fallback step: at least one trial evaluation
                    // threw during this iteration's acceptance attempts, so
                    // committing the never-evaluated fallback step risks
                    // turning the next iteration's committed-point evaluation
                    // into a fatal error. Never accept it (this deliberately
                    // overrides a watchdog-relaxed acceptance too).
                    //
                    // What happens instead mirrors the reference interior-point
                    // method's handling of a failed line search, in its order:
                    //
                    //   1. If the CURRENT (committed) iterate already satisfies
                    //      the acceptable convergence tier, stop here and report
                    //      the acceptable level rather than aborting. A single
                    //      transient evaluation excursion must not throw away a
                    //      solve that is already at a usable point — the common
                    //      case being a near-feasible warm start, where the
                    //      restoration guard below would refuse entry anyway.
                    //      The failed step is discarded (alpha = 0), the iterate
                    //      stays un-accepted, and the loop finishes this
                    //      iteration's bookkeeping before exiting normally.
                    //   2. Otherwise enter feasibility restoration, when a
                    //      strategy is configured, inactive, and entry-permitted
                    //      — skipping the soft pre-stage, whose trial is the very
                    //      step that could not be evaluated.
                    //   3. Otherwise abort with the latched evaluation error
                    //      wrapped in solver context.
                    //
                    // Citer carries this iterate's residuals here:
                    // fill_residual_info() wrote them from the committed XSL
                    // before the factorization above, and nothing since has
                    // touched XSL (the `XSL += alpha*DXSL` commit is below the
                    // loop's exit check).
                    const int ncons_ue = this->equal_cons_ + this->inequal_cons_;
                    const double violation_ue = RHS.tail(ncons_ue).template lpNorm<1>();
                    if (psiopt_iterate_acceptable(Citer, settings_)) {
                        alpha = 0.0;
                        Citer.accepted_ = false;
                        exit_at_acceptable = true;
                    } else if (this->restoration_ && !this->restoration_->is_active() &&
                               this->restoration_->entry_permitted(violation_ue, ctx)) {
                        this->enter_feasibility_restoration(XSL, RHS, prim_obj, barr_obj, mu);
                        // A stage resumed after an episode restarts its stall window,
                        // and this entry becomes the handback the stall exit measures
                        // net progress against.
                        feas_stall.note_dispatch(violation_ue);
                        feas_stall.reset_window();
                        alpha = 0.0;
                        Citer.accepted_ = false;
                        // Deliberately overwrites resolved_depth on a watchdog-resolved
                        // path too (kRecoveryDepthWatchdog -> kRecoveryDepthRestoration),
                        // so the recovery-depth histogram attributes this iteration to
                        // restoration, not to the watchdog relaxation it superseded.
                        resolved_depth = kRecoveryDepthRestoration;
                    } else {
                        // alpha was reduced once more after the last rejected rung, so
                        // the smallest fraction actually evaluated is alpha * alpha_red_.
                        throw std::runtime_error(fmt::format(
                            "PSIOPT: line search failed at iteration {} because the NLP could "
                            "not be evaluated at the trial steps ({} evaluation failure(s) this "
                            "iteration; smallest trial step fraction attempted {:.3e}). "
                            "Feasibility restoration (restoration_mode) was unavailable to "
                            "recover: not configured, entry refused, or already active. Last "
                            "evaluation error: {}",
                            Citer.iter, this->eval_error_log_.count_ - eval_errs_before,
                            alpha * settings_.alpha_red_, this->eval_error_log_.last_message_));
                    }
                }
                break;
            case RecoveryChain::Action::kRetry:
                // The recovery chain committed a corrected/reverted step into
                // DXSL/alpha.
                break;
            case RecoveryChain::Action::kSwitchToFeasibility: {
                // Enter feasibility-restoration mode. The recovery link only
                // SIGNALS (it mutates nothing); the actual mode entry happens
                // here. Discard the rejected step (alpha = 0 no-ops the
                // XSL += alpha*DXSL commit below). The shared orchestration builds
                // the TRUE-objective (θ, f) entry measures at this point (this
                // iterate was evaluated in optimality mode; restoration begins on
                // the NEXT iteration), dispatches to the proximal or nested entry
                // path (the nested path additionally stashes μ, sets μ ←
                // entry_mu(), resets the governor, and applies the verified
                // multiplier init), notifies the acceptance strategy, and resets
                // the recovery chain (WatchdogRecovery's armed-state/counters and
                // objective-scale-bound revert snapshot must not survive the
                // switch — same precedent as run_phase_sequence()'s per-phase
                // reset). FeasibilitySwitchRecovery is the only link that produces
                // this Action, and only when restoration_ is non-null and inactive
                // (so the calls below are safe).
                this->enter_feasibility_restoration(XSL, RHS, prim_obj, barr_obj, mu);
                // A stage resumed after an episode restarts its stall window, and
                // this entry becomes the handback the stall exit measures net
                // progress against.
                feas_stall.note_dispatch(
                    RHS.tail(this->equal_cons_ + this->inequal_cons_).template lpNorm<1>());
                feas_stall.reset_window();
                alpha = 0.0;
                break;
            }
            case RecoveryChain::Action::kSoftFeasibilityStep: {
                // Nested soft feasibility pre-stage. Before committing to the
                // full l1 restoration phase, try the full fraction-to-boundary
                // step on the current search direction (DXSL already carries the
                // fraction-to-boundary scaling) under a primal-dual-error
                // reduction test. If the trial reduces the primal-dual error,
                // take the full step and stay in the pre-stage (the successive-
                // soft-iteration counter persists in FeasibilitySwitchRecovery;
                // the pre-stage exits when a later iteration's ordinary
                // acceptance test recovers on its own, resetting the counter via
                // notify_step_accepted). Otherwise escalate to the full mode
                // entry here — the same enter_feasibility_restoration the
                // kSwitchToFeasibility case runs, so the acceptance-strategy
                // feasibility notification is issued exactly once, at that entry,
                // and never during the pre-stage. The soft step is an ordinary
                // optimality-phase step (restoration is not active yet), so it is
                // evaluated under the current algmode/obj_scale/eval-time μ. Only
                // FeasibilitySwitchRecovery produces this Action, and only for a
                // nested strategy that is inactive and entry-permitted.
                if (this->try_soft_feasibility_step(algmode, obj_scale, eval_mu, XSL, DXSL, Temp,
                                                    RHS, RHS2, PGX)) {
                    // Take the full fraction-to-boundary step; the outer loop's
                    // XSL += alpha*DXSL commit applies it.
                    alpha = 1.0;
                    Citer.accepted_ = true;
                } else {
                    this->enter_feasibility_restoration(XSL, RHS, prim_obj, barr_obj, mu);
                    // A stage resumed after an episode restarts its stall window, and
                    // this entry becomes the handback the stall exit measures net
                    // progress against.
                    feas_stall.note_dispatch(
                        RHS.tail(this->equal_cons_ + this->inequal_cons_).template lpNorm<1>());
                    feas_stall.reset_window();
                    alpha = 0.0;
                }
                break;
            }
            case RecoveryChain::Action::kGiveUp:
                throw std::logic_error(
                    "PSIOPT: recovery gave up on the step, but no give-up handling exists yet "
                    "(no recovery link can produce this Action)");
            }
            this->result_.recovery_depth_histogram_[resolved_depth]++;
        } else if (GoodStep && Citer.accepted_) {
            // Mirrors should_dispatch_recovery's gate (GoodStep && !accepted_)
            // for its complement: a genuinely accepted iteration, where the
            // rejection hook above was skipped. See notify_step_accepted() on
            // RecoveryChain (recovery_chain.h) for what a link may do with
            // this -- WatchdogRecovery resets its consecutive-shortened-
            // iteration count here (watchdog.h).
            this->recovery_->notify_step_accepted();
        }

        // Re-arm the one-shot second-level re-center budget on any accepted step
        // (nested l1 restoration only, disclosure (f) in l1_restoration.h): an
        // accepted step ends the consecutive-failure run the one-shot guard
        // protects against, so a later ladder exhaustion may re-center again.
        // Reached by both branches above (recovery-resolved accept and the
        // no-recovery accept). Dead off the nested path (nested_active false).
        if (nested_active && Citer.accepted_)
            this->resto_recentered_ = false;

        Citer.alpha_p_ = alphap;
        Citer.alpha_d_ = alphad;
        Citer.alpha_t_ = alpha;
        Citer.eval_exceptions_ = this->eval_error_log_.count_ - eval_errs_before;

        this->fill_iter_info(v_xsl, v_rhs, prim_obj, barr_obj, mu, Citer);
        iters.push_back(Citer);

        // Suspend best-iterate tracking while restoration is active (dead on the
        // default path: restoration_ null). A feasibility-mode iterate's
        // prim_obj/kkt_inf are proximal-scale and must not compete with
        // true-objective iterates for "best" — otherwise return_best_ could
        // report a mixed-scale winner.
        if (settings_.return_best_ && !(this->restoration_ && this->restoration_->is_active())) {
            double critval;
            switch (settings_.best_criteria_) {
            case BestCriteriaModes::ECONS:
                critval = iters.back().econ_inf_;
                break;
            case BestCriteriaModes::ICONS:
                critval = iters.back().icon_inf_;
                break;
            case BestCriteriaModes::KKT:
                critval = iters.back().kkt_inf_;
                break;
            case BestCriteriaModes::OBJ:
                critval = iters.back().prim_obj_;
                break;
            default:
                throw std::invalid_argument("Unknown BestCriteriaModes");
            }
            if (critval <= BestCriteriaVal || i == 0) {
                BestCriteriaVal = critval;
                BestXSL = XSL;
                BestRHS = RHS;
                BestIter = i;
            }
        }

        if (this->late_callback_enabled_) {
            CBtimer.start();
            this->late_callback_(iters.back(), XSL, RHS);
            CBtimer.stop();
        }

        ExitCode = this->converge_check(iters);
        if (!GoodStep)
            ExitCode = ConvergenceFlags::DIVERGING;
        // Un-evaluable exhaustion at an already-acceptable iterate (see the
        // bypass above): report the acceptable level so the exit block below
        // terminates the loop. converge_check() only reaches ACCEPTABLE after a
        // sustained run of acceptable iterates; this iterate is acceptable but
        // the solve cannot continue, which is exactly the reference method's
        // "current point is acceptable, stop here" verdict. A stronger verdict
        // already reached (CONVERGED) is left alone; DIVERGING is unreachable
        // here both because the bypass only runs on a usable step direction
        // (the !GoodStep override above is mutually exclusive with it) and
        // because an acceptable iterate is finite and inside the divergence
        // thresholds (validate() enforces acc <= div).
        if (exit_at_acceptable && ExitCode == ConvergenceFlags::NOTCONVERGED)
            ExitCode = ConvergenceFlags::ACCEPTABLE;

        if (settings_.print_level_ == 0) {
            Printtimer.start();
            this->print_last_iterate(iters);
            Printtimer.stop();
        }

        // exit_stage_stalled (see the stall dispatch above) forces the exit
        // without touching ExitCode: unlike exit_at_acceptable it does not
        // upgrade the verdict, so converge_check's own answer (NOTCONVERGED,
        // or better if this iterate happens to qualify) is what gets reported.
        if (ExitCode == ConvergenceFlags::CONVERGED || ExitCode == ConvergenceFlags::ACCEPTABLE ||
            ExitCode == ConvergenceFlags::DIVERGING || exit_stage_stalled ||
            i == (settings_.max_iters_ - 1)) {

            if (ExitCode != ConvergenceFlags::CONVERGED && settings_.return_best_) {
                XSL = BestXSL;
                RHS = BestRHS;
            }

            this->result_.converge_flag_ = ExitCode;
            break;
        }

        // Apply step
        XSL += alpha * DXSL;

        // Commit the recovered elastic step alongside the outer primal/dual step.
        // Dead unless a nested restoration strategy is active. The outer primal
        // block was fraction-to-boundary scaled by alphap and the dual block by
        // alphad inside compute_step, then both damped by the backtrack alpha; the
        // elastic slacks (n,p) share the primal damping alpha·alphap and their
        // bound multipliers (z_n,z_p) the dual damping alpha·alphad, so the
        // condensed elastic variables move in lockstep with the KKT variables they
        // were eliminated from. Reached only on committed steps (this line is
        // skipped on the terminating iteration, exactly like the XSL commit).
        if (nested_active) {
            this->restoration_->apply_elastic_step(alpha * alphap, alpha * alphad);
        }
    }

    // Teardown invariant (dead on the default path: restoration_ is null). Any
    // return path that is still in feasibility mode (max_iters, divergence)
    // exits restoration and notifies the acceptance strategy of the switch back
    // to optimality BEFORE alg_impl returns, so the phase-boundary reset() in
    // run_phase_sequence() always sees optimality mode (the acceptance reset
    // invariant). The in-loop exit paths above already tore down; this is the
    // catch-all for the loop's break/fall-through exits (max_iters exhausted, or
    // DIVERGING, while restoration was still active).
    //
    // restoration_was_active/restoration_true_obj (declared near the top of
    // this function) are set here too, for the obj_val_ override further down:
    // by the time obj_val_ is set, exit_restoration() has already flipped
    // is_active() false, so `restoration_ && restoration_->is_active()` can no
    // longer be re-tested there.
    if (this->restoration_ && this->restoration_->is_active()) {
        // The barrier auxiliary is not recomputed here (the phase is ending
        // and the next reset() clears everything). The objective, however,
        // must be the TRUE objective, not φ_prox (this->restoration_->
        // proximal_objective(...)): notify_switch_to_optimality augments this
        // pair into the restored OPTIMALITY filter/funnel, whose accumulated
        // pairs are all true-objective-scale, and obj_val_ below must report
        // a meaningful number rather than a solver-internal one. Re-evaluated
        // once via the same helper the in-loop exit arms use.
        double teardown_theta;
        if (this->restoration_->is_nested()) {
            // Nested: the RHS constraint rows carry the condensed r̃, so the raw
            // original-problem infeasibility comes from the seam's saved residuals
            // (populated by the final active iteration's eval seam). Restore the
            // stashed outer μ and reset the governor so this catch-all teardown
            // leaves the solver in optimality mode with the outer barrier
            // parameter before run_phase_sequence()'s phase-boundary reset.
            teardown_theta = 0.0;
            if (this->equal_cons_ > 0)
                teardown_theta = std::max(
                    teardown_theta, this->resto_ec_scratch_.template lpNorm<Eigen::Infinity>());
            if (this->inequal_cons_ > 0)
                teardown_theta = std::max(
                    teardown_theta, this->resto_ic_scratch_.template lpNorm<Eigen::Infinity>());
            mu = this->stashed_mu_;
            this->governor_->reset();
        } else {
            teardown_theta = v_rhs.all_cons().template lpNorm<1>();
        }
        ProgressMeasures measures =
            this->build_restoration_exit_measures(obj_scale, teardown_theta, v_xsl.primals(), 0.0);
        restoration_was_active = true;
        restoration_true_obj = measures.objective;
        // No note_iteration() here: this teardown catches the max_iters /
        // divergence exits, whose final feasibility-mode iterate was already
        // counted by the in-loop stay-in-mode note_iteration() before the loop
        // broke (the decision-driven in-loop exits, which return before that
        // point, are the ones that count their exit iteration explicitly).
        this->restoration_->exit_restoration();
        this->acceptance_->notify_switch_to_optimality(measures);
        // Reset the recovery chain across the mode switch (see the
        // kSwitchToFeasibility entry rationale). Once per transition.
        this->recovery_->reset();
    }

    if (algmode == AlgorithmModes::OPT) {
        this->result_.obj_val_ = iters.back().prim_obj_;
    } else {
        Funtimer.start();
        this->result_.obj_val_ = 0;
        this->nlp_->eval_obj(obj_scale, v_xsl.primals(), this->result_.obj_val_);
        Funtimer.stop();
    }

    if (restoration_was_active) {
        // Override the algmode==OPT branch's iters.back().prim_obj_ above,
        // which is φ_prox for the last iterate evaluated while restoration was
        // active -- obj_val_ must report the true objective at the returned
        // primals.
        this->result_.obj_val_ = restoration_true_obj;
    }

    this->result_.primals_ = v_xsl.primals();

    // Proximal primal-dual regularization diagnostics: report the shifts from
    // the last FACTORIZED iteration of this phase (tracked in alg_impl locals;
    // iters.back() is the wrong source -- on a converged exit it is the
    // non-factorized convergence probe, whose fields still hold the sentinel).
    // There is no dedicated component object here with its own
    // append_diagnostics() hook to collect this from after alg_impl() returns
    // (unlike the acceptance_/governor_/restoration_ diagnostics collected in
    // run_phase_sequence()), since the shifts are alg_impl-local mode state.
    // Sentinel -1.0 stays untouched (from reset_accumulators()) when the mode
    // is off, matching the classic path's byte-identical guarantee, and also
    // when the mode is on but the phase converged before its first
    // factorization (no shift was ever applied). Same last-phase-wins
    // semantics as the other diagnostic fields: a multi-phase call ends with
    // the LAST phase's alg_impl call's values.
    if (settings_.inertia_mode_ == InertiaModes::proximal_regularization) {
        this->result_.last_prox_reg_primal_ = last_prox_primal;
        this->result_.last_prox_reg_dual_ = last_prox_dual;
    }

    // Trial-evaluation exception diagnostic: the message of the most recent
    // evaluation failure the acceptance machinery absorbed. Unlike the shifts
    // above there is no mode gate — the log is per-SOLVE (reset alongside
    // result_.reset_accumulators()), not per-phase, so a phase that absorbed
    // nothing after an earlier phase did leaves the earlier message standing,
    // and an entirely clean solve leaves the empty sentinel untouched.
    if (this->eval_error_log_.count_ > 0)
        this->result_.last_eval_exception_ = this->eval_error_log_.last_message_;

    if (this->equal_cons_ > 0) {
        this->result_.eq_cons_ = v_rhs.eq_cons();
        this->result_.eq_lmults_ = v_xsl.eq_lmults();
    }
    if (this->inequal_cons_ > 0) {
        this->result_.iq_cons_ = v_rhs.iq_cons() - v_xsl.slacks();
        this->result_.iq_lmults_ = v_xsl.iq_lmults();
    }

    Runtimer.stop();
    this->result_.iter_num_ += iters.size();
    double qptime = double(QPtimer.count<std::chrono::microseconds>()) / 1000000.0;
    double nlptime = double(Funtimer.count<std::chrono::microseconds>()) / 1000000.0;
    double tottime = double(Runtimer.count<std::chrono::microseconds>()) / 1000000.0;

    this->result_.func_time_ += nlptime;
    this->result_.kkt_time_ += qptime;
    double printtime = double(Printtimer.count<std::chrono::microseconds>()) / 1000000.0;
    this->result_.print_time_ += printtime;

    // Print exit statistics
    assert(!iters.empty());
    assert(!settings_.return_best_ || BestIter < static_cast<int>(iters.size()));
    int retiter = (settings_.return_best_ ? BestIter : static_cast<int>(iters.size()) - 1);
    print_exit_stats(ExitCode, iters[retiter], iters.size(), tottime * 1000, nlptime * 1000,
                     qptime * 1000, printtime * 1000);

    return XSL;
}

Eigen::VectorXd tycho::solvers::PSIOPT::init_impl(const Eigen::VectorXd &x, double mu,
                                                  bool docompute) {

    tycho::utils::Timer kktt;
    kktt.start();

    Eigen::VectorXd XSL(this->kkt_dim_);
    XSL.setZero();
    XSL.head(this->primal_vars_) = x;

    Eigen::VectorXd RHS(this->kkt_dim_);
    RHS.setZero();
    double val = 0;
    this->nlp_->set_primal_diags(1.0);
    if (this->inequal_cons_ > 0) {
        this->nlp_->set_slacks_ones();
    }
    // INIT mode never runs with restoration active (init_impl is the one-shot
    // multiplier initializer), so the μ argument is inert here; pass mu for
    // consistency with the live phase parameter.
    this->eval_nlp(AlgorithmModes::INIT, settings_.obj_scale_, XSL, val,
                   RHS.head(this->primal_vars_), RHS, this->kkt_sol_.get_matrix(), mu);

    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_rhs = kkt_view(RHS);

    Eigen::VectorXd hp(this->slack_vars_);

    for (int i = 0; i < this->slack_vars_; i++) {
        double fxi = v_rhs.iq_cons()[i];
        if (fxi < -settings_.bound_push_) {
            v_xsl.slacks()[i] = std::abs(fxi);
        } else {
            v_xsl.slacks()[i] = settings_.bound_push_;
        }
        hp[i] = 1.0;
        v_xsl.iq_lmults()[i] = mu / v_xsl.slacks()[i];
    }

    RHS.tail(this->equal_cons_ + this->inequal_cons_).setZero();

    if (this->inequal_cons_ > 0)
        this->nlp_->assign_kkt_slack_hessian(hp, this->kkt_sol_.get_matrix());
    if (settings_.print_level_ < 2) {
        print_beginning("KKT-Matrix Analysis ");
    }

    if (docompute)
        this->kkt_sol_.compute_internal();
    else
        this->kkt_sol_.refactorize_internal();
    kktt.stop();

    double pretime = double(kktt.count<std::chrono::microseconds>()) / 1000000.0;
    this->result_.pre_time_ += pretime;

    this->result_.factor_flops_ = this->kkt_sol_.flops_;
    this->result_.factor_mem_ = this->kkt_sol_.mem_;

    if (settings_.print_level_ < 2) {
        auto cyan = fmt::fg(fmt::color::cyan);
        if (docompute) {
            fmt::print(" LDLT Factor Size      : ");
            fmt::print(cyan, "{0:<10}\n", this->result_.factor_mem_);
            if (this->result_.factor_flops_ > 0) {
                fmt::print(" LDLT Factor FLOPs     : ");
                fmt::print(cyan, "{0} MFLOPs\n", this->result_.factor_flops_);
            }
        }
        fmt::print(" Analysis/Reorder Time : ");
        fmt::print(cyan, "{0:.3f} ms\n", pretime * 1000);
        print_finished("KKT-Matrix Analysis ");
    }

    // See the solve-into comment in alg_impl: direct-assign + in-place negate
    // avoids the extra temporary that `-kkt_sol_.solve(RHS)` forces.
    Eigen::VectorXd dx = this->kkt_sol_.solve(RHS);
    dx = -dx;
    KKTVector v_dx = kkt_view(dx);

    if (equal_cons_ > 0)
        v_xsl.eq_lmults() = v_dx.eq_lmults();
    if (this->inequal_cons_ > 0)
        this->nlp_->set_slack_diags(0.0);
    this->nlp_->set_primal_diags(0.0);

    return XSL;
}

Eigen::VectorXd tycho::solvers::PSIOPT::run_phase_sequence(const Eigen::VectorXd &x,
                                                           std::initializer_list<PhaseStep> steps) {
    if (!this->nlp_) {
        throw std::runtime_error("PSIOPT::run_phase_sequence: no NLP has been set. "
                                 "Call set_nlp() before optimize/solve.");
    }
    if (x.size() != primal_vars_) {
        throw std::invalid_argument(
            fmt::format("PSIOPT: initial guess has {} elements, expected {} primal variables",
                        x.size(), primal_vars_));
    }

    this->result_.reset_accumulators();
    this->eval_error_log_.reset();
    settings_.validate();

    // Rebuild acceptance_/mechanism_/governor_/recovery_ from the
    // just-validated Settings on every solve entry, not just on
    // (re)transcription (set_nlp() no longer builds them) — see
    // rebuild_globalization_components()'s doc comment for why this must run
    // per solve rather than per transcription, and for the neutrality
    // argument on the default (all-off) path. nlp_ is guaranteed non-null
    // here (checked above), and set_nlp() has always already run (same
    // guarantee), so the SolverContext captures this call takes are final.
    this->rebuild_globalization_components();

    // Re-apply the QP threading setting on every solve entry, not just in
    // set_nlp() (which only runs on transcribe): a single-thread pin left on
    // this thread by another component (e.g. Jet's per-job pin — thread-local
    // BLASSetThreading on macOS 15+, or detail::MklLocalPinGuard in jet.h on
    // the MKL side) must not silently single-thread reused solves.
    //
    // This re-apply is NOT redundant now that the MKL setter below is
    // thread-local rather than global — if anything it is MORE load-bearing:
    // a nonzero thread-local override takes priority over the process-global
    // value on that thread until explicitly reset, so a *global* re-apply
    // could never have actually overridden a lingering local pin in the
    // first place. A solve driven through Jet::map runs jet_run() ->
    // solve()/optimize() -> run_phase_sequence() on the pool worker thread
    // *while still inside* detail::MklLocalPinGuard's scope (jet.h), which
    // has pinned that thread's local MKL thread count to 1. The explicit
    // thread-local set here is what makes this thread (whichever one is
    // calling) actually run Pardiso with this driver's own qp_threads_,
    // instead of silently inheriting whatever local value a previous
    // occupant left behind. In the Jet::map case specifically,
    // jet_initialize() forces qp_threads_ == 1 (via set_num_partitions(1, 1)),
    // so this re-apply sets the local value to 1 — consistent with, and
    // redundant to, the guard's own pin — but for any other thread reusing a
    // pool worker outside of Jet, this call is what restores the driver's
    // intended thread count on its own calling thread.
#ifdef USE_ACCELERATE_SPARSE
    accelerate_set_num_threads(settings_.qp_threads_);
#else
    // Return value (previous local count) intentionally discarded; see the
    // set_nlp() call site above for the fire-and-forget rationale.
    mkl_set_num_threads_local(settings_.qp_threads_);
#endif

    if (settings_.print_level_ == 0)
        print_stats();
    if (settings_.print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();

    tycho::utils::Timer t;
    t.start();

    bool docompute = analyze_kkt_matrix();
    Eigen::VectorXd XSL = this->init_impl(x, settings_.init_mu_, docompute);

    auto it = steps.begin();
    auto end = steps.end();
    while (it != end) {
        const auto &step = *it;
        ++it;
        bool is_last = (it == end);

        // Conditional steps only run if the previous phase didn't converge
        if (step.conditional_ && this->result_.converge_flag_ == ConvergenceFlags::CONVERGED)
            continue;

        if (settings_.print_level_ < 2)
            print_beginning(step.label_);

        // Phase-boundary reset: each globalization component's μ-event/
        // phase-change hook (see e.g. recovery_chain.h's ownership-rule
        // note). reset() was never actually invoked anywhere before this —
        // every implementation that could be live through recovery_ (or
        // acceptance_/mechanism_/governor_) has an empty reset() body EXCEPT
        // WatchdogRecovery (ClassicMeritAcceptance, BacktrackingLineSearch,
        // ClassicAdaptiveGovernor, NoopRecovery, SocRecovery,
        // ExtendedBacktrackRecovery, and ChainedRecovery are all no-ops), so
        // adding these calls is behavior-neutral for every configuration
        // except one: WatchdogRecovery, the one component with real
        // per-solve state, needs its counters/arm-state/snapshot cleared at
        // each new phase (OPT, then a conditional SOE, etc.) rather than
        // carried over from the previous phase.
        this->acceptance_->reset();
        this->mechanism_->reset();
        this->governor_->reset();
        this->recovery_->reset();
        // Restoration reset is null-guarded (restoration_ exists only under a
        // restoration mode). alg_impl's teardown guarantees restoration is
        // inactive by the time this runs, so reset() here only clears the
        // per-phase entry snapshot and diagnostic counters. The solver-side
        // nested-lifecycle bookkeeping (stashed outer μ, first-iteration guard,
        // κ_resto ratchet baseline) is cleared here too — this is the
        // phase-boundary reset, distinct from the μ-event reset() mid-phase that
        // deliberately preserves the stash (see the members' reset-invariant
        // note in psiopt.h).
        if (this->restoration_) {
            this->restoration_->reset();
            this->stashed_mu_ = 0.0;
            this->resto_first_iter_ = false;
            this->resto_theta_orig_prev_ = 0.0;
            this->resto_recentered_ = false;
        }

        XSL = this->alg_impl(step.alg_mode_, step.bar_mode_, step.ls_mode_, settings_.obj_scale_,
                             settings_.init_mu_, XSL);

        // Solver-level observability: collect this phase's acceptance-
        // strategy diagnostics (funnel width / filter size+resets — see
        // AcceptanceStrategy::append_diagnostics()) right after alg_impl()
        // returns and BEFORE the next loop iteration's acceptance_->reset()
        // above clears any per-phase state (e.g. FilterAcceptance's reset
        // counters). A multi-phase call therefore ends with the LAST phase's
        // values, like every other SolveResult field alg_impl overwrites.
        // The default no-op body means this is write-only-neutral on the
        // classic/merit paths.
        this->acceptance_->append_diagnostics(this->result_);

        // Same collection point, same last-phase-wins semantics, for the
        // barrier governor's own diagnostics (monotone-mode switch/iteration
        // counts — see BarrierGovernor::append_diagnostics()). The default
        // no-op body means this is write-only-neutral unless
        // barrier_governor=monitored is selected.
        this->governor_->append_diagnostics(this->result_);

        // Same collection point and last-phase-wins semantics for the
        // feasibility-restoration diagnostics (entry count / iterations-in-mode
        // — see RestorationStrategy::append_diagnostics()). Null-guarded: the
        // SolveResult::last_feas_rest_* fields keep their -1 sentinels when
        // restoration_mode_ == off (no strategy constructed).
        if (this->restoration_)
            this->restoration_->append_diagnostics(this->result_);

        if (settings_.print_level_ < 2)
            print_finished(step.label_);

        // If a phase diverged, skip subsequent phases — result_.primals_ may contain
        // garbage and feeding it into init_impl for the next phase would be pointless.
        if (result_.converge_flag_ == ConvergenceFlags::DIVERGING) {
            if (settings_.print_level_ < 3)
                fmt::print(fmt::fg(fmt::color::yellow),
                           "Phase diverged; skipping remaining phases.\n");
            break;
        }

        // Re-init for the next phase using stored primals
        if (!is_last) {
            XSL = this->init_impl(result_.primals_, settings_.init_mu_, false);
        }
    }

    t.stop();
    double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
    this->result_.total_time_ = tottime / 1000.0;

    if (settings_.print_level_ < 2) {
        print_timing_summary();
        fmt::print(" PSIOPT Total Time            : ");
        fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.3f} ms\n", tottime);
        print_finished("PSIOPT ");
        print_header();
    }

    return result_.primals_;
}

Eigen::VectorXd tycho::solvers::PSIOPT::optimize(const Eigen::VectorXd &x) {
    return run_phase_sequence(x, {{AlgorithmModes::OPT, settings_.opt_bar_mode_,
                                   settings_.opt_ls_mode_, "Optimization Algorithm "}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve(const Eigen::VectorXd &x) {
    return run_phase_sequence(x, {{settings_.soe_mode_, settings_.soe_bar_mode_,
                                   settings_.soe_ls_mode_, "Solve Algorithm "}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve_optimize(const Eigen::VectorXd &x) {
    return run_phase_sequence(x, {{settings_.soe_mode_, settings_.soe_bar_mode_,
                                   settings_.soe_ls_mode_, "Solve Algorithm "},
                                  {AlgorithmModes::OPT, settings_.opt_bar_mode_,
                                   settings_.opt_ls_mode_, "Optimization Algorithm "}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::optimize_solve(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x, {{AlgorithmModes::OPT, settings_.opt_bar_mode_, settings_.opt_ls_mode_,
             "Optimization Algorithm "},
            {settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_,
             "Solve Algorithm ", /*conditional_=*/true}});
}

Eigen::VectorXd tycho::solvers::PSIOPT::solve_optimize_solve(const Eigen::VectorXd &x) {
    return run_phase_sequence(
        x,
        {{settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_, "Solve Algorithm "},
         {AlgorithmModes::OPT, settings_.opt_bar_mode_, settings_.opt_ls_mode_,
          "Optimization Algorithm "},
         {settings_.soe_mode_, settings_.soe_bar_mode_, settings_.soe_ls_mode_, "Solve Algorithm ",
          /*conditional_=*/true}});
}
