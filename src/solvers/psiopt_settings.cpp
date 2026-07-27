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
//   - Configuration fields grouped into Settings struct
//   - Validated setter methods added
//   - Split out of psiopt.cpp: the string-to-enum converters, the validated
//     setters and Settings::validate() share no state and no idea with the
//     algorithm, and account for the first six hundred lines of a heavy TU
// =============================================================================
//
// PSIOPT settings: the string-to-enum converters the Python surface uses, the
// validated set_*() methods, and Settings::validate(). Nothing here touches the
// solve; the algorithm lives in psiopt.cpp, the iteration/exit reporting in
// psiopt_print.cpp, and the globalization components in
// psiopt_globalization.cpp.

#include "tycho/detail/solvers/psiopt.h"
#include "tycho/detail/solvers/psiopt_presets.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

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

// Shared validation helpers for both the individual set_*() methods and
// Settings::validate(): every numeric-field invariant below is checked twice
// (once at the point of assignment, once again over the whole struct at
// run_phase_sequence() entry), and both checks must enforce the identical
// condition and report the identical message -- these four helpers are the
// single home for that pairing, so the two call sites cannot drift apart.
void pos_finite(double v, const char *name) {
    if (!std::isfinite(v) || v <= 0.0)
        throw std::invalid_argument(
            fmt::format("{} must be finite and positive, got {}", name, v));
}

void pos_int(int v, const char *name) {
    if (v < 1)
        throw std::invalid_argument(fmt::format("{} must be >= 1, got {}", name, v));
}

void in_open_unit(double v, const char *name) {
    if (v <= 0.0 || v >= 1.0)
        throw std::invalid_argument(fmt::format("{} must be in (0, 1), got {}", name, v));
}

void greater_than(double v, double bound, const char *name) {
    if (v <= bound)
        throw std::invalid_argument(
            fmt::format("{} must be greater than {}, got {}", name, bound, v));
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
    pos_int(max_iters, "max_iters");
    settings_.max_iters_ = max_iters;
}

void tycho::solvers::PSIOPT::set_max_acc_iters(int max_acc_iters) {
    pos_int(max_acc_iters, "max_acc_iters");
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
    in_open_unit(bound_fraction, "bound_fraction");
    settings_.bound_fraction_ = bound_fraction;
}

void tycho::solvers::PSIOPT::set_bound_push(double bound_push) {
    greater_than(bound_push, 0.0, "bound_push");
    settings_.bound_push_ = bound_push;
}

void tycho::solvers::PSIOPT::set_alpha_red(double ared) {
    greater_than(ared, 1.0, "alpha_red");
    settings_.alpha_red_ = ared;
}

void tycho::solvers::PSIOPT::set_delta_h(double delta_h) {
    greater_than(delta_h, 0.0, "delta_h");
    settings_.delta_h_ = delta_h;
}

void tycho::solvers::PSIOPT::set_incr_h(double incr_h) {
    greater_than(incr_h, 1.0, "incr_h");
    settings_.incr_h_ = incr_h;
}

void tycho::solvers::PSIOPT::set_decr_h(double decr_h) {
    in_open_unit(decr_h, "decr_h");
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
    // pos_finite/pos_int/in_open_unit/greater_than are the file-scope helpers
    // defined above (shared with the individual set_*() methods, so a field's
    // invariant and message can never drift between the two call sites).

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
    in_open_unit(bound_fraction_, "bound_fraction");
    greater_than(bound_push_, 0.0, "bound_push");
    pos_finite(neg_slack_reset_, "neg_slack_reset");
    greater_than(alpha_red_, 1.0, "alpha_red");

    // --- Hessian perturbation ---
    greater_than(delta_h_, 0.0, "delta_h");
    greater_than(incr_h_, 1.0, "incr_h");
    in_open_unit(decr_h_, "decr_h");

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
// Named configuration presets
// =============================================================================

void tycho::solvers::PSIOPT::apply_preset(std::string_view name) {
    for (const auto &entry : kPSIOPTPresets) {
        if (entry.name_ != name)
            continue;
        const PSIOPTPresetFields &f = entry.fields_;
        settings_.acceptance_strategy_ = f.acceptance_strategy_;
        settings_.merit_penalty_rule_ = f.merit_penalty_rule_;
        settings_.barrier_governor_ = f.barrier_governor_;
        settings_.never_monotone_ = f.never_monotone_;
        settings_.restoration_mode_ = f.restoration_mode_;
        settings_.inertia_mode_ = f.inertia_mode_;
        settings_.max_soc_ = f.max_soc_;
        settings_.ls_extended_iters_ = f.ls_extended_iters_;
        settings_.watchdog_ = f.watchdog_;
        return;
    }

    // Unrecognized name: fold the full valid-name list into the exception
    // message (T6) rather than printing it separately -- kPSIOPTPresets is the
    // single source for both this message and any future binding docstring.
    std::string valid_names;
    for (std::size_t i = 0; i < kPSIOPTPresets.size(); ++i) {
        if (i != 0)
            valid_names += ", ";
        valid_names += kPSIOPTPresets[i].name_;
    }
    throw std::invalid_argument(
        fmt::format("Unrecognized PSIOPT preset '{}'. Valid options are: {}", name, valid_names));
}
