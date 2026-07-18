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
#include "tycho/detail/solvers/globalization/merit_acceptance.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/noop_recovery.h"
#include "tycho/detail/solvers/globalization/soc.h"
#include "tycho/detail/solvers/globalization/watchdog.h"
#include "tycho/detail/solvers/globalization/restoration.h"

#ifndef USE_ACCELERATE_SPARSE
#include <mkl.h>
#endif

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
    if (ls_extended_iters_ < 0)
        throw std::invalid_argument(fmt::format(
            "ls_extended_iters must be non-negative, got {}", ls_extended_iters_));

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

    // (Re)build the classic merit acceptance strategy wired to a
    // SolverContext view of this solver. Rebuilt here (rather than once in the
    // constructor) so the SolverContext's captured nlp_ raw pointer tracks the
    // NLP just installed; dims/settings/scratch are captured by reference and
    // stay live. PSIOPT owns acceptance_, so the SolverContext's references
    // cannot outlive their referents.
    this->acceptance_ = std::make_unique<ClassicMeritAcceptance>(
        SolverContext{this->nlp_.get(), this->kkt_sol_, this->settings_, this->primal_vars_,
                      this->slack_vars_, this->equal_cons_, this->inequal_cons_, this->kkt_dim_,
                      this->stli_scratch_, this->hp_scratch_, this->best_xsl_scratch_,
                      this->best_rhs_scratch_});

    // The step-length globalization mechanism. Stateless (holds
    // NO solver state per GlobalizationMechanism's ownership rule) — every call
    // receives the live SolverContext as an explicit parameter — so it is
    // constructed with no context here; alg_impl builds the SolverContext view
    // it passes to compute_step / max_primal_dual_step.
    this->mechanism_ = std::make_unique<BacktrackingLineSearch>();

    // The barrier-parameter governor. Stateless (holds NO solver
    // state per BarrierGovernor's ownership rule) — every update_barrier() call
    // receives the live SolverContext and the GlobalizationMechanism as explicit
    // parameters — so it is constructed with no context here; alg_impl builds
    // the SolverContext view and passes *mechanism_ to update_barrier.
    this->governor_ = std::make_unique<ClassicAdaptiveGovernor>();

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
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    std::fill_n(KKTmat.valuePtr(), KKTmat.nonZeros(), 0.0);

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

tycho::ConvergenceFlags tycho::solvers::PSIOPT::converge_check(std::vector<IterateInfo> &iters) {
    assert(!iters.empty() && "converge_check called with empty iteration history");
    ConvergenceFlags Flag = ConvergenceFlags::CONVERGED;
    IterateInfo last = iters.back();
    bool KKTFeas = (last.kkt_inf_ < settings_.kkt_tol_);
    bool EConFeas = (last.econ_inf_ < settings_.econ_tol_);
    bool IConFeas = (last.icon_inf_ < settings_.icon_tol_);
    bool BarFeas = (last.barr_inf_ < settings_.bar_tol_);

    bool KKTDiv = (last.kkt_inf_ > settings_.div_kkt_tol_) || !std::isfinite(last.kkt_inf_);
    bool EConDiv = (last.econ_inf_ > settings_.div_econ_tol_) || !std::isfinite(last.econ_inf_);
    bool IConDiv = (last.icon_inf_ > settings_.div_icon_tol_) || !std::isfinite(last.icon_inf_);
    bool BarDiv = (last.barr_inf_ > settings_.div_bar_tol_) || !std::isfinite(last.barr_inf_);

    if (KKTDiv || EConDiv || IConDiv || BarDiv) {
        Flag = ConvergenceFlags::DIVERGING;
        return Flag;
    } else if (KKTFeas && EConFeas && IConFeas && BarFeas) {
        Flag = ConvergenceFlags::CONVERGED;
        return Flag;
    } else if (int(iters.size()) > settings_.max_acc_iters_) {
        int nfeas = 0;
        for (int i = 0; i < settings_.max_acc_iters_; i++) {
            last = iters[int(iters.size()) - i - 1];
            KKTFeas = (last.kkt_inf_ < settings_.acc_kkt_tol_);
            EConFeas = (last.econ_inf_ < settings_.acc_econ_tol_);
            IConFeas = (last.icon_inf_ < settings_.acc_icon_tol_);
            BarFeas = (last.barr_inf_ < settings_.acc_bar_tol_);
            if (KKTFeas && EConFeas && IConFeas && BarFeas)
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
                                        double incpurt, double &finalpert, double &cumpert) {
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

    if (Zfac || docompute) {
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
    SolverContext ctx{this->nlp_.get(),    this->kkt_sol_,          this->settings_,
                      this->primal_vars_,  this->slack_vars_,       this->equal_cons_,
                      this->inequal_cons_, this->kkt_dim_,          this->stli_scratch_,
                      this->hp_scratch_,   this->best_xsl_scratch_, this->best_rhs_scratch_};

    tycho::utils::Timer Runtimer;
    tycho::utils::Timer Funtimer;
    tycho::utils::Timer QPtimer;
    tycho::utils::Timer CBtimer; // Callback time falls into misc_time_ implicitly (misc = total -
                                 // pre - kkt - func - print)
    tycho::utils::Timer Printtimer;

    double Hpert0 = settings_.delta_h_;
    std::vector<IterateInfo> iters;
    iters.reserve(settings_.max_iters_);
    ConvergenceFlags ExitCode = ConvergenceFlags::NOTCONVERGED;
    bool FirstPert = true;

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

        this->eval_nlp(algmode, obj_scale, XSL, prim_obj, PGX, RHS, this->kkt_sol_.get_matrix());

        if (this->inequal_cons_ > 0) {
            this->apply_reset_slacks(v_xsl.slacks(), v_rhs.iq_cons());
            this->barrier_hessian(this->kkt_sol_.get_matrix(), v_xsl.slacks(), v_xsl.iq_lmults(),
                                  mu);
            this->complementarity(v_xsl.slacks(), v_xsl.iq_lmults(), avgcomp, mincomp, maxcomp);
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

            if (settings_.return_best_) {
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

        Citer.h_facs_ = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert, nhpert_cum);
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

        // Update barrier parameter and compute search direction. The whole
        // PROBE/LOQO switch + common clamp/objective/gradient tail is now
        // ClassicAdaptiveGovernor::update_barrier; the
        // `if (inequal_cons_ > 0)` guard stays here, exactly as the block was
        // guarded before extraction, so the governor is invoked only when there
        // are inequality constraints (barrier terms). The PROBE predictor's KKT
        // solve moves INTO the governor; the REAL step solve below (a distinct
        // second solve) stays here. avgcomp/mincomp feed the mu oracles;
        // *mechanism_ lets the PROBE predictor reuse the step-scaling.
        if (this->inequal_cons_ > 0) {
            mu = governor_->update_barrier(barmode, mu, avgcomp, mincomp, XSL, RHS, DXSL, Temp,
                                           *mechanism_, ctx, barr_obj);
        }

        // The REAL step solve (distinct from the PROBE predictor solve, which
        // moved into ClassicAdaptiveGovernor::update_barrier — see its solve-into
        // comment): direct assignment + in-place negate avoids the extra
        // temporary that `-kkt_sol_.solve(RHS)` forces.
        DXSL = this->kkt_sol_.solve(RHS);
        DXSL = -DXSL;
        bool GoodStep = std::isfinite(DXSL.squaredNorm());
        QPtimer.stop();

        // Line search. lsobjscale is hoisted out of the GoodStep block below so
        // the recovery-chain hook can forward the identical merit objective
        // scale (obj_scale * lsobjscale) to its acceptance re-test; its value is
        // a pure select on algmode (0.0 for SOE/OPTNO, else 1.0), so hoisting the
        // declaration does not change the value passed to compute_step.
        double lsobjscale =
            algmode == AlgorithmModes::SOE || algmode == AlgorithmModes::OPTNO ? 0.0 : 1.0;
        Funtimer.start();
        if (GoodStep) {
            // compute_step fuses the fraction-to-boundary scaling (former
            // `if (inequal_cons_ > 0) max_primal_dual_step(...)`, now guarded
            // identically inside compute_step and MUTATING DXSL in place) and
            // the acceptance backtrack on the scaled DXSL. This is the riskiest
            // FP-order seam: negate -> block-scale by
            // alphap/alphad -> `xsl + alpha*dxsl` trial -> `XSL += alpha*DXSL`.
            alpha = mechanism_->compute_step(lsmode, obj_scale * lsobjscale, mu, prim_obj, barr_obj,
                                             XSL, DXSL, Temp, RHS, RHS2, *acceptance_, alphap,
                                             alphad, Citer, iters, ctx);

        } else {
            Citer.h_facs_ = -1;
        }

        Funtimer.stop();

        // Recovery-chain hook. This is where a rejected step's
        // recovery gets a say -- SOC -> extended-backtrack -> watchdog-revert
        // dispatch from this point (see set_nlp's wiring comment); the
        // feasibility switch remains a future link. The inertia/perturbation
        // ladder above (factor_impl's Zfac cycling + escalation) is a
        // SEPARATE mechanism and stays out of this chain until the
        // proximal-regularization inertia mode is implemented -- it is NOT
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
        // By default recovery_ is a NoopRecovery (set_nlp installs it whenever
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
                Citer, iters, ctx, *acceptance_, *mechanism_, lsmode, obj_scale * lsobjscale, mu,
                prim_obj, barr_obj, XSL, DXSL, Temp, RHS, RHS2, alpha, alphap, alphad,
                this->result_.soc_steps_taken_, resolved_depth,
                this->result_.watchdog_activations_);
            switch (recovery_action) {
            case RecoveryChain::Action::kAcceptAsIs:
                // Take the step compute_step produced (DXSL/alpha unchanged).
                break;
            case RecoveryChain::Action::kRetry:
                // The recovery chain committed a corrected/reverted step into
                // DXSL/alpha.
                break;
            case RecoveryChain::Action::kSwitchToFeasibility:
                throw std::logic_error(
                    "PSIOPT: recovery requested a feasibility-restoration switch, but no "
                    "restoration strategy exists yet (no recovery link can produce this Action)");
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

        Citer.alpha_p_ = alphap;
        Citer.alpha_d_ = alphad;
        Citer.alpha_t_ = alpha;

        this->fill_iter_info(v_xsl, v_rhs, prim_obj, barr_obj, mu, Citer);
        iters.push_back(Citer);

        if (settings_.return_best_) {
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

        if (settings_.print_level_ == 0) {
            Printtimer.start();
            this->print_last_iterate(iters);
            Printtimer.stop();
        }

        if (ExitCode == ConvergenceFlags::CONVERGED || ExitCode == ConvergenceFlags::ACCEPTABLE ||
            ExitCode == ConvergenceFlags::DIVERGING || i == (settings_.max_iters_ - 1)) {

            if (ExitCode != ConvergenceFlags::CONVERGED && settings_.return_best_) {
                XSL = BestXSL;
                RHS = BestRHS;
            }

            this->result_.converge_flag_ = ExitCode;
            break;
        }

        // Apply step
        XSL += alpha * DXSL;
    }

    if (algmode == AlgorithmModes::OPT) {
        this->result_.obj_val_ = iters.back().prim_obj_;
    } else {
        Funtimer.start();
        this->result_.obj_val_ = 0;
        this->nlp_->eval_obj(obj_scale, v_xsl.primals(), this->result_.obj_val_);
        Funtimer.stop();
    }

    this->result_.primals_ = v_xsl.primals();

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
    this->eval_nlp(AlgorithmModes::INIT, settings_.obj_scale_, XSL, val,
                   RHS.head(this->primal_vars_), RHS, this->kkt_sol_.get_matrix());

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
    settings_.validate();

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

        XSL = this->alg_impl(step.alg_mode_, step.bar_mode_, step.ls_mode_, settings_.obj_scale_,
                             settings_.init_mu_, XSL);

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
