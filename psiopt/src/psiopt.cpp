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
//   - Printing extracted to psiopt_print.cpp
//   - Settings converters/setters/validation extracted to psiopt_settings.cpp
// =============================================================================

#include "tycho/detail/solvers/psiopt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "tycho/detail/solvers/barrier_math.h"
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
    // Same exhaustive-switch hardening as the Accelerate branch's qp_ord_
    // switch above (T6): a value outside QPOrderingModes' three cases would
    // otherwise land in Pardiso's iparm ordering slot as an unvalidated int.
    switch (settings_.qp_ord_) {
    case QPOrderingModes::MINDEG:
    case QPOrderingModes::METIS:
    case QPOrderingModes::PARMETIS:
        this->kkt_sol_.ord_ = static_cast<int>(settings_.qp_ord_);
        break;
    default:
        throw std::invalid_argument("Unknown QPOrderingMode");
    }
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

bool tycho::solvers::PSIOPT::claim_kkt_analysis() {
    bool docompute = true;
    if (this->qp_analyzed_ && !(settings_.force_qp_analysis_)) {
        docompute = false;
    } else {
        this->qp_analyzed_ = true;
    }
    return docompute;
}

// =============================================================================
// Release
// =============================================================================

void tycho::solvers::PSIOPT::release() {
    this->kkt_sol_.release();
    this->qp_analyzed_ = false;
    // The bound set lives in the NLP being released; the multipliers indexed it.
    this->bounds_ = nullptr;
    this->bound_duals_ = BoundDualState{};
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
    detail::apply_reset_slacks(S, FXI, this->slack_vars_, settings_.neg_slack_reset_);
}

// max_step_to_boundary was extracted verbatim into BacktrackingLineSearch
// (src/solvers/psiopt_globalization.cpp).

void tycho::solvers::PSIOPT::complementarity(Eigen::Ref<Eigen::VectorXd> X,
                                             Eigen::Ref<Eigen::VectorXd> S,
                                             Eigen::Ref<Eigen::VectorXd> LI, double &avgcomp,
                                             double &mincomp, double &maxcomp) const {
    // Keep the exact Eigen .sum()/minCoeff()/maxCoeff() reduction expressions
    // over the slack pairs unchanged. avgcomp feeds mu (see mpc_mu/loqo_mu call
    // sites), so a hand-fused loop that reorders the sum could perturb the
    // reduction by a ULP under fast-math and change iterates -- forbidden. The
    // buffer hoist removed the per-call heap allocation of StLI without touching
    // that ordering, and the variable-bound extension below likewise leaves it
    // alone: the bound pairs are reduced SEPARATELY and combined by count, never
    // folded into this reduction.
    //
    // The emptiness guard is new and is not a behaviour change on any path that
    // existed before: this function was only ever called with inequality
    // constraints present. It exists because a problem whose only barrier terms
    // are variable-bound terms reaches here with no slack block at all, and the
    // reductions below are undefined on an empty vector.
    int base_count = 0;
    if (S.size() > 0) {
        this->stli_scratch_.resize(S.size());
        this->stli_scratch_ = S.cwiseProduct(LI);
        mincomp = this->stli_scratch_.minCoeff();
        maxcomp = this->stli_scratch_.maxCoeff();
        avgcomp = this->stli_scratch_.sum() / double(this->stli_scratch_.size());
        base_count = static_cast<int>(this->stli_scratch_.size());
    }

    // Variable-bound pairs join the account the barrier oracles and the barrier
    // residual read: a bounded variable's (x-l)*z_L and (u-x)*z_U are
    // complementarity products in exactly the sense s*lambda is, and mu has to
    // be driven by all of them or the bounded coordinates never centre. Dead on
    // a problem with no bound set.
    if (this->bounds_)
        detail::augment_bound_complementarity(X, *this->bounds_, this->bound_duals_, base_count,
                                              avgcomp, mincomp, maxcomp);
}

int tycho::solvers::PSIOPT::complementarity_pair_count(int slack_count) const {
    if (!this->bounds_)
        return slack_count;
    return slack_count + static_cast<int>(this->bounds_->lower_idx_.size()) +
           static_cast<int>(this->bounds_->upper_idx_.size());
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
// Native variable-bound helpers. Every one is a no-op when bounds_ is null,
// which is the whole story on a problem that declares no variable bounds.
// =============================================================================

void tycho::solvers::PSIOPT::push_initial_point_interior(EigenRef<Eigen::VectorXd> x, double mu0) {
    this->bound_duals_ = BoundDualState{};
    if (!this->bounds_)
        return;

    if (x.size() != this->primal_vars_)
        throw std::logic_error(
            fmt::format("PSIOPT: interior push expected a {0}-element reduced primal vector "
                        "(got {1})",
                        this->primal_vars_, x.size()));

    const BoundSet &b = *this->bounds_;
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());

    // Dense by-variable view of the recorded bounds. The two lists are built in
    // one pass over the variables and so happen to be sorted, but nothing in
    // BoundSet's contract says so, and the two-sided cap needs BOTH endpoints of
    // the same variable -- so the pairing goes through an explicit lookup rather
    // than an assumed merge. Allocated once per solve; the push runs once.
    constexpr double kInf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(this->primal_vars_, -kInf);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(this->primal_vars_, kInf);
    for (int k = 0; k < nl; k++)
        lower[b.lower_idx_[k]] = b.lower_val_[k];
    for (int k = 0; k < nu; k++)
        upper[b.upper_idx_[k]] = b.upper_val_[k];

    const double kappa1 = settings_.bound_push_;
    const double kappa2 = settings_.bound_interval_push_;

    // Lower pushes first, then upper, matching the reference order. A two-sided
    // variable takes p_L + p_U <= 2*kappa2*(u-l) of its own interval, so with
    // kappa2 below one half the projections cannot cross each other.
    for (int k = 0; k < nl; k++) {
        const int i = b.lower_idx_[k];
        const double l = b.lower_val_[k];
        double p = kappa1 * std::max(1.0, std::abs(l));
        if (upper[i] < kInf)
            p = std::min(p, kappa2 * (upper[i] - l));
        x[i] = std::max(x[i], l + p);
    }
    for (int k = 0; k < nu; k++) {
        const int i = b.upper_idx_[k];
        const double u = b.upper_val_[k];
        double p = kappa1 * std::max(1.0, std::abs(u));
        if (lower[i] > -kInf)
            p = std::min(p, kappa2 * (u - lower[i]));
        x[i] = std::min(x[i], u - p);
    }

    // Multipliers seeded from the barrier parameter and the (now interior)
    // distances, capped so a point started very close to a bound cannot seed an
    // enormous multiplier.
    this->bound_duals_.z_lower_.resize(nl);
    this->bound_duals_.dz_lower_ = Eigen::VectorXd::Zero(nl);
    for (int k = 0; k < nl; k++) {
        const int i = b.lower_idx_[k];
        this->bound_duals_.z_lower_[k] =
            std::min(kBoundMultInitCap, mu0 / (x[i] - b.lower_val_[k]));
    }
    this->bound_duals_.z_upper_.resize(nu);
    this->bound_duals_.dz_upper_ = Eigen::VectorXd::Zero(nu);
    for (int k = 0; k < nu; k++) {
        const int i = b.upper_idx_[k];
        this->bound_duals_.z_upper_[k] =
            std::min(kBoundMultInitCap, mu0 / (b.upper_val_[k] - x[i]));
    }
}

void tycho::solvers::PSIOPT::compute_bound_dual_direction(ConstEigenRef<Eigen::VectorXd> x,
                                                          ConstEigenRef<Eigen::VectorXd> dx,
                                                          double mu) {
    if (!this->bounds_)
        return;
    const BoundSet &b = *this->bounds_;
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());
    for (int k = 0; k < nl; k++) {
        const int i = b.lower_idx_[k];
        const double d = x[i] - b.lower_val_[k];
        const double z = this->bound_duals_.z_lower_[k];
        this->bound_duals_.dz_lower_[k] = mu / d - z - (z / d) * dx[i];
    }
    // Upper-bound distance shrinks as x grows, so the curvature term enters with
    // the opposite sign to the lower-bound one.
    for (int k = 0; k < nu; k++) {
        const int i = b.upper_idx_[k];
        const double d = b.upper_val_[k] - x[i];
        const double z = this->bound_duals_.z_upper_[k];
        this->bound_duals_.dz_upper_[k] = mu / d - z + (z / d) * dx[i];
    }
}

void tycho::solvers::PSIOPT::apply_bound_dual_step(double alphad, KKTVector &xsl_new, double mu,
                                                   bool monotone_mu) {
    if (!this->bounds_)
        return;
    const BoundSet &b = *this->bounds_;
    const int nl = static_cast<int>(b.lower_idx_.size());
    const int nu = static_cast<int>(b.upper_idx_.size());
    auto x_new = xsl_new.primals();

    // Take the step first: the clamp below is a correction applied to the
    // multipliers this iterate actually landed on, and the free-mu barrier
    // parameter it uses is measured from them.
    for (int k = 0; k < nl; k++)
        this->bound_duals_.z_lower_[k] += alphad * this->bound_duals_.dz_lower_[k];
    for (int k = 0; k < nu; k++)
        this->bound_duals_.z_upper_[k] += alphad * this->bound_duals_.dz_upper_[k];

    // The barrier parameter the clamp is taken at (Ipopt IpIpoptAlg.cpp,
    // correct_bound_multiplier): the barrier parameter itself under a monotone
    // schedule, and otherwise the average complementarity at the new point --
    // over EVERY complementary pair, the inequality slack/multiplier pairs as
    // well as the bound pairs -- capped at kFreeModeClipMuCap.
    double mu_clip = mu;
    if (!monotone_mu) {
        double sum = 0.0;
        int count = 0;
        auto s = xsl_new.slacks();
        auto li = xsl_new.iq_lmults();
        for (int i = 0; i < this->slack_vars_; i++) {
            sum += s[i] * li[i];
            count++;
        }
        for (int k = 0; k < nl; k++) {
            sum += this->bound_duals_.z_lower_[k] * (x_new[b.lower_idx_[k]] - b.lower_val_[k]);
            count++;
        }
        for (int k = 0; k < nu; k++) {
            sum += this->bound_duals_.z_upper_[k] * (b.upper_val_[k] - x_new[b.upper_idx_[k]]);
            count++;
        }
        // An active nested restoration phase's elastic pairs belong in this
        // average for exactly the reason they belong in complementarity()'s:
        // they are complementary at restoration scale long after the original
        // pairs collapse to solve-tolerance, and an average that omits them
        // describes a central path the iterate is nowhere near. The aggregate
        // this clamp uses is then the same one the barrier oracle is being
        // driven by. Dead off the nested path.
        if (this->restoration_ && this->restoration_->is_active() &&
            this->restoration_->is_nested()) {
            double esum = 0.0;
            double emin = 0.0;
            double emax = 0.0;
            int ecount = 0;
            this->restoration_->nested_complementarity(esum, emin, emax, ecount);
            sum += esum;
            count += ecount;
        }

        // count > 0 is guaranteed: bounds_ is non-null only for a non-empty set.
        // Every product in the average is positive, because the
        // fraction-to-boundary rule keeps every slack, every inequality
        // multiplier and every bound multiplier strictly positive -- which is
        // what makes the average a usable barrier parameter at all.
        mu_clip = std::min(sum / double(count), kFreeModeClipMuCap);
    }

    // The clamp is measured at the NEW x: it is a safeguard on how far the
    // committed multipliers may sit off the primal-dual central path of the
    // point they now belong to, not of the point they were computed at.
    auto clamp = [mu_clip](double z, double d) {
        return std::max(std::min(z, kKappaSigma * mu_clip / d), mu_clip / (kKappaSigma * d));
    };
    for (int k = 0; k < nl; k++) {
        const int i = b.lower_idx_[k];
        this->bound_duals_.z_lower_[k] =
            clamp(this->bound_duals_.z_lower_[k], x_new[i] - b.lower_val_[k]);
    }
    for (int k = 0; k < nu; k++) {
        const int i = b.upper_idx_[k];
        this->bound_duals_.z_upper_[k] =
            clamp(this->bound_duals_.z_upper_[k], b.upper_val_[k] - x_new[i]);
    }

    // The direction has been consumed; clear it. dz is only meaningful between
    // the Newton solve that produced it and this commit, and the
    // fraction-to-boundary rule now READS it -- so a direction left standing
    // here would be re-consumed by the next iteration's PROBE predictor scaling,
    // which runs before that iteration's direction is computed. Zeroing makes
    // that staleness structurally impossible rather than merely unlikely: an
    // all-zero dz caps nothing, which is the right answer for a predictor step
    // that takes no bound-dual step at all.
    this->bound_duals_.dz_lower_.setZero();
    this->bound_duals_.dz_upper_.setZero();
}

void tycho::solvers::PSIOPT::add_bound_sigma(ConstEigenRef<Eigen::VectorXd> x,
                                             EigenRef<Eigen::VectorXd> diag) const {
    if (!this->bounds_)
        return;
    detail::accumulate_bound_sigma(x, *this->bounds_, this->bound_duals_, diag);
}

void tycho::solvers::PSIOPT::install_primal_diags_with_sigma(ConstEigenRef<Eigen::VectorXd> x,
                                                             double base) {
    if (!this->bounds_) {
        this->nlp_->set_primal_diags(base);
        return;
    }
    this->bound_sigma_scratch_ = Eigen::VectorXd::Constant(this->primal_vars_, base);
    this->add_bound_sigma(x, this->bound_sigma_scratch_);
    this->nlp_->set_primal_diags(this->bound_sigma_scratch_);
}

double
tycho::solvers::PSIOPT::dual_infeasibility_inf(ConstEigenRef<Eigen::VectorXd> prim_base) const {
    if (!this->bounds_)
        return prim_base.lpNorm<Eigen::Infinity>();
    this->bound_resid_scratch_ = prim_base;
    detail::accumulate_bound_dual_terms(*this->bounds_, this->bound_duals_,
                                        this->bound_resid_scratch_);
    return this->bound_resid_scratch_.lpNorm<Eigen::Infinity>();
}

// =============================================================================
// NLP eval dispatch methods
// =============================================================================

template <class Fn>
void tycho::solvers::PSIOPT::eval_dispatch(Fn fn, double obj_scale, ConstEigenRef<VectorXd> XSL,
                                           double &val, EigenRef<VectorXd> GX,
                                           EigenRef<VectorXd> AGXS_FX,
                                           Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    ((*this->nlp_).*fn)(
        obj_scale, XSL.head(primal_vars_), XSL.segment(primal_vars_ + slack_vars_, equal_cons_),
        XSL.tail(inequal_cons_), val, GX.head(primal_vars_), AGXS_FX.head(primal_vars_),
        AGXS_FX.segment(primal_vars_ + slack_vars_, equal_cons_), AGXS_FX.tail(inequal_cons_),
        KKTmat);
}

void tycho::solvers::PSIOPT::eval_kkt(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->eval_dispatch(&NonLinearProgram::eval_kkt, obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
}

void tycho::solvers::PSIOPT::eval_kkt_no(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                         EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                         Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->eval_dispatch(&NonLinearProgram::eval_kkt_no, obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
}

void tycho::solvers::PSIOPT::eval_aug(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->eval_dispatch(&NonLinearProgram::eval_aug, obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
}

void tycho::solvers::PSIOPT::eval_soe(double obj_scale, ConstEigenRef<VectorXd> XSL, double &val,
                                      EigenRef<VectorXd> GX, EigenRef<VectorXd> AGXS_FX,
                                      Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat) {
    this->eval_dispatch(&NonLinearProgram::eval_soe, obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
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

// Re-reads the problem dimensions from the NLP. The primal width is the SOLVER's
// -- the NLP's variable count minus whatever the fixed-variable treatment
// eliminated -- so every vector this solver sizes, every KKTVector segment and
// the KKT matrix all agree on one space. full_primal_vars_ keeps the problem's
// own count, which is the only width the outside world ever sees: it is what an
// initial guess must have and what a returned solution is expanded back to.
// Called by set_nlp and again at solve entry whenever a configuration changed
// the reduction.
void tycho::solvers::PSIOPT::refresh_nlp_dimensions() {
    this->full_primal_vars_ = this->nlp_->primal_vars_;
    this->primal_vars_ = this->nlp_->reduced_primal_vars();
    this->equal_cons_ = this->nlp_->equal_cons_;
    this->inequal_cons_ = this->nlp_->inequal_cons_;
    this->slack_vars_ = this->nlp_->slack_vars_;
    this->kkt_dim_ = this->nlp_->kkt_dim_;
    if (kkt_dim_ != primal_vars_ + slack_vars_ + equal_cons_ + inequal_cons_)
        throw std::logic_error(
            fmt::format("PSIOPT: NLP kkt_dim ({}) != solver primal_vars ({}) + slack_vars ({}) "
                        "+ equal_cons ({}) + inequal_cons ({})",
                        kkt_dim_, primal_vars_, slack_vars_, equal_cons_, inequal_cons_));
}

void tycho::solvers::PSIOPT::set_nlp(std::shared_ptr<NonLinearProgram> np) {
    if (!np)
        throw std::invalid_argument("PSIOPT::set_nlp: NonLinearProgram pointer must not be null");
    this->nlp_ = np;
    // Any bound set this solver was pointing at belonged to the previous NLP,
    // and the multipliers indexed it. The next solve's configuration step
    // re-reads both.
    this->bounds_ = nullptr;
    this->bound_duals_ = BoundDualState{};
    this->refresh_nlp_dimensions();

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

// (Re)builds the five globalization components from the CURRENT Settings
// (acceptance_/mechanism_/governor_/recovery_, always constructed, plus the
// optional restoration_). Called once per run_phase_sequence(), right after the
// variable-treatment configuration and before the first phase — i.e. once per
// solve invocation (optimize()/solve()/solve_optimize()/etc.
// all route through run_phase_sequence()) — rather than only from set_nlp()
// (which runs only on (re)transcription). This makes construction-time knobs
// (acceptance_strategy, max_soc, ls_extended_iters, watchdog,
// merit_penalty_rule, barrier_governor, restoration_mode) live at the next
// solve even when no set_nlp() call intervenes, matching every other Settings
// field (which alg_impl/governor_/etc. already read live off settings_ each
// iteration). Before this fix, these knobs were snapshotted at whichever
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
// equal_cons_/inequal_cons_/kkt_dim_ by const reference) reproduce
// bit-identical captures to the old per-transcription construction.
//
// Those dimension captures are BY REFERENCE, which is load-bearing now that the
// fixed-variable treatment can narrow the problem after this call: the
// configuration step later in run_phase_sequence() may rewrite primal_vars_ and
// kkt_dim_, and every component reads the updated values through its reference.
// A component that copied them here would be left describing a problem that no
// longer exists.
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
                          this->stli_scratch_, this->restoration_.get(), &this->eval_error_log_,
                          this->bounds_, &this->bound_duals_});
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
    // The z-FORM dual infeasibility. On a problem without variable bounds this
    // is the prim_grad() norm it always was; with bounds it additionally carries
    // -z_L + z_U, which the RHS block itself deliberately does not (that block
    // has to serve as the condensed Newton right-hand side, in the mu-form).
    // Both call sites of this function run OUTSIDE alg_impl's Newton bracket, so
    // prim_grad() is the base block here.
    iter.kkt_inf_ = this->dual_infeasibility_inf(rhs.prim_grad());

    double avgcomp = 0;
    double mincomp = 0;
    double maxcomp = 0;
    if (inequal_cons_ > 0) {
        iter.icon_inf_ = rhs.iq_cons().lpNorm<Eigen::Infinity>();
        iter.max_i_mult_ = xsl.iq_lmults().lpNorm<Eigen::Infinity>();
    }
    // The barrier account covers the inequality slack pairs AND the variable-
    // bound pairs, so it runs whenever either exists. A bounded problem with no
    // inequality constraints still has complementarity to report, and leaving
    // barr_inf_ at zero there would make the convergence check's barrier tier
    // pass vacuously -- the solve would report converged with its bound
    // multipliers still far off the central path. Byte-identical off the bound
    // path: same statements, same order, same guard value.
    if (inequal_cons_ > 0 || this->bounds_) {
        this->complementarity(xsl.primals(), xsl.slacks(), xsl.iq_lmults(), avgcomp, mincomp,
                              maxcomp);
        // While a nested restoration phase is active, the barrier error the
        // convergence check and the monitored governor's KKT-error monitor read
        // must reflect the elastic pairs too (dead no-op otherwise).
        this->augment_complementarity_nested(
            avgcomp, mincomp, maxcomp,
            this->complementarity_pair_count(static_cast<int>(xsl.slacks().size())));

        iter.barr_inf_ = maxcomp;
    }
    if (equal_cons_ > 0) {
        iter.econ_inf_ = rhs.eq_cons().lpNorm<Eigen::Infinity>();
        iter.max_e_mult_ = xsl.eq_lmults().lpNorm<Eigen::Infinity>();
    }
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
            this->add_bound_sigma(XSL.head(primal_vars_), this->resto_pdiag_scratch_);
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
        if (this->bounds_) {
            this->bound_sigma_scratch_ = this->restoration_->proximal_diagonal();
            this->add_bound_sigma(XSL.head(primal_vars_), this->bound_sigma_scratch_);
            this->nlp_->set_primal_diags(this->bound_sigma_scratch_);
        } else {
            this->nlp_->set_primal_diags(this->restoration_->proximal_diagonal());
        }
        eval_kkt_no(0.0, XSL, val, GX, AGXS_FX, KKTmat);
        this->nlp_->set_primal_diags(0.0);
        val = this->restoration_->proximal_objective(XSL.head(primal_vars_));
        this->restoration_->add_proximal_gradient(XSL.head(primal_vars_), GX.head(primal_vars_));
        return;
    }

    // Condensed variable-bound curvature. Eliminating the bound-multiplier rows
    // leaves Sigma on the Hessian (1,1) diagonal, which is exactly the
    // primal-diagonal slot the modes below already write -- so it composes with
    // whatever base each mode wants (0 for the optimality modes, 1 for SOE, the
    // restoration bases above) instead of needing rows of its own, and the KKT
    // dimension does not grow. The inertia ladder's own perturb_kkt_p_diags
    // accumulates on top of it afterwards, unchanged.
    //
    // The OPT/OPTNO arms guard on bounds_ rather than routing through the shared
    // helper: their base is 0.0, which is what the slots already hold, so without
    // bounds they must issue no diagonal write at all.
    //
    // AlgorithmModes::INIT is deliberately absent. It is not an iteration -- it
    // is init_impl's one-shot multiplier initializer, which sets the primal
    // diagonal to 1.0 itself and solves a least-squares system for the equality
    // multipliers; folding bound curvature into that system would change the
    // multipliers a solve starts from, which is not what condensing the bound
    // rows out of the ITERATION's KKT system is about.
    switch (algmode) {
    case AlgorithmModes::OPT:
        if (this->bounds_)
            this->install_primal_diags_with_sigma(XSL.head(primal_vars_), 0.0);
        eval_kkt(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        if (this->bounds_)
            this->nlp_->set_primal_diags(0.0);
        break;
    case AlgorithmModes::OPTNO:
        if (this->bounds_)
            this->install_primal_diags_with_sigma(XSL.head(primal_vars_), 0.0);
        eval_kkt_no(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        if (this->bounds_)
            this->nlp_->set_primal_diags(0.0);
        break;
    case AlgorithmModes::INIT:
        eval_aug(obj_scale, XSL, val, GX, AGXS_FX, KKTmat);
        break;
    case AlgorithmModes::SOE:
        this->install_primal_diags_with_sigma(XSL.head(primal_vars_), 1.0);
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
    entry.infeasibility = this->constraint_violation_l1(v_rhs);
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

// Restoration-entry dispatch: one owner of the note_dispatch / entry /
// reset_window ordering. See the declaration in psiopt.h.
void tycho::solvers::PSIOPT::dispatch_restoration_entry(Eigen::VectorXd &XSL, Eigen::VectorXd &RHS,
                                                        double prim_obj, double barr_obj,
                                                        double &mu, double theta,
                                                        FeasibilityStallDetector &feas_stall) {
    // A stage resumed after an episode restarts its stall window, and this entry
    // becomes the handback the stall exit measures net progress against.
    feas_stall.note_dispatch(theta);
    this->enter_feasibility_restoration(XSL, RHS, prim_obj, barr_obj, mu);
    feas_stall.reset_window();
}

// Restoration-exit protocol. See the declaration in psiopt.h for why this order
// is load-bearing.
void tycho::solvers::PSIOPT::leave_restoration(const ProgressMeasures &measures,
                                               bool restore_stashed_mu, double &mu) {
    if (restore_stashed_mu) {
        mu = this->stashed_mu_;
        this->governor_->reset();
    }
    this->restoration_->exit_restoration();
    this->acceptance_->notify_switch_to_optimality(measures);
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
    // trial. (2) Newton complementarity step under the STASHED outer μ (Ipopt
    // ComputeBoundMultiplierStep) for EVERY bound-multiplier family the solver
    // carries: the inequality slacks and, when the problem declares variable
    // bounds, both sides of those. The general step
    // Δz = [(d_curr − d_trial)·z + μ_outer]/d_curr − z is taken at
    // d_curr == d_trial, giving Δz = μ_outer/d − z.
    //
    // For the slack family that reduction is forced: no pre-restoration s is
    // retained, so there is no other d_curr to form. For the bound family it is
    // a CHOICE — the nested phase's proximal term keeps the entry x as its
    // reference, so a genuine (x_curr − x_trial) could be formed. Taking the
    // same reduction is the better of the two: one re-anchoring rule across the
    // families beats being faithful in one and simplified in the other, and the
    // whole point of the step is to put every multiplier back on the outer
    // barrier parameter's scale, which μ_outer/d does directly.
    //
    // ONE shared dual fraction-to-boundary damping across all families (Ipopt
    // takes a single alpha_dual over all four of its), so the family whose cap
    // binds shortens the step for every family — τ = bound_fraction_, the same
    // τ the solver's own dual steps use, through the same kernel. (3) If the
    // largest updated multiplier ACROSS ALL FAMILIES exceeds
    // kBoundMultResetThreshold, reset them all to 1 (Ipopt resets every family,
    // not just the one that tripped it). The reset value survives the next
    // commit's κ_Σ clip untouched — that bracket is ten orders wide on each
    // side — so it is faithful and inert rather than immediately undone.
    //
    // Every family's step is formed BEFORE any is applied: the shared fraction
    // is a minimum over all of them, so stepping one early would damp it by a
    // fraction taken without the others in view.
    const double tau = settings_.bound_fraction_;
    const BoundSet *bounds = this->bounds_;
    const int nl = bounds ? static_cast<int>(bounds->lower_idx_.size()) : 0;
    const int nu = bounds ? static_cast<int>(bounds->upper_idx_.size()) : 0;
    double alpha_dual = 1.0;

    if (this->inequal_cons_ > 0) {
        auto s = v_xsl.slacks();
        auto z = v_xsl.iq_lmults();
        this->resto_dz_scratch_.resize(this->inequal_cons_);
        this->resto_dz_scratch_ = this->stashed_mu_ * s.cwiseInverse() - z;
        alpha_dual = std::min(alpha_dual, detail::max_step_to_boundary(z, this->resto_dz_scratch_,
                                                                       tau, this->inequal_cons_));
    }
    if (nl > 0) {
        auto x = v_xsl.primals();
        this->resto_bound_dz_lower_scratch_.resize(nl);
        for (int k = 0; k < nl; k++) {
            const double d = x[bounds->lower_idx_[k]] - bounds->lower_val_[k];
            this->resto_bound_dz_lower_scratch_[k] =
                this->stashed_mu_ / d - this->bound_duals_.z_lower_[k];
        }
        alpha_dual = std::min(
            alpha_dual, detail::max_step_to_boundary(this->bound_duals_.z_lower_,
                                                     this->resto_bound_dz_lower_scratch_, tau, nl));
    }
    if (nu > 0) {
        auto x = v_xsl.primals();
        this->resto_bound_dz_upper_scratch_.resize(nu);
        for (int k = 0; k < nu; k++) {
            const double d = bounds->upper_val_[k] - x[bounds->upper_idx_[k]];
            this->resto_bound_dz_upper_scratch_[k] =
                this->stashed_mu_ / d - this->bound_duals_.z_upper_[k];
        }
        alpha_dual = std::min(
            alpha_dual, detail::max_step_to_boundary(this->bound_duals_.z_upper_,
                                                     this->resto_bound_dz_upper_scratch_, tau, nu));
    }

    double bound_mult_max = 0.0;
    if (this->inequal_cons_ > 0) {
        auto z = v_xsl.iq_lmults();
        z += alpha_dual * this->resto_dz_scratch_;
        bound_mult_max = std::max(bound_mult_max, z.cwiseAbs().maxCoeff());
    }
    if (nl > 0) {
        this->bound_duals_.z_lower_ += alpha_dual * this->resto_bound_dz_lower_scratch_;
        bound_mult_max =
            std::max(bound_mult_max, this->bound_duals_.z_lower_.cwiseAbs().maxCoeff());
    }
    if (nu > 0) {
        this->bound_duals_.z_upper_ += alpha_dual * this->resto_bound_dz_upper_scratch_;
        bound_mult_max =
            std::max(bound_mult_max, this->bound_duals_.z_upper_.cwiseAbs().maxCoeff());
    }
    if (bound_mult_max > kBoundMultResetThreshold) {
        if (this->inequal_cons_ > 0)
            v_xsl.iq_lmults().setConstant(1.0);
        if (nl > 0)
            this->bound_duals_.z_lower_.setConstant(1.0);
        if (nu > 0)
            this->bound_duals_.z_upper_.setConstant(1.0);
    }

    // (4) Equality constraint multipliers ← 0 (Ipopt least_square_mults at the
    // shipped-default reset threshold 0 sets y_c/y_d to zero; verified against
    // IpDefaultIterateInitializer).
    if (this->equal_cons_ > 0)
        v_xsl.eq_lmults().setZero();

    // (5) The shared exit protocol, with the stashed outer μ restored: exit
    // restoration and notify the acceptance strategy of the switch back to
    // optimality with true-objective exit measures (the loop's prim_obj is φ_l1
    // while active — never valid for the optimality filter/funnel; re-evaluated
    // here), then reset the recovery chain. BestXSL tracking resumes implicitly
    // once is_active() flips false.
    this->leave_restoration(
        this->build_restoration_exit_measures(obj_scale, theta_orig, v_xsl.primals(), barr_obj),
        true, mu);
}

// Per-iteration κ_resto ratchet: the original-problem infeasibility must fall to
// at most max(kKappaResto · previous-iteration value, econ_tol_) (Ipopt
// RestoConvCheck::CheckConvergence's orig_inf_pr_max, single-tolerance floor).
bool tycho::solvers::PSIOPT::resto_ratchet_passes(double theta_orig) const {
    return theta_orig <= std::max(kKappaResto * this->resto_theta_orig_prev_, settings_.econ_tol_);
}

// ‖c‖₁ over a KKT vector's constraint block. See the declaration in psiopt.h.
double tycho::solvers::PSIOPT::constraint_violation_l1(KKTVector &v) const {
    return v.all_cons().template lpNorm<1>();
}

// Original-problem infeasibility (∞-norm) from the nested restoration eval
// seam's saved raw residuals. Two separate reductions, deliberately not fused;
// see the declaration in psiopt.h.
double tycho::solvers::PSIOPT::original_infeasibility_inf() const {
    double theta_orig = 0.0;
    if (this->equal_cons_ > 0)
        theta_orig =
            std::max(theta_orig, this->resto_ec_scratch_.template lpNorm<Eigen::Infinity>());
    if (this->inequal_cons_ > 0)
        theta_orig =
            std::max(theta_orig, this->resto_ic_scratch_.template lpNorm<Eigen::Infinity>());
    return theta_orig;
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
double tycho::solvers::PSIOPT::primal_dual_error(KKTVector &xsl, KKTVector &rhs,
                                                 ConstEigenRef<Eigen::VectorXd> prim_base,
                                                 double mu) const {
    double err = 0.0;
    if (this->primal_vars_ > 0)
        err = std::max(err, this->dual_infeasibility_inf(prim_base));
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
    //
    // Its primal block is staged in the mu-form, though -- this call sits inside
    // alg_impl's Newton bracket -- so the BASE block comes from the snapshot the
    // bracket took. Measuring the current point mu-form while the trial below is
    // measured base-form would put extra -mu/(x-l) + mu/(u-x) mass on one side
    // of the reduction test only, biasing it toward accepting the soft step and
    // away from escalating to restoration. Off the bound path the staging is a
    // no-op and the two expressions name the same numbers.
    const double curr_pd =
        this->bounds_ ? this->primal_dual_error(v_xsl, v_rhs, this->bound_grad_scratch_, mu)
                      : this->primal_dual_error(v_xsl, v_rhs, v_rhs.prim_grad(), mu);
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
    // The trial RHS was just built by the eval above and never staged, so its
    // primal block IS the base block -- the same form the current point is
    // measured in above.
    const double trial_pd = this->primal_dual_error(v_xsl2, v_rhs2, v_rhs2.prim_grad(), mu);

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
                                        double base_prox, double dual_shift, bool &exhausted) {
    auto Inertia = [&]() {
        return this->kkt_sol_.neigs() - (this->equal_cons_ + this->inequal_cons_);
    };
    // Full Ipopt inertia-correction condition (Algorithm IC, Wächter & Biegler
    // 2006): accept only inertia exactly (kkt_dim - m, m, 0). Singular() is the
    // rank-deficiency part; Inertia() != 0 covers both excess (the only case the
    // pre-2026-07 ladder corrected) and missing negative eigenvalues.
    auto Singular = [&]() {
        return (this->kkt_sol_.neigs() + this->kkt_sol_.peigs() - this->kkt_dim_) != 0;
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
    auto PerturbC = [&](double p) {
        this->nlp_->perturb_kkt_c_diags(p, this->kkt_sol_.get_matrix());
    };
    // On-demand dual regularization (delta_c). dual_shift is the magnitude
    // available to this call (0.0 while a nested l1 restoration phase owns the
    // constraint-row diagonals -- the caller suppresses it). The proximal branch
    // applies it up-front as part of the base matrix; the classic branch applies
    // it here, at most once per call, the first time a factorization reports
    // the singularity signal (rank deficiency, or neigs < m -- see
    // SingularitySignal below) -- it lands in the matrix and takes effect at the next
    // Refactor(), so a singular base costs one ladder rung (a small delta_w
    // rides along with delta_c, matching Ipopt, which raises both on
    // singularity). dc_latched_ is the Ipopt hess/jac-degenerate adaptation:
    // sticky per phase, it makes later calls pre-apply delta_c at the base
    // attempt instead of re-paying the singular factorization every iteration.
    bool dc_applied = false;
    auto EngageDualReg = [&]() {
        if (!dc_applied && dual_shift != 0.0) {
            PerturbC(-dual_shift);
            dc_applied = true;
            this->dc_latched_ = true;
        }
    };
    int IncEigs = 0;
    cumpert = 0.0;
    // Engagement signal for on-demand delta_c: fires on Singular() (rank
    // deficiency, zeigs != 0) OR IncEigs < 0 (neigs < m). By Gould's inertia
    // theorem, In(KKT) = In(Z^T H Z) + (m, m, 0) for a full-rank constraint
    // Jacobian, so neigs < m with zeigs == 0 cannot occur for a full-rank J --
    // it is a masked rank-deficiency report from a pivot-perturbing backend
    // (e.g. MKL Pardiso's static pivot perturbation). delta_w cannot correct
    // this case (Weyl's inequality: +p*I on the primal diagonal is PSD, so
    // neigs is non-increasing in p, never able to grow to m), so treat it as
    // the singularity signal it is and let delta_c (the one shift that raises
    // neigs) engage.
    auto SingularitySignal = [&]() { return Singular() || IncEigs < 0; };

    // Latched pre-apply, classic mode only: once dc_latched_ is set (a prior
    // call this phase observed SingularitySignal()), pre-apply delta_c before
    // this rung's factorization even runs -- hoisted above the Zfac/docompute
    // gate below (unlike the mode branches, this must fire on EVERY classic
    // call regardless of the fast_factor_alg_ cycling heuristic, or the latch
    // would be bypassed on ~3 of every 4 iterations, defeating its purpose).
    // Proximal applies delta_c itself (see its branch below) and never
    // consults the latch.
    if (settings_.inertia_mode_ != InertiaModes::proximal_regularization && this->dc_latched_)
        EngageDualReg();

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
        Perturb(base_prox);
        if (dual_shift != 0.0) {
            PerturbC(-dual_shift);
            dc_applied = true;
        }
        if (!docompute)
            Refactor();
        else
            Compute();
        CheckInfo();
        RankDef();
        IncEigs = Inertia();
        finalpert = 0.0;
        // A singular or wrong-inertia base factorization enters the ladder.
        if (IncEigs == 0 && !Singular())
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
        if (SingularitySignal())
            EngageDualReg();
        if (IncEigs == 0 && !Singular())
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

        if (SingularitySignal())
            EngageDualReg();
        if (IncEigs == 0 && !Singular())
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
                   "inertia p/n/z = {}/{}/{}, expected {}/{}/0)\n",
                   settings_.max_refac_, this->kkt_sol_.peigs(), this->kkt_sol_.neigs(),
                   this->kkt_dim_ - this->kkt_sol_.peigs() - this->kkt_sol_.neigs(),
                   this->kkt_dim_ - (this->equal_cons_ + this->inequal_cons_),
                   this->equal_cons_ + this->inequal_cons_);
    exhausted = true;
    return settings_.max_refac_;
}

// Best-iterate scoring + snapshot for the return_best_ path. See the declaration
// in psiopt.h; the return_best_ / restoration-active guard stays at each call
// site.
void tycho::solvers::PSIOPT::track_best_iterate(const IterateInfo &iter, int i,
                                                const VectorXd &XSL, const VectorXd &RHS,
                                                double &BestCriteriaVal, int &BestIter) {
    double critval;
    switch (settings_.best_criteria_) {
    case BestCriteriaModes::ECONS:
        critval = iter.econ_inf_;
        break;
    case BestCriteriaModes::ICONS:
        critval = iter.icon_inf_;
        break;
    case BestCriteriaModes::KKT:
        critval = iter.kkt_inf_;
        break;
    case BestCriteriaModes::OBJ:
        critval = iter.prim_obj_;
        break;
    default:
        throw std::invalid_argument("Unknown BestCriteriaModes");
    }
    if (critval <= BestCriteriaVal || i == 0) {
        BestCriteriaVal = critval;
        this->best_xsl_scratch_ = XSL;
        this->best_rhs_scratch_ = RHS;
        BestIter = i;
    }
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
    // Fresh phase: re-probe rank rather than inheriting the previous phase's
    // degeneracy diagnosis.
    this->dc_latched_ = false;

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
    SolverContext ctx{this->nlp_.get(),         this->kkt_sol_,         this->settings_,
                      this->primal_vars_,       this->slack_vars_,      this->equal_cons_,
                      this->inequal_cons_,      this->kkt_dim_,         this->stli_scratch_,
                      this->restoration_.get(), &this->eval_error_log_, this->bounds_,
                      &this->bound_duals_};

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
    // (never set true unless restoration_ is non-null). Three terminal
    // restoration exits leave the loop with the last-filled
    // iters.back().prim_obj_ still at φ_prox (the proximal objective
    // substituted while restoration was active): the nested mode's in-loop
    // "converged to a locally infeasible point" break, the proximal mode's
    // equivalent in-loop break, and the post-loop max-iters/DIVERGING-while-
    // active catch-all. All three record the true objective (re-evaluated via
    // build_restoration_exit_measures) here so the unconditional algmode==OPT
    // obj_val_ assignment below the main loop can be corrected afterward,
    // once, in a single place.
    bool restoration_was_active = false;
    double restoration_true_obj = 0.0;

    Runtimer.start();
    for (int i = 0; i < settings_.max_iters_; i++) {
        IterateInfo Citer;
        Citer.iter_ = i;

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
        }
        // The complementarity account and the barrier-parameter update below run
        // whenever there is ANY barrier term to drive -- inequality slacks or
        // variable bounds. The slack-reset and slack-Hessian work above stays
        // behind the narrower guard: those are about the slack block alone, and
        // there is no slack block here. Byte-identical off the bound path.
        if (this->inequal_cons_ > 0 || this->bounds_) {
            this->complementarity(v_xsl.primals(), v_xsl.slacks(), v_xsl.iq_lmults(), avgcomp,
                                  mincomp, maxcomp);
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
            this->augment_complementarity_nested(
                avgcomp, mincomp, maxcomp,
                this->complementarity_pair_count(static_cast<int>(v_xsl.slacks().size())));
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

        // RHS PRIMAL-BLOCK STAGING (native variable bounds). From here the
        // primal block holds grad f + J'lambda and NOTHING bound-related. That is
        // deliberate, and it is the same vector in three different roles this
        // iteration:
        //
        //   1. RESIDUAL. The convergence account wants the dual infeasibility
        //      grad f + J'lambda - z_L + z_U. It is never accumulated here --
        //      dual_infeasibility_inf() adds the z-form in scratch instead, at
        //      both consuming sites (fill_residual_info now, fill_iter_info at
        //      the bottom of the loop).
        //   2. AFFINE NEWTON RIGHT-HAND SIDE. The PROBE predictor solve inside
        //      update_barrier() below is the mu = 0 system, whose condensed
        //      primal right-hand side is exactly grad f + J'lambda -- the bound
        //      terms cancel out at mu = 0, precisely as the slack block's
        //      predictor form (lambda, not lambda - mu/s) does. So the predictor
        //      needs no bound work at all, and gets none.
        //   3. CORRECTOR NEWTON RIGHT-HAND SIDE. The real step solve needs the
        //      mu-form, grad f + J'lambda - mu/(x-l) + mu/(u-x). It is installed
        //      just before that solve, at the barrier parameter the solve is
        //      formed at, and copied back out after the step is settled.
        //
        // Roles 1 and 3 disagree away from the central path, which is why the
        // mu-form lives in a bracket rather than being written once here.

        // Check convergence before factorizing the converged iterate: every
        // residual converge_check() consumes is now
        // fully determined -- kkt_inf_ reads prim_grad() (just updated above; the
        // barrier writes later this iteration target the *disjoint* dual_grad()
        // block, see tycho::solvers::KKTVector::prim_grad()/dual_grad()
        // (include/tycho/detail/solvers/kkt_vector.h) for their segment
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
            // Set by whichever arm below finds the restoration subproblem
            // converged/stalled at a still-infeasible point. Both arms then share
            // the single teardown-and-report block after the if/else: the arms
            // differ only in how θ is measured (nested reads the seam's saved raw
            // residuals; proximal reads the live KKT constraint block) and in
            // whether the stashed outer μ has to be restored — which the shared
            // block reads straight off is_nested(), since exit_restoration() has
            // not run yet at that point.
            bool locally_infeasible = false;
            double locally_infeasible_theta = 0.0;
            double locally_infeasible_threshold = 0.0;

            if (this->restoration_->is_nested()) {
                // Nested l1 restoration exit tests (Ipopt RestoConvCheck structure):
                // first-iteration guard, then the per-iteration κ_resto ratchet, then
                // the acceptance-strategy exit test; all three pass → the full
                // multiplier re-entry sequence. θ_orig is the ORIGINAL-problem
                // infeasibility at the current point. The eval seam replaced the RHS
                // constraint rows with the condensed r̃, so the raw residuals come
                // from the seam's saved copies (resto_ec_/ic_scratch_, populated this
                // same iteration before the r̃ overwrite) — no extra NLP evaluation.
                const double theta_orig = this->original_infeasibility_inf();

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
                    // Locally infeasible: hand off to the shared teardown below
                    // (which restores μ and resets the governor for this arm) and
                    // stop NOT-converged. No multiplier re-entry — the phase failed.
                    locally_infeasible = true;
                    locally_infeasible_theta = theta_orig;
                    locally_infeasible_threshold = resto_failure_threshold;
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
                cur.infeasibility = this->constraint_violation_l1(v_rhs);
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
                        this->leave_restoration(
                            this->build_restoration_exit_measures(obj_scale, cur.infeasibility,
                                                                  v_xsl.primals(), barr_obj),
                            false, mu);
                        iters.pop_back();
                        QPtimer.stop();
                        continue;
                    }
                    // Proximal subproblem converged/stalled at a still-infeasible
                    // point: the problem is locally infeasible (Ipopt's
                    // restoration-convergence failure classification). Hand off to
                    // the shared teardown below, which tears restoration down BEFORE
                    // returning so the phase-boundary reset() sees optimality mode,
                    // then stops with the NOT-converged verdict.
                    locally_infeasible = true;
                    locally_infeasible_theta = cur.infeasibility;
                    locally_infeasible_threshold = resto_failure_threshold;
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
                        this->leave_restoration(
                            this->build_restoration_exit_measures(obj_scale, cur.infeasibility,
                                                                  v_xsl.primals(), barr_obj),
                            false, mu);
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

            if (locally_infeasible) {
                // The restoration subproblem converged/stalled at a still-infeasible
                // point: the problem is locally infeasible (Ipopt's restoration-
                // convergence failure classification). Tear restoration down BEFORE
                // returning so the phase-boundary reset() sees optimality mode, then
                // stop with the NOT-converged verdict. Shared by both modes; the
                // nested arm additionally restores the stashed outer μ and resets
                // the governor (the proximal arm never stashed one).
                {
                    // The notify measures record the point restoration exited (the
                    // live iterate) — the filter augment describes the exit point
                    // itself, independent of the return_best_ reporting substitution
                    // below.
                    ProgressMeasures exit_measures = this->build_restoration_exit_measures(
                        obj_scale, locally_infeasible_theta, v_xsl.primals(), barr_obj);
                    restoration_was_active = true;
                    // Count this exit iteration in the in-mode total (see the
                    // near-feasible exits above).
                    this->restoration_->note_iteration();
                    this->leave_restoration(exit_measures, this->restoration_->is_nested(), mu);
                }
                iters.back().mu_ = mu;
                QPtimer.stop();
                ExitCode = ConvergenceFlags::NOTCONVERGED;
                if (settings_.return_best_) {
                    XSL = BestXSL;
                    RHS = BestRHS;
                }
                // obj_val_ must describe the RETURNED primals: evaluate after the
                // return_best_ substitution above (which may have replaced XSL),
                // unlike the notify measures, which record the exit point. With
                // return_best_ off the two evaluations coincide.
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
                               locally_infeasible_theta, locally_infeasible_threshold);
                break;
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
            if (settings_.return_best_ && !(this->restoration_ && this->restoration_->is_active()))
                this->track_best_iterate(iters.back(), i, XSL, RHS, BestCriteriaVal, BestIter);

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
            const double theta_fs = this->constraint_violation_l1(v_rhs);
            if (feas_stall.observe(theta_fs)) {
                if (this->restoration_->entry_permitted(theta_fs, ctx)) {
                    // Entry measures are (theta, 0, 0) here: prim_obj and barr_obj are
                    // still their pre-factorization 0.0 initializers (eval_soe/eval_kkt_no
                    // never write the objective, and the barrier objective is computed
                    // further down), matching the other pre-factorization restoration
                    // seams above and unlike the post-line-search seams, which pass a
                    // live barrier objective.
                    this->dispatch_restoration_entry(XSL, RHS, prim_obj, barr_obj, mu, theta_fs,
                                                     feas_stall);
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

        // δ_c availability is computed for BOTH inertia modes: the classic
        // ladder engages it on demand when a factorization reports rank
        // deficiency (see factor_impl). Suppressed while a nested l1
        // restoration phase owns the constraint-row diagonals
        // (inertia_regularization.h).
        const bool dc_suppressed = this->restoration_ && this->restoration_->is_active() &&
                                   this->restoration_->is_nested();
        double base_prox = 0.0;
        double dual_shift = dc_suppressed ? 0.0 : tycho::solvers::dual_regularization(mu);
        if (settings_.inertia_mode_ == InertiaModes::proximal_regularization)
            base_prox = rho_k;

        bool kkt_exhausted = false;
        Citer.h_facs_ = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert, nhpert_cum,
                                          base_prox, dual_shift, kkt_exhausted);

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
        // The barrier parameter is updated whenever there is a barrier term to
        // drive. That used to mean inequality slacks; it now also means variable
        // bounds. Without the added disjunct a problem whose only barrier terms
        // are bound terms would hold mu at init_mu_ for the whole phase, its
        // bound complementarity would floor at that value, and the solve could
        // not reach the barrier tolerance at all. The oracles handle an empty
        // slack block on their own -- the objective sums nothing, the dual
        // gradient writes an empty segment, and loqo_mu reads the aggregates the
        // bound pairs now populate. Provably false off the bound path.
        if (this->inequal_cons_ > 0 || this->bounds_) {
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

        // Variable-bound contribution to THIS iterate's barrier objective. Every
        // consumer of barr_obj is a merit/acceptance account, and each of them
        // compares it against a trial-point barrier objective that carries the
        // matching bound term (added at the three trial evaluators in
        // psiopt_globalization.cpp), so the two sides stay in the same units.
        //
        // Added here rather than inside the governors, which is where the SLACK
        // barrier objective is produced, for one reason: that whole block is
        // guarded on inequal_cons_ > 0, and a problem whose only barrier terms
        // come from variable bounds has no slacks and never reaches a governor.
        // This is the single site that produces the current iterate's barr_obj,
        // so the bound term is added exactly once whichever governor ran.
        if (this->bounds_)
            barr_obj += detail::bound_barrier_objective(v_xsl.primals(), *this->bounds_, mu);

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

        // [NEWTON FORM IN] Install the mu-form variable-bound terms on the primal
        // right-hand side for the corrector solve (see the staging comment above
        // the convergence check). The block is snapshotted first so the restore
        // after the step is an exact copy back rather than an add-then-subtract
        // round trip, which would not return the same bits. Uses step_mu, the
        // barrier parameter the rest of this iteration's step algebra is formed
        // at; it differs from mu only inside a nested restoration phase.
        if (this->bounds_) {
            this->bound_grad_scratch_ = v_rhs.prim_grad();
            detail::accumulate_bound_barrier_gradient(v_xsl.primals(), *this->bounds_, step_mu,
                                                      v_rhs.prim_grad());
        }

        // The REAL step solve (distinct from the PROBE predictor solve, which
        // moved into ClassicAdaptiveGovernor::update_barrier — see its solve-into
        // comment): direct assignment + in-place negate avoids the extra
        // temporary that `-kkt_sol_.solve(RHS)` forces.
        DXSL = this->kkt_sol_.solve(RHS);
        DXSL = -DXSL;
        bool GoodStep = std::isfinite(DXSL.squaredNorm());

        // Bound-multiplier Newton step, recovered from the primal step the
        // condensed system just produced. Taken here, on the RAW solution, before
        // compute_step's fraction-to-boundary rule scales DXSL in place -- the
        // elimination that produced Sigma was written against the unscaled dx.
        if (this->bounds_ && GoodStep) {
            KKTVector v_dxsl_bounds = kkt_view(DXSL);
            this->compute_bound_dual_direction(v_xsl.primals(), v_dxsl_bounds.primals(), step_mu);
        }

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
        // Set by the exhausted-inertia-correction dispatch below: the KKT
        // factorization never reached correct inertia and neither an elastic
        // re-center nor a restoration entry was available to resolve it.
        // Terminates the phase as SINGULAR_KKT at the loop tail (same idiom as
        // exit_at_acceptable / exit_stage_stalled).
        bool singular_abort = false;

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

        // Inertia-correction exhaustion (Ipopt-faithful fail-the-step): a step
        // solved on a factorization that never reached correct inertia must not
        // be accepted on merit. The rejection is forced here -- the bookkeeping
        // below (and the acceptance accounting) reads accepted_ -- and the
        // dedicated dispatch below decides what happens instead. The non-finite
        // case (!GoodStep) already exits as DIVERGING at the loop tail -- the
        // non-finite verdict dominates and needs no special-casing here.
        if (kkt_exhausted)
            Citer.accepted_ = false;

        Funtimer.stop();

        // Recovery-chain hook. This is where a rejected step's
        // recovery gets a say -- SOC -> extended-backtrack -> watchdog-revert
        // -> feasibility-switch dispatch from this point (see
        // rebuild_globalization_components's wiring comment for how the
        // chain is assembled from settings_). The
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
        // step is accepted, so the hook is never invoked at all. It is
        // additionally gated on !kkt_exhausted: an exhausted factorization's
        // forced rejection is owned by the dedicated dispatch below, which runs
        // INSTEAD of the chain.
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
        //
        // Exhausted inertia correction takes its own route, bypassing the chain.
        // Every merit-retry link in it -- extended backtracking and the
        // watchdog's relaxed acceptance re-test or re-accept the SAME direction
        // the exhausted factorization produced; SOC and the soft feasibility
        // pre-stage re-solve or re-trial on that same never-correct-inertia
        // factorization -- so any of them could "resolve" the forced rejection
        // by committing a step no correct-inertia system vetted (an
        // indefinite-curvature direction can decrease the merit perfectly well
        // while walking into a saddle). Only two outcomes count as genuine
        // resolutions here, and neither commits such a step: re-centering the
        // elastic pairs of an active nested l1 phase, and entering feasibility
        // restoration. Failing both, the phase aborts. The reference method
        // treats the same condition as a step-computation error and goes to
        // restoration or gives up; it never re-tests the direction. Trial
        // evaluations that threw during this iteration's line search do not
        // divert this route: SINGULAR_KKT is decisive over the un-evaluable
        // dispatch's acceptable-tier exit (see the loop tail), and the
        // exception count still lands in the iterate via eval_exceptions_.
        if (kkt_exhausted && GoodStep) {
            int resolved_depth = kRecoveryDepthUnresolved;
            if (nested_active && this->try_recenter_elastics(step_mu)) {
                // Nested l1 restoration active: re-center the elastic pairs in
                // closed form at the current phase mu and discard the step
                // (alpha = 0 no-ops the XSL += alpha*DXSL commit below; the
                // re-centered elastics change the NEXT iteration's condensed
                // system). try_recenter_elastics enforces the one-shot budget,
                // so a second consecutive exhaustion falls through to the
                // restoration/abort decision below.
                alpha = 0.0;
                resolved_depth = kRecoveryDepthRestoration;
            } else {
                const double violation_sk = this->constraint_violation_l1(v_rhs);
                if (this->restoration_ && !this->restoration_->is_active() &&
                    this->restoration_->entry_permitted(violation_sk, ctx)) {
                    // Enter feasibility restoration, skipping the soft
                    // pre-stage, whose trial would be the very direction the
                    // exhausted factorization produced -- the same precedent as
                    // the un-evaluable-step dispatch in the chain below.
                    this->dispatch_restoration_entry(XSL, RHS, prim_obj, barr_obj, mu, violation_sk,
                                                     feas_stall);
                    alpha = 0.0;
                    resolved_depth = kRecoveryDepthRestoration;
                } else {
                    // Nothing can resolve it: discard the step and abort the
                    // phase as SINGULAR_KKT at the loop tail.
                    alpha = 0.0;
                    singular_abort = true;
                }
            }
            this->result_.recovery_depth_histogram_[resolved_depth]++;
        } else if (should_dispatch_recovery(GoodStep, Citer)) {
            int resolved_depth = kRecoveryDepthUnresolved;
            const RecoveryChain::Action recovery_action = this->recovery_->on_step_rejected(
                Citer, iters, ctx, *acceptance_, *mechanism_, lsmode, obj_scale * lsobjscale,
                step_mu, prim_obj, barr_obj, XSL, DXSL, Temp, RHS, RHS2, alpha, alphap, alphad,
                this->result_.soc_steps_taken_, resolved_depth,
                this->result_.watchdog_activations_);
            switch (recovery_action) {
            case RecoveryChain::Action::kAcceptAsIs:
                // No link resolved the ordinary rejection: take the step
                // compute_step produced (DXSL/alpha unchanged). Exhausted
                // inertia correction never reaches this switch — it takes the
                // dedicated dispatch above. EXCEPTION (dead off the nested
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
                    const double violation_ue = this->constraint_violation_l1(v_rhs);
                    if (psiopt_iterate_acceptable(Citer, settings_)) {
                        alpha = 0.0;
                        Citer.accepted_ = false;
                        exit_at_acceptable = true;
                    } else if (this->restoration_ && !this->restoration_->is_active() &&
                               this->restoration_->entry_permitted(violation_ue, ctx)) {
                        this->dispatch_restoration_entry(XSL, RHS, prim_obj, barr_obj, mu,
                                                         violation_ue, feas_stall);
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
                            Citer.iter_, this->eval_error_log_.count_ - eval_errs_before,
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
                this->dispatch_restoration_entry(XSL, RHS, prim_obj, barr_obj, mu,
                                                 this->constraint_violation_l1(v_rhs), feas_stall);
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
                    this->dispatch_restoration_entry(XSL, RHS, prim_obj, barr_obj, mu,
                                                     this->constraint_violation_l1(v_rhs),
                                                     feas_stall);
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

        // [NEWTON FORM OUT] The step is settled -- every consumer of the mu-form
        // primal block has run (the corrector solve, the line search's
        // directional derivative, and any recovery link that re-solved on the
        // live factorization). Put the block back to grad f + J'lambda so the
        // residual account below reads what it is documented to read, and so the
        // return_best_ snapshot stores a right-hand side in the same form the
        // top of the next iteration would build.
        if (this->bounds_)
            v_rhs.prim_grad() = this->bound_grad_scratch_;

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
        if (settings_.return_best_ && !(this->restoration_ && this->restoration_->is_active()))
            this->track_best_iterate(iters.back(), i, XSL, RHS, BestCriteriaVal, BestIter);

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

        // SINGULAR_KKT is decisive: an inertia-correction failure is a
        // step-computation error (Ipopt Error_In_Step_Computation), reported as
        // such even at an otherwise-acceptable iterate.
        if (singular_abort)
            ExitCode = ConvergenceFlags::SINGULAR_KKT;

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
            ExitCode == ConvergenceFlags::DIVERGING || ExitCode == ConvergenceFlags::SINGULAR_KKT ||
            exit_stage_stalled || i == (settings_.max_iters_ - 1)) {

            if (ExitCode != ConvergenceFlags::CONVERGED && settings_.return_best_) {
                XSL = BestXSL;
                RHS = BestRHS;
            }

            this->result_.converge_flag_ = ExitCode;
            break;
        }

        // Apply step
        XSL += alpha * DXSL;

        // The ONE iterate-commit site, and therefore the only place the bound
        // multipliers move ALONG dz. (They are written at one other site, the
        // nested restoration return, which re-anchors them on the stashed outer
        // barrier parameter applying no dz and moving no x — a different event
        // class, and the same one that has always rewritten the slack
        // multipliers. See the two-event note on bound_duals_ in psiopt.h.)
        // Every recovery outcome funnels through the line
        // above: an accepted first trial, a second-order correction, an extended
        // backtrack, a restoration dispatch or a re-center (both alpha = 0), and
        // the watchdog revert (which restores its snapshot into XSL -- and, so
        // the pair stays consistent, its snapshot of the bound multipliers too --
        // and leaves alpha = 0). The dual fraction is alpha*alphad, matching the
        // damping the KKT dual blocks took inside compute_step and the elastic
        // commit below; on the alpha = 0 paths the multipliers do not move, and
        // the kappa_sigma clamp simply re-projects whatever x and z now are.
        //
        // Which barrier parameter that clamp is taken at depends on the barrier
        // schedule in force this iteration: a monotone one holds mu fixed for
        // the whole subproblem, so mu describes the iterate; a free-mode oracle
        // proposes mu for the NEXT step, so the clamp measures the iterate's own
        // average complementarity instead. A phase with no inequality
        // constraints runs no governor at all, which is a fixed-mu schedule by
        // definition; force_monotone_barrier is the nested-restoration route
        // through update_barrier_monotone; otherwise the governor reports its
        // own live mode (always free for classic_adaptive).
        if (this->bounds_) {
            const bool monotone_clip_mu = this->inequal_cons_ == 0 || force_monotone_barrier ||
                                          this->governor_->in_monotone_mode();
            this->apply_bound_dual_step(alpha * alphad, v_xsl, step_mu, monotone_clip_mu);
        }

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
            // (populated by the final active iteration's eval seam). The stashed
            // outer μ is restored (and the governor reset) by leave_restoration
            // below, so this catch-all teardown leaves the solver in optimality
            // mode with the outer barrier parameter before run_phase_sequence()'s
            // phase-boundary reset.
            teardown_theta = this->original_infeasibility_inf();
        } else {
            teardown_theta = this->constraint_violation_l1(v_rhs);
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
        this->leave_restoration(measures, this->restoration_->is_nested(), mu);
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

void tycho::solvers::PSIOPT::validate_staged_multipliers(const Eigen::VectorXd &eq_mults,
                                                         const Eigen::VectorXd &iq_mults) {
    // eq_mults is accepted at either of two sizes: the problem's user-facing
    // equality row count (the common case -- what a caller building on
    // NLPProblem/starting_multipliers() or hand-transcribing their own rows
    // sees), or the post-treatment count that additionally counts one
    // internal fixing row per fixed variable under the MakeConstraint
    // treatment (NonLinearProgram::user_equal_cons_ vs. equal_cons_ -- see
    // install_fixed_variable_rows). apply_staged_multipliers below applies
    // the same rule when actually installing the (by-then-validated) seed.
    const int user_eq = this->nlp_->user_equal_cons_;
    const bool eq_size_ok = eq_mults.size() == user_eq || eq_mults.size() == this->equal_cons_;
    if (!eq_size_ok || iq_mults.size() != this->inequal_cons_) {
        throw std::invalid_argument(fmt::format(
            "PSIOPT: seeded multipliers sized ({} eq, {} iq) do not match the problem's {} user "
            "equality rows ({} eq once the fixed-variable treatment's internal rows are "
            "included, {} iq)",
            eq_mults.size(), iq_mults.size(), user_eq, this->equal_cons_, this->inequal_cons_));
    }
    if (!eq_mults.allFinite() || !iq_mults.allFinite()) {
        throw std::invalid_argument("PSIOPT: seeded multipliers contain a non-finite value");
    }
}

void tycho::solvers::PSIOPT::apply_staged_multipliers(Eigen::VectorXd &XSL,
                                                      const Eigen::VectorXd &eq_mults,
                                                      const Eigen::VectorXd &iq_mults) {
    // Install-only: validate_staged_multipliers() (always called first, at
    // run_phase_sequence entry) has already rejected a mis-sized or
    // non-finite seed, so eq_mults/iq_mults are known-good here. A seed sized
    // to the user equality row count is zero-padded across the
    // MakeConstraint treatment's internal fixing rows, which occupy the TAIL
    // of the equality row space (see validate_staged_multipliers and
    // install_fixed_variable_rows); a seed already sized to the
    // post-treatment count is installed as-is.
    const int user_eq = this->nlp_->user_equal_cons_;
    KKTVector v_xsl = kkt_view(XSL);
    if (this->equal_cons_ > 0) {
        Eigen::VectorXd clamped_eq =
            eq_mults.cwiseMax(-kSeededMultInitMax).cwiseMin(kSeededMultInitMax);
        if (eq_mults.size() == this->equal_cons_) {
            v_xsl.eq_lmults() = clamped_eq;
        } else {
            v_xsl.eq_lmults().head(user_eq) = clamped_eq;
            if (this->equal_cons_ > user_eq) {
                v_xsl.eq_lmults().tail(this->equal_cons_ - user_eq).setZero();
            }
        }
    }
    for (int i = 0; i < this->inequal_cons_; i++) {
        v_xsl.iq_lmults()[i] = std::clamp(iq_mults[i], kSeededIqMultFloor, kSeededMultInitMax);
    }
}

Eigen::VectorXd tycho::solvers::PSIOPT::run_phase_sequence(const Eigen::VectorXd &x,
                                                           std::initializer_list<PhaseStep> steps) {
    // Disarm any staged multiplier seed immediately, before anything below --
    // the nlp_/x-size checks just after this, settings_.validate(), the
    // variable-treatment reconfiguration, ... -- gets a chance to throw and
    // leave a stale seed armed for an unrelated later call. The local copies
    // below live only for the duration of this call; nothing they hold
    // survives past its return either way, applied or not.
    bool have_seed = this->mults_staged_;
    Eigen::VectorXd seed_eq_mults;
    Eigen::VectorXd seed_iq_mults;
    // The phase (if any) the seed applies to: the first OPT/OPTNO-mode phase
    // in the requested sequence, whichever position that is. A solve-only
    // sequence (bare solve()) has no such phase, so the seed is simply never
    // applied -- SOE ignores the multiplier block it would have seeded, so
    // there is nothing meaningful to apply it to.
    int first_opt_phase_idx = -1;
    if (have_seed) {
        seed_eq_mults = std::move(this->staged_eq_mults_);
        seed_iq_mults = std::move(this->staged_iq_mults_);
        this->staged_eq_mults_.resize(0);
        this->staged_iq_mults_.resize(0);
        this->mults_staged_ = false;

        int idx = 0;
        for (const auto &step : steps) {
            if (step.alg_mode_ == AlgorithmModes::OPT || step.alg_mode_ == AlgorithmModes::OPTNO) {
                first_opt_phase_idx = idx;
                break;
            }
            ++idx;
        }
    }

    if (!this->nlp_) {
        throw std::runtime_error("PSIOPT::run_phase_sequence: no NLP has been set. "
                                 "Call set_nlp() before optimize/solve.");
    }
    if (x.size() != full_primal_vars_) {
        throw std::invalid_argument(
            fmt::format("PSIOPT: initial guess has {} elements, expected {} primal variables",
                        x.size(), full_primal_vars_));
    }

    this->result_.reset_accumulators();
    this->eval_error_log_.reset();
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

    // Classify the NLP's variable bounds for this solve, once, before any
    // evaluation: free / lower-only / upper-only / two-sided / fixed, with the
    // fixed ones handed to the configured treatment -- eliminated from the
    // solver's variable space (make_parameter, the default), given an internal
    // equality row each (make_constraint), or kept as two-sided variables with
    // their bounds pushed apart (relax_bounds). Both arguments come from the
    // Settings validated at the top of this function, so a solver switching
    // treatment between two solves re-classifies here and nowhere else.
    //
    // The call is idempotent and reports whether it rebuilt anything. When it
    // did, the problem the solver is about to factorize is a different size than
    // the one set_nlp analyzed, so the dimensions have to be re-read and the
    // sparsity pattern recomputed -- and the symbolic analysis redone, since the
    // matrix it was computed for no longer exists. A solver instance solving
    // repeatedly against unchanged bounds takes none of this.
    //
    // The bound-set pointer is cleared FIRST and re-read only after the call
    // returns. A configuration that throws leaves a rejected classification
    // behind on the NLP, and a caller that catches it and solves again must not
    // find this solver still pointing at it; clearing before and re-reading
    // after makes the success path the only one that can produce a non-null
    // bounds_. A set with nothing in it is left null, so "has variable-bound
    // barrier terms" and "bounds_ != nullptr" are the same question everywhere.
    this->bounds_ = nullptr;
    if (this->nlp_->configure_variable_treatment(settings_.fixed_variable_treatment_,
                                                 settings_.bound_relax_factor_)) {
        this->refresh_nlp_dimensions();
        this->nlp_->analyze_sparsity(this->kkt_sol_.get_matrix());
#ifdef USE_ACCELERATE_SPARSE
        this->kkt_sol_.reinitialize_internal_matrix_representation();
#endif
        this->qp_analyzed_ = false;
    }

    // A staged seed is validated here, right after variable-treatment
    // reconfiguration -- equal_cons_/inequal_cons_/user_equal_cons_ are final
    // for this call at this point, whether or not the block above actually
    // ran. Validating this early (rather than at the point of installation,
    // possibly deep inside a solve-first phase sequence) means a bad seed is
    // rejected before the entry init_impl/factorization even runs, on every
    // entry point -- not after paying for a whole SOE phase first on
    // solve()/solve_optimize()/solve_optimize_solve().
    if (have_seed) {
        this->validate_staged_multipliers(seed_eq_mults, seed_iq_mults);
    }

    if (this->nlp_->variable_bound_set().any())
        this->bounds_ = &this->nlp_->variable_bound_set();

    // Rebuild acceptance_/mechanism_/governor_/recovery_ (and, when
    // restoration_mode_ != off, restoration_) from the just-validated Settings
    // on every solve entry, not just on (re)transcription (set_nlp() no longer
    // builds them) — see rebuild_globalization_components()'s doc comment for
    // why this must run per solve rather than per transcription, and for the
    // neutrality argument on the default (all-off) path. nlp_ is guaranteed
    // non-null here (checked above), and set_nlp() has always already run (same
    // guarantee), so the SolverContext captures this call takes are final.
    //
    // Placed AFTER the variable-treatment configuration above, not before it:
    // ClassicMeritAcceptance is the one component that holds its SolverContext
    // by value, and that context carries the bound-set pointer BY VALUE, so it
    // has to be built once the classification that pointer refers to is settled.
    // Nothing reads any of the five components in between (their only consumers
    // are alg_impl's dispatch and the per-phase reset() calls, both below), and
    // the dimension captures were already by reference precisely because this
    // configuration step can narrow the problem.
    this->rebuild_globalization_components();

    if (settings_.print_level_ == 0)
        print_stats();
    if (settings_.print_level_ < 2) {
        print_header();
        print_beginning("PSIOPT ");
    }
    this->ensure_solver_initialized();

    tycho::utils::Timer t;
    t.start();

    // Into the solver's space: the eliminated coordinates are dropped from the
    // guess here, and their values live in the NLP until the solution is
    // expanded again at the return below.
    Eigen::VectorXd x_solver(this->primal_vars_);
    this->nlp_->gather_reduced_x(x, x_solver);

    // Into the interior of the declared variable bounds, and seed the bound
    // multipliers there. Runs once per solve, after the gather (so it works in
    // the same reduced space the bound set is recorded in) and before any
    // evaluation -- every barrier term below divides by a distance to a bound,
    // and this is what makes those distances strictly positive to begin with. A
    // guess sitting on or outside a bound is projected, not rejected. Resets the
    // multiplier state to empty on a problem with no bounds.
    this->push_initial_point_interior(x_solver, settings_.init_mu_);

    bool docompute = claim_kkt_analysis();
    Eigen::VectorXd XSL = this->init_impl(x_solver, settings_.init_mu_, docompute);

    int phase_idx = 0;
    auto it = steps.begin();
    auto end = steps.end();
    while (it != end) {
        const auto &step = *it;
        ++it;
        bool is_last = (it == end);
        // Captured (and phase_idx advanced) before the conditional-skip check
        // below, so a skipped conditional step still keeps phase_idx aligned
        // with the position first_opt_phase_idx was computed against.
        const int current_phase_idx = phase_idx++;

        // Single application site: whichever XSL is current when the loop
        // reaches the first OPT/OPTNO-mode phase -- the entry init_impl's
        // XSL if that phase is first, an inter-phase re-init's XSL otherwise
        // -- gets the seed installed here, before anything else about this
        // phase (including the conditional-skip check right below) runs.
        // Skip-proof by construction: there is no separate "apply after the
        // reinit that precedes phase N" site to fall out of sync with a
        // skipped step -- the check fires (or doesn't) purely off this
        // iteration's own current_phase_idx.
        if (have_seed && current_phase_idx == first_opt_phase_idx) {
            this->apply_staged_multipliers(XSL, seed_eq_mults, seed_iq_mults);
            have_seed = false;
        }

        // Conditional steps only run if the previous phase didn't converge
        if (step.conditional_ && this->result_.converge_flag_ == ConvergenceFlags::CONVERGED)
            continue;

        if (settings_.print_level_ < 2)
            print_beginning(step.label_);

        // Phase-boundary reset: each globalization component's μ-event/
        // phase-change hook (see e.g. recovery_chain.h's ownership-rule
        // note). reset() was never actually invoked anywhere before this. On
        // the all-default path every live implementation (ClassicMeritAcceptance,
        // BacktrackingLineSearch, ClassicAdaptiveGovernor, NoopRecovery) still
        // has an empty reset() body, so adding these calls remains
        // behavior-neutral there. Several opt-in implementations now carry
        // real per-solve state this reset genuinely clears, though:
        // WatchdogRecovery's counters/arm-state/snapshot, ModernMeritAcceptance's
        // penalty state, SwitchingAcceptance's bounds_initialized_ flag plus
        // FilterAcceptance/FunnelAcceptance's working filter/width (via the
        // virtual reset_bounds() it dispatches to), and MonitoredBarrierGovernor's
        // monotone-mode tracking — each cleared at every new phase (OPT, then a
        // conditional SOE, etc.) rather than carried over from the previous
        // phase.
        this->acceptance_->reset();
        this->mechanism_->reset();
        this->governor_->reset();
        this->recovery_->reset();
        // The bound-multiplier DIRECTION, cleared for the same reason the commit
        // clears it: it is meaningful only between the Newton solve that
        // produced it and the commit that consumes it, and the
        // fraction-to-boundary rule reads it. A phase's final iteration always
        // breaks ABOVE the commit, so the last direction of the phase just
        // ending is never consumed and never zeroed -- and the next phase's
        // first PROBE predictor scales before it has a direction of its own.
        // Without this, that predictor shortens the dual fraction against a
        // direction belonging to a step that was never taken, in a phase that is
        // over. The multipliers themselves carry across, as the iterate does.
        this->bound_duals_.dz_lower_.setZero();
        this->bound_duals_.dz_upper_.setZero();
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

        // If a phase reached DIVERGING or anything at least as severe, skip
        // subsequent phases. For DIVERGING itself, result_.primals_ may
        // contain garbage and feeding it into init_impl for the next phase
        // would be pointless. SINGULAR_KKT's primals are NOT garbage (the
        // forced-rejected step was discarded, alpha = 0.0), but the
        // most-severe verdict reached so far must not be silently
        // overwritten by a later phase's own converge_flag_ (severity order
        // via operator<=> in psiopt_fwd.h) -- DIVERGING and anything more
        // severe ends the sequence.
        if (result_.converge_flag_ >= ConvergenceFlags::DIVERGING) {
            if (settings_.print_level_ < 3)
                fmt::print(fmt::fg(fmt::color::yellow),
                           "Phase diverged; skipping remaining phases.\n");
            break;
        }

        // Re-init for the next phase using stored primals. Still in the
        // solver's space -- the expansion happens once, at the return below.
        // (Any seed the loop-top check above just installed for THIS phase
        // is not undone here -- this re-init only runs when advancing to the
        // NEXT phase, i.e. after alg_impl has already consumed the seeded
        // XSL for the current one.)
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

    // THE reinsertion seam. Everything above ran in the solver's space; from
    // here on the solution is in the caller's, with each eliminated variable
    // back in its own coordinate at its declared value. Nothing else in this
    // solver expands anything, and nothing needs to: result_.primals_ is the one
    // primal vector that leaves.
    //
    // What is NOT expanded, deliberately: an eliminated variable's bound
    // multiplier is not reported. Its value is the stationarity residual in the
    // eliminated coordinate -- the gradient entry that the reduced problem
    // simply has no row for -- and surfacing it belongs with the bound
    // multipliers the barrier work introduces, which is where it will be
    // computed alongside every other bound's.
    if (this->nlp_->is_reduced()) {
        Eigen::VectorXd primals_full(this->full_primal_vars_);
        this->nlp_->scatter_full_x(result_.primals_, primals_full);
        result_.primals_ = primals_full;
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
