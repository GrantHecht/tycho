// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: definitions for
// ClassicMeritAcceptance. The classic_line_search dispatcher and the
// ls_lang / ls_l1 / ls_auglang merit variants plus their
// eval_trial_point_occ / compute_penalties / secondary_accept helpers are
// moved VERBATIM from src/solvers/psiopt.cpp (statement order and operand
// order preserved exactly — the merge gate is a bit-identical CBWR
// iteration-count comparison). The only edits are context-plumbing renames:
// former PSIOPT member reads (settings_, equal_cons_, inequal_cons_, nlp_)
// now go through the SolverContext reference ctx_. Of the four barrier/eval
// helpers, apply_reset_slacks/barrier_objective/barrier_gradient forward to
// the shared inline kernels in barrier_math.h (as do PSIOPT's own
// identically-named methods); eval_rhs has no shared counterpart and stays a
// real body forwarding to ctx_.nlp_->eval_rhs (see merit_acceptance.h's
// byte-identity design note).
//
// This file also hosts BacktrackingLineSearch (the step-length
// mechanism) — verbatim today's max_step_to_boundary / max_primal_dual_step,
// reading through the SolverContext passed to each call. See
// backtracking_line_search.h's riskiest-seam design note.
//
// This file also hosts ClassicAdaptiveGovernor (the barrier-parameter
// update) — verbatim today's PROBE/LOQO barmode switch + common clamp/objective/
// gradient tail, plus the loqo_mu / mpc_mu oracles it consumes. Its barrier_*
// helpers forward to the shared inline kernels in barrier_math.h; complementarity
// stays a real, TOKEN-IDENTICAL copy of PSIOPT's own (including its
// ULP-load-bearing .sum() warning), since its avgcomp/mincomp feed mu and are not
// candidates for the shared header. See classic_adaptive_governor.h's
// PROBE-impurity design note.
//
// This file also hosts the second batch of live RecoveryChain links:
// ExtendedBacktrackRecovery, WatchdogRecovery, and the ChainedRecovery
// composition — see watchdog.h's file docstring for the full design.
//
// This file also hosts the feasibility-restoration strategies:
// ProximalSwitchRestoration (the proximal mode-switch) and NestedL1Restoration
// (the condensed elastic l1 phase) — see proximal_restoration.h and
// l1_restoration.h for the formulations and citations, and restoration.h for
// the wiring overview.
// =============================================================================

#include "tycho/detail/solvers/barrier_math.h"
#include "tycho/detail/solvers/eval_error_log.h"
#include "tycho/detail/solvers/globalization/backtracking_line_search.h"
#include "tycho/detail/solvers/globalization/classic_adaptive_governor.h"
#include "tycho/detail/solvers/globalization/feasibility_switch_recovery.h"
#include "tycho/detail/solvers/globalization/filter_acceptance.h"
#include "tycho/detail/solvers/globalization/funnel_acceptance.h"
#include "tycho/detail/solvers/globalization/l1_restoration.h"
#include "tycho/detail/solvers/globalization/merit_acceptance.h"
#include "tycho/detail/solvers/globalization/modern_merit.h"
#include "tycho/detail/solvers/globalization/monitored_governor.h"
#include "tycho/detail/solvers/globalization/proximal_restoration.h"
#include "tycho/detail/solvers/globalization/soc.h"
#include "tycho/detail/solvers/globalization/switching_acceptance.h"
#include "tycho/detail/solvers/globalization/watchdog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tycho::solvers {

namespace {
// File-local helpers shared by every wrapped trial-evaluation catch site
// below (ls_lang/ls_l1/ls_auglang/generic_line_search): both null-guard on
// `log` so a SolverContext built without an EvalErrorLog (e.g. a bare
// unit-test context) stays a silent no-op, matching the inline record()/
// record_unknown() calls they replace.
void note_eval_error(EvalErrorLog *log, const char *what) {
    if (log)
        log->record(what);
}

void note_eval_error_unknown(EvalErrorLog *log) {
    if (log)
        log->record_unknown();
}
} // namespace

// ============================================================================
// Generic interface — is_iterate_acceptable is unused on the classic merit
// path (see header). T6: it throws rather than return a fabricated answer;
// the shipped FilterAcceptance/FunnelAcceptance/ModernMeritAcceptance
// strategies are the ones that give it a real body, since they are the ones
// that actually drive it. is_infeasibility_sufficiently_reduced below is a
// separate case: it has a real body right here, because it IS driven on the
// classic path (see alg_impl, psiopt.cpp).
// ============================================================================
bool ClassicMeritAcceptance::is_iterate_acceptable(const ProgressMeasures &current,
                                                   const ProgressMeasures &trial,
                                                   const ProgressMeasures &predicted_reduction,
                                                   double objective_multiplier,
                                                   double step_length) {
    (void)current;
    (void)trial;
    (void)predicted_reduction;
    (void)objective_multiplier;
    (void)step_length;
    throw std::logic_error("ClassicMeritAcceptance::is_iterate_acceptable is unused on the classic "
                           "merit path (acceptance is fused inside classic_line_search); it is "
                           "driven only by the shipped filter/funnel/modern-merit (WMNO) "
                           "acceptance strategies, which give it a real body");
}

bool ClassicMeritAcceptance::is_infeasibility_sufficiently_reduced(
    const ProgressMeasures &reference, const ProgressMeasures &trial) const {
    // Ipopt IpRestoConvCheck::CheckConvergence (coin-or/Ipopt 72a29c9):
    //   orig_inf_pr_max = Max(kappa_resto_ * orig_curr_inf_pr,
    //                         Min(orig_ip_data->tol(), orig_constr_viol_tol_));
    //   ... exit when orig_trial_inf_pr <= orig_inf_pr_max.
    // orig_curr_inf_pr is the infeasibility at the frozen entry point =
    // reference.infeasibility; orig_trial_inf_pr = trial.infeasibility.
    //
    // Ipopt floors the relative target with Min(tol, constr_viol_tol) — its
    // separate optimality and constraint-violation tolerances. Tycho carries a
    // single constraint-violation tolerance (settings_.econ_tol_, the same field
    // RestorationStrategy::entry_permitted reads via near_feasible()), so the floor here is
    // that one tolerance — a disclosed single-tolerance adaptation of Ipopt's
    // two-tolerance minimum. Classic merit has no Uno counterpart; the Ipopt
    // relative-reduction shape is the reference.
    const double floor = std::max(kKappaResto * reference.infeasibility, ctx_.settings_.econ_tol_);
    return trial.infeasibility <= floor;
}

// ============================================================================
// Barrier/eval helpers. apply_reset_slacks/barrier_objective/barrier_gradient
// are one-line forwarders into the shared kernels in barrier_math.h;
// eval_rhs has no shared counterpart and stays a real body forwarding to
// ctx_.nlp_->eval_rhs (see merit_acceptance.h's byte-identity design note).
// ============================================================================

void ClassicMeritAcceptance::eval_rhs(double obj_scale,
                                      const Eigen::Ref<const Eigen::VectorXd> &XSL, double &val,
                                      Eigen::Ref<Eigen::VectorXd> GX,
                                      Eigen::Ref<Eigen::VectorXd> AGXS_FX) {
    ctx_.nlp_->eval_rhs(obj_scale, XSL.head(ctx_.primal_vars_),
                        XSL.segment(ctx_.primal_vars_ + ctx_.slack_vars_, ctx_.equal_cons_),
                        XSL.tail(ctx_.inequal_cons_), val, GX.head(ctx_.primal_vars_),
                        AGXS_FX.head(ctx_.primal_vars_),
                        AGXS_FX.segment(ctx_.primal_vars_ + ctx_.slack_vars_, ctx_.equal_cons_),
                        AGXS_FX.tail(ctx_.inequal_cons_));
}

void ClassicMeritAcceptance::apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S,
                                                Eigen::Ref<Eigen::VectorXd> FXI) const {
    detail::apply_reset_slacks(S, FXI, ctx_.slack_vars_, ctx_.settings_.neg_slack_reset_);
}

double ClassicMeritAcceptance::barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu) const {
    return detail::barrier_objective(S, mu, ctx_.inequal_cons_);
}

void ClassicMeritAcceptance::barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                                              Eigen::Ref<Eigen::VectorXd> LI, double mu,
                                              Eigen::Ref<Eigen::VectorXd> AGS) const {
    detail::barrier_gradient(S, LI, mu, AGS);
}

// ============================================================================
// Line search — shared helpers
// ============================================================================

void ClassicMeritAcceptance::eval_trial_point_occ(double obj_scale, double mu, double alpha,
                                                  KKTVector &xsl, KKTVector &dxsl, KKTVector &xsl2,
                                                  KKTVector &rhs2, double &ptest, double &btest) {
    xsl2.data() = xsl.data() + alpha * dxsl.data();
    rhs2.data().setZero();
    ctx_.nlp_->eval_occ(obj_scale, xsl2.primals(), ptest, rhs2.eq_cons(), rhs2.iq_cons());
    // Feasibility-restoration trial seam (dead on the default path:
    // ctx_.restoration_ is null). Shared by the L1 and AUGLANG variants. While
    // active, obj_scale is 0 (user objective contributes exactly 0.0) and the
    // restoration objective at the trial primals is added — matching the uniform
    // objective substitution the eval seam applies to prim_obj. The nested mode
    // additionally uses the live-μ, step-fraction-α elastic trial objective and
    // shifts the constraint residuals by the elastic slacks so the merit measures
    // the restoration subproblem's infeasibility θ_elastic = ‖c + n − p‖.
    if (ctx_.restoration_ && ctx_.restoration_->is_active()) {
        if (ctx_.restoration_->is_nested())
            ptest += ctx_.restoration_->trial_objective(mu, alpha, xsl2.primals());
        else
            ptest += ctx_.restoration_->proximal_objective(xsl2.primals());
    }
    // Trial paths keep the reset-completed inequality residual even while a
    // nested restoration is active (the main evaluation seam uses the direct,
    // unreset g+s there). The asymmetry is intentional: the trial merit only
    // ranks step candidates, while the exit/convergence decisions read the
    // main-seam measure, and the reset completion is conservative for
    // strictly-feasible rows.
    this->apply_reset_slacks(xsl2.slacks(), rhs2.iq_cons());
    btest = this->barrier_objective(xsl2.slacks(), mu);
    // Variable-bound barrier at the TRIAL primals. The current-point barr_obj
    // this is compared against carries the matching term (added once in
    // alg_impl), so both sides of the merit test are in the same units. Null off
    // the bound path, where btest is unchanged.
    if (ctx_.bounds_)
        btest += detail::bound_barrier_objective(xsl2.primals(), *ctx_.bounds_, mu);
    if (ctx_.restoration_ && ctx_.restoration_->is_active() && ctx_.restoration_->is_nested()) {
        resto_eq_shift_scratch_.resize(ctx_.equal_cons_);
        resto_iq_shift_scratch_.resize(ctx_.inequal_cons_);
        ctx_.restoration_->trial_residual_shift(alpha, resto_eq_shift_scratch_,
                                                resto_iq_shift_scratch_);
        rhs2.eq_cons() += resto_eq_shift_scratch_;
        rhs2.iq_cons() += resto_iq_shift_scratch_;
    }
}

auto ClassicMeritAcceptance::compute_penalties(KKTVector &xsl, KKTVector &rhs) const
    -> PenaltyTerms {
    return {xsl.lmults().cwiseAbs().dot(rhs.all_cons().cwiseAbs()), rhs.all_cons().squaredNorm(),
            rhs.all_cons().template lpNorm<Eigen::Infinity>()};
}

bool ClassicMeritAcceptance::secondary_accept(double ptest, double prim_obj,
                                              const PenaltyTerms &test,
                                              const PenaltyTerms &init) const {
    return ptest < prim_obj && (test.l2_ < init.l2_ || test.linf_ < init.linf_);
}

// ============================================================================
// Line search — variant implementations
// ============================================================================

double ClassicMeritAcceptance::ls_lang(double obj_scale, double mu, double prim_obj,
                                       double barr_obj, KKTVector &xsl, KKTVector &dxsl,
                                       KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                                       IterateInfo &citer) {
    double alpha = 1.0;
    double LangInit = prim_obj + barr_obj + xsl.lmults().dot(rhs.all_cons());

    for (int j = 0; j < ctx_.settings_.max_ls_iters_; j++) {
        double ptest = 0;
        double btest = 0;
        xsl2.data() = xsl.data() + alpha * dxsl.data();
        rhs2.data().setZero();
        bool evaluable = true;
        try {
            this->eval_rhs(obj_scale, xsl2.data(), ptest, rhs2.data(), rhs2.data());
        } catch (const std::exception &e) {
            note_eval_error(ctx_.eval_errors_, e.what());
            evaluable = false;
        } catch (...) {
            note_eval_error_unknown(ctx_.eval_errors_);
            evaluable = false;
        }
        if (!evaluable) {
            // Un-evaluable trial: treat exactly like a merit rejection (the merit
            // value is unknowable), and continue down the alpha ladder. The
            // infeasibility signal is deliberately left at its sentinel — there
            // is no evaluated trial to measure it on, and the sentinel suppresses
            // the second-order correction trigger for this rejection. merit_val_
            // is likewise not written on a throwing rung, so it retains the
            // previous evaluable rung's value (or the IterateInfo default when
            // the first rung throws).
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
            citer.first_rejection_iter_ =
                citer.first_rejection_iter_ < 0 ? j : citer.first_rejection_iter_;
            continue;
        }
        // Feasibility-restoration trial seam (dead on the default path:
        // ctx_.restoration_ is null). While active, obj_scale is 0 (the user
        // objective contributes exactly 0.0 to ptest via lsobjscale) and the
        // restoration objective at the trial primals is added instead — matching
        // the uniform objective substitution the eval seam applies to prim_obj.
        // The nested mode uses the live-μ, α-fraction elastic trial objective and
        // shifts the residuals by the elastic slacks (see eval_trial_point_occ).
        if (ctx_.restoration_ && ctx_.restoration_->is_active()) {
            if (ctx_.restoration_->is_nested())
                ptest += ctx_.restoration_->trial_objective(mu, alpha, xsl2.primals());
            else
                ptest += ctx_.restoration_->proximal_objective(xsl2.primals());
        }
        this->apply_reset_slacks(xsl2.slacks(), rhs2.iq_cons());
        btest = this->barrier_objective(xsl2.slacks(), mu);
        // Variable-bound barrier at the TRIAL primals — see eval_trial_point_occ.
        if (ctx_.bounds_)
            btest += detail::bound_barrier_objective(xsl2.primals(), *ctx_.bounds_, mu);
        this->barrier_gradient(xsl2.slacks(), xsl2.iq_lmults(), mu, rhs2.dual_grad());
        if (ctx_.restoration_ && ctx_.restoration_->is_active() && ctx_.restoration_->is_nested()) {
            resto_eq_shift_scratch_.resize(ctx_.equal_cons_);
            resto_iq_shift_scratch_.resize(ctx_.inequal_cons_);
            ctx_.restoration_->trial_residual_shift(alpha, resto_eq_shift_scratch_,
                                                    resto_iq_shift_scratch_);
            rhs2.eq_cons() += resto_eq_shift_scratch_;
            rhs2.iq_cons() += resto_iq_shift_scratch_;
        }
        double LangTest = ptest + btest + xsl2.lmults().dot(rhs2.all_cons());
        if (LangTest < LangInit) {
            citer.ls_iters_ = j;
            citer.accepted_ = true;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
            // Signal-only: record the first rejection's backtracking index.
            // The self-referential select keeps the first value (the LANG
            // variant computes no infeasibility scalar, so theta is left at its
            // default). Writes touch only the diagnostic signal fields; no
            // classic line-search state is read or modified here.
            citer.first_rejection_iter_ =
                citer.first_rejection_iter_ < 0 ? j : citer.first_rejection_iter_;
        }
    }
    return alpha;
}

double ClassicMeritAcceptance::ls_l1(double obj_scale, double mu, double prim_obj, double barr_obj,
                                     KKTVector &xsl, KKTVector &dxsl, KKTVector &xsl2,
                                     KKTVector &rhs, KKTVector &rhs2, IterateInfo &citer) {
    double alpha = 1.0;
    double vv = rhs.prim_dual_grad().dot(dxsl.primals_slacks());
    double cv = dxsl.lmults().dot(rhs.all_cons());

    PenaltyTerms init = compute_penalties(xsl, rhs);

    // Branch-first: avoid computing a transient inf/nan when init.l2_ == 0.0
    // (the division result is immediately discarded by the guard below anyway;
    // final sc is bit-identical to the previous compute-then-overwrite order).
    double sc = (init.l2_ == 0.0) ? 1.0 : .1 + std::abs(vv - cv) / init.l2_;

    double LangInit = prim_obj + barr_obj + init.l1_ + init.l2_ * sc;

    for (int j = 0; j < ctx_.settings_.max_ls_iters_; j++) {
        double ptest = 0;
        double btest = 0;
        bool evaluable = true;
        try {
            eval_trial_point_occ(obj_scale, mu, alpha, xsl, dxsl, xsl2, rhs2, ptest, btest);
        } catch (const std::exception &e) {
            note_eval_error(ctx_.eval_errors_, e.what());
            evaluable = false;
        } catch (...) {
            note_eval_error_unknown(ctx_.eval_errors_);
            evaluable = false;
        }
        if (!evaluable) {
            // Un-evaluable trial — see ls_lang's rationale.
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
            citer.first_rejection_iter_ =
                citer.first_rejection_iter_ < 0 ? j : citer.first_rejection_iter_;
            continue;
        }

        double LangTest = ptest + btest;
        PenaltyTerms test = compute_penalties(xsl, rhs2);
        LangTest += test.l1_ + test.l2_ * sc;

        citer.merit_val_ = LangTest;
        if (LangTest < LangInit || secondary_accept(ptest, prim_obj, test, init)) {
            citer.ls_iters_ = j;
            citer.accepted_ = true;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
            // Signal-only: record the first rejection's index and the trial's
            // already-computed L2 infeasibility (test.l2_). The self-referential
            // selects keep the first values. Writes touch only the diagnostic
            // signal fields; no classic line-search state is read or modified.
            citer.theta_at_first_rejection_ =
                citer.first_rejection_iter_ < 0 ? test.l2_ : citer.theta_at_first_rejection_;
            citer.first_rejection_iter_ =
                citer.first_rejection_iter_ < 0 ? j : citer.first_rejection_iter_;
        }
    }
    return alpha;
}

// ls_auglang's tolerance-filtered L1 penalty only accumulates a row's
// contribution once its residual exceeds this multiple of the row's own
// convergence tolerance (econ_tol_/icon_tol_) -- a coarser filter than the
// tolerance itself, so residuals within one order of magnitude of converged
// are excluded from the L1 penalty term.
constexpr double kLsAuglangTolFilterMultiplier = 10.0;

double ClassicMeritAcceptance::ls_auglang(double obj_scale, double mu, double prim_obj,
                                          double barr_obj, KKTVector &xsl, KKTVector &dxsl,
                                          KKTVector &xsl2, KKTVector &rhs, KKTVector &rhs2,
                                          IterateInfo &citer) {
    double alpha = 1.0;
    double vv = rhs.prim_dual_grad().dot(dxsl.primals_slacks());
    double cv = dxsl.lmults().dot(rhs.all_cons());

    PenaltyTerms init = compute_penalties(xsl, rhs);

    // Branch-first: avoid computing a transient inf/nan when init.l2_ == 0.0
    // (the division result is immediately discarded by the guard below anyway;
    // final sc is bit-identical to the previous compute-then-overwrite order).
    double sc = (init.l2_ == 0.0) ? 1.0 : .01 + std::abs(vv - cv) / init.l2_;

    double LangInit = prim_obj + barr_obj + init.l1_ + init.l2_ * sc;

    for (int j = 0; j < ctx_.settings_.max_ls_iters_; j++) {
        double ptest = 0;
        double btest = 0;
        bool evaluable = true;
        try {
            eval_trial_point_occ(obj_scale, mu, alpha, xsl, dxsl, xsl2, rhs2, ptest, btest);
        } catch (const std::exception &e) {
            note_eval_error(ctx_.eval_errors_, e.what());
            evaluable = false;
        } catch (...) {
            note_eval_error_unknown(ctx_.eval_errors_);
            evaluable = false;
        }
        if (!evaluable) {
            // Un-evaluable trial — see ls_lang's rationale.
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
            citer.first_rejection_iter_ =
                citer.first_rejection_iter_ < 0 ? j : citer.first_rejection_iter_;
            continue;
        }

        double LangTest = ptest + btest;

        // Tolerance-filtered L1 penalty
        double TestL1Pen = 0;
        for (int i = 0; i < ctx_.equal_cons_; i++) {
            double eqerr = std::abs(rhs2.eq_cons()[i]);
            double eqmul = std::abs(xsl.eq_lmults()[i]);
            if (eqerr > ctx_.settings_.econ_tol_ * kLsAuglangTolFilterMultiplier) {
                TestL1Pen += eqerr * eqmul;
            }
        }
        for (int i = 0; i < ctx_.inequal_cons_; i++) {
            double iqerr = std::abs(rhs2.iq_cons()[i]);
            double iqmul = std::abs(xsl.iq_lmults()[i]);
            if (iqerr > ctx_.settings_.icon_tol_ * kLsAuglangTolFilterMultiplier) {
                TestL1Pen += iqerr * iqmul;
            }
        }

        double TestL2Pen = rhs2.all_cons().squaredNorm();
        double TestLinfPenalty = rhs2.all_cons().template lpNorm<Eigen::Infinity>();

        // Zero L2 when within tolerance threshold
        const double zero_l2_threshold =
            ctx_.settings_.econ_tol_ * ctx_.settings_.econ_tol_ * ctx_.equal_cons_ +
            ctx_.settings_.icon_tol_ * ctx_.settings_.icon_tol_ * ctx_.inequal_cons_;
        if (TestL2Pen < zero_l2_threshold) {
            TestL2Pen = 0;
        }

        LangTest += TestL1Pen + TestL2Pen * sc;

        PenaltyTerms test{TestL1Pen, TestL2Pen, TestLinfPenalty};
        citer.merit_val_ = LangTest;
        if (LangTest < LangInit || secondary_accept(ptest, prim_obj, test, init)) {
            citer.ls_iters_ = j;
            citer.accepted_ = true;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
            // Signal-only: record the first rejection's index and the trial's
            // already-computed L2 penalty (TestL2Pen). The self-referential
            // selects keep the first values. Writes touch only the diagnostic
            // signal fields; no classic line-search state is read or modified.
            citer.theta_at_first_rejection_ =
                citer.first_rejection_iter_ < 0 ? TestL2Pen : citer.theta_at_first_rejection_;
            citer.first_rejection_iter_ =
                citer.first_rejection_iter_ < 0 ? j : citer.first_rejection_iter_;
        }
    }
    return alpha;
}

// ============================================================================
// Line search — dispatcher (verbatim today's PSIOPT::ls_impl)
// ============================================================================

double ClassicMeritAcceptance::classic_line_search(PSIOPT::LineSearchModes lsmode, double obj_scale,
                                                   double mu, double prim_obj, double barr_obj,
                                                   Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL,
                                                   Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
                                                   Eigen::VectorXd &RHS2, IterateInfo &Citer,
                                                   const std::vector<IterateInfo> &iters) {
    // Line search exhaustion (all max_ls_iters_ attempts fail the merit test) is
    // signaled by Citer.ls_iters_ == max_ls_iters_ in the iteration table.
    // On success, ls_iters_ records the 0-based index of the accepted step (j).
    // On failure, ls_iters_ is set to j+1 (attempts exhausted so far), reaching
    // max_ls_iters_ when all attempts are exhausted. This convention is shared by
    // ls_lang, ls_l1, and ls_auglang.
    // The best alpha found is still returned; alg_impl's convergence check determines
    // whether the overall iteration should continue or terminate.

    KKTVector v_xsl = kkt_view(XSL);
    KKTVector v_dxsl = kkt_view(DXSL);
    KKTVector v_xsl2 = kkt_view(XSL2);
    KKTVector v_rhs = kkt_view(RHS);
    KKTVector v_rhs2 = kkt_view(RHS2);

    switch (lsmode) {
    case PSIOPT::LineSearchModes::LANG:
        return ls_lang(obj_scale, mu, prim_obj, barr_obj, v_xsl, v_dxsl, v_xsl2, v_rhs, v_rhs2,
                       Citer);
    case PSIOPT::LineSearchModes::L1:
        return ls_l1(obj_scale, mu, prim_obj, barr_obj, v_xsl, v_dxsl, v_xsl2, v_rhs, v_rhs2,
                     Citer);
    case PSIOPT::LineSearchModes::AUGLANG:
        return ls_auglang(obj_scale, mu, prim_obj, barr_obj, v_xsl, v_dxsl, v_xsl2, v_rhs, v_rhs2,
                          Citer);
    case PSIOPT::LineSearchModes::NOLS:
        // Unreachable through the mechanism: run_acceptance_backtrack takes the
        // NOLS early-out before dispatching here. Kept as an explicit case so a
        // direct call still gets the NOLS convention rather than the default throw.
        Citer.ls_iters_ = 0;
        Citer.accepted_ = true;
        return 1.0;
    default:
        throw std::invalid_argument("Unknown LineSearchMode");
    }
}

// ============================================================================
// ModernMeritAcceptance — the modernized merit family (WMNO / flexible penalty
// rules), driven through the GENERIC AcceptanceStrategy path. See
// globalization/modern_merit.h for the full paper-derived formulation and the
// ProgressMeasures mapping; the code below is a direct transcription of the
// (accept-π), (threshold), and penalty-update equations documented there.
// ============================================================================

void ModernMeritAcceptance::reset() {
    // Working penalty state + working smallest-known tracker back to their
    // fresh-construction values (Uno MeritFunction: smallest_known_infeasibility
    // is +∞-initialized).
    nu_ = kWmnoInitPenalty;
    pi_l_ = kFlexInitPiL;
    pi_u_ = kFlexInitPiU;
    smallest_known_infeasibility_ = std::numeric_limits<double>::infinity();
    // Phase-aware, mirroring the filter/funnel: inside the feasibility phase
    // this is a μ-event reset — the stashed optimality-phase state (which the
    // exit test reduces against) and the in-feasibility flag SURVIVE, so a
    // μ-event mid-restoration cannot destroy the frozen tracker. Outside the
    // phase this is the full per-phase clear, which also drops any leftover
    // stash defensively. See the state-isolation section in modern_merit.h.
    if (!in_feasibility_phase_) {
        stashed_nu_ = kWmnoInitPenalty;
        stashed_pi_l_ = kFlexInitPiL;
        stashed_pi_u_ = kFlexInitPiU;
        stashed_smallest_known_infeasibility_ = std::numeric_limits<double>::infinity();
    }
}

bool ModernMeritAcceptance::is_infeasibility_sufficiently_reduced(
    const ProgressMeasures &reference, const ProgressMeasures &trial) const {
    // Uno MeritFunction::is_infeasibility_sufficiently_reduced (cvanaret/Uno
    // 7481abe, MeritFunction.cpp):
    //   return trial_progress.infeasibility <=
    //          sufficient_infeasibility_decrease_ratio * smallest_known_infeasibility;
    // Uno's signature also takes the reference progress but its body never reads
    // it — the smallest-known tracker is the reference this rule reduces
    // against, not the entry point. Mirror that: `reference` is ignored.
    //
    // While in the feasibility phase, reduce against the STASHED (frozen
    // optimality-phase) tracker — Uno freezes it structurally by running a
    // separate optimality instance; here the stash is that frozen copy. Reading
    // the live tracker would be self-defeating: the accept branch of
    // is_iterate_acceptable has just lowered it to include the tested point's
    // own θ, making θ ≤ ratio·live unsatisfiable for θ > 0. Outside the phase
    // the live tracker is read (well-defined; the seam only tests for exit
    // during the phase). The +∞ edge — restoration entered before any
    // optimality-phase accept leaves the stash at +∞, so ratio·(+∞) passes at
    // the first check — is the reference solver's own behavior, retained.
    (void)reference;
    const double tracker = in_feasibility_phase_ ? stashed_smallest_known_infeasibility_
                                                 : smallest_known_infeasibility_;
    return trial.infeasibility <= kSufficientInfeasibilityDecreaseRatio * tracker;
}

void ModernMeritAcceptance::notify_switch_to_feasibility(const ProgressMeasures &current_progress) {
    // T6: a second entry without an intervening exit would stash the FEASIBILITY
    // working penalties/tracker over the preserved optimality stash, silently
    // clobbering the frozen state the exit test consults — a phase-transition
    // mis-wiring, not a recoverable runtime condition.
    if (in_feasibility_phase_)
        throw std::logic_error(
            "ModernMeritAcceptance::notify_switch_to_feasibility: already in the feasibility "
            "phase (in_feasibility_phase_ is true) — a solver wiring bug called entry without "
            "an intervening notify_switch_to_optimality exit");

    // The merit rule augments nothing at the switch point (unlike the filter);
    // the merit's whole persistent state is the penalties + tracker, so the
    // switch measures are not consumed here.
    (void)current_progress;

    // Stash the optimality-phase persistent state, then enter feasibility mode
    // and reinitialize fresh working state via the reset() machinery (the flag
    // is set FIRST so the phase-aware reset() preserves the stash just written).
    stashed_nu_ = nu_;
    stashed_pi_l_ = pi_l_;
    stashed_pi_u_ = pi_u_;
    stashed_smallest_known_infeasibility_ = smallest_known_infeasibility_;
    in_feasibility_phase_ = true;
    this->reset();
}

void ModernMeritAcceptance::notify_switch_to_optimality(const ProgressMeasures &current_progress) {
    // T6: an exit without a preceding entry has no stash to restore — running
    // this body would overwrite the live optimality penalties/tracker with the
    // fresh-construction stash (or a stale one from a prior phase), a
    // phase-transition mis-wiring symmetric to the entry-side hazard above.
    if (!in_feasibility_phase_)
        throw std::logic_error(
            "ModernMeritAcceptance::notify_switch_to_optimality: not in the feasibility phase "
            "(in_feasibility_phase_ is false) — a solver wiring bug called exit without a "
            "preceding notify_switch_to_feasibility entry");

    // The merit rule augments nothing at the exit point either.
    (void)current_progress;

    // Restore the preserved optimality-phase persistent state and leave the
    // phase. The stash is intentionally NOT cleared here: it is now a harmless
    // leftover (the exit test only reads it while the flag is set, and the next
    // entry overwrites it), dropped by the next reset() OUTSIDE the phase.
    nu_ = stashed_nu_;
    pi_l_ = stashed_pi_l_;
    pi_u_ = stashed_pi_u_;
    smallest_known_infeasibility_ = stashed_smallest_known_infeasibility_;
    in_feasibility_phase_ = false;
}

bool ModernMeritAcceptance::is_iterate_acceptable(const ProgressMeasures &current,
                                                  const ProgressMeasures &trial,
                                                  const ProgressMeasures &predicted_reduction,
                                                  double objective_multiplier, double step_length) {
    // objective/auxiliary arrive already σ-scaled (parity with the classic
    // path), so the merit uses them directly; objective_multiplier is available
    // for future rules but not needed by the arithmetic here. step_length is
    // likewise accepted and ignored — see modern_merit.h's declaration comment.
    (void)objective_multiplier;
    (void)step_length;
    bool accept;
    switch (rule_) {
    case MeritPenaltyRules::wmno:
        accept = accept_wmno(current, trial, predicted_reduction);
        break;
    case MeritPenaltyRules::flexible:
        accept = accept_flexible(current, trial, predicted_reduction);
        break;
    default:
        throw std::logic_error(
            "ModernMeritAcceptance::is_iterate_acceptable: unknown MeritPenaltyRule");
    }
    // Uno MeritFunction.cpp: the smallest-known-infeasibility tracker is updated
    // by std::min() ONLY inside the accept branch of is_iterate_acceptable (the
    // restoration-exit test reduces against it). FP-inert on the default path —
    // it writes a member that nothing reads unless the exit test runs.
    if (accept)
        smallest_known_infeasibility_ =
            std::min(smallest_known_infeasibility_, trial.infeasibility);
    return accept;
}

// WMNO Math. Program. 107 (2006), §3.1. Merit Eq. (3.1), ν_TRIAL Eq. (3.5, σ=0),
// update Eq. (3.6), Armijo Eq. (3.9).
bool ModernMeritAcceptance::accept_wmno(const ProgressMeasures &current,
                                        const ProgressMeasures &trial,
                                        const ProgressMeasures &pred) {
    // Penalty update (Eq. 3.6): steplength-independent — the α factor cancels in
    // τ (both pred terms scale by α), so calling this every backtrack is
    // idempotent within an iteration and monotone across iterations. The m_θ==0
    // (feasible current) special case leaves ν unchanged (WMNO "c(z)=0 ⇒ ν⁺=ν").
    if (pred.infeasibility > 0.0) {
        const double tau = -pred.objective / ((1.0 - kWmnoRho) * pred.infeasibility);
        if (nu_ < tau)
            nu_ = tau + kWmnoPenaltyBump;
    }
    return armijo(current, trial, pred, nu_, kWmnoArmijoEta);
}

// Curtis–Nocedal IMA JNA 28(4) (2008). Interval merit Eq. (2.1), χ Eq. (3.8,
// ω=0), π_u update Eq. (3.9), endpoint acceptance (remark after Alg. 3.1), π_l
// update Eqs. (3.10)-(3.11).
bool ModernMeritAcceptance::accept_flexible(const ProgressMeasures &current,
                                            const ProgressMeasures &trial,
                                            const ProgressMeasures &pred) {
    // π_u update (Eq. 3.9): raise π_u to χ+ε only when χ exceeds it. Same
    // α-cancellation / feasible-current guard as WMNO.
    if (pred.infeasibility > 0.0) {
        const double chi = -pred.objective / ((1.0 - kFlexSigma) * pred.infeasibility);
        if (pi_u_ < chi)
            pi_u_ = chi + kFlexEpsPiU;
    }

    // Acceptance over the interval: satisfied for some π ∈ [π_l, π_u] iff
    // satisfied at either endpoint (Curtis–Nocedal, practical remark after
    // Algorithm 3.1). Test π_l first so the π_l branch of the (3.10) update is
    // known.
    const bool accept_at_l = armijo(current, trial, pred, pi_l_, kFlexArmijoEta);
    const bool accept_at_u = accept_at_l || armijo(current, trial, pred, pi_u_, kFlexArmijoEta);
    if (!accept_at_u)
        return false;

    // π_l update (Eq. 3.10) on an accepted step: if π_l already accepted
    // (regions III/IV) keep it; else (region II — accepted only at π_u) raise
    // π_l gradually toward ν(step) (Eq. 3.11), clamped to π_u.
    if (!accept_at_l) {
        const double denom = current.infeasibility - trial.infeasibility; // ‖c_k‖−‖c_{k+1}‖
        if (denom > 0.0) {
            const double num = (trial.objective + trial.auxiliary) -
                               (current.objective + current.auxiliary); // ϕ_μ(trial)−ϕ_μ(current)
            const double r = num / denom;                               // ν(step), Eq. (3.11)
            const double bump = std::max(kFlexPiLDamping * (r - pi_l_), kFlexEpsPiL);
            pi_l_ = std::min(pi_u_, pi_l_ + bump);
        }
    }
    return true;
}

namespace {
// ============================================================================
// modern_eval_trial_point — shared trial-point evaluation for the GENERIC
// acceptance loop (BacktrackingLineSearch::generic_line_search). A PARALLEL
// copy of ClassicMeritAcceptance::eval_trial_point_occ's math (same slack-reset
// + barrier convention as apply_reset_slacks / barrier_objective) — the classic
// path's own copies are deliberately NOT reused or touched, so the classic
// diff stays empty. Reaches PSIOPT state through the SolverContext only.
// Returns ptest (σ-scaled primal objective at the trial), btest (barrier term
// −μ·Σ log s), and theta (L1 constraint-norm merit infeasibility ‖c‖₁).
// ============================================================================
void modern_eval_trial_point(SolverContext &ctx, double obj_scale, double mu, double alpha,
                             const Eigen::VectorXd &XSL, const Eigen::VectorXd &DXSL,
                             Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS2, double &ptest,
                             double &btest, double &theta,
                             Eigen::VectorXd &resto_eq_shift_scratch,
                             Eigen::VectorXd &resto_iq_shift_scratch) {
    const int pv = ctx.primal_vars_;
    const int sv = ctx.slack_vars_;
    const int ec = ctx.equal_cons_;
    const int ic = ctx.inequal_cons_;

    XSL2 = XSL + alpha * DXSL;
    RHS2.setZero();
    ptest = 0.0;
    ctx.nlp_->eval_occ(obj_scale, XSL2.head(pv), ptest, RHS2.segment(pv + sv, ec), RHS2.tail(ic));
    // Feasibility-restoration trial seam (dead on the default path:
    // ctx.restoration_ is null). While active, obj_scale is 0 (user objective
    // contributes exactly 0.0) and the restoration objective at the trial primals
    // is added, so the generic acceptance loop's trial.objective mirrors the
    // uniform substitution applied to current.objective (= prim_obj). The nested
    // mode uses the live-μ, α-fraction elastic trial objective and shifts the
    // residuals by the elastic slacks so θ measures ‖c + n − p‖ (added below,
    // before θ is taken).
    if (ctx.restoration_ && ctx.restoration_->is_active()) {
        if (ctx.restoration_->is_nested())
            ptest += ctx.restoration_->trial_objective(mu, alpha, XSL2.head(pv));
        else
            ptest += ctx.restoration_->proximal_objective(XSL2.head(pv));
    }

    auto S = XSL2.segment(pv, sv);
    auto FXI = RHS2.tail(ic);
    for (int i = 0; i < sv; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < ctx.settings_.neg_slack_reset_)
            si = ctx.settings_.neg_slack_reset_;
        if (fxi < 0.0) {
            FXI[i] = 0.0;
            S[i] = std::max(std::abs(fxi), ctx.settings_.neg_slack_reset_);
        } else {
            FXI[i] += si;
        }
    }

    btest = 0.0;
    for (int i = 0; i < ic; i++)
        btest += -mu * std::log(S[i]);
    // Variable-bound barrier at the TRIAL primals, matching the term alg_impl
    // adds to the current point's barr_obj so the generic acceptance loop's
    // trial.auxiliary and current.auxiliary are in the same units. Null off the
    // bound path, where btest is unchanged.
    if (ctx.bounds_)
        btest += detail::bound_barrier_objective(XSL2.head(pv), *ctx.bounds_, mu);

    if (ctx.restoration_ && ctx.restoration_->is_active() && ctx.restoration_->is_nested()) {
        resto_eq_shift_scratch.resize(ec);
        resto_iq_shift_scratch.resize(ic);
        ctx.restoration_->trial_residual_shift(alpha, resto_eq_shift_scratch,
                                               resto_iq_shift_scratch);
        RHS2.segment(pv + sv, ec) += resto_eq_shift_scratch;
        RHS2.tail(ic) += resto_iq_shift_scratch;
    }

    theta = RHS2.tail(ec + ic).template lpNorm<1>();
}
} // namespace

// ============================================================================
// BacktrackingLineSearch — step-length mechanism. max_step_to_
// boundary and max_primal_dual_step are moved VERBATIM from src/solvers/
// psiopt.cpp (statement order and operand order preserved exactly — the merge
// gate is a bit-identical CBWR iteration-count comparison). The only edits are
// context-plumbing renames: former PSIOPT member reads (settings_.pd_step_
// strategy_, inequal_cons_, equal_cons_) now go through the SolverContext
// reference `ctx`, and the KKTVector view over the raw XSL/DXSL blocks is
// reconstructed inside max_primal_dual_step (the caller used to build and pass
// it). See backtracking_line_search.h's riskiest-seam design note.
// ============================================================================

double BacktrackingLineSearch::max_step_to_boundary(Eigen::Ref<Eigen::VectorXd> SLI,
                                                    Eigen::Ref<Eigen::VectorXd> dSLI, double bfrac,
                                                    const SolverContext &ctx) const {
    double alpha = 1.0;
    for (int i = 0; i < ctx.inequal_cons_; i++) {
        if (dSLI[i] < -bfrac * SLI[i]) {
            double an = -bfrac * SLI[i] / dSLI[i];
            if (an < alpha)
                alpha = an;
        }
    }
    return alpha;
}

void BacktrackingLineSearch::max_primal_dual_step(Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL,
                                                  double bfrac, double &alphap, double &alphad,
                                                  const SolverContext &ctx) {
    KKTVector xsl = kkt_view(XSL, ctx);
    KKTVector dxsl = kkt_view(DXSL, ctx);
    double Smax = this->max_step_to_boundary(xsl.slacks(), dxsl.slacks(), bfrac, ctx);
    double Lmax = this->max_step_to_boundary(xsl.iq_lmults(), dxsl.iq_lmults(), bfrac, ctx);

    // Nested-restoration elastic fraction-to-boundary. Dead unless a nested
    // restoration strategy is active. The condensed elastic slacks (n,p) and
    // their bound multipliers (z_n,z_p) live outside the KKT vector, so their
    // positivity is enforced here: combine their τ-caps (τ = bfrac, the same
    // fraction the slack/bound steps use) into the primal (Smax) and dual (Lmax)
    // step lengths, so the pre-scaled DXSL below and the returned alphap/alphad
    // keep every eliminated variable strictly positive. On the default and
    // proximal-switch paths ctx.restoration_ is null / not nested → unchanged.
    if (ctx.restoration_ && ctx.restoration_->is_active() && ctx.restoration_->is_nested()) {
        Smax = std::min(Smax, ctx.restoration_->primal_boundary_alpha(bfrac));
        Lmax = std::min(Lmax, ctx.restoration_->dual_boundary_alpha(bfrac));
    }

    double primstep = Smax;
    double slackstep = Smax;
    double eqmultstep = Smax;
    double iqmultstep = Lmax;

    if (ctx.settings_.pd_step_strategy_ == PSIOPT::PDStepStrategies::PrimSlackEq_Iq) {
    } else if (ctx.settings_.pd_step_strategy_ == PSIOPT::PDStepStrategies::AllMinimum) {
        double step = std::min(Smax, Lmax);
        primstep = step;
        slackstep = step;
        eqmultstep = step;
        iqmultstep = step;
    } else if (ctx.settings_.pd_step_strategy_ == PSIOPT::PDStepStrategies::PrimSlack_EqIq) {
        eqmultstep = Lmax;
    } else if (ctx.settings_.pd_step_strategy_ == PSIOPT::PDStepStrategies::MaxEq) {
        double step = std::max(Smax, Lmax);
        eqmultstep = step;
    }
    dxsl.primals() *= primstep;
    if (ctx.inequal_cons_ > 0)
        dxsl.slacks() *= slackstep;
    if (ctx.equal_cons_ > 0)
        dxsl.eq_lmults() *= eqmultstep;
    if (ctx.inequal_cons_ > 0)
        dxsl.iq_lmults() *= iqmultstep;

    alphap = Smax;
    alphad = Lmax;
}

// compute_step fuses the fraction-to-boundary scaling and the acceptance
// backtrack (riskiest seam): max_primal_dual_step MUTATES DXSL in
// place — applied when inequality slacks exist OR an active nested
// restoration's condensed elastic variables carry positivity caps — and the
// acceptance strategy then backtracks a scalar alpha on the already-scaled
// DXSL. SocRecovery::do_correction applies the same rule to corrected
// directions.
double BacktrackingLineSearch::compute_step(
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, AcceptanceStrategy &acceptance, double &alphap, double &alphad,
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx) {
    // The fraction-to-boundary scaling runs when there are inequality slacks OR
    // when a nested restoration strategy is active (whose eliminated elastic
    // variables carry their own positivity caps even for an equality-only
    // problem). The added disjunct is provably false on the default path
    // (ctx.restoration_ null → short-circuit), so the guard is byte-identical off.
    if (ctx.inequal_cons_ > 0 ||
        (ctx.restoration_ && ctx.restoration_->is_active() && ctx.restoration_->is_nested()))
        this->max_primal_dual_step(XSL, DXSL, ctx.settings_.bound_fraction_, alphap, alphad, ctx);

    // Second half: the acceptance backtrack on the scaled DXSL, dispatched
    // classic-vs-generic. Factored into run_acceptance_backtrack so the same
    // dispatch is reachable by the recovery links (SOC / extended backtracking)
    // on a prepared direction. The classic path still forwards verbatim to
    // classic_line_search with the identical arguments — byte-identical to
    // pre-modern-merit behavior.
    return this->run_acceptance_backtrack(lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, DXSL,
                                          XSL2, RHS, RHS2, acceptance, Citer, iters, ctx);
}

// Acceptance backtrack only — the classic-vs-generic dispatch, without the
// fraction-to-boundary scaling (compute_step's first half). See
// backtracking_line_search.h. The classic path forwards straight to the fused
// classic_line_search; the generic path (drives_classic_path() == false) runs
// the loop-in-mechanism generic_line_search.
double BacktrackingLineSearch::run_acceptance_backtrack(
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, AcceptanceStrategy &acceptance, IterateInfo &Citer,
    const std::vector<IterateInfo> &iters, SolverContext &ctx) {
    // NOLS short-circuits both driving paths identically, so the early-out lives
    // here rather than being repeated at the head of each. Both dispatch arms are
    // reached only through this function.
    if (lsmode == PSIOPT::LineSearchModes::NOLS) {
        Citer.ls_iters_ = 0;
        // No line search runs: the full step is taken, i.e. accepted.
        Citer.accepted_ = true;
        return 1.0;
    }
    if (acceptance.drives_classic_path())
        return acceptance.classic_line_search(lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, DXSL,
                                              XSL2, RHS, RHS2, Citer, iters);
    return this->generic_line_search(lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, DXSL, XSL2,
                                     RHS, RHS2, acceptance, Citer, ctx);
}

// Generic driving path — see backtracking_line_search.h. Loop-in-mechanism,
// judgment-in-strategy: this reproduces the classic backtracking ladder (up to
// max_ls_iters_, alpha /= alpha_red_ on reject) and the classic signal stores,
// but the accept/reject verdict comes from
// AcceptanceStrategy::is_iterate_acceptable on a ProgressMeasures triple.
double BacktrackingLineSearch::generic_line_search(
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, AcceptanceStrategy &acceptance, IterateInfo &Citer, SolverContext &ctx) {
    KKTVector v_dxsl = kkt_view(DXSL, ctx);
    KKTVector v_rhs = kkt_view(RHS, ctx);

    // Current-point measures (j-independent). infeasibility = ‖c(z_k)‖₁ (the
    // merit ‖c‖); objective/auxiliary = the smooth barrier objective split
    // f + (−μ·Σ log s). See modern_merit.h's ProgressMeasures mapping.
    ProgressMeasures current;
    current.infeasibility = v_rhs.all_cons().template lpNorm<1>();
    current.objective = prim_obj;
    current.auxiliary = barr_obj;

    // Directional derivative of the smooth objective ϕ_μ along the (already
    // fraction-to-boundary-scaled) step: ∇ϕ_μ(z_k)ᵀd = RHS.prim_dual_grad ·
    // DXSL.primals_slacks — the exact quantity ls_l1/ls_auglang call `vv`.
    const double dirderiv = v_rhs.prim_dual_grad().dot(v_dxsl.primals_slacks());

    double alpha = 1.0;
    for (int j = 0; j < ctx.settings_.max_ls_iters_; j++) {
        double ptest = 0.0;
        double btest = 0.0;
        double theta_t = 0.0;
        bool evaluable = true;
        try {
            modern_eval_trial_point(ctx, obj_scale, mu, alpha, XSL, DXSL, XSL2, RHS2, ptest, btest,
                                    theta_t, resto_eq_shift_scratch_, resto_iq_shift_scratch_);
        } catch (const std::exception &e) {
            note_eval_error(ctx.eval_errors_, e.what());
            evaluable = false;
        } catch (...) {
            note_eval_error_unknown(ctx.eval_errors_);
            evaluable = false;
        }
        if (!evaluable) {
            // Un-evaluable trial — see ls_lang's rationale.
            Citer.ls_iters_ = j + 1;
            alpha = alpha / ctx.settings_.alpha_red_;
            Citer.first_rejection_iter_ =
                Citer.first_rejection_iter_ < 0 ? j : Citer.first_rejection_iter_;
            continue;
        }
        ProgressMeasures trial{theta_t, ptest, btest};

        // Predicted reductions (α-scaled; see modern_merit.h): m_f = −α·∇ϕ_μᵀd
        // (≥0 for descent), m_θ = α·θ_c (linearized-constraint model).
        ProgressMeasures pred;
        pred.objective = -alpha * dirderiv;
        pred.infeasibility = alpha * current.infeasibility;

        // Diagnostic merit (barrier objective sans penalty; the penalty is
        // strategy-internal state). Not printed — see IterateInfo's field note.
        Citer.merit_val_ = trial.objective + trial.auxiliary;

        if (acceptance.is_iterate_acceptable(current, trial, pred, obj_scale, alpha)) {
            Citer.ls_iters_ = j;
            Citer.accepted_ = true;
            break;
        } else {
            Citer.ls_iters_ = j + 1;
            alpha = alpha / ctx.settings_.alpha_red_;
            // Signal stores mirror the classic path (self-referential selects
            // keep the FIRST rejection's values) so SOC/watchdog compose. The
            // modern path records its own merit infeasibility θ_t (L1 ‖c‖).
            Citer.theta_at_first_rejection_ =
                Citer.first_rejection_iter_ < 0 ? theta_t : Citer.theta_at_first_rejection_;
            Citer.first_rejection_iter_ =
                Citer.first_rejection_iter_ < 0 ? j : Citer.first_rejection_iter_;
        }
    }
    return alpha;
}

// ============================================================================
// ClassicAdaptiveGovernor — barrier-parameter update. The
// PROBE/LOQO barmode switch + common clamp/objective/gradient tail and the
// loqo_mu / mpc_mu oracles are moved from src/solvers/psiopt.cpp (statement
// order and operand order preserved exactly — the merge gate is a
// bit-identical CBWR iteration-count comparison). Besides context-plumbing
// renames (former PSIOPT member reads (kkt_sol_ -> ctx.kkt_solver_,
// settings_/dims -> ctx.*) and the mechanism_ base pointer -> the mechanism
// reference parameter), loqo_mu dropped its unused S/LI parameters (the
// former PSIOPT::loqo_mu took Eigen::Ref<Eigen::VectorXd> S, LI ahead of
// avgcomp/mincomp; neither was read by the body). Of the barrier_* /
// complementarity helpers below, barrier_objective/barrier_gradient are
// one-line forwarders into the shared kernels in barrier_math.h;
// complementarity remains a real, TOKEN-IDENTICAL copy of PSIOPT's own
// (including its ULP-load-bearing .sum() warning), since its avgcomp/mincomp
// feed mu and are not a candidate for the shared header. See
// classic_adaptive_governor.h's PROBE-impurity and byte-identity design notes.
// ============================================================================

void ClassicAdaptiveGovernor::complementarity(Eigen::Ref<Eigen::VectorXd> S,
                                              Eigen::Ref<Eigen::VectorXd> LI, double &avgcomp,
                                              double &mincomp, double &maxcomp,
                                              const SolverContext &ctx) const {
    // Buffer-hoist ONLY: keep the exact Eigen .sum()/minCoeff()/maxCoeff()
    // reduction expressions unchanged. avgcomp feeds mu (see mpc_mu/loqo_mu
    // call sites), so a hand-fused loop that reorders the sum could perturb
    // the reduction by a ULP under fast-math and change iterates -- forbidden.
    // This change only removes the per-call heap allocation of StLI.
    ctx.stli_scratch_.resize(S.size());
    ctx.stli_scratch_ = S.cwiseProduct(LI);
    mincomp = ctx.stli_scratch_.minCoeff();
    maxcomp = ctx.stli_scratch_.maxCoeff();
    avgcomp = ctx.stli_scratch_.sum() / double(ctx.stli_scratch_.size());
}

double ClassicAdaptiveGovernor::barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu,
                                                  const SolverContext &ctx) const {
    return detail::barrier_objective(S, mu, ctx.inequal_cons_);
}

void ClassicAdaptiveGovernor::barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                                               Eigen::Ref<Eigen::VectorXd> LI, double mu,
                                               Eigen::Ref<Eigen::VectorXd> AGS) const {
    detail::barrier_gradient(S, LI, mu, AGS);
}

void ClassicAdaptiveGovernor::barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI,
                                               Eigen::Ref<Eigen::VectorXd> AGS) const {
    AGS = LI;
}

double ClassicAdaptiveGovernor::loqo_mu(double avgcomp, double mincomp) const {
    double eta = mincomp / avgcomp;
    double sigmat = .1 * std::pow(0.05 * (1.0 - eta) / eta, 3);
    double sigma = std::min(0.8, std::abs(sigmat));
    return sigma * avgcomp;
}

double ClassicAdaptiveGovernor::mpc_mu(Eigen::Ref<Eigen::VectorXd> S,
                                       Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                                       double mincomp, const SolverContext &ctx) const {
    // The PROBE predictor recomputes complementarity on the predictor point from
    // the ORIGINAL slack/multiplier pairs only (no elastic augmentation), so the
    // ratio (navgcomp/avgcomp)³ would drive mu toward its floor during a nested
    // restoration phase. Under a free (classic_adaptive) governor that collapse is
    // avoided by construction: the alg_impl seam routes the in-phase barrier
    // update to BarrierGovernor::update_barrier_monotone (Ipopt's default
    // restoration mu_strategy) whenever the governor lacks its own safeguard, so
    // this free-mode predictor is UNREACHABLE while nested-active under such a
    // governor. Under the monitored governor it is reachable only inside that
    // governor's own guarded free mode, where the monitor forces a monotone
    // fallback on stall (and avgcomp/mincomp carry the elastic complementarity).
    double navgcomp = 0;
    double nmincomp = 0;
    double nmaxcomp = 0;
    this->complementarity(S, LI, navgcomp, nmincomp, nmaxcomp, ctx);
    return std::pow(navgcomp / avgcomp, 3) * avgcomp;
}

// Verbatim today's psiopt.cpp barmode switch (the former `if (inequal_cons_ > 0)`
// body): the guard stays at the alg_impl call site, so update_barrier assumes
// inequal_cons_ > 0. The predictor's alphap/alphad are locals here (discarded —
// see the divergence-path note in the header). `current` is ignored and
// `mu_event` is never written (free mode only; see the header).
double ClassicAdaptiveGovernor::update_barrier(PSIOPT::BarrierModes barmode, double mu_in,
                                               double avgcomp, double mincomp, Eigen::VectorXd &XSL,
                                               Eigen::VectorXd &RHS, Eigen::VectorXd &DXSL,
                                               Eigen::VectorXd &Temp,
                                               GlobalizationMechanism &mechanism,
                                               SolverContext &ctx, double &barr_obj,
                                               const IterateInfo & /*current*/,
                                               bool & /*mu_event*/) {
    KKTVector v_xsl = kkt_view(XSL, ctx);
    KKTVector v_rhs = kkt_view(RHS, ctx);
    KKTVector v_temp = kkt_view(Temp, ctx);

    double mu = mu_in;
    double alphap = 1.0; // predictor fraction-to-boundary steps — discarded (see header).
    double alphad = 1.0;

    switch (barmode) {
    case PSIOPT::BarrierModes::PROBE:
        this->barrier_gradient(v_xsl.iq_lmults(), v_rhs.dual_grad());
        // Assign the Solve<> expression directly (hits Eigen's specialized
        // Assignment<DstXprType, Solve<...>> and writes straight into DXSL,
        // no temporary) then negate in place (elementwise, alias-safe) --
        // avoids the extra kkt_dim_-sized temporary that
        // `DXSL = -kkt_sol_.solve(RHS)` forces via Solve's
        // EvalBeforeNestingBit when wrapped in a CwiseUnaryOp.
        DXSL = ctx.kkt_solver_.solve(RHS);
        DXSL = -DXSL;
        mechanism.max_primal_dual_step(XSL, DXSL, ctx.settings_.bound_fraction_, alphap, alphad,
                                       ctx);
        Temp = XSL + DXSL;
        mu = this->mpc_mu(v_temp.slacks(), v_temp.iq_lmults(), avgcomp, mincomp, ctx);

        break;
    case PSIOPT::BarrierModes::LOQO:
        mu = this->loqo_mu(avgcomp, mincomp);
        break;
    default:
        throw std::invalid_argument("Unknown BarrierMode");
    }

    mu = std::max(mu, ctx.settings_.min_mu_);
    mu = std::min(mu, ctx.settings_.max_mu_);
    barr_obj = this->barrier_objective(v_xsl.slacks(), mu, ctx);
    this->barrier_gradient(v_xsl.slacks(), v_xsl.iq_lmults(), mu, v_rhs.dual_grad());
    return mu;
}

// ============================================================================
// BarrierGovernor::update_barrier_monotone — the monotone (Fiacco-McCormick)
// barrier schedule forced while a nested l1 feasibility-restoration phase is
// active, transcribing Ipopt's default restoration mu_strategy. Non-virtual and
// shared by every governor: the configured free-mode oracle is bypassed for the
// duration of the phase (see the rationale on the declaration in
// barrier_governor.h). Reuses MonitoredBarrierGovernor's pure static
// Fiacco-McCormick helpers — the same safeguarded rule the monitored governor
// runs in its own monotone mode — so the two paths share one arithmetic.
// ============================================================================
double BarrierGovernor::update_barrier_monotone(double mu_in, Eigen::VectorXd &XSL,
                                                Eigen::VectorXd &RHS, SolverContext &ctx,
                                                double &barr_obj, const IterateInfo &current,
                                                bool &mu_event) {
    double mu = mu_in;
    mu_event = false;

    // Advance mu only when the restoration barrier subproblem is sufficiently
    // solved — the Fiacco-McCormick gate (monitored_governor.h (6)): advance iff
    // barrier_subproblem_error <= kBarrierTolFactor · mu. barrier_subproblem_error
    // reads barr_inf_, into which the elastic complementarity is folded while
    // nested-active, so the gate cannot fire while the elastics are still at
    // restoration scale — mu stays anchored at the entry resto_mu until they
    // shrink. Never decrease mu unconditionally.
    const double sub_err = MonitoredBarrierGovernor::barrier_subproblem_error(current);
    if (sub_err <= kBarrierTolFactor * mu_in) {
        const double new_mu = MonitoredBarrierGovernor::fiacco_mccormick_mu(
            mu_in, ctx.settings_.bar_tol_, ctx.settings_.kkt_tol_, ctx.settings_.min_mu_,
            ctx.settings_.max_mu_);
        if (new_mu < mu_in) {
            mu = new_mu;      // advance (a new barrier subproblem)
            mu_event = true;  // -> acceptance per-subproblem reset
        }
    }

    // Log-barrier objective/dual-gradient tail at the resulting mu — the same
    // segments (slacks/iq_lmults on XSL, dual_grad on RHS) and the same
    // arithmetic the free-mode common tail writes.
    auto slacks = XSL.segment(ctx.primal_vars_, ctx.slack_vars_);
    auto iq_lmults = XSL.tail(ctx.inequal_cons_);
    auto dual_grad = RHS.segment(ctx.primal_vars_, ctx.slack_vars_);
    double psi = 0.0;
    for (int k = 0; k < ctx.inequal_cons_; ++k) {
        psi += -mu * std::log(slacks[k]);
    }
    barr_obj = psi;
    dual_grad = iq_lmults - mu * slacks.cwiseInverse();
    return mu;
}

namespace {
// A recovery link that got a trial direction accepted commits it exactly this
// way: the trial direction and its step length replace DXSL/alpha (alg_impl's
// `XSL += alpha*DXSL` then applies them), and Citer is stamped so the recorded
// iterate reflects the taken step rather than the rejected one. SocRecovery and
// ExtendedBacktrackRecovery ran identical copies of this block.
void commit_recovered_step(Eigen::VectorXd &DXSL, double &alpha, IterateInfo &Citer,
                           const Eigen::VectorXd &dxsl_trial, double alpha_trial,
                           const IterateInfo &trial_iter) {
    DXSL = dxsl_trial;
    alpha = alpha_trial;
    Citer.accepted_ = true;
    Citer.ls_iters_ = trial_iter.ls_iters_;
    Citer.merit_val_ = trial_iter.merit_val_;
}
} // namespace

// ============================================================================
// SocRecovery — second-order correction (Wächter & Biegler 2006, §2.4). Only
// reached when SOC is enabled (max_soc_ > 0, so rebuild_globalization_
// components() built a SocRecovery)
// AND the line search rejected a usable step (the recovery-dispatch gate). See
// globalization/soc.h for the algorithm overview and the trigger/termination
// predicates driven below.
// ============================================================================

void SocRecovery::eval_trial_constraints(SolverContext &ctx, double obj_scale,
                                         const Eigen::VectorXd &XSL, const Eigen::VectorXd &dir,
                                         double alpha, Eigen::VectorXd &xsl2_scratch,
                                         Eigen::VectorXd &cons_out) {
    const int pv = ctx.primal_vars_;
    const int sv = ctx.slack_vars_;
    const int ec = ctx.equal_cons_;
    const int ic = ctx.inequal_cons_;

    xsl2_scratch = XSL + alpha * dir;

    double val = 0.0;
    cons_out.setZero();
    ctx.nlp_->eval_occ(obj_scale, xsl2_scratch.head(pv), val, cons_out.head(ec), cons_out.tail(ic));

    // Slack reset on the inequality block against the trial slacks — the same
    // convention ClassicMeritAcceptance::apply_reset_slacks / alg_impl's RHS
    // assembly use, so cons_out is directly comparable to RHS.all_cons().
    auto S = xsl2_scratch.segment(pv, sv);
    auto FXI = cons_out.tail(ic);
    for (int i = 0; i < sv; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < ctx.settings_.neg_slack_reset_) {
            si = ctx.settings_.neg_slack_reset_;
        }
        if (fxi < 0.0) {
            FXI[i] = 0.0;
            S[i] = std::max(std::abs(fxi), ctx.settings_.neg_slack_reset_);
        } else {
            FXI[i] += si;
        }
    }
}

RecoveryChain::Action SocRecovery::on_step_rejected(
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx,
    AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism,
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, double &alpha, double &alphap, double &alphad, int &soc_steps,
    int & /*resolved_depth*/, int & /*watchdog_activations*/) {
    const int max_soc = ctx.settings_.max_soc_;
    if (max_soc <= 0)
        return Action::kAcceptAsIs; // defensive: SocRecovery is only built when max_soc_ > 0.

    const int ncons = ctx.equal_cons_ + ctx.inequal_cons_;

    // Current-iterate constraint violation, in the SAME norm the driving line
    // search records into theta_at_first_rejection_ so the trigger compares like
    // with like. The classic merit path records the squared-L2 all_cons penalty
    // (test.l2_ / TestL2Pen); the generic path (filter / funnel / modern merit)
    // records the L1 merit norm ‖c‖₁ (generic_line_search's theta_t). Selecting
    // the norm by drives_classic_path() keeps the classic path byte-identical
    // (squaredNorm, unchanged) while making the generic trigger dimensionally
    // consistent. RHS's inequality block already carries the merit slack reset
    // (see the RHS assembly in psiopt.cpp), and during a nested restoration
    // phase it carries the condensed residual r̃ — the same quantity the driving
    // line search's current measure reads, so the comparison stays internally
    // consistent on that path too.
    const double current_infeasibility = acceptance.drives_classic_path()
                                             ? RHS.tail(ncons).squaredNorm()
                                             : RHS.tail(ncons).template lpNorm<1>();
    if (!soc_should_trigger(Citer, current_infeasibility))
        return Action::kAcceptAsIs;

    // Snapshot the fraction-to-boundary lengths so a rejected correction leaves
    // no diagnostic residue: only an accepted (kRetry) correction keeps the
    // corrected step's alphap/alphad.
    const double alphap_orig = alphap;
    const double alphad_orig = alphad;

    // Correction-loop scratch (local — SocRecovery holds no state).
    Eigen::VectorXd rhs_soc(RHS.size());
    Eigen::VectorXd dxsl_soc(DXSL.size());
    Eigen::VectorXd trial_cons(ncons);

    // Accumulated corrected constraint block c_soc. Seed from the first rejected
    // trial (paper: c_soc = alpha_0*c_k + c(x_k + alpha_0*d_k); the classic first
    // trial used alpha_0 = 1 on the already fraction-to-boundary-scaled DXSL).
    Eigen::VectorXd c_soc(ncons);
    try {
        eval_trial_constraints(ctx, obj_scale, XSL, DXSL, 1.0, XSL2, trial_cons);
    } catch (const std::exception &e) {
        note_eval_error(ctx.eval_errors_, e.what());
        // The correction seed cannot be evaluated; there is no point starting
        // the SOC procedure. Fall back to the ladder-exhaustion path. alphap /
        // alphad are still at their snapshot values here (only do_correction
        // mutates them), so this early return needs no restore.
        return Action::kAcceptAsIs;
    } catch (...) {
        note_eval_error_unknown(ctx.eval_errors_);
        return Action::kAcceptAsIs;
    }
    c_soc = RHS.tail(ncons) + trial_cons;

    auto do_correction = [&](int /*correction_index*/, double /*prev_violation*/) {
        // Corrected RHS: reuse the factored RHS (objective block unchanged) and
        // overwrite only the constraint block with the accumulated c_soc.
        rhs_soc = RHS;
        rhs_soc.tail(ncons) = c_soc;

        // Correction on the LIVE factorization — one back-substitution, no
        // refactor. Same sign convention as the main step (DXSL = -solve(RHS)).
        dxsl_soc = ctx.kkt_solver_.solve(rhs_soc);
        dxsl_soc = -dxsl_soc;
        if (!std::isfinite(dxsl_soc.squaredNorm()))
            return SocCorrectionOutcome{false, std::numeric_limits<double>::infinity()};

        // Fraction-to-boundary scale the corrected direction, exactly as the
        // main step path scales DXSL before its backtrack (mechanism's shared
        // entry point, under the same guard compute_step applies): there are
        // inequality slacks to keep positive, OR a nested restoration strategy
        // is active, whose eliminated elastic variables carry their own
        // positivity caps even on an equality-only problem. A corrected
        // direction is committed in place of the rejected one, so it has to be
        // held to the same bounds the rejected one was.
        if (ctx.inequal_cons_ > 0 ||
            (ctx.restoration_ && ctx.restoration_->is_active() && ctx.restoration_->is_nested()))
            mechanism.max_primal_dual_step(XSL, dxsl_soc, ctx.settings_.bound_fraction_, alphap,
                                           alphad, ctx);

        // Re-run the full acceptance backtrack on the corrected direction,
        // through the mechanism so the SAME acceptance criteria the ordinary
        // step faced are applied to the corrected trial — the classic merit
        // test on the classic path, or the generic
        // AcceptanceStrategy::is_iterate_acceptable surface (filter / funnel /
        // modern merit) on the generic path. A fresh IterateInfo captures the
        // verdict and, on rejection, the first corrected trial's infeasibility
        // (theta_at_first_rejection_, in the driving path's own norm) without
        // clobbering Citer's recorded signals.
        IterateInfo trial_iter;
        const double alpha_soc = mechanism.run_acceptance_backtrack(
            lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, dxsl_soc, XSL2, RHS, RHS2, acceptance,
            trial_iter, iters, ctx);

        if (trial_iter.accepted_) {
            commit_recovered_step(DXSL, alpha, Citer, dxsl_soc, alpha_soc, trial_iter);
            return SocCorrectionOutcome{true, 0.0};
        }

        // Rejected. Without an infeasibility reading (theta < 0) SOC cannot
        // measure progress; stop.
        const double trial_violation = trial_iter.theta_at_first_rejection_;
        if (trial_violation < 0.0)
            return SocCorrectionOutcome{false, std::numeric_limits<double>::infinity()};

        // Accumulate for a possible next round:
        // c_soc <- alpha_soc*c_soc + c(x_k + alpha_soc*d_soc).
        // An un-evaluable accumulation point leaves no corrected constraint block
        // to carry forward: end the correction loop with the same signal the
        // non-finite direction guard above uses.
        try {
            eval_trial_constraints(ctx, obj_scale, XSL, dxsl_soc, alpha_soc, XSL2, trial_cons);
        } catch (const std::exception &e) {
            note_eval_error(ctx.eval_errors_, e.what());
            return SocCorrectionOutcome{false, std::numeric_limits<double>::infinity()};
        } catch (...) {
            note_eval_error_unknown(ctx.eval_errors_);
            return SocCorrectionOutcome{false, std::numeric_limits<double>::infinity()};
        }
        c_soc = alpha_soc * c_soc + trial_cons;
        return SocCorrectionOutcome{false, trial_violation};
    };

    const Action action =
        soc_run_loop(Citer.theta_at_first_rejection_, max_soc, soc_steps, do_correction);

    if (action != Action::kRetry) {
        // Reverting to the originally-rejected step: restore the diagnostic
        // fraction-to-boundary lengths (DXSL/alpha were never touched on a
        // rejected correction).
        alphap = alphap_orig;
        alphad = alphad_orig;
    }
    return action;
}

// ============================================================================
// ExtendedBacktrackRecovery — see watchdog.h's file docstring, "Extended
// backtracking" section, for the exact mechanics (why scaling DXSL by the
// live alpha and re-driving run_acceptance_backtrack reproduces the SAME
// ladder with no redundant re-test and no new math).
// ============================================================================
RecoveryChain::Action ExtendedBacktrackRecovery::on_step_rejected(
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx,
    AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism,
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, double &alpha, double & /*alphap*/, double & /*alphad*/,
    int & /*soc_steps*/, int & /*resolved_depth*/, int & /*watchdog_activations*/) {
    const int max_extended = ctx.settings_.ls_extended_iters_;
    if (max_extended <= 0)
        return Action::kAcceptAsIs; // defensive: only built when ls_extended_iters_ > 0.

    // Extended backtracking is a pure α-schedule extension — acceptance-
    // strategy-neutral. It re-drives the acceptance backtrack (through the
    // mechanism, so the SAME acceptance criteria apply on the classic and the
    // generic paths) on the SAME rejected direction, further scaled. `scale`
    // continues the SAME ladder from the live alpha (compute_step's return
    // value, passed in as `alpha`) — NOT a restart at 1.0. DXSL itself is never
    // touched here (only read): the direction that was rejected is the SAME
    // direction extended backtracking keeps testing at smaller alpha, exactly
    // like the classic loop's own internal divisions do.
    Eigen::VectorXd dxsl_ext(DXSL.size());
    double scale = alpha;
    for (int i = 0; i < max_extended; ++i) {
        dxsl_ext = scale * DXSL;
        IterateInfo trial_iter;
        const double alpha_result = mechanism.run_acceptance_backtrack(
            lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, dxsl_ext, XSL2, RHS, RHS2, acceptance,
            trial_iter, iters, ctx);
        if (trial_iter.accepted_) {
            // The accepted step is still the original direction, further scaled.
            commit_recovered_step(DXSL, alpha, Citer, dxsl_ext, alpha_result, trial_iter);
            return Action::kRetry;
        }
        // Not accepted: run_acceptance_backtrack's own internal loop
        // (classic_line_search or generic_line_search, whichever the
        // acceptance strategy drives) already divided down to the next
        // untested rung (relative to dxsl_ext); carry that forward as the
        // next external trial's absolute scale.
        scale = alpha_result * scale;
    }
    return Action::kAcceptAsIs; // extended budget exhausted: take the original rejected step.
}

// ============================================================================
// WatchdogRecovery — drives WatchdogState against the real working set. See
// watchdog.h's file docstring, "Watchdog" section, for the full semantics.
// ============================================================================
RecoveryChain::Action WatchdogRecovery::on_step_rejected(
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx,
    AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism,
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, double &alpha, double &alphap, double &alphad, int &soc_steps,
    int &resolved_depth, int &watchdog_activations) {
    // Always-available proxy for "did the point improve" — see the file
    // docstring for why prim_obj + barr_obj (rather than a per-variant merit
    // value) is used here.
    const double merit = prim_obj + barr_obj;
    const WatchdogState::Outcome outcome = state_.record_rejected_iteration(mu, merit);

    switch (outcome) {
    case WatchdogState::Outcome::kAccumulate:
        // inner_ is enforced non-null at construction (see the class doc) --
        // no kAcceptAsIs/kRecoveryDepthUnresolved fallback branch is reachable
        // here.
        return inner_->on_step_rejected(Citer, iters, ctx, acceptance, mechanism, lsmode,
                                        obj_scale, mu, prim_obj, barr_obj, XSL, DXSL, XSL2, RHS,
                                        RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
                                        watchdog_activations);

    case WatchdogState::Outcome::kArmed:
        // Just armed: snapshot the pre-watchdog iterate (XSL as it stands
        // right now, before this iteration's relaxed-accepted step is
        // committed) so a later revert can restore it. The bound multipliers
        // are part of that iterate and are snapshotted with it. The dual state
        // pointer is wired unconditionally, so gate on the bound set itself:
        // off the bound path there are no multipliers worth copying.
        snapshot_xsl_ = XSL;
        if (ctx.bounds_)
            snapshot_bound_duals_ = *ctx.bound_duals_;
        ++watchdog_activations;
        [[fallthrough]];
    case WatchdogState::Outcome::kTrialRelax:
        Citer.accepted_ = true;
        resolved_depth = kRecoveryDepthWatchdog;
        return Action::kAcceptAsIs;

    case WatchdogState::Outcome::kTrialProgress:
        // Progress observed: the emergency is over, hand this rejection back
        // to the wrapped chain for its normal treatment. inner_ is enforced
        // non-null at construction (see the class doc) -- no
        // kAcceptAsIs/kRecoveryDepthUnresolved fallback branch is reachable
        // here.
        return inner_->on_step_rejected(Citer, iters, ctx, acceptance, mechanism, lsmode,
                                        obj_scale, mu, prim_obj, barr_obj, XSL, DXSL, XSL2, RHS,
                                        RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
                                        watchdog_activations);

    case WatchdogState::Outcome::kTrialRevert:
        // Window exhausted with no progress: revert XSL to the snapshot and
        // leave DXSL/alpha at zero so alg_impl's XSL += alpha*DXSL commit is
        // a no-op on the already-reverted iterate. DXSL is zeroed BEFORE the
        // XSL assignment (not after): XSL/DXSL are threaded through this
        // interface as independent Eigen::VectorXd& parameters, but nothing
        // in the contract (recovery_chain.h) forbids a caller from binding
        // them to the same underlying storage, and the snapshot write is the
        // one that must be the LAST write standing on that storage -- a
        // caller-supplied test double exercising exactly this aliasing is
        // what caught the ordering bug this comment now documents.
        DXSL.setZero();
        XSL = snapshot_xsl_;
        // The bound multipliers revert with the primals they belong to. Without
        // this the discarded trajectory's z survives into Sigma and the dual
        // infeasibility, and the kappa_sigma clamp is far too loose to pull it
        // back. The dz vectors are left alone: the caller's commit applies
        // alpha * alphad = 0 of them, so they never reach the reverted state.
        if (ctx.bounds_)
            *ctx.bound_duals_ = snapshot_bound_duals_;
        alpha = 0.0;
        Citer.accepted_ = true;
        resolved_depth = kRecoveryDepthWatchdog;
        return Action::kRetry;
    }
    throw std::logic_error(
        "WatchdogRecovery::on_step_rejected: unreachable WatchdogState::Outcome");
}

// ============================================================================
// ChainedRecovery — tries SOC then extended backtracking (see watchdog.h's
// class doc for the ordering rationale), stamping resolved_depth with
// whichever link's index actually resolved the rejection.
// ============================================================================
RecoveryChain::Action ChainedRecovery::on_step_rejected(
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx,
    AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism,
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, double &alpha, double &alphap, double &alphad, int &soc_steps,
    int &resolved_depth, int &watchdog_activations) {
    // Links in trial order, each paired with the depth it stamps when it resolves.
    // A null link is simply absent from the chain.
    const std::array<std::pair<RecoveryChain *, int>, 2> links = {
        std::pair<RecoveryChain *, int>{soc_.get(), kRecoveryDepthSoc},
        std::pair<RecoveryChain *, int>{extended_.get(), kRecoveryDepthExtended}};
    for (const auto &[link, depth] : links) {
        if (!link)
            continue;
        const Action action = link->on_step_rejected(
            Citer, iters, ctx, acceptance, mechanism, lsmode, obj_scale, mu, prim_obj, barr_obj,
            XSL, DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
            watchdog_activations);
        if (action != Action::kAcceptAsIs) {
            resolved_depth = depth;
            return action;
        }
    }
    resolved_depth = kRecoveryDepthUnresolved;
    return Action::kAcceptAsIs;
}

// ============================================================================
// FeasibilitySwitchRecovery — outermost link; converts a ladder-exhausted
// rejection into a feasibility-restoration mode-switch. See
// feasibility_switch_recovery.h for the design.
// ============================================================================

RecoveryChain::Action FeasibilitySwitchRecovery::on_step_rejected(
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx,
    AcceptanceStrategy &acceptance, GlobalizationMechanism &mechanism,
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, double &alpha, double &alphap, double &alphad, int &soc_steps,
    int &resolved_depth, int &watchdog_activations) {
    // Delegate the whole rejection to the inner chain first.
    const Action inner = inner_->on_step_rejected(
        Citer, iters, ctx, acceptance, mechanism, lsmode, obj_scale, mu, prim_obj, barr_obj, XSL,
        DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
        watchdog_activations);

    // Only a ladder-exhausted rejection is a candidate for a feasibility
    // switch. kAcceptAsIs alone is NOT a sufficient discriminator: it is
    // overloaded. The watchdog's trial-acceptance path also returns kAcceptAsIs
    // — but it has RESOLVED the rejection (it stamped Citer.accepted_ = true and
    // wrote resolved_depth = kRecoveryDepthWatchdog), taking the relaxed step on
    // purpose. Only the ladder-exhaustion fallback returns kAcceptAsIs while
    // leaving resolved_depth at the caller-seeded kRecoveryDepthUnresolved (any
    // link that resolved sets its own depth). So the designed discriminator is
    // the depth out-parameter, not the Action: intercept only an UNRESOLVED
    // kAcceptAsIs. Anything else — a resolved kAcceptAsIs (watchdog), or a
    // kRetry/kSwitchToFeasibility/kGiveUp — passes through untouched, with the
    // resolved_depth it already set.
    if (inner != Action::kAcceptAsIs || resolved_depth != kRecoveryDepthUnresolved)
        return inner;

    // No restoration strategy configured, or already in restoration mode:
    // there is nothing to switch into. (Both are impossible while active,
    // since alg_impl re-enters iterations in feasibility mode — but guarding
    // keeps the link correct in isolation and under the unit tests.)
    if (ctx.restoration_ == nullptr || ctx.restoration_->is_active())
        return inner;

    // Current-iterate constraint violation: L1 norm of the KKT constraint block
    // (the [eq | iq] tail of RHS), the same measure alg_impl builds the
    // restoration reference from.
    const int ncons = ctx.equal_cons_ + ctx.inequal_cons_;
    const double constraint_violation = RHS.tail(ncons).template lpNorm<1>();

    // Near-feasible guard + per-phase entry budget (RestorationStrategy::
    // entry_permitted). Either refusing keeps the inner take-as-is behavior.
    if (!ctx.restoration_->entry_permitted(constraint_violation, ctx))
        return inner;

    // The ladder is exhausted and a real restoration episode is warranted. Both
    // the soft pre-stage and the full switch belong to the restoration recovery
    // bucket. This link mutates nothing; alg_impl performs the trial step (soft)
    // or the actual mode entry (switch).
    resolved_depth = kRecoveryDepthRestoration;

    // Soft feasibility pre-stage (nested restoration only). Before committing to
    // the full restoration switch, try the full fraction-to-boundary step on the
    // current search direction under alg_impl's primal-dual-error reduction test.
    // Escalate to the real switch after more than kMaxSoftRestoIters successive
    // soft iterations. The proximal switch has no pre-stage — it switches
    // directly. See the file docstring.
    if (ctx.restoration_->is_nested()) {
        ++soft_counter_;
        if (soft_counter_ <= kMaxSoftRestoIters)
            return Action::kSoftFeasibilityStep;
    }

    // Signal the switch. alg_impl's kSwitchToFeasibility case performs the actual
    // mode entry.
    return Action::kSwitchToFeasibility;
}

// ============================================================================
// SwitchingAcceptance — the shared Wächter–Biegler switching-condition
// skeleton (filter/funnel shared base). See globalization/switching_acceptance.h
// for the full formulation; the code below is a direct transcription of the
// θ_min/θ_max derivation, the switching condition (Eq. 19), and the F-type
// Armijo test (Eq. 20) documented there.
// ============================================================================

void SwitchingAcceptance::reset() {
    bounds_initialized_ = false;
    reset_bounds();
}

bool SwitchingAcceptance::is_infeasibility_sufficiently_reduced(
    const ProgressMeasures &reference, const ProgressMeasures &trial) const {
    (void)reference;
    (void)trial;
    throw std::logic_error(
        "SwitchingAcceptance::is_infeasibility_sufficiently_reduced has no shared "
        "implementation here; FilterAcceptance and FunnelAcceptance override it with a real "
        "body, so reaching this base throw means a subclass failed to override it");
}

bool SwitchingAcceptance::compute_switching_holds(const ProgressMeasures &current,
                                                  const ProgressMeasures &predicted_reduction,
                                                  double step_length) const {
    // Eq. (19): tested only when θ_k ≤ θ_min AND the step is a descent
    // direction for φ (m_f = predicted_reduction.objective > 0).
    if (current.infeasibility <= theta_min_ && predicted_reduction.objective > 0.0) {
        const double lhs =
            step_length * std::pow(predicted_reduction.objective / step_length, kSwitchingSPhi);
        const double rhs = kSwitchingDelta * std::pow(current.infeasibility, kSwitchingSTheta);
        return lhs > rhs;
    }
    return false;
}

bool SwitchingAcceptance::armijo_holds(const ProgressMeasures &current,
                                       const ProgressMeasures &trial,
                                       const ProgressMeasures &predicted_reduction) const {
    // Eq. (20). φ(pt) = pt.objective + pt.auxiliary.
    const double phi_current = current.objective + current.auxiliary;
    const double phi_trial = trial.objective + trial.auxiliary;
    return phi_trial <= phi_current - kArmijoEtaPhi * predicted_reduction.objective;
}

bool SwitchingAcceptance::is_iterate_acceptable(const ProgressMeasures &current,
                                                const ProgressMeasures &trial,
                                                const ProgressMeasures &predicted_reduction,
                                                double objective_multiplier, double step_length) {
    // objective_multiplier is available for future rules but not needed by
    // the arithmetic here — see modern_merit.h's identical posture.
    (void)objective_multiplier;

    // Lazy per-phase initialization (see the file-top formulation): θ₀ is the
    // FIRST current.infeasibility seen since the last reset().
    if (!bounds_initialized_) {
        const double theta_0 = current.infeasibility;
        theta_min_ = kThetaMinFact * std::max(1.0, theta_0);
        theta_max_ = kThetaMaxFact * std::max(1.0, theta_0);
        initialize_bounds(theta_0);
        bounds_initialized_ = true;
    }

    // Hard ceiling (Eq. 21): rejected before any other test. Ipopt leaves its
    // filter-attribution flag untouched on a "Tmax" rejection, so the second
    // argument here is a don't-care false (see notify_trial_rejected()'s doc).
    if (trial.infeasibility > theta_max_) {
        notify_trial_rejected(RejectionCause::kCeiling, false);
        return false;
    }

    // Strategy membership (step 2): checked for EVERY trial, F-type included —
    // the filter's non-dominance test / the funnel's within-the-width test.
    if (!is_trial_acceptable_to_strategy(current, trial)) {
        // Ipopt attributes a filter rejection only when its own first test
        // (T1) PASSED. Reproduce that by SPECULATIVELY evaluating the
        // type-appropriate T1 here — the Armijo condition if the switching
        // condition would have selected an f-type trial, the H-type
        // sufficient-progress delegate otherwise (side-effect-free per its
        // contract) — and handing the verdict to notify_trial_rejected().
        const bool switching_holds_spec =
            compute_switching_holds(current, predicted_reduction, step_length);
        const bool trial_passed_progress_test =
            switching_holds_spec ? armijo_holds(current, trial, predicted_reduction)
                                 : is_h_type_progress_acceptable(current, trial);
        notify_trial_rejected(RejectionCause::kMembership, trial_passed_progress_test);
        return false;
    }

    // Switching condition (Eq. 19) selects F-type vs H-type.
    const bool switching_holds =
        compute_switching_holds(current, predicted_reduction, step_length);

    if (switching_holds) {
        // F-type: accept iff the Armijo condition on φ holds (Eq. 20).
        if (armijo_holds(current, trial, predicted_reduction)) {
            register_accepted_step(current, trial, /*h_type=*/false);
            return true;
        }
        // T1 (Armijo) failed by definition to reach this branch.
        notify_trial_rejected(RejectionCause::kArmijo, false);
        return false;
    }

    // H-type: the subclass sufficient-progress verdict (filter acceptable-to-
    // current-iterate / funnel β·width). Bookkeeping runs only on an accept.
    if (!is_h_type_progress_acceptable(current, trial)) {
        // T1 (the current-iterate test) failed by definition to reach here.
        notify_trial_rejected(RejectionCause::kHTypeProgress, false);
        return false;
    }
    register_accepted_step(current, trial, /*h_type=*/true);
    return true;
}

// ============================================================================
// FunnelAcceptance — scalar-funnel H-type strategy on the switching skeleton.
// See globalization/funnel_acceptance.h for the full formulation; the code
// below transcribes the width initialization (init), the H-type verdict (2),
// and the width update (3) documented there, following Uno's shipped default
// funnel_update_strategy = 1.
// ============================================================================

void FunnelAcceptance::initialize_bounds(double theta_0) {
    // (init): τ = max(τ̄, κ̄·θ₀).
    width_ = std::max(kFunnelInitialUpperBound, kFunnelInfeasibilityFactor * theta_0);
}

void FunnelAcceptance::reset_bounds() {
    // Always restore the uninitialized sentinel on the WORKING width so the next
    // θ₀ re-derives it.
    width_ = std::numeric_limits<double>::infinity();
    // Phase-aware (mirroring the filter): inside the feasibility phase this is a
    // μ-event reset — clear the working width only, preserving the stashed
    // optimality width and the in-feasibility flag so the exit test and the exit
    // re-base still have the frozen width to consult/restore. Outside the phase
    // this is the full per-phase clear, which also drops any leftover stash
    // defensively. See (5)/(6) in funnel_acceptance.h.
    if (!in_feasibility_phase_)
        stashed_width_ = std::numeric_limits<double>::infinity();
}

bool FunnelAcceptance::is_trial_acceptable_to_strategy(const ProgressMeasures &current,
                                                       const ProgressMeasures &trial) {
    // (2a) membership: within the funnel (θ_trial ≤ τ) — Uno Funnel::acceptable.
    // current participates only in the update, not the verdict.
    (void)current;
    return trial.infeasibility <= width_;
}

bool FunnelAcceptance::is_h_type_progress_acceptable(const ProgressMeasures &current,
                                                     const ProgressMeasures &trial) {
    // (2b) H-type sufficient reduction (θ_trial ≤ β·τ) — Uno
    // Funnel::sufficient_decrease_condition.
    (void)current;
    return trial.infeasibility <= kFunnelBeta * width_;
}

void FunnelAcceptance::register_accepted_step(const ProgressMeasures &current,
                                              const ProgressMeasures &trial, bool h_type) {
    // (3): the width tightens ONLY on an accepted H-type step (Uno calls
    // funnel.update() from the H-type branch alone); an F-type accept leaves τ
    // untouched. Uno update strategy 1.
    if (!h_type)
        return;
    const double theta_current = current.infeasibility;
    const double theta_trial = trial.infeasibility;
    if (theta_trial <= theta_current) {
        width_ = std::max(kFunnelBeta * width_,
                          convex_combination(theta_current, theta_trial, kFunnelKappa));
    } else {
        width_ = kFunnelBeta * width_;
    }
}

void FunnelAcceptance::append_diagnostics(PSIOPT::SolveResult &result) const {
    result.last_funnel_width_ = std::isfinite(width_) ? width_ : -1.0;
}

bool FunnelAcceptance::is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                                             const ProgressMeasures &trial) const {
    // Uno FunnelMethod::is_infeasibility_sufficiently_reduced (cvanaret/Uno
    // 7481abe):
    //   return funnel.acceptable(trial.infeasibility) &&
    //          trial.infeasibility <= parameters.beta * reference.infeasibility;
    // funnel.acceptable(θ) is the membership test θ ≤ width; parameters.beta is
    // the funnel's own sufficient-decrease β, reused here (kFunnelBeta) exactly
    // as Uno reuses it. While in the feasibility phase the membership half tests
    // against the STASHED (frozen optimality) width — the reference solver's
    // exit test reads its frozen optimality funnel; outside the phase it reads
    // the live width (well-defined; the seam only tests for exit during the
    // phase). See (5a) in funnel_acceptance.h.
    const double effective_width = in_feasibility_phase_ ? stashed_width_ : width_;
    return trial.infeasibility <= effective_width &&
           trial.infeasibility <= kFunnelBeta * reference.infeasibility;
}

void FunnelAcceptance::notify_switch_to_feasibility(const ProgressMeasures &current_progress) {
    // T6: a second entry without an intervening exit would stash the FEASIBILITY
    // working width over the preserved optimality width, silently clobbering the
    // frozen width the exit test and exit re-base consult — a phase-transition
    // mis-wiring, not a recoverable runtime condition.
    if (in_feasibility_phase_)
        throw std::logic_error(
            "FunnelAcceptance::notify_switch_to_feasibility: already in the feasibility "
            "phase (in_feasibility_phase_ is true) — a solver wiring bug called entry "
            "without an intervening notify_switch_to_optimality exit");

    // The funnel augments nothing at the switch point; its whole persistent
    // state is the scalar width.
    (void)current_progress;

    // Stash the optimality width, then enter feasibility mode and reinitialize a
    // fresh working width via the reset() machinery (re-arms the base's lazy θ₀
    // init so the feasibility phase derives its own θ_min/θ_max and width). The
    // flag is set FIRST so the phase-aware reset_bounds() preserves the stash
    // just written. See (5b) in funnel_acceptance.h.
    stashed_width_ = width_;
    in_feasibility_phase_ = true;
    this->reset();
}

void FunnelAcceptance::notify_switch_to_optimality(const ProgressMeasures &current_progress) {
    // T6: an exit without a preceding entry has no stashed width to restore —
    // running this body would re-base whatever stashed_width_ last held (the
    // uninitialized sentinel, or a stale stash), a phase-transition mis-wiring
    // symmetric to the entry-side hazard above.
    if (!in_feasibility_phase_)
        throw std::logic_error(
            "FunnelAcceptance::notify_switch_to_optimality: not in the feasibility phase "
            "(in_feasibility_phase_ is false) — a solver wiring bug called exit without a "
            "preceding notify_switch_to_feasibility entry");

    // Restore the preserved optimality width DIRECTLY (not via reset(), which
    // would re-arm the base's lazy θ₀ init and make the next optimality call
    // re-derive and wipe the just-restored width), THEN apply Uno's
    // update_restoration re-base to the restored width:
    //   width = convex_combination(width, current_infeasibility, kappa)
    //         = kappa*width + (1-kappa)*current_infeasibility.
    // See (5c) in funnel_acceptance.h. The stash is left as a harmless leftover,
    // dropped by the next reset() OUTSIDE the phase.
    width_ = stashed_width_;
    width_ = convex_combination(width_, current_progress.infeasibility, kFunnelKappa);
    in_feasibility_phase_ = false;
}

// ============================================================================
// FilterAcceptance — (θ, φ)-pair filter H-type strategy on the switching
// skeleton. See globalization/filter_acceptance.h for the full formulation and
// the rule-by-rule Ipopt citations; the code below transcribes the
// acceptable-to-current test (1a), the acceptable-to-filter test (1b), the
// augmentation (2), and the filter-reset heuristic (4) documented there.
// ============================================================================

void FilterAcceptance::initialize_bounds(double theta_0) {
    // Start the phase with an empty filter (Ipopt Reset; the θ_max ceiling is
    // the base's scalar bound, not a seeded filter entry — see the divergence
    // notes). θ_0 is not needed here: the filter derives no bound from it.
    (void)theta_0;
    filter_.clear();
    successive_filter_rejections_ = 0;
    n_filter_resets_ = 0;
    last_rejection_was_filter_ = false;
}

void FilterAcceptance::reset_bounds() {
    // Always empty the live working state (filter + reset-heuristic counters) so
    // the next θ₀ capture re-arms a clean per-phase filter.
    filter_.clear();
    successive_filter_rejections_ = 0;
    n_filter_resets_ = 0;
    last_rejection_was_filter_ = false;
    // Reset invariant (see (6) in filter_acceptance.h): inside the feasibility
    // phase this is a μ-event reset — clear WORKING state only, preserving the
    // stashed optimality filter and the in-feasibility flag so the exit test
    // still has the preserved filter to consult. Outside the phase this is the
    // full per-phase clear, which also drops any leftover stash/flag
    // defensively. The injected constraint tolerance is configuration, not
    // working state, and is intentionally left untouched.
    if (!in_feasibility_phase_) {
        stashed_filter_.clear();
        stashed_successive_filter_rejections_ = 0;
        stashed_n_filter_resets_ = 0;
        stashed_last_rejection_was_filter_ = false;
    }
}

bool FilterAcceptance::is_acceptable_to_current(double phi_trial, double theta_trial,
                                                double phi_current, double theta_current) {
    // (1a) barrier-objective ceiling (Ipopt IsAcceptableToCurrentIterate's
    // obj_max_inc test) — only when the barrier objective increases.
    if (phi_trial > phi_current) {
        double basval = 1.0;
        if (std::abs(phi_current) > 10.0)
            basval = std::log10(std::abs(phi_current));
        if (std::log10(phi_trial - phi_current) > kFilterObjMaxInc + basval)
            return false;
    }

    // (1a) two-condition margin test (WB Eqs. (18a)/(18b)); both margins scale
    // by θ_current, matching Ipopt. Plain ≤ (see the Compare_le divergence note).
    return theta_trial <= (1.0 - kFilterGammaTheta) * theta_current ||
           phi_trial - phi_current <= -kFilterGammaPhi * theta_current;
}

bool FilterAcceptance::is_trial_acceptable_to_strategy(const ProgressMeasures &current,
                                                       const ProgressMeasures &trial) {
    // (1b) membership: acceptable to the current filter (Ipopt
    // IsAcceptableToCurrentFilter). Run for every trial; the reset heuristic is
    // driven from notify_trial_rejected()/register_accepted_step(), not here.
    (void)current;
    const double theta_trial = trial.infeasibility;
    const double phi_trial = trial.objective + trial.auxiliary;
    return filter_.acceptable(phi_trial, theta_trial);
}

bool FilterAcceptance::is_h_type_progress_acceptable(const ProgressMeasures &current,
                                                     const ProgressMeasures &trial) {
    // (1a) acceptable to the current iterate (Ipopt IsAcceptableToCurrentIterate).
    const double theta_trial = trial.infeasibility;
    const double phi_trial = trial.objective + trial.auxiliary;
    const double theta_current = current.infeasibility;
    const double phi_current = current.objective + current.auxiliary;
    return is_acceptable_to_current(phi_trial, theta_trial, phi_current, theta_current);
}

void FilterAcceptance::notify_trial_rejected(RejectionCause cause,
                                             bool trial_passed_progress_test) {
    // Ipopt last_rejection_due_to_filter_: TRUE only when T1 (the Armijo/
    // current-iterate test) PASSED and the filter membership test failed —
    // i.e. a kMembership rejection whose speculatively-evaluated
    // trial_passed_progress_test is true. A current-iterate/Armijo rejection
    // (T1 itself failed) sets it FALSE; a θ_max ceiling rejection leaves it
    // UNCHANGED (Ipopt returns on "Tmax" before touching the flag; the second
    // argument is a don't-care here). See (4).
    switch (cause) {
    case RejectionCause::kMembership:
        last_rejection_was_filter_ = trial_passed_progress_test;
        break;
    case RejectionCause::kArmijo:
    case RejectionCause::kHTypeProgress:
        last_rejection_was_filter_ = false;
        break;
    case RejectionCause::kCeiling:
        break;
    }
}

void FilterAcceptance::register_accepted_step(const ProgressMeasures &current,
                                              const ProgressMeasures &trial, bool h_type) {
    (void)trial;
    // (4) Filter-reset heuristic (Ipopt CheckAcceptabilityOfTrialPoint tail):
    // runs once per ACCEPT, reading the last rejection's cause. A filter-caused
    // last rejection advances the successive-iteration counter (and clears the
    // filter at the trigger, honouring the per-phase cap); any other last
    // rejection zeroes it. Below the cap only, matching Ipopt's outer guard.
    if (n_filter_resets_ < kFilterMaxResets) {
        if (last_rejection_was_filter_) {
            ++successive_filter_rejections_;
            if (successive_filter_rejections_ >= kFilterResetTrigger) {
                filter_.clear();
                ++n_filter_resets_;
                successive_filter_rejections_ = 0;
            }
        } else {
            successive_filter_rejections_ = 0;
        }
    }
    last_rejection_was_filter_ = false;

    // (2) Augment ONLY on an H-type accept (Ipopt AugmentFilter, called from the
    // H-type acceptance branch): store the CURRENT (reference) iterate's margined
    // pair. An F-type accept never augments.
    if (h_type) {
        const double theta_current = current.infeasibility;
        const double phi_current = current.objective + current.auxiliary;
        filter_.augment(phi_current, theta_current);
    }
}

void FilterAcceptance::append_diagnostics(PSIOPT::SolveResult &result) const {
    result.last_filter_size_ = static_cast<int>(filter_.size());
    result.last_filter_resets_ = n_filter_resets_;
}

bool FilterAcceptance::is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                                             const ProgressMeasures &trial) const {
    // Ipopt IpRestoConvCheck::CheckConvergence + IpRestoFilterConvCheck::
    // TestOrigProgress (coin-or/Ipopt 72a29c9). The original problem's current
    // iterate is FROZEN at the restoration entry point, so its (θ, φ) is
    // `reference`; the trial's is `trial`. See (5b) in filter_acceptance.h.
    const double theta_ref = reference.infeasibility;
    const double theta_trial = trial.infeasibility;
    const double phi_ref = reference.objective + reference.auxiliary;
    const double phi_trial = trial.objective + trial.auxiliary;

    // (Tmax) relative θ-reduction with the constraint-tolerance floor:
    //   orig_inf_pr_max = Max(kappa_resto_ * orig_curr_inf_pr,
    //                         Min(orig tol, orig constr_viol_tol_));
    // Tycho's single constraint tolerance stands in for Ipopt's two-tolerance
    // minimum (injected via set_restoration_constraint_tol; see the divergence
    // note in the header).
    const double floor = std::max(kKappaResto * theta_ref, restoration_constraint_tol_);
    if (theta_trial > floor)
        return false;

    // Acceptable to the PRESERVED optimality filter (Ipopt
    // IsAcceptableToCurrentFilter). During feasibility the preserved filter is
    // the stash (the optimality filter + the entry pair added at (5a)).
    if (!stashed_filter_.acceptable(phi_trial, theta_trial))
        return false;

    // Acceptable w.r.t. the preserved (frozen entry) iterate — Ipopt
    // IsAcceptableToCurrentIterate against `reference`, margined identically to
    // the live (1a) acceptable-to-current test.
    return is_acceptable_to_current(phi_trial, theta_trial, phi_ref, theta_ref);
}

void FilterAcceptance::notify_switch_to_feasibility(const ProgressMeasures &current_progress) {
    // T6: a second entry without an intervening exit would run
    // `stashed_filter_ = filter_` while filter_ already holds the
    // FEASIBILITY working filter, silently clobbering the preserved
    // optimality stash — a phase-transition mis-wiring, not a recoverable
    // runtime condition.
    if (in_feasibility_phase_)
        throw std::logic_error(
            "FilterAcceptance::notify_switch_to_feasibility: already in the feasibility "
            "phase (in_feasibility_phase_ is true) — a solver wiring bug called entry "
            "without an intervening notify_switch_to_optimality exit");

    // (5a). Uno FilterMethod::notify_switch_to_feasibility (Ipopt
    // PrepareRestoPhaseStart analog): augment the optimality filter with the
    // entry pair BEFORE it is preserved. φ = objective + auxiliary (the
    // unconstrained merit measure), θ = infeasibility.
    const double theta_entry = current_progress.infeasibility;
    const double phi_entry = current_progress.objective + current_progress.auxiliary;
    filter_.augment(phi_entry, theta_entry);

    // Preserve the optimality-phase working state (the augmented filter + the
    // reset-heuristic counters) — the exit test consults stashed_filter_.
    stashed_filter_ = filter_;
    stashed_successive_filter_rejections_ = successive_filter_rejections_;
    stashed_n_filter_resets_ = n_filter_resets_;
    stashed_last_rejection_was_filter_ = last_rejection_was_filter_;

    // Enter feasibility mode, then reinitialize fresh working state via the
    // reset() machinery (re-arms the lazy θ₀ init so the feasibility phase
    // derives its own θ_min/θ_max). The flag is set first so the phase-aware
    // reset_bounds() preserves the stash just written above.
    in_feasibility_phase_ = true;
    this->reset();
}

void FilterAcceptance::notify_switch_to_optimality(const ProgressMeasures &current_progress) {
    // T6: an exit without a preceding entry has no stash to restore — running
    // this body would overwrite the live optimality filter with whatever
    // stashed_filter_ last held (empty, or a stale stash from a prior phase),
    // a phase-transition mis-wiring symmetric to the entry-side hazard above.
    if (!in_feasibility_phase_)
        throw std::logic_error(
            "FilterAcceptance::notify_switch_to_optimality: not in the feasibility phase "
            "(in_feasibility_phase_ is false) — a solver wiring bug called exit without a "
            "preceding notify_switch_to_feasibility entry");

    // (5c). Restore the preserved optimality-phase working state and augment the
    // restored filter with the EXIT pair (Uno FilterMethod::
    // notify_switch_to_optimality — Uno adds the switch-point pair at BOTH
    // transitions).
    const double theta_exit = current_progress.infeasibility;
    const double phi_exit = current_progress.objective + current_progress.auxiliary;

    // Restore directly (NOT via reset()): the base's lazy θ₀ init is left ARMED
    // (bounds_initialized_ stays true from the feasibility phase), because
    // re-arming it would make the next optimality is_iterate_acceptable call
    // initialize_bounds(), which clears the filter — wiping the very state just
    // restored. The θ_min/θ_max thresholds therefore retain their feasibility-
    // phase values until the next phase-boundary reset (see the base-bounds
    // divergence note); this affects only the heuristic switching/ceiling
    // scalars, not the restored filter membership/dominance semantics.
    filter_ = stashed_filter_;
    successive_filter_rejections_ = stashed_successive_filter_rejections_;
    n_filter_resets_ = stashed_n_filter_resets_;
    last_rejection_was_filter_ = stashed_last_rejection_was_filter_;
    filter_.augment(phi_exit, theta_exit);

    // Leave feasibility mode. The stash is intentionally NOT cleared here: it is
    // now a harmless leftover (the exit test only reads it while the flag is set,
    // and the next entry overwrites it). The next reset() OUTSIDE the phase drops
    // it defensively — the reset invariant, (6) in filter_acceptance.h.
    in_feasibility_phase_ = false;
}

// ============================================================================
// MonitoredBarrierGovernor — free<->monotone monitored barrier governor. See
// monitored_governor.h for the full formulation, the Ipopt source citations,
// and the μ-event / re-entry / error-norm resolutions. Every bare
// IpAdaptiveMuUpdate.cpp/IpMonotoneMuUpdate.cpp line-number citation below
// (including the AMU:/MMU: shorthand) is pinned to Ipopt releases/3.14.19
// (2695946fa79d2e84f3034e065e788933a81466eb), matching monitored_governor.h.
// ============================================================================

MonitoredBarrierGovernor::MonitoredBarrierGovernor()
    : free_delegate_(std::make_unique<ClassicAdaptiveGovernor>()) {}

MonitoredBarrierGovernor::MonitoredBarrierGovernor(std::unique_ptr<BarrierGovernor> free_delegate)
    : free_delegate_(std::move(free_delegate)) {}

MonitoredBarrierGovernor::~MonitoredBarrierGovernor() = default;

double MonitoredBarrierGovernor::monitor_error(const IterateInfo &it) {
    // (1): sum of squared ∞-norm residual parts — the Tycho mapping of Ipopt's
    // 2-norm-squared quality function (IpAdaptiveMuUpdate.cpp:657-675).
    return it.kkt_inf_ * it.kkt_inf_ + it.econ_inf_ * it.econ_inf_ +
           it.icon_inf_ * it.icon_inf_ + it.barr_inf_ * it.barr_inf_;
}

double MonitoredBarrierGovernor::barrier_subproblem_error(const IterateInfo &it) {
    // (6): ∞-norm barrier optimality error (Ipopt curr_barrier_error analog).
    return std::max({it.kkt_inf_, it.econ_inf_, it.icon_inf_, it.barr_inf_});
}

double MonitoredBarrierGovernor::fiacco_mccormick_mu(double mu, double bar_tol, double kkt_tol,
                                                     double min_mu, double max_mu) {
    // (6): IpAdaptiveMuUpdate.cpp:327-329 / IpMonotoneMuUpdate.cpp:214-215.
    double new_mu = std::min(kBarrierKappaMu * mu, std::pow(mu, kBarrierThetaMu));
    const double floor = std::min(bar_tol, kkt_tol) / (kBarrierTolFactor + 1.0);
    new_mu = std::max(new_mu, floor);
    // Consistency clamp against the configured mu bounds (ClassicAdaptiveGovernor
    // common-tail clamp; Ipopt clamps to [mu_min, mu_max], AMU:623-624).
    new_mu = std::max(new_mu, min_mu);
    new_mu = std::min(new_mu, max_mu);
    return new_mu;
}

double MonitoredBarrierGovernor::handoff_mu(double avgcomp, double min_mu, double max_mu) {
    // (4): IpAdaptiveMuUpdate.cpp:618 (NewFixedMu default oracle) + 623-624 clamp.
    double new_mu = kAdaptiveMuMonotoneInitFactor * avgcomp;
    new_mu = std::max(new_mu, min_mu);
    new_mu = std::min(new_mu, max_mu);
    return new_mu;
}

bool MonitoredBarrierGovernor::check_sufficient_progress(double curr_error) const {
    // (2): IpAdaptiveMuUpdate.cpp:452-469 (CheckSufficientProgress, KKT_ERROR).
    // Fewer than the full window of references -> trivially sufficient.
    if (static_cast<int>(refs_vals_.size()) < kAdaptiveMuKktErrorRedIters) {
        return true;
    }
    for (double ref : refs_vals_) {
        if (curr_error <= kAdaptiveMuKktErrorRedFact * ref) {
            return true;
        }
    }
    return false;
}

void MonitoredBarrierGovernor::remember_accepted(double curr_error) {
    // (3): IpAdaptiveMuUpdate.cpp:496-505 (RememberCurrentPointAsAccepted).
    if (static_cast<int>(refs_vals_.size()) >= kAdaptiveMuKktErrorRedIters) {
        refs_vals_.pop_front();
    }
    refs_vals_.push_back(curr_error);
}

MonitoredBarrierGovernor::BarrierDecision
MonitoredBarrierGovernor::decide(const IterateInfo &current, double mu_in, double avgcomp,
                                 double bar_tol, double kkt_tol, double min_mu, double max_mu) {
    const double curr_error = monitor_error(current);
    BarrierDecision d;
    d.mu = mu_in;

    if (monotone_mode_) {
        // Fixed-mu branch (IpAdaptiveMuUpdate.cpp:299-341).
        if (check_sufficient_progress(curr_error)) {
            // Re-entry to free mode (AMU:303-311); no mu_event.
            monotone_mode_ = false;
            remember_accepted(curr_error);
            // Falls through to free delegation (d.monotone == false).
        } else {
            // Remain monotone (AMU:313-340).
            ++last_monotone_iters_;
            const double sub_err = barrier_subproblem_error(current);
            if (sub_err <= kBarrierTolFactor * monotone_mu_) {
                const double new_mu =
                    fiacco_mccormick_mu(monotone_mu_, bar_tol, kkt_tol, min_mu, max_mu);
                if (new_mu < monotone_mu_) {
                    monotone_mu_ = new_mu; // advance (AMU:335)
                    d.mu_event = true;     // new subproblem -> filter reset (AMU:339)
                }
            }
            d.mu = monotone_mu_;
            d.monotone = true;
            return d;
        }
    } else {
        // Free-mu branch (IpAdaptiveMuUpdate.cpp:343-389).
        if (check_sufficient_progress(curr_error)) {
            remember_accepted(curr_error); // stay free (AMU:352-357)
        } else {
            // Handoff to monotone (AMU:358-388).
            monotone_mode_ = true;
            ++last_monotone_switches_;
            monotone_mu_ = handoff_mu(avgcomp, min_mu, max_mu); // NewFixedMu (AMU:374)
            d.mu = monotone_mu_;
            d.mu_event = true; // new subproblem -> filter reset (AMU:386)
            d.monotone = true;
            return d;
        }
    }

    // Free mode (stayed free, or just re-entered): the free oracle produces mu.
    d.monotone = false;
    return d;
}

double MonitoredBarrierGovernor::barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu,
                                                   const SolverContext &ctx) const {
    return detail::barrier_objective(S, mu, ctx.inequal_cons_);
}

void MonitoredBarrierGovernor::barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                                                Eigen::Ref<Eigen::VectorXd> LI, double mu,
                                                Eigen::Ref<Eigen::VectorXd> AGS) const {
    detail::barrier_gradient(S, LI, mu, AGS);
}

double MonitoredBarrierGovernor::update_barrier(
    PSIOPT::BarrierModes barmode, double mu_in, double avgcomp, double mincomp,
    Eigen::VectorXd &XSL, Eigen::VectorXd &RHS, Eigen::VectorXd &DXSL, Eigen::VectorXd &Temp,
    GlobalizationMechanism &mechanism, SolverContext &ctx, double &barr_obj,
    const IterateInfo &current, bool &mu_event) {
    const BarrierDecision d = decide(current, mu_in, avgcomp, ctx.settings_.bar_tol_,
                                     ctx.settings_.kkt_tol_, ctx.settings_.min_mu_,
                                     ctx.settings_.max_mu_);
    mu_event = d.mu_event;

    if (d.monotone) {
        // Monotone mode: hold μ fixed and write the barrier tail directly (the
        // same objective/dual-gradient the free-mode common tail produces). The
        // slack / inequality-multiplier / dual-gradient blocks are the same
        // contiguous segments PSIOPT::KKTVector names (slacks/iq_lmults on XSL,
        // dual_grad on RHS): segment(primal_vars_, slack_vars_) and tail(...).
        const double mu = d.mu;
        auto slacks = XSL.segment(ctx.primal_vars_, ctx.slack_vars_);
        auto iq_lmults = XSL.tail(ctx.inequal_cons_);
        auto dual_grad = RHS.segment(ctx.primal_vars_, ctx.slack_vars_);
        barr_obj = barrier_objective(slacks, mu, ctx);
        barrier_gradient(slacks, iq_lmults, mu, dual_grad);
        return mu;
    }

    // Free mode: delegate the whole barrier update (oracle + clamp + barrier
    // tail) to the composed governor. Its own mu_event (never set by the classic
    // delegate) is captured locally so it cannot leak past the free-mode path.
    bool inner_event = false;
    return free_delegate_->update_barrier(barmode, mu_in, avgcomp, mincomp, XSL, RHS, DXSL, Temp,
                                          mechanism, ctx, barr_obj, current, inner_event);
}

void MonitoredBarrierGovernor::reset() {
    refs_vals_.clear();
    monotone_mode_ = false;
    monotone_mu_ = 0.0;
    last_monotone_switches_ = 0;
    last_monotone_iters_ = 0;
    if (free_delegate_) {
        free_delegate_->reset();
    }
}

void MonitoredBarrierGovernor::append_diagnostics(PSIOPT::SolveResult &result) const {
    result.last_monotone_switches_ = last_monotone_switches_;
    result.last_monotone_iters_ = last_monotone_iters_;
}

// ============================================================================
// ProximalSwitchRestoration — proximal feasibility mode-switch. See
// proximal_restoration.h for the full formulation and the Uno/Ipopt source
// citations; the code below transcribes the frozen-ζ snapshot (1), the
// per-coordinate scaling and proximal term/gradient (2), and the near-
// feasible + budget entry guard (3)/(4) documented there.
// ============================================================================

void ProximalSwitchRestoration::enter_restoration(const ProgressMeasures &reference,
                                                  const Eigen::Ref<const Eigen::VectorXd> &primals,
                                                  double mu) {
    reference_ = reference;
    x_r_ = primals;
    // (1): ζ = resto_proximity_weight * sqrt(mu), set ONCE here from the mu
    // live at this call — never re-derived from a later, live mu.
    zeta_ = kRestoProximityWeight * std::sqrt(mu);
    // (2): d_i = min(1, 1/|x_R_i|); diagonal_ = zeta * d_i^2.
    d_.resize(x_r_.size());
    for (Eigen::Index i = 0; i < x_r_.size(); ++i) {
        d_[i] = std::min(1.0, 1.0 / std::abs(x_r_[i]));
    }
    diagonal_ = zeta_ * d_.array().square();
    active_ = true;
    ++entries_;
}

void ProximalSwitchRestoration::reset() {
    active_ = false;
    x_r_.resize(0);
    d_.resize(0);
    diagonal_.resize(0);
    zeta_ = 0.0;
    reference_ = ProgressMeasures{};
    entries_ = 0;
    iterations_in_mode_ = 0;
}

double ProximalSwitchRestoration::proximal_objective(
    const Eigen::Ref<const Eigen::VectorXd> &primals) const {
    // (2): P(x) = (ζ/2) * sum_i d_i^2 * (x_i - x_R_i)^2 = 0.5 * sum_i
    // diagonal_[i] * (x_i - x_R_i)^2, using the cached diagonal_ = ζ*d_i^2.
    const Eigen::VectorXd delta = primals - x_r_;
    return 0.5 * diagonal_.dot(delta.cwiseProduct(delta));
}

void ProximalSwitchRestoration::add_proximal_gradient(
    const Eigen::Ref<const Eigen::VectorXd> &primals, Eigen::Ref<Eigen::VectorXd> grad_out) const {
    // (2): dP/dx_i = ζ * d_i^2 * (x_i - x_R_i) = diagonal_[i] * (x_i - x_R_i).
    grad_out += diagonal_.cwiseProduct(primals - x_r_);
}

// entry_permitted() (virtual, shared default body) and append_diagnostics()
// (non-virtual) are both inherited unoverridden from RestorationStrategy
// (restoration.h) — both concrete strategies share the identical
// entries_/iterations_in_mode_ pair.

// ============================================================================
// NestedL1Restoration — condensed l1 elastic feasibility restoration. See
// l1_restoration.h for the full formulation, the pinned Ipopt citations per
// formula, and the disclosed design decisions. The code below transcribes the
// closed-form slack init (4), the per-iteration condensation (pivot + r̃) and
// step recovery (5), the live-μ proximal objective/gradient/diagonal (2)/(3),
// and the fraction-to-boundary caps documented there.
// ============================================================================

void NestedL1Restoration::enter_restoration(const ProgressMeasures &reference,
                                            const Eigen::Ref<const Eigen::VectorXd> &primals,
                                            double mu) {
    // Guard-only entry: the nested solver path enters through enter_nested();
    // this records just the reference/center and the entry counters so the
    // budget guard and reference() accessor work if this base entry is used.
    (void)mu;
    reference_ = reference;
    x_r_ = primals;
    active_ = true;
    ++entries_;
}

void NestedL1Restoration::reset() {
    active_ = false;
    reference_ = ProgressMeasures{};
    x_r_.resize(0);
    dr2_.resize(0);
    resto_mu_ = 0.0;
    n_e_.resize(0);
    p_e_.resize(0);
    z_ne_.resize(0);
    z_pe_.resize(0);
    n_i_.resize(0);
    p_i_.resize(0);
    z_ni_.resize(0);
    z_pi_.resize(0);
    dn_e_.resize(0);
    dp_e_.resize(0);
    dzn_e_.resize(0);
    dzp_e_.resize(0);
    dn_i_.resize(0);
    dp_i_.resize(0);
    dzn_i_.resize(0);
    dzp_i_.resize(0);
    e_pivots_.resize(0);
    i_pivots_.resize(0);
    entries_ = 0;
    iterations_in_mode_ = 0;
    recenter_calls_ = 0;
}

double
NestedL1Restoration::proximal_objective(const Eigen::Ref<const Eigen::VectorXd> &primals) const {
    (void)primals;
    throw std::logic_error("NestedL1Restoration does not use the frozen-proximal-coefficient "
                           "objective surface; the live-mu nested_objective is used instead");
}

void NestedL1Restoration::add_proximal_gradient(const Eigen::Ref<const Eigen::VectorXd> &primals,
                                                Eigen::Ref<Eigen::VectorXd> grad_out) const {
    (void)primals;
    (void)grad_out;
    throw std::logic_error("NestedL1Restoration does not use the frozen-proximal-coefficient "
                           "gradient surface; the live-mu add_nested_gradient is used instead");
}

const Eigen::VectorXd &NestedL1Restoration::proximal_diagonal() const {
    throw std::logic_error("NestedL1Restoration does not use the frozen-proximal-coefficient "
                           "Hessian surface; the live-mu nested_primal_diagonal is used instead");
}

// entry_permitted() (virtual, shared default body) and append_diagnostics()
// (non-virtual) are both inherited unoverridden from RestorationStrategy
// (restoration.h) — same near-feasible guard + per-phase entry budget as the
// proximal switch (single-measure adaptation disclosure (d)); both concrete
// strategies share the identical entries_/iterations_in_mode_ pair.

void NestedL1Restoration::init_channel(const Eigen::Ref<const Eigen::VectorXd> &residuals,
                                       double mu, Eigen::VectorXd &n, Eigen::VectorXd &p,
                                       Eigen::VectorXd &zn, Eigen::VectorXd &zp) const {
    const Eigen::Index m = residuals.size();
    n.resize(m);
    p.resize(m);
    zn.resize(m);
    zp.resize(m);
    for (Eigen::Index i = 0; i < m; ++i) {
        // (4): closed-form positive-root init at the given barrier parameter
        // (resto_mu_ at entry; the live μ at a second-level re-center).
        const ElasticSlackInit s = l1_elastic_slack_init(residuals[i], mu, kRestoPenaltyParameter);
        n[i] = s.n;
        p[i] = s.p;
        zn[i] = s.zn;
        zp[i] = s.zp;
    }
}

void NestedL1Restoration::update_pivots(const Eigen::VectorXd &n, const Eigen::VectorXd &p,
                                        const Eigen::VectorXd &zn, const Eigen::VectorXd &zp,
                                        Eigen::VectorXd &pivots) {
    // (5): pivot = n/z_n + p/z_p (POSITIVE; the seam negates it into (y,y)).
    pivots = n.cwiseQuotient(zn) + p.cwiseQuotient(zp);
}

void NestedL1Restoration::enter_nested(const ProgressMeasures &reference,
                                       const Eigen::Ref<const Eigen::VectorXd> &primals,
                                       const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
                                       const Eigen::Ref<const Eigen::VectorXd> &iq_residuals,
                                       double outer_mu) {
    reference_ = reference;
    x_r_ = primals;

    // (3): D_R^2 = diag(1/max(1,|x_R_i|))^2, cached.
    dr2_.resize(x_r_.size());
    for (Eigen::Index i = 0; i < x_r_.size(); ++i) {
        const double di = 1.0 / std::max(1.0, std::abs(x_r_[i]));
        dr2_[i] = di * di;
    }

    // (4): resto_mu = max(outer_mu, ||h||_inf, ||g+s||_inf).
    double m = outer_mu;
    if (eq_residuals.size() > 0) {
        m = std::max(m, eq_residuals.cwiseAbs().maxCoeff());
    }
    if (iq_residuals.size() > 0) {
        m = std::max(m, iq_residuals.cwiseAbs().maxCoeff());
    }
    resto_mu_ = m;

    // (4): closed-form elastic init per channel at the entry barrier parameter.
    init_channel(eq_residuals, resto_mu_, n_e_, p_e_, z_ne_, z_pe_);
    init_channel(iq_residuals, resto_mu_, n_i_, p_i_, z_ni_, z_pi_);

    // (5): pivots from the fresh elastic state.
    update_pivots(n_e_, p_e_, z_ne_, z_pe_, e_pivots_);
    update_pivots(n_i_, p_i_, z_ni_, z_pi_, i_pivots_);

    // Recovered steps start at zero (no step recovered until the first solve).
    dn_e_ = Eigen::VectorXd::Zero(n_e_.size());
    dp_e_ = Eigen::VectorXd::Zero(p_e_.size());
    dzn_e_ = Eigen::VectorXd::Zero(z_ne_.size());
    dzp_e_ = Eigen::VectorXd::Zero(z_pe_.size());
    dn_i_ = Eigen::VectorXd::Zero(n_i_.size());
    dp_i_ = Eigen::VectorXd::Zero(p_i_.size());
    dzn_i_ = Eigen::VectorXd::Zero(z_ni_.size());
    dzp_i_ = Eigen::VectorXd::Zero(z_pi_.size());

    active_ = true;
    ++entries_;
}

void NestedL1Restoration::nested_complementarity(double &sum, double &min_comp, double &max_comp,
                                                 int &count) const {
    sum = 0.0;
    count = 0;
    min_comp = std::numeric_limits<double>::infinity();
    max_comp = -std::numeric_limits<double>::infinity();
    // Fold one channel's elastic complementarity products (v = n or p, z the
    // paired bound multiplier) into the aggregates. The cwiseProduct reduction
    // expressions evaluate lazily — no heap temporary is materialized.
    auto accumulate = [&](const Eigen::VectorXd &v, const Eigen::VectorXd &z) {
        if (v.size() == 0)
            return;
        const auto prod = v.cwiseProduct(z);
        sum += prod.sum();
        min_comp = std::min(min_comp, prod.minCoeff());
        max_comp = std::max(max_comp, prod.maxCoeff());
        count += static_cast<int>(v.size());
    };
    accumulate(n_e_, z_ne_);
    accumulate(p_e_, z_pe_);
    accumulate(n_i_, z_ni_);
    accumulate(p_i_, z_pi_);
}

void NestedL1Restoration::condensed_residuals(
    double mu, const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
    const Eigen::Ref<const Eigen::VectorXd> &iq_residuals,
    const Eigen::Ref<const Eigen::VectorXd> &eq_lmults,
    const Eigen::Ref<const Eigen::VectorXd> &iq_lmults, Eigen::Ref<Eigen::VectorXd> eq_rtilde_out,
    Eigen::Ref<Eigen::VectorXd> iq_rtilde_out) const {
    const double rho = kRestoPenaltyParameter;
    // (5): r̃ = (c+n−p) + μ/z_n − (n/z_n)(ρ+y) − μ/z_p + (p/z_p)(ρ−y).
    auto condense = [&](const Eigen::Ref<const Eigen::VectorXd> &c, const Eigen::VectorXd &n,
                        const Eigen::VectorXd &p, const Eigen::VectorXd &zn,
                        const Eigen::VectorXd &zp, const Eigen::Ref<const Eigen::VectorXd> &y,
                        Eigen::Ref<Eigen::VectorXd> out) {
        for (Eigen::Index i = 0; i < c.size(); ++i) {
            out[i] = (c[i] + n[i] - p[i]) + mu / zn[i] - (n[i] / zn[i]) * (rho + y[i]) -
                     mu / zp[i] + (p[i] / zp[i]) * (rho - y[i]);
        }
    };
    condense(eq_residuals, n_e_, p_e_, z_ne_, z_pe_, eq_lmults, eq_rtilde_out);
    condense(iq_residuals, n_i_, p_i_, z_ni_, z_pi_, iq_lmults, iq_rtilde_out);
}

double NestedL1Restoration::nested_objective(double mu,
                                             const Eigen::Ref<const Eigen::VectorXd> &primals)
    const {
    // (2)/(3): ρ·Σ(n+p) + (η(μ)/2)‖D_R(x−x_R)‖², η(μ) recomputed from μ.
    const double eta = kRestoProximityWeight * std::sqrt(mu);
    const double slack_sum = n_e_.sum() + p_e_.sum() + n_i_.sum() + p_i_.sum();
    const Eigen::VectorXd dx = primals - x_r_;
    const double prox = 0.5 * eta * dr2_.dot(dx.cwiseProduct(dx));
    return kRestoPenaltyParameter * slack_sum + prox;
}

void NestedL1Restoration::add_nested_gradient(double mu,
                                              const Eigen::Ref<const Eigen::VectorXd> &primals,
                                              Eigen::Ref<Eigen::VectorXd> grad_out) const {
    // (2)/(3): ∇_x = η(μ)·D_R²·(x−x_R); accumulated (+=) into grad_out.
    const double eta = kRestoProximityWeight * std::sqrt(mu);
    grad_out += eta * dr2_.cwiseProduct(primals - x_r_);
}

void NestedL1Restoration::nested_primal_diagonal(double mu,
                                                 Eigen::Ref<Eigen::VectorXd> diag_out) const {
    // (2)/(3): proximal Hessian diagonal on the x rows, η(μ)·D_R² (written).
    const double eta = kRestoProximityWeight * std::sqrt(mu);
    diag_out = eta * dr2_;
}

void NestedL1Restoration::recover_elastic_steps(
    double mu, const Eigen::Ref<const Eigen::VectorXd> &eq_lmults,
    const Eigen::Ref<const Eigen::VectorXd> &iq_lmults,
    const Eigen::Ref<const Eigen::VectorXd> &eq_dy,
    const Eigen::Ref<const Eigen::VectorXd> &iq_dy) {
    const double rho = kRestoPenaltyParameter;
    // (5): Δn = μ/z_n − (n/z_n)(ρ+y) − (n/z_n)Δy;
    //      Δp = μ/z_p − (p/z_p)(ρ−y) + (p/z_p)Δy;
    //      Δz_n = Δy + ρ + y − z_n;  Δz_p = −Δy + ρ − y − z_p.
    auto recover = [&](const Eigen::VectorXd &n, const Eigen::VectorXd &p,
                       const Eigen::VectorXd &zn, const Eigen::VectorXd &zp,
                       const Eigen::Ref<const Eigen::VectorXd> &y,
                       const Eigen::Ref<const Eigen::VectorXd> &dy, Eigen::VectorXd &dn,
                       Eigen::VectorXd &dp, Eigen::VectorXd &dzn, Eigen::VectorXd &dzp) {
        const Eigen::Index m = n.size();
        dn.resize(m);
        dp.resize(m);
        dzn.resize(m);
        dzp.resize(m);
        for (Eigen::Index i = 0; i < m; ++i) {
            dn[i] = mu / zn[i] - (n[i] / zn[i]) * (rho + y[i]) - (n[i] / zn[i]) * dy[i];
            dp[i] = mu / zp[i] - (p[i] / zp[i]) * (rho - y[i]) + (p[i] / zp[i]) * dy[i];
            dzn[i] = dy[i] + rho + y[i] - zn[i];
            dzp[i] = -dy[i] + rho - y[i] - zp[i];
        }
    };
    recover(n_e_, p_e_, z_ne_, z_pe_, eq_lmults, eq_dy, dn_e_, dp_e_, dzn_e_, dzp_e_);
    recover(n_i_, p_i_, z_ni_, z_pi_, iq_lmults, iq_dy, dn_i_, dp_i_, dzn_i_, dzp_i_);
}

void NestedL1Restoration::recenter_elastics(
    double mu, const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
    const Eigen::Ref<const Eigen::VectorXd> &iq_residuals) {
    // (f): second-level closed-form re-solve of the separable elastic subproblem
    // holding x and s fixed. Re-center BOTH channels' pairs at the LIVE μ from the
    // current raw residuals, reusing the entry-init quadratic (init_channel) — the
    // pairs land on the exact n·z=μ pairing the pivots/recovery are built from
    // (disclosure (f): z is re-centered alongside n,p, unlike Ipopt's keep-z).
    init_channel(eq_residuals, mu, n_e_, p_e_, z_ne_, z_pe_);
    init_channel(iq_residuals, mu, n_i_, p_i_, z_ni_, z_pi_);

    // Pivots track the re-centered state; the stale recovered steps are dropped
    // (the next iteration recovers fresh steps from the re-centered pairs).
    update_pivots(n_e_, p_e_, z_ne_, z_pe_, e_pivots_);
    update_pivots(n_i_, p_i_, z_ni_, z_pi_, i_pivots_);
    dn_e_ = Eigen::VectorXd::Zero(n_e_.size());
    dp_e_ = Eigen::VectorXd::Zero(p_e_.size());
    dzn_e_ = Eigen::VectorXd::Zero(z_ne_.size());
    dzp_e_ = Eigen::VectorXd::Zero(z_pe_.size());
    dn_i_ = Eigen::VectorXd::Zero(n_i_.size());
    dp_i_ = Eigen::VectorXd::Zero(p_i_.size());
    dzn_i_ = Eigen::VectorXd::Zero(z_ni_.size());
    dzp_i_ = Eigen::VectorXd::Zero(z_pi_.size());

    ++recenter_calls_;
}

namespace {
// Fraction-to-boundary tau cap over one positive-variable / step pair: the
// largest alpha in (0,1] keeping v + alpha*dv >= (1-tau)*v for every entry.
double l1resto_tau_cap(double alpha, const Eigen::VectorXd &v, const Eigen::VectorXd &dv,
                       double tau) {
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (dv[i] < 0.0) {
            alpha = std::min(alpha, -tau * v[i] / dv[i]);
        }
    }
    return alpha;
}
} // namespace

double NestedL1Restoration::primal_boundary_alpha(double tau) const {
    double alpha = 1.0;
    alpha = l1resto_tau_cap(alpha, n_e_, dn_e_, tau);
    alpha = l1resto_tau_cap(alpha, p_e_, dp_e_, tau);
    alpha = l1resto_tau_cap(alpha, n_i_, dn_i_, tau);
    alpha = l1resto_tau_cap(alpha, p_i_, dp_i_, tau);
    return alpha;
}

double NestedL1Restoration::dual_boundary_alpha(double tau) const {
    double alpha = 1.0;
    alpha = l1resto_tau_cap(alpha, z_ne_, dzn_e_, tau);
    alpha = l1resto_tau_cap(alpha, z_pe_, dzp_e_, tau);
    alpha = l1resto_tau_cap(alpha, z_ni_, dzn_i_, tau);
    alpha = l1resto_tau_cap(alpha, z_pi_, dzp_i_, tau);
    return alpha;
}

void NestedL1Restoration::apply_elastic_step(double alpha_primal, double alpha_dual) {
    n_e_ += alpha_primal * dn_e_;
    p_e_ += alpha_primal * dp_e_;
    z_ne_ += alpha_dual * dzn_e_;
    z_pe_ += alpha_dual * dzp_e_;
    n_i_ += alpha_primal * dn_i_;
    p_i_ += alpha_primal * dp_i_;
    z_ni_ += alpha_dual * dzn_i_;
    z_pi_ += alpha_dual * dzp_i_;
    // Pivots track the moved elastic state.
    update_pivots(n_e_, p_e_, z_ne_, z_pe_, e_pivots_);
    update_pivots(n_i_, p_i_, z_ni_, z_pi_, i_pivots_);
}

double NestedL1Restoration::trial_objective(double mu, double alpha,
                                            const Eigen::Ref<const Eigen::VectorXd> &trial_primals)
    const {
    const double eta = kRestoProximityWeight * std::sqrt(mu);
    // Slacks along the step at fraction alpha: n+αΔn, p+αΔp (both channels).
    const double slack_sum = (n_e_ + alpha * dn_e_).sum() + (p_e_ + alpha * dp_e_).sum() +
                             (n_i_ + alpha * dn_i_).sum() + (p_i_ + alpha * dp_i_).sum();
    const Eigen::VectorXd dx = trial_primals - x_r_;
    const double prox = 0.5 * eta * dr2_.dot(dx.cwiseProduct(dx));
    return kRestoPenaltyParameter * slack_sum + prox;
}

void NestedL1Restoration::trial_residual_shift(double alpha,
                                               Eigen::Ref<Eigen::VectorXd> eq_shift_out,
                                               Eigen::Ref<Eigen::VectorXd> iq_shift_out) const {
    // shift = (n + αΔn) − (p + αΔp), added to the raw constraint residuals.
    eq_shift_out = (n_e_ + alpha * dn_e_) - (p_e_ + alpha * dp_e_);
    iq_shift_out = (n_i_ + alpha * dn_i_) - (p_i_ + alpha * dp_i_);
}

} // namespace tycho::solvers
