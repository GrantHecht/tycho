// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
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
// now go through the SolverContext reference ctx_. The four barrier/eval
// helpers (eval_rhs, apply_reset_slacks, barrier_objective, barrier_gradient)
// are verbatim copies of the identically-named PSIOPT methods, reading
// through ctx_ (see merit_acceptance.h's byte-identity design note).
//
// This file also hosts BacktrackingLineSearch (the step-length
// mechanism) — verbatim today's max_step_to_boundary / max_primal_dual_step,
// reading through the SolverContext passed to each call. See
// backtracking_line_search.h's riskiest-seam design note.
//
// This file also hosts ClassicAdaptiveGovernor (the barrier-parameter
// update) — verbatim today's PROBE/LOQO barmode switch + common clamp/objective/
// gradient tail, plus the loqo_mu / mpc_mu oracles and verbatim copies of the
// barrier_* / complementarity helpers it consumes (the complementarity copy is
// TOKEN-IDENTICAL including its ULP-load-bearing .sum() warning). See
// classic_adaptive_governor.h's PROBE-impurity design note.
// =============================================================================

#include "tycho/detail/solvers/globalization/backtracking_line_search.h"
#include "tycho/detail/solvers/globalization/classic_adaptive_governor.h"
#include "tycho/detail/solvers/globalization/merit_acceptance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tycho::solvers {

// ============================================================================
// Generic interface — unused on the classic merit path (see header). T6:
// these throw rather than return a fabricated answer; a future filter/funnel
// strategy gives them real bodies when it actually drives them.
// ============================================================================
bool ClassicMeritAcceptance::is_iterate_acceptable(const ProgressMeasures &current,
                                                   const ProgressMeasures &trial,
                                                   const ProgressMeasures &predicted_reduction,
                                                   double objective_multiplier) {
    (void)current;
    (void)trial;
    (void)predicted_reduction;
    (void)objective_multiplier;
    throw std::logic_error("ClassicMeritAcceptance::is_iterate_acceptable is unused on the classic "
                           "merit path (acceptance is fused inside classic_line_search); it is "
                           "driven only by G2+ filter/funnel strategies");
}

bool ClassicMeritAcceptance::is_infeasibility_sufficiently_reduced(
    const ProgressMeasures &reference, const ProgressMeasures &trial) const {
    (void)reference;
    (void)trial;
    throw std::logic_error(
        "ClassicMeritAcceptance::is_infeasibility_sufficiently_reduced is unused "
        "on the classic merit path; it is driven only by a G5 restoration "
        "strategy");
}

// ============================================================================
// Barrier/eval helpers — VERBATIM copies of the PSIOPT methods, reading
// through ctx_ (nlp_/settings_/dims) instead of PSIOPT members.
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
    for (int i = 0; i < ctx_.slack_vars_; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < ctx_.settings_.neg_slack_reset_) {
            si = ctx_.settings_.neg_slack_reset_;
        }

        if (fxi < 0.0) {
            FXI[i] = 0.0;
            S[i] = std::max(std::abs(fxi), ctx_.settings_.neg_slack_reset_);
        } else {
            FXI[i] += si;
        }
    }
}

double ClassicMeritAcceptance::barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu) const {
    double psi = 0;
    for (int i = 0; i < ctx_.inequal_cons_; i++) {
        psi += -mu * std::log(S[i]);
    }
    return psi;
}

void ClassicMeritAcceptance::barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                                              Eigen::Ref<Eigen::VectorXd> LI, double mu,
                                              Eigen::Ref<Eigen::VectorXd> AGS) const {
    AGS = LI - mu * (S.cwiseInverse());
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
    this->apply_reset_slacks(xsl2.slacks(), rhs2.iq_cons());
    btest = this->barrier_objective(xsl2.slacks(), mu);
}

auto ClassicMeritAcceptance::compute_penalties(KKTVector &xsl, KKTVector &rhs) const
    -> PenaltyTerms {
    return {xsl.lmults().cwiseAbs().dot(rhs.all_cons().cwiseAbs()), rhs.all_cons().squaredNorm(),
            rhs.all_cons().template lpNorm<Eigen::Infinity>()};
}

bool ClassicMeritAcceptance::secondary_accept(double ptest, double prim_obj,
                                              const PenaltyTerms &test,
                                              const PenaltyTerms &init) const {
    return (ptest < prim_obj && test.l2_ < init.l2_) ||
           (ptest < prim_obj && test.linf_ < init.linf_);
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
        this->eval_rhs(obj_scale, xsl2.data(), ptest, rhs2.data(), rhs2.data());
        this->apply_reset_slacks(xsl2.slacks(), rhs2.iq_cons());
        btest = this->barrier_objective(xsl2.slacks(), mu);
        this->barrier_gradient(xsl2.slacks(), xsl2.iq_lmults(), mu, rhs2.dual_grad());
        double LangTest = ptest + btest + xsl2.lmults().dot(rhs2.all_cons());
        if (LangTest < LangInit) {
            citer.ls_iters_ = j;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
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
        eval_trial_point_occ(obj_scale, mu, alpha, xsl, dxsl, xsl2, rhs2, ptest, btest);

        double LangTest = ptest + btest;
        PenaltyTerms test = compute_penalties(xsl, rhs2);
        LangTest += test.l1_ + test.l2_ * sc;

        citer.merit_val_ = LangTest;
        if (LangTest < LangInit || secondary_accept(ptest, prim_obj, test, init)) {
            citer.ls_iters_ = j;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
        }
    }
    return alpha;
}

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
        eval_trial_point_occ(obj_scale, mu, alpha, xsl, dxsl, xsl2, rhs2, ptest, btest);

        double LangTest = ptest + btest;

        // Tolerance-filtered L1 penalty
        double TestL1Pen = 0;
        for (int i = 0; i < ctx_.equal_cons_; i++) {
            double eqerr = std::abs(rhs2.eq_cons()[i]);
            double eqmul = std::abs(xsl.eq_lmults()[i]);
            if (eqerr > ctx_.settings_.econ_tol_ * 10) {
                TestL1Pen += eqerr * eqmul;
            }
        }
        for (int i = 0; i < ctx_.inequal_cons_; i++) {
            double iqerr = std::abs(rhs2.iq_cons()[i]);
            double iqmul = std::abs(xsl.iq_lmults()[i]);
            if (iqerr > ctx_.settings_.icon_tol_ * 10) {
                TestL1Pen += iqerr * iqmul;
            }
        }

        double TestL2Pen = rhs2.all_cons().squaredNorm();
        double TestLinfPenalty = rhs2.all_cons().template lpNorm<Eigen::Infinity>();

        // Zero L2 when within tolerance threshold
        if (TestL2Pen <
            ctx_.settings_.econ_tol_ * ctx_.settings_.econ_tol_ * ctx_.equal_cons_ +
                ctx_.settings_.icon_tol_ * ctx_.settings_.icon_tol_ * ctx_.inequal_cons_) {
            TestL2Pen = 0;
        }

        LangTest += TestL1Pen + TestL2Pen * sc;

        PenaltyTerms test{TestL1Pen, TestL2Pen, TestLinfPenalty};
        citer.merit_val_ = LangTest;
        if (LangTest < LangInit || secondary_accept(ptest, prim_obj, test, init)) {
            citer.ls_iters_ = j;
            break;
        } else {
            citer.ls_iters_ = j + 1;
            alpha = alpha / ctx_.settings_.alpha_red_;
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
        Citer.ls_iters_ = 0;
        return 1.0;
    default:
        throw std::invalid_argument("Unknown LineSearchMode");
    }
}

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
// place — guarded exactly as the original alg_impl main-path call
// (`if (inequal_cons_ > 0)`) — and the acceptance strategy then backtracks a
// scalar alpha on the already-scaled DXSL.
double BacktrackingLineSearch::compute_step(
    PSIOPT::LineSearchModes lsmode, double obj_scale, double mu, double prim_obj, double barr_obj,
    Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
    Eigen::VectorXd &RHS2, AcceptanceStrategy &acceptance, double &alphap, double &alphad,
    IterateInfo &Citer, const std::vector<IterateInfo> &iters, SolverContext &ctx) {
    if (ctx.inequal_cons_ > 0)
        this->max_primal_dual_step(XSL, DXSL, ctx.settings_.bound_fraction_, alphap, alphad, ctx);

    return acceptance.classic_line_search(lsmode, obj_scale, mu, prim_obj, barr_obj, XSL, DXSL,
                                          XSL2, RHS, RHS2, Citer, iters);
}

// ============================================================================
// ClassicAdaptiveGovernor — barrier-parameter update. The
// PROBE/LOQO barmode switch + common clamp/objective/gradient tail and the
// loqo_mu / mpc_mu oracles are moved VERBATIM from src/solvers/psiopt.cpp
// (statement order and operand order preserved exactly — the merge gate is a
// bit-identical CBWR iteration-count comparison). The only edits are context-
// plumbing renames: former PSIOPT member reads (kkt_sol_ -> ctx.kkt_solver_,
// settings_/dims -> ctx.*) and the mechanism_ base pointer -> the mechanism
// reference parameter. The barrier_* / complementarity helpers below are
// verbatim copies of the identically-named PSIOPT methods (the complementarity
// copy is TOKEN-IDENTICAL including its ULP warning). See classic_adaptive_
// governor.h's PROBE-impurity and byte-identity design notes.
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
    double psi = 0;
    for (int i = 0; i < ctx.inequal_cons_; i++) {
        psi += -mu * std::log(S[i]);
    }
    return psi;
}

void ClassicAdaptiveGovernor::barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                                               Eigen::Ref<Eigen::VectorXd> LI, double mu,
                                               Eigen::Ref<Eigen::VectorXd> AGS) const {
    AGS = LI - mu * (S.cwiseInverse());
}

void ClassicAdaptiveGovernor::barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI,
                                               Eigen::Ref<Eigen::VectorXd> AGS) const {
    AGS = LI;
}

double ClassicAdaptiveGovernor::loqo_mu(Eigen::Ref<Eigen::VectorXd> S,
                                        Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                                        double mincomp) const {
    double eta = mincomp / avgcomp;
    double sigmat = .1 * std::pow(0.05 * (1.0 - eta) / eta, 3);
    double sigma = std::min(0.8, std::abs(sigmat));
    return sigma * avgcomp;
}

double ClassicAdaptiveGovernor::mpc_mu(Eigen::Ref<Eigen::VectorXd> S,
                                       Eigen::Ref<Eigen::VectorXd> LI, double avgcomp,
                                       double mincomp, const SolverContext &ctx) const {
    double navgcomp = 0;
    double nmincomp = 0;
    double nmaxcomp = 0;
    this->complementarity(S, LI, navgcomp, nmincomp, nmaxcomp, ctx);
    return std::pow(navgcomp / avgcomp, 3) * avgcomp;
}

// Verbatim today's psiopt.cpp barmode switch (the former `if (inequal_cons_ > 0)`
// body): the guard stays at the alg_impl call site, so update_barrier assumes
// inequal_cons_ > 0. The predictor's alphap/alphad are locals here (discarded —
// see the divergence-path note in the header).
double ClassicAdaptiveGovernor::update_barrier(PSIOPT::BarrierModes barmode, double mu_in,
                                               double avgcomp, double mincomp, Eigen::VectorXd &XSL,
                                               Eigen::VectorXd &RHS, Eigen::VectorXd &DXSL,
                                               Eigen::VectorXd &Temp,
                                               GlobalizationMechanism &mechanism,
                                               SolverContext &ctx, double &barr_obj) {
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
        mu = this->loqo_mu(v_xsl.slacks(), v_xsl.iq_lmults(), avgcomp, mincomp);
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

} // namespace tycho::solvers
