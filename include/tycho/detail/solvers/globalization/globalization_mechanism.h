// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this interface is the
// backtracking line search (classic) with recovery-dispatch hooks. A
// trust-region mechanism was investigated as an alternative implementation
// and cut without being built — see
// docs/dev/analysis/2026-07-e2-g7-tr-decision.md for the investigation, the
// decision, and its reversal condition; this interface carries no
// TR-specific accommodation as a result.
//
// This file: the GlobalizationMechanism interface (pure virtual except
// run_acceptance_backtrack()'s throwing default — see below). The
// implementation that ships alongside it (BacktrackingLineSearch, not part
// of this file) is verbatim today's max_primal_dual_step() fraction-to-
// boundary scaling followed by a classic_line_search() dispatch on the
// AcceptanceStrategy it is given — see the riskiest-seam note below.
//
// Ownership rule: a GlobalizationMechanism holds NO solver state. reset() is
// the μ-event/phase-change hook (mirrors AcceptanceStrategy::reset()); the
// shipped implementation's reset() is a no-op, since it carries no state to
// clear.

#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"
// PSIOPT::LineSearchModes (forwarded to AcceptanceStrategy::classic_line_search)
// requires the complete PSIOPT class; see acceptance_strategy.h's include note.
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// GlobalizationMechanism — owns the fraction-to-boundary + backtracking
// sequence for one step proposal, then dispatches acceptance.
// =============================================================================
class GlobalizationMechanism {
  public:
    virtual ~GlobalizationMechanism() = default;

    // Riskiest seam in the whole extraction: today's max_primal_dual_step()
    // SCALES DXSL's primal/slack/multiplier blocks IN PLACE (see
    // BacktrackingLineSearch::max_primal_dual_step, psiopt_globalization.cpp),
    // between the negate (DXSL = -kkt_sol_.solve(RHS)) and the
    // merit backtrack — and the backtrack's trial points (xsl + alpha*dxsl)
    // and the eventual commit (XSL += alpha*DXSL) both operate on that SAME,
    // already-scaled DXSL. This interface therefore fuses "compute the
    // fraction-to-boundary step (alphap, alphad), scale DXSL by it, then run
    // the acceptance-strategy backtrack on the scaled DXSL" into one call —
    // splitting "scale" from "backtrack" across two interface calls would let
    // a future component boundary reorder those multiplications, which
    // breaks the CBWR bit-identical-iteration-count gate. DXSL, XSL are
    // therefore passed by mutable
    // reference and DXSL absolutely IS mutated in place by any real
    // implementation of this method (documented here, not just at the call
    // site, precisely because that in-place mutation is the load-bearing
    // behavior the whole seam depends on).
    //
    // KKTVector is inaccessible outside PSIOPT (see acceptance_strategy.h's
    // note); XSL/DXSL/XSL2/RHS/RHS2 are therefore the same raw
    // Eigen::VectorXd blocks ls_impl/max_primal_dual_step operate on today.
    // bfrac and pd_step_strategy_ (today's max_primal_dual_step inputs) are
    // NOT separate parameters here — they are read from `ctx.settings_`
    // (bound_fraction_ / pd_step_strategy_), since they are persistent
    // Settings, not per-iteration transients.
    //
    // lsmode/obj_scale/mu/prim_obj/barr_obj are forwarded verbatim to
    // `acceptance.classic_line_search(...)` (or an equivalent generic-path
    // call, once future acceptance strategies exist) once the
    // fraction-to-boundary scaling above has been applied. alphap/alphad are
    // out-parameters (today's max_primal_dual_step out-params); the return
    // value is the final backtracked alpha (today's ls_impl return value).
    virtual double compute_step(PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                                 double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                                 Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                 Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                                 AcceptanceStrategy &acceptance, double &alphap, double &alphad,
                                 IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                                 SolverContext &ctx) = 0;

    // Fraction-to-boundary primal/dual step (today's max_primal_dual_step),
    // exposed on the interface so alg_impl's BarrierModes::PROBE predictor
    // block can drive the SAME step-scaling through the mechanism_ base
    // pointer WITHOUT the acceptance backtrack (compute_step's second half).
    // PROBE's mpc_mu() consumes a predictor DXSL that has already been scaled
    // to the boundary (see ClassicAdaptiveGovernor::update_barrier's PROBE
    // case, psiopt_globalization.cpp); that scaling is barrier/IPM logic
    // independent of the globalization strategy, so it belongs on this
    // interface. Reconstructs the KKTVector view over the
    // raw XSL/DXSL blocks internally and MUTATES DXSL in place; bfrac, the
    // problem dims, and pd_step_strategy_ are read from `ctx`. alphap/alphad
    // are out-parameters (today's max_primal_dual_step out-params).
    //
    // NOTE: compute_step already applies this step (guarded by
    // inequal_cons_ > 0) as its first half; this standalone entry point exists
    // for callers that need the scaling without a backtrack. Two live callers
    // beyond compute_step itself: ClassicAdaptiveGovernor::update_barrier's
    // PROBE predictor block (same operands, same position — and the path
    // MonitoredBarrierGovernor's free-mode delegate also reaches through),
    // and SocRecovery::do_correction, which re-scales a corrected direction
    // the same way before re-testing acceptance.
    //
    // Divergence-path note: this call runs inside alg_impl's
    // GoodStep branch; pre-extraction it ran unconditionally. On a non-finite
    // DXSL the old code could record alpha_p_/alpha_d_ values derived from
    // -Inf/NaN entries (e.g. 0.0) on the terminal DIVERGING iterate, where the
    // new code records the 1.0 init values. Print-only exposure on diverging
    // runs; iterates, mu, and iteration counts are unaffected (DXSL is
    // discarded before the state commit on that path).
    virtual void max_primal_dual_step(Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, double bfrac,
                                       double &alphap, double &alphad, const SolverContext &ctx) = 0;

    // Run ONLY the acceptance backtrack on an already-fraction-to-boundary-
    // scaled DXSL, dispatching classic-vs-generic per the strategy — i.e.
    // compute_step's SECOND half (the backtrack) without its first half (the
    // fraction-to-boundary scaling). compute_step calls this after applying the
    // scaling; the recovery links (SOC / extended backtracking) call it to
    // re-drive acceptance on a corrected or further-scaled direction they have
    // already prepared, so a corrected trial is tested against the SAME
    // acceptance criteria the ordinary step was — the classic merit test on the
    // classic path, or the generic AcceptanceStrategy::is_iterate_acceptable
    // surface (filter / funnel / modern merit) on the generic path.
    // This is the seam that lets SOC and extended backtracking compose with
    // every acceptance strategy rather than only classic merit: they must NOT
    // call AcceptanceStrategy::classic_line_search directly (it throws on the
    // generic strategies), and only the mechanism can reach the generic
    // driving path (which owns the trial-point evaluation). `iters` is
    // forwarded for the classic dispatcher's signature; the generic path does
    // not read it.
    //
    // Default body is a T6-style logic error (matching
    // AcceptanceStrategy::classic_line_search): a mechanism that hosts a
    // generic driving path must override it. It is never reached on any
    // configuration that does not enable a recovery link that re-drives
    // acceptance.
    virtual double run_acceptance_backtrack(PSIOPT::LineSearchModes lsmode, double obj_scale,
                                            double mu, double prim_obj, double barr_obj,
                                            Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL,
                                            Eigen::VectorXd &XSL2, Eigen::VectorXd &RHS,
                                            Eigen::VectorXd &RHS2, AcceptanceStrategy &acceptance,
                                            IterateInfo &Citer,
                                            const std::vector<IterateInfo> &iters,
                                            SolverContext &ctx) {
        (void)lsmode;
        (void)obj_scale;
        (void)mu;
        (void)prim_obj;
        (void)barr_obj;
        (void)XSL;
        (void)DXSL;
        (void)XSL2;
        (void)RHS;
        (void)RHS2;
        (void)acceptance;
        (void)Citer;
        (void)iters;
        (void)ctx;
        throw std::logic_error("GlobalizationMechanism::run_acceptance_backtrack is only "
                               "implemented by mechanisms that host an acceptance backtrack "
                               "(BacktrackingLineSearch)");
    }

    // μ-event / phase-change reset hook — see the ownership-rule note above.
    virtual void reset() = 0;
};

} // namespace tycho::solvers
