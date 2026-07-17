// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction. Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("globalization_mechanism.h — Backtracking line search (classic) with
// recovery-dispatch hooks; §5b TR joins in G7"). Ground-truth recon:
// docs/superpowers/plans/2026-07-16-e2-g1-dossier.md §2 ("Fraction-to-
// boundary & step coupling") and §8 ("Riskiest seam" callout.
//
// G1 (this file): pure interface declaration, no implementation. The
// eventual G1 implementation (BacktrackingLineSearch, not part of this task)
// is verbatim today's max_primal_dual_step() fraction-to-boundary scaling
// followed by a classic_line_search() dispatch on the AcceptanceStrategy it
// is given — see the riskiest-seam note below. G7 (spec §4) adds a TR
// (trust-region) mechanism as an alternative implementation of this same
// interface.
//
// Ownership rule: a GlobalizationMechanism holds NO solver state. reset() is
// the μ-event/phase-change hook (mirrors AcceptanceStrategy::reset(); a
// future TR mechanism uses it to reset its radius — spec §4 G7).

#pragma once

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

    // Riskiest seam in the whole G1 extraction (dossier §2/§8): today's
    // max_primal_dual_step() SCALES DXSL's primal/slack/multiplier blocks IN
    // PLACE (psiopt.cpp:864-870), between the negate (DXSL = -kkt_sol_.solve(
    // RHS)) and the merit backtrack — and the backtrack's trial points
    // (xsl + alpha*dxsl) and the eventual commit (XSL += alpha*DXSL) both
    // operate on that SAME, already-scaled DXSL. This interface therefore
    // fuses "compute the fraction-to-boundary step (alphap, alphad), scale
    // DXSL by it, then run the acceptance-strategy backtrack on the scaled
    // DXSL" into one call — splitting "scale" from "backtrack" across two
    // interface calls would let a future component boundary reorder those
    // multiplications, which breaks the CBWR bit-identical-iteration-count
    // gate (spec §3, "G1 gate"). DXSL, XSL are therefore passed by mutable
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
    // call, once G2+ strategies exist) once the fraction-to-boundary scaling
    // above has been applied. alphap/alphad are out-parameters (today's
    // max_primal_dual_step out-params, dossier §2); the return value is the
    // final backtracked alpha (today's ls_impl return value).
    virtual double compute_step(PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                                 double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                                 Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                 Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2,
                                 AcceptanceStrategy &acceptance, double &alphap, double &alphad,
                                 IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                                 SolverContext &ctx) = 0;

    // Fraction-to-boundary primal/dual step (today's max_primal_dual_step,
    // dossier §2), exposed on the interface so alg_impl's BarrierModes::PROBE
    // predictor block can drive the SAME step-scaling through the mechanism_
    // base pointer WITHOUT the acceptance backtrack (compute_step's second
    // half). PROBE's mpc_mu() consumes a predictor DXSL that has already been
    // scaled to the boundary (dossier §3, psiopt.cpp:1323-1324); that scaling
    // is barrier/IPM logic independent of the globalization strategy, so it
    // belongs on this interface (a future TR mechanism, spec §4 G7, must
    // likewise provide it for PROBE). Reconstructs the KKTVector view over the
    // raw XSL/DXSL blocks internally and MUTATES DXSL in place; bfrac, the
    // problem dims, and pd_step_strategy_ are read from `ctx`. alphap/alphad
    // are out-parameters (today's max_primal_dual_step out-params).
    //
    // NOTE: compute_step already applies this step (guarded by
    // inequal_cons_ > 0) as its first half; this standalone entry point exists
    // ONLY for the PROBE predictor, which needs the scaling without a backtrack.
    virtual void max_primal_dual_step(Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, double bfrac,
                                       double &alphap, double &alphad, const SolverContext &ctx) = 0;

    // μ-event / phase-change reset hook — see the ownership-rule note above.
    virtual void reset() = 0;
};

} // namespace tycho::solvers
