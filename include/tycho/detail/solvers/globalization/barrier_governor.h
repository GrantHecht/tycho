// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction. Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("barrier_governor.h — Free<->monotone state machine. G1 ships a stub that
// always reports 'free' (LOQO/PROBE oracles untouched); G4 makes it real")
// and §4 ("G4 — BarrierGovernor"). Ground-truth recon:
// docs/superpowers/plans/2026-07-16-e2-g1-dossier.md §3 ("Barrier update").
//
// G1 (this file): pure interface declaration, no implementation. The
// eventual G1 implementation (ClassicAdaptiveGovernor, not part of this
// task) is verbatim today's PROBE/LOQO block (psiopt.cpp:1310-1340) and
// always reports in_monotone_mode() == false (the default below). G4 (spec
// §4) is what actually implements the free<->monotone switch this interface
// exists to support.
//
// Ownership rule: a BarrierGovernor holds NO solver state (no persistent mu_
// member, etc.) — mu is always passed in (mu_in) and returned, never cached.
// reset() is the μ-event/phase-change hook (mirrors AcceptanceStrategy /
// GlobalizationMechanism); G4's monotone-mode bookkeeping (the KKT-error
// sufficient-decrease window, spec §4) is exactly the kind of state reset()
// exists to clear.

#pragma once

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/solver_context.h"
// PSIOPT::BarrierModes requires the complete PSIOPT class; see
// acceptance_strategy.h's include note for why this is a plain,
// non-circular include (psiopt.h does not include this directory back).
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// BarrierGovernor — computes the next barrier parameter mu and its
// contribution to the objective/gradient.
// =============================================================================
class BarrierGovernor {
  public:
    virtual ~BarrierGovernor() = default;

    // Mirrors today's psiopt.cpp:1310-1340 block. avgcomp/mincomp are the
    // complementarity measures PSIOPT::complementarity() already computed
    // this iteration (dossier §3: "computed once per iteration ... before
    // factorization" — NOT recomputed here). barmode selects PROBE
    // (Mehrotra predictor-corrector) vs. LOQO exactly as today's switch does.
    //
    // PROBE is NOT a pure function of (mu_in, avgcomp, mincomp): its
    // predictor step needs a fresh KKT solve (DXSL = -kkt_sol_.solve(RHS),
    // using the already-factored system) and reuses the fraction-to-boundary
    // scaling (today calls max_primal_dual_step on the predictor DXSL) to
    // form the predictor point Temp = XSL + DXSL before computing
    // mpc_mu(Temp.slacks(), Temp.iq_lmults(), avgcomp, mincomp) — hence this
    // signature takes the live KKT solver and dims (via SolverContext) plus
    // explicit RHS/DXSL/Temp work vectors rather than being a free function
    // of scalars. XSL is needed for LOQO's mu = loqo_mu(slacks, iq_lmults,
    // avgcomp, mincomp) and for the final barrier_objective(slacks, mu) /
    // barrier_gradient(slacks, iq_lmults, mu, dual_grad) common tail that
    // runs after either branch. KKTVector is inaccessible outside PSIOPT
    // (see acceptance_strategy.h's note), so XSL/RHS/DXSL/Temp are the same
    // raw Eigen::VectorXd blocks the current code operates on via KKTVector
    // views constructed from SolverContext's dims.
    //
    // mu_in is accepted for API symmetry with a future monotone-mode
    // implementation that blends with the previous mu (G4, Fiacco-McCormick
    // rule); G1's free-mode PROBE/LOQO oracles do not read it (they compute
    // an entirely new mu from avgcomp/mincomp, then the common tail clamps
    // it against ctx.settings_.min_mu_/max_mu_) — unused on the classic path,
    // analogous to AcceptanceStrategy's generic-interface stubs.
    //
    // Returns the new (already-clamped) mu; barr_obj is an out-parameter
    // (today's barr_obj local, set by the common tail's barrier_objective()
    // call).
    virtual double update_barrier(PSIOPT::BarrierModes barmode, double mu_in, double avgcomp,
                                   double mincomp, Eigen::VectorXd &XSL, Eigen::VectorXd &RHS,
                                   Eigen::VectorXd &DXSL, Eigen::VectorXd &Temp, SolverContext &ctx,
                                   double &barr_obj) = 0;

    // Free vs. monotone mode query (spec §4, G4). G1 has no monotone mode at
    // all, so every G1 implementation reports "free" unconditionally; G4
    // overrides this once the free<->monotone state machine exists.
    virtual bool in_monotone_mode() const { return false; }

    // μ-event / phase-change reset hook — see the ownership-rule note above.
    virtual void reset() = 0;
};

} // namespace tycho::solvers
