// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: BarrierGovernor is the
// free<->monotone barrier-update state machine. This header ships an
// interface plus a stub implementation that always reports "free" (the
// LOQO/PROBE oracles are untouched); a future free<->monotone barrier
// governor implements the real state machine.
//
// This file: pure interface declaration, no implementation. The
// implementation that ships alongside it (ClassicAdaptiveGovernor, not part
// of this file) is verbatim today's PROBE/LOQO block (psiopt.cpp:1310-1340)
// and always reports in_monotone_mode() == false (the default below). A
// future free<->monotone barrier governor is what actually implements the
// free<->monotone switch this interface exists to support.
//
// Ownership rule: a BarrierGovernor holds NO solver state (no persistent mu_
// member, etc.) — mu is always passed in (mu_in) and returned, never cached.
// reset() is the μ-event/phase-change hook (mirrors AcceptanceStrategy /
// GlobalizationMechanism); a future free<->monotone barrier governor's
// monotone-mode bookkeeping (the KKT-error sufficient-decrease window) is
// exactly the kind of state reset() exists to clear.

#pragma once

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"
// PSIOPT::BarrierModes requires the complete PSIOPT class; see
// acceptance_strategy.h's include note for why this is a plain,
// non-circular include (psiopt.h does not include this directory back).
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// Forward declaration only: update_barrier takes a GlobalizationMechanism by
// reference so PROBE's predictor can drive the SAME fraction-to-boundary
// step-scaling (GlobalizationMechanism::max_primal_dual_step) that alg_impl's
// main-path step uses — this is the "second caller of that entry point"
// globalization_mechanism.h anticipates. A reference parameter needs only the
// incomplete type here; the concrete BacktrackingLineSearch is visible at the
// psiopt_globalization.cpp definition site.
class GlobalizationMechanism;

// =============================================================================
// BarrierGovernor — computes the next barrier parameter mu and its
// contribution to the objective/gradient.
// =============================================================================
class BarrierGovernor {
  public:
    virtual ~BarrierGovernor() = default;

    // Mirrors today's psiopt.cpp:1310-1340 block. avgcomp/mincomp are the
    // complementarity measures PSIOPT::complementarity() already computed
    // this iteration (computed once per iteration, before factorization —
    // NOT recomputed here). barmode selects PROBE (Mehrotra
    // predictor-corrector) vs. LOQO exactly as today's switch does.
    //
    // PROBE is NOT a pure function of (mu_in, avgcomp, mincomp): its
    // predictor step needs a fresh KKT solve (DXSL = -kkt_sol_.solve(RHS),
    // using the already-factored system) and reuses the fraction-to-boundary
    // scaling (calls mechanism.max_primal_dual_step on the predictor DXSL) to
    // form the predictor point Temp = XSL + DXSL before computing
    // mpc_mu(Temp.slacks(), Temp.iq_lmults(), avgcomp, mincomp) — hence this
    // signature takes the live KKT solver and dims (via SolverContext), the
    // GlobalizationMechanism (to reuse its step-scaling), plus explicit
    // RHS/DXSL/Temp work vectors rather than being a free function
    // of scalars. The predictor's alphap/alphad out-values are NOT surfaced
    // here — they are computed into locals and discarded, because on the
    // classic (GoodStep) path the main-path step overwrites them before they
    // are recorded (see the divergence-path note in the classic implementation
    // header). XSL is needed for LOQO's mu = loqo_mu(slacks, iq_lmults,
    // avgcomp, mincomp) and for the final barrier_objective(slacks, mu) /
    // barrier_gradient(slacks, iq_lmults, mu, dual_grad) common tail that
    // runs after either branch. KKTVector is inaccessible outside PSIOPT
    // (see acceptance_strategy.h's note), so XSL/RHS/DXSL/Temp are the same
    // raw Eigen::VectorXd blocks the current code operates on via KKTVector
    // views constructed from SolverContext's dims.
    //
    // mu_in is accepted for API symmetry with a future monotone-mode
    // implementation that blends with the previous mu (Fiacco-McCormick
    // rule); the free-mode PROBE/LOQO oracles implemented today do not read
    // it (they compute an entirely new mu from avgcomp/mincomp, then the
    // common tail clamps it against ctx.settings_.min_mu_/max_mu_) — unused
    // on the classic path, analogous to AcceptanceStrategy's generic-interface
    // stubs.
    //
    // Returns the new (already-clamped) mu; barr_obj is an out-parameter
    // (today's barr_obj local, set by the common tail's barrier_objective()
    // call).
    //
    // `current` is the in-progress iteration's IterateInfo whose residual
    // fields (kkt_inf_/econ_inf_/icon_inf_/barr_inf_) were filled by this
    // iteration's convergence check — it is NOT yet in the solver's iteration
    // history at this point in the loop (it is re-appended after the line
    // search), which is exactly why it is passed explicitly. A monitored
    // free<->monotone governor reads these residuals to decide the
    // free<->monotone switch; the classic free-mode oracles ignore it
    // entirely.
    //
    // `mu_event` is an out-signal (the caller passes it initialized to false):
    // an implementation sets it true when its monotone mode begins a new barrier
    // subproblem with a fresh barrier parameter, which is the acceptance
    // strategy's per-barrier-subproblem reset trigger (the caller clears the
    // acceptance filter/funnel before the iteration's line search runs). The
    // classic free-mode oracles never set it, so on the default path the
    // caller's reset branch is dead and the solve stays bit-identical.
    virtual double update_barrier(PSIOPT::BarrierModes barmode, double mu_in, double avgcomp,
                                  double mincomp, Eigen::VectorXd &XSL, Eigen::VectorXd &RHS,
                                  Eigen::VectorXd &DXSL, Eigen::VectorXd &Temp,
                                  GlobalizationMechanism &mechanism, SolverContext &ctx,
                                  double &barr_obj, const IterateInfo &current,
                                  bool &mu_event) = 0;

    // Free vs. monotone mode query. No implementation shipped today has a
    // monotone mode, so every implementation reports "free" unconditionally;
    // a future free<->monotone barrier governor overrides this once the
    // free<->monotone state machine exists.
    virtual bool in_monotone_mode() const { return false; }

    // μ-event / phase-change reset hook — see the ownership-rule note above.
    virtual void reset() = 0;
};

} // namespace tycho::solvers
