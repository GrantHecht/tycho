// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: RecoveryChain is the
// ordered dispatch on step rejection: second-order correction (SOC) ->
// extended backtracking -> watchdog revert -> feasibility switch. This
// header ships an empty-chain implementation matching today's give-up
// behavior.
//
// This file: pure interface declaration, no implementation. IMPORTANT scope
// note: the inertia/perturbation LADDER itself (factor_impl's Zfac cycling +
// the 8x/(1/3) escalation) is NOT what this interface wraps — it stays
// inside PSIOPT::factor_impl until the proximal-regularization inertia mode
// is implemented. This interface is the POST-REJECTION dispatcher: what to
// do once a trial step has already been rejected by an AcceptanceStrategy.
// The implementation shipped alongside it (NoopRecovery, noop_recovery.h) is
// an empty chain that always returns kAcceptAsIs, i.e. today's behavior: the
// capped backtrack's surviving alpha is simply taken forward (PSIOPT has no
// post-rejection recovery today; there is no give-up branch at this point).
// A future change fills in the real SOC -> extended-backtrack ->
// watchdog -> feasibility-switch dispatch.
//
// Ownership rule: a RecoveryChain holds NO solver state (no persistent
// watchdog counters, etc. until a live recovery dispatcher actually needs
// them, and even then they live behind reset(), not as ambient global
// state). reset() is the μ-event/phase-change hook (mirrors the other three
// interfaces); a future watchdog is explicitly reset on μ change.

#pragma once

#include <vector>

#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"

namespace tycho::solvers {

// =============================================================================
// RecoveryChain — ordered dispatch invoked after an AcceptanceStrategy
// rejects a trial step.
// =============================================================================
class RecoveryChain {
  public:
    virtual ~RecoveryChain() = default;

    // Action a recovery chain link may take on a rejected step:
    //   kAcceptAsIs        — override the rejection, take the step anyway.
    //   kRetry             — try again this iteration (e.g. after a SOC
    //                        correction or an extended-backtrack step).
    //   kSwitchToFeasibility — hand off to a restoration strategy (inert
    //                        until a RestorationStrategy exists).
    //   kGiveUp            — no recovery available. NOTE: today's classic
    //                        behavior is kAcceptAsIs (NoopRecovery) — the
    //                        capped backtrack's surviving alpha is taken;
    //                        there is no give-up branch in the current loop.
    enum class Action { kAcceptAsIs, kRetry, kSwitchToFeasibility, kGiveUp };

    // Citer is the just-rejected iterate's record (mutable: a real
    // implementation may annotate it, e.g. marking that a SOC correction
    // ran); `iters` is the read-only iteration history a chain link may
    // consult (the Zfac heuristic already reads
    // `iters[...].h_facs_`, establishing precedent for recovery-adjacent
    // code needing history access, though that specific heuristic itself
    // stays in factor_impl per the scope note above). SolverContext gives
    // access to settings_ (delta_h_/incr_h_/decr_h_/max_refac_ govern the
    // ladder a future inertia_mode shares state with) and the KKT solver a
    // SOC re-solve would reuse.
    virtual Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                                     SolverContext &ctx) = 0;

    // μ-event / phase-change reset hook — see the ownership-rule note above.
    virtual void reset() = 0;
};

// Recovery-dispatch gate. The RecoveryChain hook is driven only when the trial
// step was actually rejected by the acceptance strategy (the line search's
// out-signal reports not-accepted) AND the KKT step direction was usable
// (good_step). An accepted step — full or backtracked — never reaches the hook,
// and the non-finite-direction path (which runs no line search) is excluded
// too. Factored out of alg_impl so the gate condition has a single definition,
// callable in isolation by the unit test that guards it.
inline bool should_dispatch_recovery(bool good_step, const IterateInfo &citer) {
    return good_step && !citer.accepted_;
}

} // namespace tycho::solvers
