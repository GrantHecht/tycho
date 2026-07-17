// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction. Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("recovery_chain.h — Ordered dispatch on step rejection: SOC -> extended
// backtracking -> watchdog revert -> feasibility switch. G1: empty chain
// (today's give-up behavior)") and §4 ("G2 — ... Recovery chain"). Ground-
// truth recon: docs/superpowers/plans/2026-07-16-e2-g1-dossier.md §4
// ("Inertia / perturbation ladder") and §8 ("RecoveryChain" verdict.
//
// G1 (this file): pure interface declaration, no implementation. IMPORTANT
// scope note (also in the brief): the inertia/perturbation LADDER itself
// (factor_impl's Zfac cycling + the 8x/(1/3) escalation, dossier §4) is NOT
// what this interface wraps — it stays inside PSIOPT::factor_impl until G6
// (inertia_mode, spec §4). This interface is the POST-REJECTION dispatcher:
// what to do once a trial step has already been rejected by an
// AcceptanceStrategy. G1's implementation (NoopRecovery, noop_recovery.h) is
// an empty chain that always returns kAcceptAsIs, i.e. today's behavior: the
// capped backtrack's surviving alpha is simply taken forward (PSIOPT has no
// post-rejection recovery today; there is no give-up branch at this point).
// G2 fills in the real SOC -> extended-backtrack ->
// watchdog -> feasibility-switch dispatch (spec §4).
//
// Ownership rule: a RecoveryChain holds NO solver state (no persistent
// watchdog counters, etc. until G2 actually needs them, and even then they
// live behind reset(), not as ambient global state). reset() is the
// μ-event/phase-change hook (mirrors the other three interfaces); G2's
// watchdog is explicitly reset on μ change (spec §4, "reset on μ change").

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

    // Action a recovery chain link may take on a rejected step (spec §4, G2):
    //   kAcceptAsIs        — override the rejection, take the step anyway.
    //   kRetry             — try again this iteration (e.g. after a SOC
    //                        correction or an extended-backtrack step).
    //   kSwitchToFeasibility — hand off to a restoration strategy (inert
    //                        until G5's RestorationStrategy exists, spec §4).
    //   kGiveUp            — no recovery available; today's only behavior.
    enum class Action { kAcceptAsIs, kRetry, kSwitchToFeasibility, kGiveUp };

    // Citer is the just-rejected iterate's record (mutable: a real
    // implementation may annotate it, e.g. marking that a SOC correction
    // ran); `iters` is the read-only iteration history a chain link may
    // consult (dossier §4: the Zfac heuristic already reads
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

} // namespace tycho::solvers
