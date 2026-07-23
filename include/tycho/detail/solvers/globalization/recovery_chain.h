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
// post-rejection recovery by default; there is no give-up branch at this
// point). SocRecovery (soc.h) is the first live implementor: an opt-in
// (Settings::max_soc_ > 0) second-order correction that re-solves on the live
// factorization and returns kRetry with a corrected step. ChainedRecovery,
// ExtendedBacktrackRecovery, and WatchdogRecovery (globalization/watchdog.h)
// are the second batch of live links: ChainedRecovery composes SOC and
// extended backtracking in order (SOC first — see its class doc for why),
// WatchdogRecovery wraps whatever chain is configured as an outer decorator.
// The feasibility switch remains a future link (kSwitchToFeasibility below is
// still unproduced by any link).
//
// Ownership rule: a RecoveryChain holds NO solver state (no persistent
// watchdog counters, etc. until a live recovery dispatcher actually needs
// them, and even then they live behind reset(), not as ambient global
// state). reset() is the μ-event/phase-change hook (mirrors the other three
// interfaces); WatchdogRecovery is explicitly reset on μ change (see
// watchdog.h) in addition to the ordinary phase-boundary reset() call.

#pragma once

#include <vector>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"
// PSIOPT::LineSearchModes (forwarded to the acceptance re-test during a
// second-order correction) requires the complete PSIOPT class; pulled in
// transitively via the two globalization headers above, which already include
// psiopt.h. See solver_context.h's one-directional include-discipline note.

namespace tycho::solvers {

// Recovery-dispatch depth: which link (if any) actually resolved a given
// rejection. Written by ChainedRecovery/WatchdogRecovery (globalization/
// watchdog.h) into the `resolved_depth` out-parameter of on_step_rejected
// below; individual links (SocRecovery, ExtendedBacktrackRecovery) do not
// write it themselves — only the composing wrapper knows which position in
// the dispatch order actually won. Backs PSIOPT::SolveResult::
// recovery_depth_histogram_[d] (psiopt.h).
inline constexpr int kRecoveryDepthSoc = 0;
inline constexpr int kRecoveryDepthExtended = 1;
inline constexpr int kRecoveryDepthWatchdog = 2;
inline constexpr int kRecoveryDepthUnresolved = 3; // classic give-up: no link resolved it.
// Feasibility-restoration mode-switch: written by FeasibilitySwitchRecovery
// (globalization/feasibility_switch_recovery.h) when it converts an inner
// kAcceptAsIs into a kSwitchToFeasibility. Only reachable when restoration_mode_
// == proximal_switch; PSIOPT::SolveResult::recovery_depth_histogram_ (psiopt.h)
// is sized to 5 to hold this bucket.
inline constexpr int kRecoveryDepthRestoration = 4;

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

    // Citer is the just-rejected iterate's record (mutable: an implementation
    // may annotate it, e.g. stamping accepted_ when a correction is taken, and
    // reads its trigger signals first_rejection_iter_/theta_at_first_rejection_);
    // `iters` is the read-only iteration history a chain link may consult (the
    // Zfac heuristic already reads `iters[...].h_facs_`, establishing precedent
    // for recovery-adjacent code needing history access, though that specific
    // heuristic itself stays in factor_impl per the scope note above).
    // SolverContext gives access to settings_ (max_soc_ governs the SOC cap),
    // nlp_ (constraint evaluation at a trial point), and the still-LIVE KKT
    // factorization the second-order correction re-solves against (no refactor).
    //
    // The remaining parameters are the live per-iteration working set (the same
    // objects compute_step just operated on, threaded verbatim so a recovery
    // link can build and re-test a corrected step). The interface places NO
    // no-aliasing precondition on the five VectorXd& parameters: the production
    // caller passes distinct buffers, but test doubles legitimately bind one
    // shared buffer to several slots, so implementations must order their
    // writes to be aliasing-robust (see WatchdogRecovery's revert):
    //   acceptance/mechanism        — re-run the full acceptance backtrack, and
    //                                 the fraction-to-boundary scaling, on a
    //                                 corrected direction.
    //   lsmode/obj_scale/mu/prim_obj/barr_obj — the merit transients forwarded
    //                                 to that re-test.
    //   XSL/DXSL/XSL2/RHS/RHS2       — the KKT-layout state and scratch blocks.
    //                                 A link that returns kRetry must leave the
    //                                 accepted corrected step in DXSL (and its
    //                                 length in `alpha`) so alg_impl's
    //                                 `XSL += alpha*DXSL` commit applies it; on
    //                                 any other Action DXSL/alpha are left as
    //                                 compute_step produced them.
    //   alpha/alphap/alphad         — the accepted step length and the
    //                                 fraction-to-boundary primal/dual lengths.
    //   soc_steps                   — accumulator a link increments once per
    //                                 correction back-substitution (diagnostic).
    //   resolved_depth               — out-parameter, caller-seeded to
    //                                 kRecoveryDepthUnresolved before the call.
    //                                 Only ChainedRecovery/WatchdogRecovery
    //                                 write it (see the constants above);
    //                                 SocRecovery/ExtendedBacktrackRecovery
    //                                 accept the parameter but leave it alone.
    //   watchdog_activations         — accumulator WatchdogRecovery increments
    //                                 once per arm event (diagnostic; every
    //                                 other link ignores it).
    virtual Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                                     SolverContext &ctx, AcceptanceStrategy &acceptance,
                                     GlobalizationMechanism &mechanism,
                                     PSIOPT::LineSearchModes lsmode, double obj_scale, double mu,
                                     double prim_obj, double barr_obj, Eigen::VectorXd &XSL,
                                     Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                                     Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2, double &alpha,
                                     double &alphap, double &alphad, int &soc_steps,
                                     int &resolved_depth, int &watchdog_activations) = 0;

    // Called once per genuinely ACCEPTED iteration -- i.e. the rejection hook
    // above was skipped this iteration because should_dispatch_recovery was
    // false (GoodStep && Citer.accepted_; see alg_impl's call site in
    // psiopt.cpp, in the branch mirroring should_dispatch_recovery's gate).
    // Implementations MAY use this to reset counters tied to real progress
    // (e.g. WatchdogRecovery's consecutive-shortened-iteration count -- see
    // watchdog.h); an implementation must NOT touch solver state (XSL/DXSL/
    // etc -- none of which are even passed here) or mutate Citer/iters/ctx.
    // Default: empty body -- NoopRecovery and every other currently-stateless
    // link inherit it unchanged (behavior-neutral).
    virtual void notify_step_accepted() {}

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
