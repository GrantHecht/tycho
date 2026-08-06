// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: RecoveryChain is the
// ordered dispatch on step rejection: second-order correction (SOC) ->
// extended backtracking -> watchdog revert -> feasibility switch. This
// header ships an empty-chain implementation matching pre-extraction
// give-up behavior on the all-default path.
//
// This file: the RecoveryChain interface itself (pure virtual except
// notify_step_accepted's default no-op body), plus one free function with a
// real body — should_dispatch_recovery() below, the shared gate condition
// alg_impl uses to decide whether to invoke the chain at all. IMPORTANT
// scope note: the inertia/perturbation LADDER itself (factor_impl's Zfac
// cycling + the 8x/(1/3) escalation) is NOT what this interface wraps — it
// stays inside PSIOPT::factor_impl; a future inertia-dispatch stage may fold
// it into this chain. This interface is the POST-REJECTION dispatcher: what
// to do once a trial step has already been rejected by an AcceptanceStrategy.
// NoopRecovery (noop_recovery.h) is the all-default-path implementation: an
// empty chain that always returns kAcceptAsIs, i.e. pre-extraction behavior —
// the capped backtrack's surviving alpha is simply taken forward. SocRecovery
// (soc.h) is the first live implementor: an opt-in (Settings::max_soc_ > 0)
// second-order correction that re-solves on the live factorization and
// returns kRetry with a corrected step. ChainedRecovery, ExtendedBacktrackRecovery,
// and WatchdogRecovery (globalization/watchdog.h) are the second batch of
// live links: ChainedRecovery composes SOC and extended backtracking in
// order (SOC first — see its class doc for why), WatchdogRecovery wraps
// whatever chain is configured as an outer decorator. FeasibilitySwitchRecovery
// (feasibility_switch_recovery.h) is the outermost link whenever
// restoration_mode_ != off: it converts an inner chain's ladder-exhausted
// kAcceptAsIs into kSwitchToFeasibility, so that Action is live, not future.
//
// Ownership rule: the RecoveryChain interface itself defines no persistent
// state — the base contract is stateless. Two concrete links are the
// exception: WatchdogRecovery (WatchdogState plus an XSL snapshot vector)
// and FeasibilitySwitchRecovery (its soft-pre-stage counter) hold real
// per-solve state, each documented at its own declaration, and both cleared
// behind reset(), not as ambient global state. reset() is the μ-event/
// phase-change hook (mirrors the other three interfaces); WatchdogRecovery
// is explicitly reset on μ change (see watchdog.h) in addition to the
// ordinary phase-boundary reset() call.

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
// watchdog.h) and by FeasibilitySwitchRecovery (feasibility_switch_recovery.h)
// into the `resolved_depth` out-parameter of on_step_rejected below;
// individual links (SocRecovery, ExtendedBacktrackRecovery) do not write it
// themselves — only a composing/wrapping link knows which position in the
// dispatch order actually won. PSIOPT::alg_impl() also overwrites it
// directly, outside the chain call, in two feasibility-restoration branches:
// the nested elastic re-centering fallback (try_recenter_elastics, guarded
// on resolved_depth still being the unresolved sentinel) and the
// un-evaluable-fallback entry, the latter so the histogram attributes that
// iteration to restoration rather than to whatever depth the chain itself
// resolved (even a watchdog-resolved one). The soft-feasibility-step
// escalation is not a third site — FeasibilitySwitchRecovery stamps the
// depth itself before returning that action. Backs PSIOPT::SolveResult::
// recovery_depth_histogram_[d] (psiopt.h).
inline constexpr int kRecoveryDepthSoc = 0;
inline constexpr int kRecoveryDepthExtended = 1;
inline constexpr int kRecoveryDepthWatchdog = 2;
inline constexpr int kRecoveryDepthUnresolved = 3; // classic give-up: no link resolved it.
// Feasibility-restoration mode-switch: written by FeasibilitySwitchRecovery
// (globalization/feasibility_switch_recovery.h) when it converts an inner
// kAcceptAsIs into a kSwitchToFeasibility, and by PSIOPT::alg_impl() directly
// (see the note above). Reachable whenever restoration_mode_ != off
// (proximal_switch or l1_nested); PSIOPT::SolveResult::
// recovery_depth_histogram_ (psiopt.h) is sized to 5 to hold this bucket.
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
    //   kSwitchToFeasibility — hand off to a restoration strategy
    //                        (ProximalSwitchRestoration or NestedL1Restoration,
    //                        selected by restoration_mode_).
    //   kSoftFeasibilityStep — take the full fraction-to-boundary step on the
    //                        current search direction as a soft feasibility
    //                        pre-stage (evaluated by alg_impl under a
    //                        primal-dual-error reduction test) before committing
    //                        to a full restoration switch. Produced only by
    //                        FeasibilitySwitchRecovery, and only for a nested
    //                        restoration strategy; see its file docstring.
    //   kGiveUp            — no recovery available. NOTE: today's classic
    //                        behavior is kAcceptAsIs (NoopRecovery) — the
    //                        capped backtrack's surviving alpha is taken;
    //                        there is no give-up branch in the current loop.
    enum class Action { kAcceptAsIs, kRetry, kSwitchToFeasibility, kSoftFeasibilityStep, kGiveUp };

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
    //                                 ChainedRecovery/WatchdogRecovery and
    //                                 FeasibilitySwitchRecovery write it (see
    //                                 the constants above); SocRecovery/
    //                                 ExtendedBacktrackRecovery accept the
    //                                 parameter but leave it alone.
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

// Recovery-dispatch gate. The RecoveryChain hook is driven only when
// citer.accepted_ is false AND the KKT step direction was usable (good_step).
// accepted_ reads false for two reasons: the acceptance strategy actually
// rejected the trial (the line search's out-signal reports not-accepted), or
// psiopt.cpp's alg_impl force-rejects a trial the acceptance strategy DID
// accept when this iteration's factorization exhausted the inertia-correction
// ladder (the `if (kkt_exhausted) Citer.accepted_ = false;` site, run before
// this gate is checked) -- so a genuinely accepted step can still reach the
// hook in that case. The non-finite-direction path (which runs no line
// search) is excluded via good_step regardless. Factored out of alg_impl so
// the gate condition has a single definition, callable in isolation by the
// unit test that guards it.
inline bool should_dispatch_recovery(bool good_step, const IterateInfo &citer) {
    return good_step && !citer.accepted_;
}

} // namespace tycho::solvers
