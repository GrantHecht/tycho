// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this header declares
// RestorationStrategy, the interface a feasibility-restoration mode-switch
// implements. Today's PSIOPT has no restoration machinery at all
// (settings_.max_feas_rest_ is a Settings field that no algorithm code reads
// yet); this interface therefore has no "today's behavior" section to
// preserve, unlike the other component headers in this directory.
//
// This file: interface declaration only, finalized against the first
// concrete implementation, ProximalSwitchRestoration (proximal_restoration.h).
// The feasibility-restoration plan ships as a trio of strategies — a proximal
// feasibility mode-switch (this one), a nested l1 proximal restoration, and
// an elastic/penalty relaxation — and this interface is shaped to host all
// three, though only the proximal switch exists so far. NO solver wiring
// exists yet: nothing constructs a RestorationStrategy on any solve path, and
// RecoveryChain::Action::kSwitchToFeasibility (recovery_chain.h) still has
// nowhere to dispatch to. Do not infer wiring behavior from this header alone.
//
// Ownership rule: like the other globalization interfaces, a
// RestorationStrategy caches NO live solver state beyond what defines its own
// mode — for the proximal switch that is exactly the primal snapshot / frozen
// proximal coefficient / per-coordinate scaling captured at entry (see
// proximal_restoration.h). Everything else it needs is either an explicit
// per-call parameter or reached through a SolverContext reference passed to
// the call — never stored across calls beyond that entry snapshot.

#pragma once

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
// PSIOPT::SolveResult requires the complete PSIOPT class; see
// acceptance_strategy.h's include note for why this is a plain, non-circular
// include (psiopt.h does not include this directory back).
#include "tycho/detail/solvers/psiopt.h"

namespace tycho::solvers {

// =============================================================================
// RestorationStrategy — feasibility-restoration mode-switch interface.
// =============================================================================
class RestorationStrategy {
  public:
    virtual ~RestorationStrategy() = default;

    // Enter restoration mode. `reference` is the (θ,f) pair of the point
    // restoration was entered from — the eventual exit test (AcceptanceStrategy::
    // is_infeasibility_sufficiently_reduced, acceptance_strategy.h) compares a
    // later trial against it via reference() below. `primals` is snapshotted
    // (copied, not referenced) as the restoration mode's defining center point;
    // `mu` is the barrier parameter live at the moment of entry, used to derive
    // whatever mode-specific state is frozen at switch time (for
    // ProximalSwitchRestoration: the proximal coefficient ζ, set ONCE here and
    // never re-derived from a later, live μ).
    virtual void enter_restoration(const ProgressMeasures &reference,
                                    const Eigen::Ref<const Eigen::VectorXd> &primals,
                                    double mu) = 0;

    // Leave restoration mode. No solver-side multiplier reset lives here (that
    // is future solver-wiring work, out of scope for this component) — this
    // call's own contract is limited to deactivating the mode.
    virtual void exit_restoration() = 0;

    // Whether restoration mode is currently active.
    virtual bool is_active() const = 0;

    // Full clear: deactivates, drops the entry snapshot, and zeroes the
    // per-phase counters (entries / iterations-in-mode below). μ-event /
    // phase-change reset hook, matching the other four globalization
    // interfaces.
    virtual void reset() = 0;

    // --- Evaluation surface the solver seam consumes while active ---

    // The proximal term's contribution to the objective at `primals`.
    virtual double proximal_objective(const Eigen::Ref<const Eigen::VectorXd> &primals) const = 0;

    // Accumulates the proximal term's gradient contribution into `grad_out`
    // (added, not overwritten — the caller's existing objective gradient is
    // already in `grad_out`).
    virtual void add_proximal_gradient(const Eigen::Ref<const Eigen::VectorXd> &primals,
                                        Eigen::Ref<Eigen::VectorXd> grad_out) const = 0;

    // The proximal term's (diagonal) contribution to the primal Hessian block.
    virtual const Eigen::VectorXd &proximal_diagonal() const = 0;

    // Entry-permission test: may the solver enter restoration right now, given
    // the current constraint violation? False refuses entry — either because
    // the point is already near-feasible (a real restoration episode is not
    // warranted) or because this phase's restoration budget
    // (ctx.settings_.max_feas_rest_) is exhausted. See
    // proximal_restoration.h's kNearFeasibleGuardFactor citation for the exact
    // guard rule.
    virtual bool entry_permitted(double constraint_violation, const SolverContext &ctx) const = 0;

    // The (θ,f) pair restoration was entered from — see `reference` above.
    virtual const ProgressMeasures &reference() const = 0;

    // Increments the per-phase iterations-in-mode counter. Called by the
    // (not-yet-wired) solver seam once per iteration while restoration is
    // active; unused until that wiring lands.
    virtual void note_iteration() = 0;

    // Solver-level observability hook: writes this strategy's diagnostic
    // state into `result`. Mirrors AcceptanceStrategy::append_diagnostics() /
    // BarrierGovernor::append_diagnostics() — same write-only contract, same
    // last-phase-wins semantics once a multi-phase caller collects it. The
    // default body is a no-op: with no RestorationStrategy constructed at all
    // (today's only solve path), the corresponding SolveResult fields stay at
    // their -1 sentinel (see PSIOPT::SolveResult::last_feas_rest_entries_ /
    // last_feas_rest_iters_ in psiopt.h).
    virtual void append_diagnostics(PSIOPT::SolveResult &result) const { (void)result; }
};

} // namespace tycho::solvers
