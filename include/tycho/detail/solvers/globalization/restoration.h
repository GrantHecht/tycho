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
// feasibility mode-switch (proximal_restoration.h), a nested l1 proximal
// restoration (l1_restoration.h), and an elastic/penalty relaxation (not yet
// implemented) — and this interface is shaped to host all three. The first two
// are wired: rebuild_globalization_components constructs the strategy selected
// by Settings::restoration_mode_, and RecoveryChain::Action::kSwitchToFeasibility
// (recovery_chain.h) dispatches into the entry orchestration in alg_impl.
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

#include <stdexcept>

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

    // -------------------------------------------------------------------------
    // Nested restoration surface.
    //
    // A second family of restoration strategies solves an l1 elastic
    // reformulation of the feasibility problem (min ρ·Σ(n+p) + proximal) with
    // the elastic slack pairs (n,p) and their bound multipliers condensed out of
    // the KKT system analytically. That machinery has no counterpart in the
    // proximal mode-switch above, so it lives behind is_nested(): the solver seam
    // consults these methods ONLY when is_nested() reports true.
    //
    // The default bodies below therefore throw — reaching one on a strategy that
    // is not nested marks a wiring bug, not a recoverable condition. Concrete
    // proximal-switch strategies inherit these throwing defaults untouched (they
    // are never reached through the is_nested() gate); the nested l1 strategy
    // overrides every one.
    // -------------------------------------------------------------------------

    // Whether this strategy uses the nested elastic-condensation surface below.
    virtual bool is_nested() const { return false; }

    // Enter the nested phase from the given entry point. `reference` is the
    // (θ,f) pair; `primals` is snapshotted as the proximal center x_R;
    // `eq_residuals`/`iq_residuals` are the constraint residual values at entry
    // (equality h(x); inequality g(x)+s). `outer_mu` is the live outer barrier
    // parameter, one input to the entry barrier parameter (see entry_mu()).
    virtual void enter_nested(const ProgressMeasures &reference,
                              const Eigen::Ref<const Eigen::VectorXd> &primals,
                              const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
                              const Eigen::Ref<const Eigen::VectorXd> &iq_residuals,
                              double outer_mu) {
        (void)reference;
        (void)primals;
        (void)eq_residuals;
        (void)iq_residuals;
        (void)outer_mu;
        throw std::logic_error(
            "RestorationStrategy::enter_nested called on a strategy that does not "
            "implement the nested restoration surface");
    }

    // The restoration barrier parameter computed at entry (also the phase's
    // starting barrier parameter).
    virtual double entry_mu() const {
        throw std::logic_error(
            "RestorationStrategy::entry_mu called on a strategy that does not "
            "implement the nested restoration surface");
    }

    // Per-row diagonal pivots landing in the KKT constraint-row slots — POSITIVE
    // vectors (the solver seam negates them into the (y,y) diagonal entries).
    virtual const Eigen::VectorXd &e_pivots() const {
        throw std::logic_error(
            "RestorationStrategy::e_pivots called on a strategy that does not "
            "implement the nested restoration surface");
    }
    virtual const Eigen::VectorXd &i_pivots() const {
        throw std::logic_error(
            "RestorationStrategy::i_pivots called on a strategy that does not "
            "implement the nested restoration surface");
    }

    // Aggregate the elastic bound-variable complementarity products (n·z_n and
    // p·z_p over every equality- and inequality-channel row) into a sum,
    // per-element min, per-element max, and count over ONLY those elastic pairs.
    // All four outputs are (re)initialized by the call: sum and count start at
    // zero, min/max at +/-infinity, so when no elastic rows exist count is zero
    // and min/max stay at their infinite sentinels (the caller must guard on
    // count). These pairs are part of the restoration barrier subproblem exactly
    // as the original slack/multiplier pairs are part of the outer barrier
    // subproblem; the barrier-parameter machinery must see them, or a
    // late-entered phase (whose original complementarity is already at
    // solve-tolerance) drives the barrier parameter to its floor while the
    // elastic pairs are still at restoration scale.
    virtual void nested_complementarity(double &sum, double &min_comp, double &max_comp,
                                        int &count) const {
        (void)sum;
        (void)min_comp;
        (void)max_comp;
        (void)count;
        throw std::logic_error(
            "RestorationStrategy::nested_complementarity called on a strategy that "
            "does not implement the nested restoration surface");
    }

    // Condensed constraint-row right-hand sides (r̃), given the live barrier
    // parameter and the CURRENT residual values and multipliers.
    virtual void condensed_residuals(double mu,
                                     const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
                                     const Eigen::Ref<const Eigen::VectorXd> &iq_residuals,
                                     const Eigen::Ref<const Eigen::VectorXd> &eq_lmults,
                                     const Eigen::Ref<const Eigen::VectorXd> &iq_lmults,
                                     Eigen::Ref<Eigen::VectorXd> eq_rtilde_out,
                                     Eigen::Ref<Eigen::VectorXd> iq_rtilde_out) const {
        (void)mu;
        (void)eq_residuals;
        (void)iq_residuals;
        (void)eq_lmults;
        (void)iq_lmults;
        (void)eq_rtilde_out;
        (void)iq_rtilde_out;
        throw std::logic_error(
            "RestorationStrategy::condensed_residuals called on a strategy that does "
            "not implement the nested restoration surface");
    }

    // The proximal objective/gradient/Hessian-diagonal pieces of the nested
    // reformulation, evaluated with the LIVE barrier parameter (η recomputed
    // from `mu` on every call, unlike the frozen-ζ proximal-switch trio above).
    virtual double nested_objective(double mu,
                                    const Eigen::Ref<const Eigen::VectorXd> &primals) const {
        (void)mu;
        (void)primals;
        throw std::logic_error(
            "RestorationStrategy::nested_objective called on a strategy that does not "
            "implement the nested restoration surface");
    }
    virtual void add_nested_gradient(double mu,
                                     const Eigen::Ref<const Eigen::VectorXd> &primals,
                                     Eigen::Ref<Eigen::VectorXd> grad_out) const {
        (void)mu;
        (void)primals;
        (void)grad_out;
        throw std::logic_error(
            "RestorationStrategy::add_nested_gradient called on a strategy that does "
            "not implement the nested restoration surface");
    }
    virtual void nested_primal_diagonal(double mu, Eigen::Ref<Eigen::VectorXd> diag_out) const {
        (void)mu;
        (void)diag_out;
        throw std::logic_error(
            "RestorationStrategy::nested_primal_diagonal called on a strategy that does "
            "not implement the nested restoration surface");
    }

    // Recover the elastic slack / bound-multiplier steps from the constraint
    // multiplier steps (Δy) produced by the condensed KKT solve.
    virtual void recover_elastic_steps(double mu,
                                       const Eigen::Ref<const Eigen::VectorXd> &eq_lmults,
                                       const Eigen::Ref<const Eigen::VectorXd> &iq_lmults,
                                       const Eigen::Ref<const Eigen::VectorXd> &eq_dy,
                                       const Eigen::Ref<const Eigen::VectorXd> &iq_dy) {
        (void)mu;
        (void)eq_lmults;
        (void)iq_lmults;
        (void)eq_dy;
        (void)iq_dy;
        throw std::logic_error(
            "RestorationStrategy::recover_elastic_steps called on a strategy that does "
            "not implement the nested restoration surface");
    }

    // Second-level elastic re-centering fallback. When the in-phase line search
    // exhausts the recovery ladder, re-solve the separable elastic subproblem in
    // closed form holding x and s FIXED — the same per-row quadratic the entry
    // initializer uses, evaluated at the LIVE barrier parameter `mu` and the
    // CURRENT raw constraint residuals (equality h(x); inequality g(x)+s) — and
    // adopt the re-centered pairs as the live elastic state (n,p from the closed
    // form; z_n=μ/n, z_p=μ/p). See l1_restoration.h's disclosure for why the
    // condensed representation re-centers z alongside n,p rather than keeping the
    // stale multipliers (Ipopt's RestoRestorationPhase leaves z untouched because
    // its z are real variables the next Newton step updates).
    virtual void recenter_elastics(double mu,
                                   const Eigen::Ref<const Eigen::VectorXd> &eq_residuals,
                                   const Eigen::Ref<const Eigen::VectorXd> &iq_residuals) {
        (void)mu;
        (void)eq_residuals;
        (void)iq_residuals;
        throw std::logic_error(
            "RestorationStrategy::recenter_elastics called on a strategy that does "
            "not implement the nested restoration surface");
    }

    // Fraction-to-boundary caps for the recovered elastic steps: primal cap from
    // the slacks (n,p), dual cap from their bound multipliers (z_n,z_p).
    virtual double primal_boundary_alpha(double tau) const {
        (void)tau;
        throw std::logic_error(
            "RestorationStrategy::primal_boundary_alpha called on a strategy that does "
            "not implement the nested restoration surface");
    }
    virtual double dual_boundary_alpha(double tau) const {
        (void)tau;
        throw std::logic_error(
            "RestorationStrategy::dual_boundary_alpha called on a strategy that does "
            "not implement the nested restoration surface");
    }

    // Commit the recovered elastic steps at the accepted step fractions.
    virtual void apply_elastic_step(double alpha_primal, double alpha_dual) {
        (void)alpha_primal;
        (void)alpha_dual;
        throw std::logic_error(
            "RestorationStrategy::apply_elastic_step called on a strategy that does not "
            "implement the nested restoration surface");
    }

    // Trial-path measures at step fraction alpha, for acceptance during the phase.
    virtual double trial_objective(double mu, double alpha,
                                   const Eigen::Ref<const Eigen::VectorXd> &trial_primals) const {
        (void)mu;
        (void)alpha;
        (void)trial_primals;
        throw std::logic_error(
            "RestorationStrategy::trial_objective called on a strategy that does not "
            "implement the nested restoration surface");
    }
    // shift = (n + αΔn) − (p + αΔp), added to the raw constraint residuals.
    virtual void trial_residual_shift(double alpha, Eigen::Ref<Eigen::VectorXd> eq_shift_out,
                                      Eigen::Ref<Eigen::VectorXd> iq_shift_out) const {
        (void)alpha;
        (void)eq_shift_out;
        (void)iq_shift_out;
        throw std::logic_error(
            "RestorationStrategy::trial_residual_shift called on a strategy that does not "
            "implement the nested restoration surface");
    }
};

} // namespace tycho::solvers
