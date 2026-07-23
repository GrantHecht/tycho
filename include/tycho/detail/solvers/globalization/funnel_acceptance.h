// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FunnelAcceptance — a concrete SwitchingAcceptance strategy that replaces the
// filter's SET of (θ, f) pairs with ONE scalar: an upper bound on the
// constraint violation, the "funnel width" τ, monotonically tightened as
// accepted iterates stay inside the funnel. The base gates EVERY trial (F-type
// included) on funnel membership (θ_trial ≤ τ), so at the strategy level every
// accepted iterate lies inside the funnel; each accepted feasibility-improving
// (H-type) step then tightens τ. (One residual edge — a solver recovery
// fallback accepting a strategy-rejected trial — is disclosed in note (3).)
// This is the funnel counterpart to the filter strategy, sharing the
// Wächter–Biegler switching skeleton in SwitchingAcceptance (θ_min/θ_max
// ceiling, switching condition, F-type Armijo); this class supplies ONLY the
// funnel-specific membership + H-type verdicts and the width bookkeeping through
// the base's subclass hooks.
//
// Opt-in via Settings::acceptance_strategy_ == funnel; the default
// classic_merit path stays bit-identical.
//
// =============================================================================
// FORMULATION — derived from the two primary sources (fetched + read, not from
// memory):
//   [KLV]  Kiessling, Leyffer & Vanaret, "A Unified Funnel Restoration SQP
//          Algorithm", arXiv:2409.09208. Uses h(x) = ‖c(x)‖₁ for the
//          constraint violation and τ for the funnel width.
//   [Uno]  Vanaret's Uno solver, uno/ingredients/globalization_strategies/
//          switching_methods/funnel_methods/{Funnel,FunnelMethod}.{cpp,hpp}.
//          The constants below are Uno's shipped option defaults (cited by
//          option name); each rule cites the Uno function that implements it.
// =============================================================================
//
// ProgressMeasures mapping (same convention as switching_acceptance.h):
//   current.infeasibility / trial.infeasibility = θ = h(x) at z_k / z_k + α·d
//   (objective/auxiliary participate only in the base's F-type Armijo test;
//    the funnel's H-type rules below read infeasibility alone — matching Uno's
//    Funnel::acceptable / Funnel::sufficient_decrease_condition, which take a
//    single trial_infeasibility argument.)
//
// (1) Funnel-width initialization from θ₀ (KLV Eq. (9); Uno
//     FunnelMethod::initialize + FunnelMethodParameters):
//
//        τ = max( τ̄ , κ̄ · θ₀ ),                                       (init)
//
//     with the absolute floor τ̄ = kFunnelInitialUpperBound (Uno option
//     "funnel_ubd" = 1.0) and the multiplier κ̄ = kFunnelInfeasibilityFactor
//     (Uno option "funnel_fact" = 1.5, i.e. κ̄ > 1 per KLV Eq. (9)). θ₀ is the
//     infeasibility of the FIRST current iterate seen after a reset() — the
//     base captures it lazily and hands it to initialize_bounds() here.
//
// (2) Acceptance verdicts, split across the base's two hooks exactly as Uno's
//     FunnelMethod::is_iterate_acceptable nests them:
//
//     (2a) MEMBERSHIP, run for EVERY trial (F-type included) — Uno's outer
//          "if (funnel.acceptable(trial_infeasibility))":
//
//            θ_trial ≤ τ          (within the funnel — KLV Eq. (8), Uno
//                                  Funnel::acceptable).
//
//     (2b) H-TYPE sufficient reduction, run only on the H-type path — Uno's
//          "else if (funnel.sufficient_decrease_condition(trial_infeasibility))":
//
//            θ_trial ≤ β · τ      (KLV Eq. (12), Uno
//                                  sufficient_decrease_condition),
//
//          with the margin β = kFunnelBeta (Uno option "funnel_beta" = 0.9999,
//          β ∈ (0,1) per KLV Eq. (12)). Since β < 1, (2b) implies (2a), so an
//          accepted H-type step is strictly inside the funnel.
//
// (3) Width update on an ACCEPTED H-type step (Uno Funnel::update with the
//     default update_strategy = 1; Uno option "funnel_update_strategy" = 1):
//
//        if θ_trial ≤ θ_current:
//            τ⁺ = max( β·τ , κ·θ_current + (1−κ)·θ_trial )
//        else:
//            τ⁺ = β·τ,
//
//     with the convex-combination coefficient κ = kFunnelKappa (Uno option
//     "funnel_kappa" = 0.5). The convex combination is Uno's
//     convex_combination(current, trial, κ) = κ·current + (1−κ)·trial, floored
//     at β·τ so a single step never over-tightens.
//
//     Because the base gates EVERY accepted iterate on membership (2a), every
//     accepted iterate satisfies θ_current ≤ τ, so the strict-decrease argument
//     holds unconditionally at the strategy level: an accepted H-type step has
//     θ_trial ≤ β·τ < τ, and with θ_current ≤ τ both branches give τ⁺ < τ. The
//     width is therefore monotonically non-increasing across strategy-accepted
//     sequences — an unconditional property of this update rule under the
//     membership gate (see the one residual edge below).
//
//     ONE residual edge, disclosed. When the backtracking ladder exhausts, the
//     solver's recovery fallback (outside this strategy) may accept the last
//     trial regardless of the funnel verdict (accept-as-is). Such an iterate can
//     lie OUTSIDE the funnel (θ_current ≫ τ), and a later accepted H-type step
//     then reads that oversized θ_current in the convex-combination branch,
//     which can EXCEED the old width: e.g. τ = 1.5, θ_current = 100, θ_trial =
//     0.5 → convex = 0.5·100 + 0.5·0.5 = 50.25, β·τ ≈ 1.5, so τ⁺ = 50.25 — a
//     transient RE-WIDENING. This is unreachable THROUGH the strategy itself
//     (membership blocks every strategy-level accept from landing outside the
//     funnel) and disappears once feasibility restoration is implemented. The
//     pinning test drives register_accepted_step() directly with an out-of-funnel
//     current to hold this recovery-fallback edge fixed.
//
// (4) When the width updates: ONLY on an accepted H-type step. The base runs
//     register_accepted_step() on every accept but the funnel updates τ only
//     when its h_type flag is set; an F-type accept (switching + Armijo, but
//     still membership-gated) leaves τ untouched, and a rejected trial performs
//     no update. This matches Uno, where funnel.update() is called from the
//     single H-type acceptance branch of FunnelMethod::is_iterate_acceptable.
//
// (5) Feasibility-restoration hooks (Uno FunnelMethod, cvanaret/Uno 7481abe).
//     The reference solver constructs a SEPARATE globalization-strategy instance
//     for the feasibility phase, so the optimality funnel's width is
//     structurally FROZEN during restoration and both the exit test and the
//     exit re-base operate on that frozen width. Tycho drives one object across
//     both phases, so the entry hook STASHES the optimality width (and re-arms
//     the base's lazy θ₀ init) to reproduce that freeze; the exit hook restores
//     it before re-basing.
//
//     (5a) Restoration-exit test, is_infeasibility_sufficiently_reduced —
//          FunnelMethod::is_infeasibility_sufficiently_reduced:
//
//            funnel.acceptable(θ_trial)  AND  θ_trial ≤ β·θ_ref
//          = θ_trial ≤ τ_frozen         AND  θ_trial ≤ kFunnelBeta·θ_ref,
//
//          reusing the existing sufficient-decrease margin β (Uno reuses the
//          same parameters.beta). θ_ref is the infeasibility at the restoration
//          entry point (reference.infeasibility). While in the feasibility phase
//          the membership half tests against the STASHED (frozen optimality)
//          width, not the live feasibility-phase width — the reference solver's
//          exit test reads the frozen optimality funnel. Outside the phase it
//          reads the live width (well-defined; the seam only tests for exit
//          during the phase).
//
//     (5b) notify_switch_to_feasibility (entry) — STASH the optimality width and
//          the base's switching state (via a reset() that re-arms the lazy θ₀
//          init), set the in-feasibility flag, and reinitialize a fresh
//          feasibility-phase width (lazily re-derived from the feasibility-phase
//          θ₀ on the first is_iterate_acceptable call, exactly as the filter
//          does at entry). The reference solver's FunnelMethod::
//          notify_switch_to_feasibility body IS an empty no-op — but that no-op
//          is correct ONLY because its optimality funnel instance is untouched
//          by the feasibility phase (a separate instance runs there). Tycho's
//          single-instance design would otherwise let feasibility-phase h-type
//          accepts keep tightening the very width the exit test and exit re-base
//          consult, so it must stash to reproduce the reference's structural
//          freeze — the stash IS the single-instance equivalent of that no-op.
//
//     (5c) notify_switch_to_optimality (exit) — Uno Funnel::update_restoration,
//          applied to the RESTORED optimality width:
//
//            τ ← τ_stashed
//            τ⁺ = κ·τ + (1−κ)·θ_exit
//               = convex_combination(τ, θ_exit, κ),   κ = kFunnelKappa,
//
//          re-basing the (restored) optimality width toward the (reduced) exit
//          infeasibility as restoration hands control back to the optimality
//          phase. Restoring directly (NOT via reset()) leaves the base's lazy
//          θ₀ init armed-as-initialized so the next optimality call does not
//          re-derive and wipe the just-restored width. No extra guard beyond
//          Uno's: the +∞-width edge (restoration entered before the first
//          is_iterate_acceptable call ⇒ stashed width +∞) is FP-inert
//          (κ·∞ + finite = ∞ — no NaN, the funnel simply stays wide).
//
// Divergences from the sources, disclosed:
//   • KLV Eq. (13) states the update as τ⁺ = (1−κ)·θ_trial + κ·τ (a convex
//     combination of the OLD WIDTH and the trial). That formula is Uno's
//     update_strategy = 2, NOT its default. This implementation follows Uno's
//     shipped default (strategy 1, above), whose constants are the ones adopted
//     here; both rules share the same width-decrease guarantee.
//   • Uno's optional "acceptable with respect to the current iterate" refinement
//     (options funnel_gamma, funnel_require_acceptance_wrt_current_iterate) is
//     OFF by default (funnel_require_acceptance_wrt_current_iterate = false) and
//     is not implemented here; its constant is omitted to avoid dead state.
//
// Ownership: like the other generic strategies, no SolverContext reference and
// no NLP eval — a pure function of its ProgressMeasures arguments plus the
// scalar width state. reset() (via the base) re-arms the lazy θ₀ init and calls
// reset_bounds(), which restores the width to its uninitialized sentinel.
//
// Definitions live in src/solvers/psiopt_globalization.cpp.

#pragma once

#include <limits>
#include <stdexcept>

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/switching_acceptance.h"

namespace tycho::solvers {

// =============================================================================
// Funnel constants (kPascalCase; each cites its source equation + Uno option).
// =============================================================================

// (init) absolute floor τ̄ (KLV Eq. (9); Uno option "funnel_ubd" = 1.0).
inline constexpr double kFunnelInitialUpperBound = 1.0;
// (init) multiplier κ̄ > 1 on θ₀ (KLV Eq. (9); Uno option "funnel_fact" = 1.5).
inline constexpr double kFunnelInfeasibilityFactor = 1.5;
// Margin β ∈ (0,1) for the sufficient-reduction test and the update floor
// (KLV Eq. (12); Uno option "funnel_beta" = 0.9999).
inline constexpr double kFunnelBeta = 0.9999;
// Convex-combination coefficient κ ∈ (0,1) in the width update
// (KLV Eq. (13) / Uno strategy 1; Uno option "funnel_kappa" = 0.5).
inline constexpr double kFunnelKappa = 0.5;

// =============================================================================
// FunnelAcceptance — scalar-funnel H-type strategy on the switching skeleton.
// =============================================================================
class FunnelAcceptance final : public SwitchingAcceptance {
  public:
    // Current funnel width τ (diagnostics + unit tests). Before the first
    // is_iterate_acceptable() after a reset() this is the uninitialized
    // sentinel (+∞); it is derived from θ₀ on the first call.
    double funnel_width() const { return width_; }

    // Solver-level observability hook (see AcceptanceStrategy::
    // append_diagnostics): reports the current width_ into
    // SolveResult::last_funnel_width_, or the -1.0 sentinel if width_ is
    // uninitialized (the pathological case of a phase that never calls
    // is_iterate_acceptable, e.g. converges at the initial iterate).
    void append_diagnostics(PSIOPT::SolveResult &result) const override;

    // --- Feasibility-restoration hooks (see (5) in the file-top formulation) ---
    // (5a) Restoration-exit test: funnel-membership (against the stashed width
    // while in the feasibility phase) AND θ_trial ≤ β·θ_ref.
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                               const ProgressMeasures &trial) const override;
    // (5b) Entry: stash the optimality width, set the flag, reinitialize fresh.
    void notify_switch_to_feasibility(const ProgressMeasures &current_progress) override;
    // (5c) Exit: restore the stashed width, then re-base τ ← κ·τ + (1−κ)·θ_exit.
    void notify_switch_to_optimality(const ProgressMeasures &current_progress) override;

    // --- Restoration diagnostics (unit tests) ---
    bool in_feasibility_phase() const { return in_feasibility_phase_; }
    double stashed_funnel_width() const { return stashed_width_; }

  protected:
    // --- SwitchingAcceptance hooks (see the file-top formulation) ---
    // (init): τ = max(τ̄, κ̄·θ₀).
    void initialize_bounds(double theta_0) override;
    // Restore the uninitialized sentinel so the next θ₀ re-derives the width.
    // Phase-aware (mirroring the filter): inside the feasibility phase a μ-event
    // clears only the working width, preserving the stashed optimality width and
    // the in-feasibility flag; outside the phase it also drops the stash.
    void reset_bounds() override;
    // (2a) membership: within the funnel (θ_trial ≤ τ) — every trial.
    bool is_trial_acceptable_to_strategy(const ProgressMeasures &current,
                                         const ProgressMeasures &trial) override;
    // (2b) H-type sufficient reduction (θ_trial ≤ β·τ).
    bool is_h_type_progress_acceptable(const ProgressMeasures &current,
                                       const ProgressMeasures &trial) override;
    // (3): shrink the width on an accepted H-type step (update strategy 1); an
    // F-type accept leaves the width untouched.
    void register_accepted_step(const ProgressMeasures &current, const ProgressMeasures &trial,
                                bool h_type) override;

  private:
    // Uno's convex_combination(a, b, coeff) = coeff·a + (1−coeff)·b.
    static double convex_combination(double a, double b, double coefficient) {
        return coefficient * a + (1.0 - coefficient) * b;
    }

    // Scalar funnel width τ; +∞ until the first θ₀ capture (uninitialized
    // sentinel, matching Uno's Funnel::width default).
    double width_ = std::numeric_limits<double>::infinity();

    // --- Feasibility-restoration state (see (5) in the file-top doc) ---
    // The PRESERVED optimality-phase width, stashed at
    // notify_switch_to_feasibility and restored at notify_switch_to_optimality.
    // The exit test's membership half consults it while in the feasibility phase.
    double stashed_width_ = std::numeric_limits<double>::infinity();
    // Set at entry, cleared at exit. Makes reset_bounds() phase-aware: a μ-event
    // reset mid-feasibility-phase must preserve the stash + this flag.
    bool in_feasibility_phase_ = false;
};

} // namespace tycho::solvers
