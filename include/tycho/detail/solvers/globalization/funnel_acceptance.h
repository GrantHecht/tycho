// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FunnelAcceptance — a concrete SwitchingAcceptance strategy that replaces the
// filter's SET of (θ, f) pairs with ONE scalar: a monotonically non-increasing
// upper bound on the constraint violation, the "funnel width" τ. A trial's
// infeasibility must stay under (a fixed margin below) τ; each accepted
// feasibility-improving step tightens τ. This is the funnel counterpart to the
// filter strategy, sharing the Wächter–Biegler switching skeleton in
// SwitchingAcceptance (θ_min/θ_max ceiling, switching condition, F-type Armijo);
// this class supplies ONLY the funnel-specific H-type verdict and the width
// bookkeeping through the base's subclass hooks.
//
// Nothing constructs FunnelAcceptance in production yet — it is opt-in and not
// wired into Settings; these definitions exist so the strategy is testable in
// isolation through ProgressMeasures + its own width state.
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
// (2) H-type acceptance verdict (Uno FunnelMethod::is_iterate_acceptable, the
//     Funnel::acceptable ∧ Funnel::sufficient_decrease_condition branch;
//     KLV Eqs. (8) and (12)):
//
//        θ_trial ≤ τ          (within the funnel — KLV Eq. (8), Uno acceptable)
//      AND
//        θ_trial ≤ β · τ      (sufficient reduction — KLV Eq. (12), Uno
//                              sufficient_decrease_condition),
//
//     with the margin β = kFunnelBeta (Uno option "funnel_beta" = 0.9999,
//     β ∈ (0,1) per KLV Eq. (12)). Since β < 1 the second inequality implies
//     the first; both are stated to mirror the sources exactly and to make the
//     "within the funnel" gate explicit.
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
//     at β·τ so a single step never over-tightens. Because an accepted H-type
//     step guarantees θ_trial ≤ β·τ < τ and the current iterate satisfies
//     θ_current ≤ τ, both branches give τ⁺ < τ: the width is strictly
//     decreasing, hence monotonically non-increasing across a run.
//
// (4) When the width updates: ONLY on an accepted H-type step. The base runs
//     register_accepted_h_type() exactly there; an F-type accept (switching +
//     Armijo, handled entirely by the base) leaves τ untouched, and a rejected
//     H-type trial performs no update. This matches Uno, where funnel.update()
//     is called from the single H-type acceptance branch of
//     FunnelMethod::is_iterate_acceptable.
//
// Divergences from the sources, disclosed:
//   • KLV Eq. (13) states the update as τ⁺ = (1−κ)·θ_trial + κ·τ (a convex
//     combination of the OLD WIDTH and the trial). That formula is Uno's
//     update_strategy = 2, NOT its default. This implementation follows Uno's
//     shipped default (strategy 1, above), whose constants are the ones adopted
//     here; both rules are monotone non-increasing.
//   • Uno gates BOTH step types on funnel.acceptable (the whole
//     is_iterate_acceptable body sits inside "if (funnel.acceptable(...))").
//     In this shared skeleton the base consults the subclass only on the H-type
//     path, so an F-type step is accepted on switching + Armijo alone, without
//     a funnel-width check — the same structure the Wächter–Biegler filter
//     companion uses. The H-type verdict above therefore carries the full
//     within-the-funnel gate. As a result, an f-type step is bounded only by
//     theta_max, so the violation may transiently exceed the funnel width on
//     f-type accepts; the strict funnel invariant (every accepted iterate inside
//     the funnel) is enforced on h-type steps only.
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

  protected:
    // --- SwitchingAcceptance hooks (see the file-top formulation) ---
    // (init): τ = max(τ̄, κ̄·θ₀).
    void initialize_bounds(double theta_0) override;
    // Restore the uninitialized sentinel so the next θ₀ re-derives the width.
    void reset_bounds() override;
    // (2): H-type verdict — within the funnel AND sufficient reduction.
    bool is_infeasibility_acceptable(const ProgressMeasures &current,
                                     const ProgressMeasures &trial) override;
    // (3): shrink the width on an accepted H-type step (update strategy 1).
    void register_accepted_h_type(const ProgressMeasures &current,
                                  const ProgressMeasures &trial) override;

  private:
    // Uno's convex_combination(a, b, coeff) = coeff·a + (1−coeff)·b.
    static double convex_combination(double a, double b, double coefficient) {
        return coefficient * a + (1.0 - coefficient) * b;
    }

    // Scalar funnel width τ; +∞ until the first θ₀ capture (uninitialized
    // sentinel, matching Uno's Funnel::width default).
    double width_ = std::numeric_limits<double>::infinity();
};

} // namespace tycho::solvers
