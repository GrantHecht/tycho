// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// ModernMeritAcceptance — the first consumer of the GENERIC AcceptanceStrategy
// surface (is_iterate_acceptable / reset), a SECOND acceptance strategy that
// lives ALONGSIDE ClassicMeritAcceptance (it does NOT edit or replace it).
// Opt-in via Settings::acceptance_strategy_ == merit; the default classic_merit
// path stays bit-identical.
//
// Driven by the EXISTING BacktrackingLineSearch mechanism: when the selected
// AcceptanceStrategy reports drives_classic_path() == false, compute_step runs
// the GENERIC loop (trial-point eval -> build ProgressMeasures ->
// is_iterate_acceptable -> alpha /= alpha_red_ on reject) instead of the fused
// classic_line_search. Loop-in-mechanism, judgment-in-strategy.
//
// =============================================================================
// FORMULATION — derived from the two papers (fetched + read, not from memory).
// =============================================================================
//
// Both rules use the standard interior-point merit
//
//        φ_π(z) = ϕ_μ(z) + π·‖c(z)‖,                                   (merit)
//
// where ϕ_μ(z) = f(x) − μ·Σ_i log s_i is the barrier objective and ‖c(z)‖ is the
// constraint-violation norm.  This is WMNO Eq. (3.1) verbatim (with penalty ν)
// and Curtis–Nocedal Eq. (2.1) verbatim (with penalty π, SQP objective f in
// place of ϕ_μ — we adapt "f" to the barrier objective ϕ_μ for the IPM).
//
// ProgressMeasures mapping (documented so the (θ,f,aux) triple is unambiguous):
//   current.infeasibility  = θ_c = ‖c(z_k)‖₁         (the ‖c‖ in the merit)
//   current.objective      = f(x_k)  (σ-scaled)      (the f in ϕ_μ)
//   current.auxiliary      = −μ·Σ log s   at z_k     (the barrier term of ϕ_μ)
//   trial.*                = the same three at z_k + α·d
//   ⇒ ϕ_μ(pt) = pt.objective + pt.auxiliary ;  φ_π(pt) = ϕ_μ(pt) + π·pt.infeasibility
//   predicted_reduction.objective     = m_f = −α·(∇ϕ_μ(z_k)ᵀd)   (≥0 for descent)
//   predicted_reduction.infeasibility = m_θ = α·θ_c              (linearized-c model)
//   predicted_reduction.auxiliary     = 0 (unused)
//   objective_multiplier              = σ (obj_scale); objective/aux arrive
//                                       pre-scaled so the merit uses them as-is
//                                       (parity with the classic path).
// The α factor cancels in every penalty-threshold ratio below, so the penalty
// update is steplength-independent (it reflects the full computed step), exactly
// as both papers intend (penalty fixed once per iteration, then held during the
// backtrack).
//
// Predicted reduction of the merit for penalty π (WMNO Eq. (3.2) with σ=0,
// α-scaled; the SQP curvature term ½dᵀWd is dropped — WMNO explicitly sanction
// the σ=0 model as "given by (3.5) but with σ always equal to 0", and the
// Lagrangian Hessian W is not available in the line search):
//
//        pred_π = m_f + π·m_θ = predicted_reduction.objective
//                             + π·predicted_reduction.infeasibility.
//
// Actual reduction:  ared_π = φ_π(current) − φ_π(trial).
// Acceptance (WMNO ratio test §3.1 / Armijo Eq. (3.9); Flex Armijo Eq. (3.12)):
//
//        ared_π ≥ η·pred_π.                                          (accept-π)
//
// (WMNO's Armijo φ_ν(z+αd) ≤ φ_ν(z)+ηα·Dφ_ν and this ared≥η·pred form are
// algebraically identical once Dφ_ν = ∇ϕ_μᵀd − ν‖c‖ (Eq. 3.7) and pred are
// α-scaled — see the report's derivation trail.)
//
// Penalty threshold χ / ν_TRIAL (WMNO Eq. (3.5) with σ=0; Flex Eq. (3.8) ω=0):
//
//        τ = (∇ϕ_μᵀd) / ((1−ρ)·‖c‖)
//          = (−predicted_reduction.objective)
//            / ((1−ρ)·predicted_reduction.infeasibility).          (threshold)
//
//   • WMNO penalty ν update (Eq. (3.6)): ν⁺ = ν if ν ≥ τ, else τ + 1.
//   • Flexible π_u update (Eq. (3.9)):  π_u⁺ = π_u if π_u ≥ τ, else τ + ε_u.
//   • Flexible ACCEPTANCE: accept iff (accept-π) holds for π = π_l OR π = π_u
//     (Curtis–Nocedal's own practical reduction: "satisfied for π∈[π_l,π_u] iff
//     satisfied for either π=π_l or π=π_u").
//   • Flexible π_l update (Eq. (3.10)) applied when a step is accepted:
//       if (accept-π) held for π = π_l:  π_l⁺ = π_l                 (regions III/IV)
//       else (accepted only at π_u, region II):
//         π_l⁺ = min{ π_u , π_l + max{ 0.1·(r − π_l), ε_l } }, with
//         r = ν(step) = (ϕ_μ(trial) − ϕ_μ(current)) / (θ_c − θ_t)     (Eq. 3.11)
//         computed only when θ_c − θ_t > 0 (region II ⇒ infeasibility fell).
//
// Feasible-current special case (WMNO "when c(z)=0, set ν⁺=ν"; Flex "‖c‖=0 ⇒
// π_u⁺=π_u"): when m_θ == 0 the threshold τ is undefined and no penalty update
// is performed; the acceptance test still evaluates (accept-π) with pred_π=m_f.
//
// Paper locations (verbatim, for the review tier):
//   WMNO  — Waltz, Morales, Nocedal & Orban, "An interior algorithm for
//           nonlinear optimization that combines line search and trust region
//           steps", Math. Program. 107:391-408 (2006).  §3.1 "Merit Function":
//           merit Eq. (3.1); pred model Eqs. (3.2)-(3.3); condition
//           pred ≥ ρν‖c‖ Eq. (3.4); ν_TRIAL Eq. (3.5); update Eq. (3.6);
//           directional derivative Eqs. (3.7)-(3.8); Armijo Eq. (3.9); ρ=0.1,
//           η=1e-8 (§ Algorithm 2.1 constants).
//   Flex  — Curtis & Nocedal, "Flexible penalty functions for nonlinear
//           constrained optimization", IMA J. Numer. Anal. 28(4):749-769 (2008).
//           merit + interval Eq. (2.1); model/χ Eqs. (3.5)-(3.8); π_u update
//           Eq. (3.9); π_l update Eq. (3.10); ν(step) Eq. (3.11); Armijo over
//           the interval Eq. (3.12); directional derivative Eq. (3.13); πᵐ
//           Eq. (3.14); endpoint-equivalence remark after Algorithm 3.1.
//
// Ownership: unlike ClassicMeritAcceptance this component DOES carry per-solve
// state — the penalty parameter(s).  reset() (the μ-event / phase-change hook)
// restores them to their initial values.  is_iterate_acceptable() is otherwise
// a pure function of its ProgressMeasures arguments and the penalty state (no
// SolverContext, no eval): the mechanism owns trial-point evaluation, so this
// strategy is unit-testable in isolation.
//
// =============================================================================
// FEASIBILITY-RESTORATION STATE ISOLATION — reproducing the reference
// solver's two-instance structure inside one object.
// =============================================================================
// The reference (Uno) constructs a SEPARATE globalization-strategy instance for
// the feasibility phase, so every piece of optimality-phase merit state — the
// smallest-known-infeasibility tracker, the WMNO penalty ν, and the flexible
// interval (π_l, π_u) — is structurally FROZEN while restoration runs, and the
// restoration-exit test (is_infeasibility_sufficiently_reduced) reduces against
// that frozen tracker. Tycho drives one object across both phases, so it must
// stash the optimality-phase state at restoration entry and restore it at exit
// to get the same behavior:
//
//   • notify_switch_to_feasibility (entry): stash ALL persistent state (the
//     tracker + ν + π_l + π_u), set the in-feasibility flag, and reinitialize
//     the working state to its fresh-construction values so the feasibility
//     phase runs its own penalties/tracker without contaminating the frozen
//     optimality-phase copy.
//   • is_infeasibility_sufficiently_reduced: while in the feasibility phase it
//     reduces the trial against the STASHED (frozen) tracker, never the live
//     working one — the live one is updated by the tested point's own θ on
//     every feasibility-phase accept, which would make θ ≤ ratio·live
//     unsatisfiable for θ > 0. Outside the phase it reads the live tracker
//     (well-defined; the exit test is only ever called during the phase).
//   • notify_switch_to_optimality (exit): restore the stash and clear the flag.
//   • reset(): phase-aware. During the phase a μ-event clears only the working
//     state, preserving the stash the exit test consults; outside the phase it
//     is the full clear and additionally drops the stash defensively.
//
// Retained edge (the reference's own behavior): if restoration is entered
// before any optimality-phase accept, the stashed tracker is still +∞, so the
// exit test's ratio·(+∞) = +∞ passes at the first check. This matches the
// reference and is kept.
//
// Definitions live in src/solvers/psiopt_globalization.cpp.

#pragma once

#include <limits>
#include <stdexcept>

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/psiopt_fwd.h"

namespace tycho::solvers {

// =============================================================================
// Paper constants (kPascalCase; each cites its source equation/section).
// =============================================================================

// WMNO Eq. (3.4)/(3.5): pred(d_z) ≥ ρν‖c‖, ρ ∈ (0,1); "In our tests we use the
// value ρ = 0.1." (Math. Program. 107, §3.1.)
inline constexpr double kWmnoRho = 0.1;
// WMNO Eq. (3.6): ν⁺ = τ + 1 when ν < τ (the additive "+1" bump).
inline constexpr double kWmnoPenaltyBump = 1.0;
// WMNO Armijo Eq. (3.9) / Algorithm 2.1 constants: "we choose η = 1e-8".
inline constexpr double kWmnoArmijoEta = 1.0e-8;
// Initial penalty ν₀ > 0 (WMNO require ν > 0; a neutral start that grows
// monotonically via Eq. (3.6)).  Cleared to this by reset().
inline constexpr double kWmnoInitPenalty = 1.0;

// Flexible χ threshold Eq. (3.8): denominator (1−σ)‖c‖, σ ∈ (0,1). Same role as
// WMNO's ρ; we use the matching value.  (IMA JNA 28(4).)
inline constexpr double kFlexSigma = 0.1;
// Flexible Armijo Eq. (3.12): 0 < η < 1 (matched to WMNO's).
inline constexpr double kFlexArmijoEta = 1.0e-8;
// Flexible π_u update Eq. (3.9): π_u⁺ = χ + ε, "where ε > 0 is a small constant".
inline constexpr double kFlexEpsPiU = 1.0e-8;
// Flexible π_l update Eq. (3.10): the additive floor ε_l, "some small constant".
inline constexpr double kFlexEpsPiL = 1.0e-8;
// Flexible π_l update Eq. (3.10): the damping factor 0.1 ("so that the value for
// π_l will increase only gradually").
inline constexpr double kFlexPiLDamping = 0.1;
// Flexible initial interval (Algorithm 3.1: "0 < π₀ˡ ⩽ π₋₁ᵘ", π_l small,
// "initialize π_u to a large value").  Cleared to these by reset().
inline constexpr double kFlexInitPiL = 1.0e-8;
inline constexpr double kFlexInitPiU = 1.0e8;

// Restoration-exit sufficient-infeasibility-decrease ratio: the trial's
// infeasibility must fall to this fraction of the smallest infeasibility seen so
// far to leave feasibility restoration. Uno option
// "sufficient_infeasibility_decrease_ratio", shipped default 0.9
// (cvanaret/Uno 7481abe, DefaultOptions.cpp; consumed verbatim in
// MeritFunction::is_infeasibility_sufficiently_reduced).
inline constexpr double kSufficientInfeasibilityDecreaseRatio = 0.9;

// =============================================================================
// ModernMeritAcceptance — modernized merit family (WMNO / flexible), driven
// through the generic AcceptanceStrategy path.
// =============================================================================
class ModernMeritAcceptance : public AcceptanceStrategy {
  public:
    explicit ModernMeritAcceptance(MeritPenaltyRules rule) : rule_(rule) { reset(); }

    // The real generic acceptance test (see the file-top formulation). Pure in
    // its ProgressMeasures arguments plus the penalty state; may mutate the
    // penalty state (WMNO ν / flexible π_l, π_u) per the paper update rules.
    // step_length is accepted and ignored: the penalty math above is
    // step-length-independent (the alpha factor cancels in every threshold
    // and Armijo ratio — see the file-top formulation's cancellation note).
    bool is_iterate_acceptable(const ProgressMeasures &current, const ProgressMeasures &trial,
                               const ProgressMeasures &predicted_reduction,
                               double objective_multiplier, double step_length) override;

    // Restoration-exit test — Uno MeritFunction::is_infeasibility_sufficiently_
    // reduced: θ_trial ≤ kSufficientInfeasibilityDecreaseRatio ·
    // smallest_known_infeasibility. The `reference` argument is accepted and
    // ignored — Uno's signature takes the reference progress but its body reads
    // only the trial and the internal smallest-known tracker (the tracker, not
    // the entry point, is the reference this rule reduces against). While in the
    // feasibility phase the tracker read is the STASHED (frozen optimality-phase)
    // one — see the state-isolation section above; outside the phase it is the
    // live tracker. See the definition in psiopt_globalization.cpp.
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                               const ProgressMeasures &trial) const override;

    // μ-event / phase-change hook: clear the WORKING penalty state AND the
    // working smallest-known-infeasibility tracker to their fresh-construction
    // values. Unlike the classic strategy this is NOT a no-op — the penalty
    // parameter and the tracker are per-solve state. Phase-aware (see the
    // state-isolation section above): during the feasibility phase a μ-event
    // clears only the working state, preserving the stashed optimality-phase
    // state the exit test consults; outside the phase it is the full clear and
    // additionally drops the stash defensively. The default (restoration-off)
    // path never enters the phase, so reset() is the full clear there.
    void reset() override;

    // Selects the GENERIC driving path in compute_step (see acceptance_strategy.h).
    bool drives_classic_path() const override { return false; }

    // --- Restoration mode-switch hooks (see the state-isolation section) ---
    // Entry: stash the optimality-phase state, set the flag, reinitialize fresh.
    void notify_switch_to_feasibility(const ProgressMeasures &current_progress) override;
    // Exit: restore the stashed optimality-phase state and clear the flag.
    void notify_switch_to_optimality(const ProgressMeasures &current_progress) override;

    // --- Penalty-state accessors (diagnostics + unit tests) ---
    double wmno_penalty() const { return nu_; }
    double flex_pi_l() const { return pi_l_; }
    double flex_pi_u() const { return pi_u_; }
    double smallest_known_infeasibility() const { return smallest_known_infeasibility_; }

    // --- Restoration-state accessors (diagnostics + unit tests) ---
    bool in_feasibility_phase() const { return in_feasibility_phase_; }
    double stashed_wmno_penalty() const { return stashed_nu_; }
    double stashed_flex_pi_l() const { return stashed_pi_l_; }
    double stashed_flex_pi_u() const { return stashed_pi_u_; }
    double stashed_smallest_known_infeasibility() const {
        return stashed_smallest_known_infeasibility_;
    }

  private:
    // Shared merit primitives (pure arithmetic on ProgressMeasures).
    static double merit(const ProgressMeasures &pt, double penalty) {
        return pt.objective + pt.auxiliary + penalty * pt.infeasibility;
    }
    // (accept-π): ared_π ≥ η·pred_π  (see file-top). pred_π = m_f + π·m_θ.
    static bool armijo(const ProgressMeasures &current, const ProgressMeasures &trial,
                       const ProgressMeasures &pred, double penalty, double eta) {
        const double ared = merit(current, penalty) - merit(trial, penalty);
        const double pred_pi = pred.objective + penalty * pred.infeasibility;
        return ared >= eta * pred_pi;
    }

    bool accept_wmno(const ProgressMeasures &current, const ProgressMeasures &trial,
                     const ProgressMeasures &pred);
    bool accept_flexible(const ProgressMeasures &current, const ProgressMeasures &trial,
                         const ProgressMeasures &pred);

    MeritPenaltyRules rule_;

    // Per-solve penalty state (reset() clears to the kInit* constants).
    double nu_ = kWmnoInitPenalty;    // WMNO single penalty ν.
    double pi_l_ = kFlexInitPiL;      // flexible lower penalty π_l.
    double pi_u_ = kFlexInitPiU;      // flexible upper penalty π_u.

    // Restoration-exit tracker (Uno MeritFunction::smallest_known_infeasibility).
    // +∞-initialized; updated by std::min() ONLY in the accept branch of
    // is_iterate_acceptable; cleared back to +∞ by reset(). FP-inert on the
    // default path: it writes a member on every accept but is never read unless
    // is_infeasibility_sufficiently_reduced (the restoration-exit test) runs.
    double smallest_known_infeasibility_ = std::numeric_limits<double>::infinity();

    // --- Feasibility-restoration state isolation (see the state-isolation
    // section in the file-top doc) ---
    // The PRESERVED optimality-phase persistent state, stashed at
    // notify_switch_to_feasibility and restored at notify_switch_to_optimality.
    // The exit test reduces against stashed_smallest_known_infeasibility_ while
    // in the feasibility phase.
    double stashed_nu_ = kWmnoInitPenalty;
    double stashed_pi_l_ = kFlexInitPiL;
    double stashed_pi_u_ = kFlexInitPiU;
    double stashed_smallest_known_infeasibility_ = std::numeric_limits<double>::infinity();
    // Set at entry, cleared at exit. Makes reset() phase-aware: a μ-event reset
    // mid-feasibility-phase must preserve the stash + this flag.
    bool in_feasibility_phase_ = false;
};

} // namespace tycho::solvers
