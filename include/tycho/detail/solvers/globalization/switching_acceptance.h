// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// SwitchingAcceptance — the shared Wächter–Biegler switching-condition
// skeleton used by BOTH the filter strategy and the funnel strategy (each
// supplies its own bound-tracking data structure through the subclass hooks
// below). Neither concrete strategy exists yet; this is the TEMPLATE METHOD
// both inherit — nothing constructs SwitchingAcceptance itself.
//
// Driven through the GENERIC AcceptanceStrategy path exactly like
// ModernMeritAcceptance (drives_classic_path() == false): compute_step runs
// the loop, is_iterate_acceptable renders the verdict.
//
// =============================================================================
// FORMULATION — Wächter & Biegler, "On the implementation of an interior-point
// filter line-search algorithm for large-scale nonlinear programming",
// Math. Program. 106(1):25-57 (2006). Constants match the Ipopt reference
// implementation's defaults (IpFilterLSAcceptor.cpp).
// =============================================================================
//
// ProgressMeasures mapping (same convention as modern_merit.h):
//   current.infeasibility / trial.infeasibility = θ at z_k / z_k + α·d
//   φ(pt) = pt.objective + pt.auxiliary                  (barrier objective)
//   predicted_reduction.objective = m_f = −α·∇φ_kᵀd_k     (α-scaled model
//     reduction, positive for descent; the RAW directional derivative
//     −∇φ_kᵀd_k recovers as m_f / step_length)
//   step_length = α_k ∈ (0, 1]
//
// θ_min / θ_max (Eq. (21) and the paragraph preceding Eq. (19)): both are
// fixed once per phase, as a multiple of max(1, θ₀) where θ₀ is the
// infeasibility of the FIRST current iterate the strategy sees after a
// reset() — captured by a lazy per-phase initialization (below), not passed
// in explicitly:
//   θ_min = kThetaMinFact · max(1, θ₀)   (pre-Eq.-19 threshold: only below
//                                          this may the switching condition fire)
//   θ_max = kThetaMaxFact · max(1, θ₀)   (Eq. (21): hard ceiling — any trial
//                                          with θ(trial) > θ_max is rejected
//                                          outright, before either the
//                                          switching test or the H-type
//                                          delegate runs)
//
// Switching condition (Eq. (19), tested only when θ_k ≤ θ_min AND the step is
// a descent direction for φ, i.e. −∇φ_kᵀd_k > 0):
//
//        α_k · (−∇φ_kᵀd_k)^{s_φ} > δ · θ_k^{s_θ}                        (Eq. 19)
//
// which in α-scaled ProgressMeasures form (m_f = α_k·(−∇φ_kᵀd_k), so
// (−∇φ_kᵀd_k) = m_f / α_k) is:
//
//        α_k · (m_f / α_k)^{s_φ} > δ · θ_k^{s_θ}.
//
// When the switching condition holds, the trial is F-TYPE: accepted iff the
// Armijo condition on φ holds (Eq. (20)):
//
//        φ(trial) ≤ φ(current) − η_φ · m_f                              (Eq. 20)
//
// (WB state Eq. (20) as φ(x_k+α_k d_k) ≤ φ_k + η_φ·α_k·∇φ_kᵀd_k; substituting
// ∇φ_kᵀd_k = −m_f/α_k and multiplying through by α_k gives the −η_φ·m_f form
// used here directly against the α-scaled predicted reduction, matching the
// modern_merit.h convention of working in α-scaled ProgressMeasures.)
//
// When the switching condition does NOT hold — either θ_k > θ_min, or the
// step is not a descent direction for φ, or Eq. (19)'s inequality itself
// fails — the trial is H-TYPE: the verdict is delegated to the subclass
// (filter membership test / funnel width test) via the pure virtual
// is_infeasibility_acceptable(); on a true verdict the subclass bookkeeping
// hook register_accepted_h_type() runs before true is returned (a filter
// augments its list there; a funnel shrinks its width there).
//
// Subclass hooks (pure virtual — this class is never instantiated directly):
//   initialize_bounds(theta_0)    — called once, lazily, the first time
//                                    is_iterate_acceptable() runs after a
//                                    reset() (i.e. at the start of each
//                                    phase); lets the filter clear its list /
//                                    the funnel set its initial width from θ₀.
//   reset_bounds()                — the reset() companion: clears whatever
//                                    initialize_bounds() set up so the next
//                                    is_iterate_acceptable() call re-arms the
//                                    lazy init.
//   is_infeasibility_acceptable(current, trial) — the H-TYPE verdict.
//   register_accepted_h_type(current, trial)    — bookkeeping run only when
//                                    an H-TYPE trial is accepted.
//
// Design note — the minimal-step / feasibility-restoration trigger (WB
// Eq. (23), α_min as a fraction of a computed minimal step size, e.g.
// alpha_min_frac = 0.05) is deliberately NOT implemented here: its only
// purpose is to detect "the line search cannot make further progress" and
// enter feasibility restoration, which does not exist yet. Until then, a
// trial that keeps failing the tests above simply keeps backtracking through
// the generic ladder's existing step-length floor and recovery-chain
// accept-as-is behavior. The trigger lands alongside feasibility restoration,
// when implemented.
//
// Ownership: like ModernMeritAcceptance, no SolverContext reference and no
// NLP eval — pure function of its ProgressMeasures arguments plus its own
// θ_min/θ_max state (and whatever state the concrete subclass adds). reset()
// is the μ-event / phase-change hook: it clears the lazy-init flag and defers
// to the subclass's reset_bounds().
//
// Definitions live in src/solvers/psiopt_globalization.cpp.

#pragma once

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/progress_measures.h"

namespace tycho::solvers {

// =============================================================================
// Paper constants (kPascalCase; each cites its source equation).
// =============================================================================

// Eq. (19) switching-condition constants; Ipopt IpFilterLSAcceptor.cpp
// defaults: s_phi_ = 2.3, s_theta_ = 1.1, delta_ = 1.0.
inline constexpr double kSwitchingDelta = 1.0;
inline constexpr double kSwitchingSPhi = 2.3;
inline constexpr double kSwitchingSTheta = 1.1;

// Eq. (20) Armijo-on-φ constant; Ipopt IpFilterLSAcceptor.cpp default:
// eta_phi_ = 1e-8 (same value as modern_merit.h's kWmnoArmijoEta).
inline constexpr double kArmijoEtaPhi = 1.0e-8;

// θ_min / θ_max factors (Eq. (21) and the paragraph preceding Eq. (19));
// Ipopt IpFilterLSAcceptor.cpp defaults: theta_min_fact_ = 1e-4,
// theta_max_fact_ = 1e4.
inline constexpr double kThetaMinFact = 1.0e-4;
inline constexpr double kThetaMaxFact = 1.0e4;

// =============================================================================
// SwitchingAcceptance — shared filter/funnel skeleton (template method).
// =============================================================================
class SwitchingAcceptance : public AcceptanceStrategy {
  public:
    // The switching-condition template method (see the file-top formulation).
    bool is_iterate_acceptable(const ProgressMeasures &current, const ProgressMeasures &trial,
                               const ProgressMeasures &predicted_reduction,
                               double objective_multiplier, double step_length) override;

    // Restoration-exit test — unused until a feasibility-restoration strategy
    // drives it; throws (T6) rather than fabricate an answer. Same posture as
    // ModernMeritAcceptance's.
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &reference,
                                               const ProgressMeasures &trial) const override;

    // μ-event / phase-change hook: re-arm the lazy θ_min/θ_max init and defer
    // to the subclass's own bound-tracking reset.
    void reset() override;

    // Selects the GENERIC driving path in compute_step (see acceptance_strategy.h).
    bool drives_classic_path() const override { return false; }

    // --- Bound accessors (diagnostics + unit tests) ---
    double theta_min() const { return theta_min_; }
    double theta_max() const { return theta_max_; }

  protected:
    // Abstract base: only a concrete filter/funnel subclass may construct.
    SwitchingAcceptance() = default;

    // --- Subclass hooks (see the file-top formulation) ---
    virtual void initialize_bounds(double theta_0) = 0;
    virtual void reset_bounds() = 0;
    virtual bool is_infeasibility_acceptable(const ProgressMeasures &current,
                                             const ProgressMeasures &trial) = 0;
    virtual void register_accepted_h_type(const ProgressMeasures &current,
                                          const ProgressMeasures &trial) = 0;

  private:
    bool bounds_initialized_ = false; // Lazy-init flag; re-armed by reset().
    double theta_min_ = 0.0;
    double theta_max_ = 0.0;
};

} // namespace tycho::solvers
