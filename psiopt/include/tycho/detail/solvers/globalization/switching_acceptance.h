// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
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
// fails — the trial is H-TYPE: its sufficient-progress verdict is delegated to
// the subclass (the filter's acceptable-to-current-iterate test / the funnel's
// β·width test) via is_h_type_progress_acceptable().
//
// ACCEPTANCE ORDER (template method; EVERY trial passes through it in order):
//   1. θ_max ceiling (Eq. 21): θ(trial) > θ_max ⇒ reject (cause kCeiling),
//      before any other test.
//   2. STRATEGY MEMBERSHIP, every trial (F-type included): the subclass verdict
//      is_trial_acceptable_to_strategy() — the filter's non-dominance test / the
//      funnel's within-the-width test. Failure ⇒ reject (cause kMembership).
//      Wächter–Biegler step A-5.4, Ipopt's CheckAcceptabilityOfTrialPoint, and
//      Uno's FunnelMethod all gate every trial on membership, not only H-type.
//      Ipopt attributes a rejection to the filter only when its own first test
//      (T1 — the Armijo condition for an f-type trial, the acceptable-to-
//      current-iterate test otherwise) PASSED and the filter test failed. To
//      reproduce that exactly despite checking membership first, a membership
//      rejection here SPECULATIVELY evaluates the type-appropriate T1 (side-
//      effect-free) and hands the result to notify_trial_rejected() as its
//      second argument — see (5).
//   3. Switching condition (Eq. 19) selects F-type vs H-type:
//        • F-TYPE (switching holds): accept iff the Armijo condition on φ holds
//          (Eq. 20); else reject (cause kArmijo).
//        • H-TYPE (switching fails): accept iff is_h_type_progress_acceptable()
//          holds; else reject (cause kHTypeProgress).
//   4. On any accept: register_accepted_step(current, trial, h_type) runs the
//      subclass bookkeeping (a filter augments only on an H-type accept and
//      folds in its per-iteration reset heuristic; a funnel tightens its width
//      only on an H-type accept). h_type is false for an F-type accept.
//   5. On any rejection: notify_trial_rejected(cause, trial_passed_progress_test)
//      lets a stateful subclass (the filter) run per-iteration bookkeeping
//      keyed on the last rejection. trial_passed_progress_test is meaningful
//      only for cause == kMembership (the speculative T1 result from (2));
//      for kArmijo/kHTypeProgress it is trivially false (T1 failed by
//      definition to reach that branch), and for kCeiling it is false and
//      MUST be ignored (Ipopt leaves its attribution flag untouched on a
//      θ_max rejection).
//
// Subclass hooks (this class is never instantiated directly):
//   initialize_bounds(theta_0)    — called once, lazily, the first time
//                                    is_iterate_acceptable() runs after a
//                                    reset() (i.e. at the start of each
//                                    phase); lets the filter clear its list /
//                                    the funnel set its initial width from θ₀.
//   reset_bounds()                — the reset() companion: clears whatever
//                                    initialize_bounds() set up so the next
//                                    is_iterate_acceptable() call re-arms the
//                                    lazy init.
//   is_trial_acceptable_to_strategy(current, trial) — MEMBERSHIP verdict, run
//                                    for every trial (step 2).
//   is_h_type_progress_acceptable(current, trial)   — H-TYPE sufficient-progress
//                                    verdict, run only on the H-type path.
//   register_accepted_step(current, trial, h_type)  — bookkeeping run on any
//                                    accept; h_type distinguishes F- from H-type.
//   notify_trial_rejected(cause, trial_passed_progress_test) — bookkeeping run
//                                    on any rejection (default no-op; the
//                                    filter overrides it). The second
//                                    argument is the speculative T1 result,
//                                    meaningful only for cause == kMembership
//                                    (see (5) above).
//
// Design note — the minimal-step / feasibility-restoration trigger (WB
// Eq. (23), α_min as a fraction of a computed minimal step size, e.g.
// alpha_min_frac = 0.05) is deliberately NOT implemented here: RecoveryChain
// already owns "the line search cannot make further progress" dispatch
// through a different route — FeasibilitySwitchRecovery converts the
// ladder-exhausted kAcceptAsIs any inner chain link produces into
// kSwitchToFeasibility (feasibility_switch_recovery.h), which is exactly the
// entry alg_impl needs to switch into restoration. A second, α_min-based
// trigger inside this class would be redundant with that dispatch, not a
// missing prerequisite — so it stays unimplemented by design, not because
// feasibility restoration is unavailable. Until such a trigger is added (if
// ever), a trial that keeps failing the tests above simply keeps
// backtracking through the generic ladder's existing step-length floor and
// the RecoveryChain dispatch above.
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
// RejectionCause — why the template method rejected a trial, handed to the
// subclass's notify_trial_rejected() hook (see the acceptance order above).
// =============================================================================
enum class RejectionCause {
    kCeiling,      // θ(trial) > θ_max (Eq. 21).
    kMembership,   // failed the strategy membership test (step 2).
    kArmijo,       // F-type: failed the Armijo condition on φ (Eq. 20).
    kHTypeProgress // H-type: failed the sufficient-progress verdict.
};

// =============================================================================
// SwitchingAcceptance — shared filter/funnel skeleton (template method).
// =============================================================================
class SwitchingAcceptance : public AcceptanceStrategy {
  public:
    // The switching-condition template method (see the file-top formulation).
    bool is_iterate_acceptable(const ProgressMeasures &current, const ProgressMeasures &trial,
                               const ProgressMeasures &predicted_reduction,
                               double objective_multiplier, double step_length) override;

    // Restoration-exit test — driven once a feasibility-restoration strategy
    // is active: alg_impl calls it from one of two mode-specific call sites
    // to decide whether a trial has reduced infeasibility enough to leave
    // restoration and resume optimality (the nested l1 phase's
    // κ_resto-ratchet exit and the proximal phase's relative-θ-reduction
    // exit, psiopt.cpp). Base SwitchingAcceptance still throws (T6) rather
    // than fabricate an answer; FilterAcceptance/FunnelAcceptance override it
    // with a real body.
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
    // Membership verdict, run for EVERY trial (step 2 of the acceptance order).
    virtual bool is_trial_acceptable_to_strategy(const ProgressMeasures &current,
                                                 const ProgressMeasures &trial) = 0;
    // H-type sufficient-progress verdict, run only on the H-type path.
    virtual bool is_h_type_progress_acceptable(const ProgressMeasures &current,
                                               const ProgressMeasures &trial) = 0;
    // Bookkeeping on any accept; h_type is false for an F-type accept.
    virtual void register_accepted_step(const ProgressMeasures &current,
                                        const ProgressMeasures &trial, bool h_type) = 0;
    // Bookkeeping on any rejection; default no-op (only the filter overrides
    // it). trial_passed_progress_test is Ipopt's speculative T1 result and is
    // meaningful only when cause == kMembership (see the file-top ordering
    // note (5)); it is a trivially-implied constant for every other cause.
    virtual void notify_trial_rejected(RejectionCause cause, bool trial_passed_progress_test) {
        (void)cause;
        (void)trial_passed_progress_test;
    }

  private:
    // Eq. (19), factored out so the membership-reject branch can evaluate it
    // speculatively (to select which T1 to run) without duplicating the
    // arithmetic the template method also runs at step 3.
    bool compute_switching_holds(const ProgressMeasures &current,
                                 const ProgressMeasures &predicted_reduction,
                                 double step_length) const;
    // Eq. (20), factored out for the same reason (used at step 3 AND,
    // speculatively, in the membership-reject branch's T1 evaluation).
    bool armijo_holds(const ProgressMeasures &current, const ProgressMeasures &trial,
                      const ProgressMeasures &predicted_reduction) const;

    bool bounds_initialized_ = false; // Lazy-init flag; re-armed by reset().
    double theta_min_ = 0.0;
    double theta_max_ = 0.0;
};

} // namespace tycho::solvers
