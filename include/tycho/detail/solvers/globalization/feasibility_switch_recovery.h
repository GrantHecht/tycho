// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FeasibilitySwitchRecovery — the outermost RecoveryChain link that converts a
// ladder-exhausted step rejection into a feasibility-restoration mode-switch.
// It is built (by rebuild_globalization_components) whenever a restoration
// mode is enabled (restoration_mode_ != off — both the proximal mode-switch
// and the nested l1 restoration), wrapping whatever inner chain the other
// settings produced (NoopRecovery, ChainedRecovery, or a WatchdogRecovery-
// decorated chain). For a nested strategy it additionally runs the soft
// pre-stage documented below before escalating to the full switch.
//
// Behavior: delegate the whole rejection to the inner chain first. If the inner
// chain RESOLVES the rejection (returns kRetry / kSwitchToFeasibility / kGiveUp,
// or a kAcceptAsIs whose resolved_depth was already stamped by a link that
// took the relaxed step on purpose — e.g. the watchdog's trial-acceptance
// path) the outcome passes through untouched — the restoration switch is
// strictly a last resort, tried only when nothing else salvaged the step. The
// LOAD-BEARING discriminator is therefore the resolved_depth out-parameter,
// not the Action alone: kAcceptAsIs is overloaded between "ladder exhausted,
// nothing resolved it" (resolved_depth still the caller-seeded
// kRecoveryDepthUnresolved) and "a link resolved this on purpose but its
// resolution also happens to be kAcceptAsIs" (resolved_depth already set).
// FeasibilitySwitchRecovery intercepts ONLY the unresolved case and, if a
// real restoration episode is warranted — the current point is not already
// near-feasible AND the per-phase entry budget is not exhausted
// (RestorationStrategy::entry_permitted) AND restoration is not already active —
// returns kSwitchToFeasibility so alg_impl enters restoration mode. Otherwise it
// returns the inner kAcceptAsIs unchanged, and the step is taken as before.
//
// This link performs NO mutation: it does not enter restoration, snapshot
// primals, or touch the working set. alg_impl's kSwitchToFeasibility case is the
// single place the actual mode entry happens (enter_restoration + the
// acceptance switch notification). The constraint violation entry_permitted
// consults is the L1 norm of the current KKT constraint block (RHS.all_cons),
// the same measure alg_impl builds the restoration reference from.
//
// Soft feasibility pre-stage (nested restoration only). Before committing to a
// full restoration switch, a NESTED restoration strategy first gets a soft
// pre-stage: instead of returning kSwitchToFeasibility at the exhaustion point,
// this link returns kSoftFeasibilityStep, which alg_impl handles by taking the
// full fraction-to-boundary step on the current search direction and testing it
// under a primal-dual-error reduction rule. A successive-iteration counter lives
// here; after more than kMaxSoftRestoIters soft iterations in a row it escalates
// by returning the real kSwitchToFeasibility. The counter resets whenever a
// regular step is accepted (notify_step_accepted, the pre-stage exiting because
// the ordinary optimality-phase acceptance test recovered) or at any mode-switch
// reset(). Adapted from Ipopt's soft restoration phase (coin-or/Ipopt 72a29c9,
// src/Algorithm/IpBacktrackingLineSearch.cpp: TrySoftRestoStep + the
// in_soft_resto_phase_ continuation). Placement differs from Ipopt with a
// consequence: Ipopt tries soft restoration the moment its backtracking line
// search fails; this solver has a second-order-correction / watchdog recovery
// ladder Ipopt lacks and tries the soft pre-stage only once that ladder is
// exhausted — so soft steps are attempted strictly later than in Ipopt, the
// conservative composition with the extra machinery. The proximal-switch
// restoration mode (is_nested() false) has NO pre-stage: it switches directly,
// byte-for-byte as before.
//
// Acceptance-strategy notification timing. The single acceptance-strategy
// feasibility notification (notify_switch_to_feasibility, which for the filter/
// funnel/modern-merit strategies stashes optimality-phase state and augments the
// filter, and which for the modern-merit strategy throws on a repeated entry)
// stays at the full restoration entry in alg_impl — it is issued exactly once
// per restoration episode, and never during the soft pre-stage. The soft steps
// are ordinary optimality-phase steps: keeping the acceptance strategy in the
// optimality phase across the pre-stage is what lets the pre-stage exit be the
// outer loop's ordinary optimality-phase acceptance recovering on its own (the
// counter-resetting notify_step_accepted path), and guarantees the augmentation
// is not issued twice.
//
// DISCLOSED CONSEQUENCE of this timing (deviation from Ipopt, which calls
// PrepareRestoPhaseStart -- a filter-only augmentation -- at soft-stage START,
// IpBacktrackingLineSearch.cpp): during the pre-stage the trigger point's
// (theta, phi) pair has NOT yet been added to the filter, so soft steps are
// tested against the un-augmented optimality-phase acceptance state; under
// Ipopt they must clear the just-augmented filter. A soft step this solver
// accepts could therefore be one Ipopt would have rejected (and vice versa
// never -- augmentation only shrinks the acceptable region). The pre-stage
// cannot loop on this: the successive counter is cleared only by a genuine
// optimality acceptance, so at most kMaxSoftRestoIters such steps occur
// before escalation, and escalation performs the augmentation.
//
// Ownership: holds only the inner chain (per RecoveryChain's ownership rule)
// plus the soft-pre-stage counter (this link's own recovery state, cleared by
// reset()). reset()/notify_step_accepted() thread straight through to the inner
// chain in addition to clearing the counter. Definitions live in
// src/solvers/psiopt_globalization.cpp.

#pragma once

#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/recovery_chain.h"

// Test fixture (declared for the friend grant below).
class NestedLifecycleHarness;

namespace tycho::solvers {

// Soft feasibility pre-stage constants (coin-or/Ipopt 72a29c9,
// src/Algorithm/IpBacktrackingLineSearch.cpp option registration). A soft step
// is accepted while its trial primal-dual error is at most this fraction of the
// current primal-dual error (Ipopt soft_resto_pderror_reduction_factor, shipped
// default 1 - 1e-4; the factor-0 "disable soft restoration" branch is not
// transcribed — this solver ships the pre-stage unconditionally for the nested
// mode with no knob).
inline constexpr double kSoftRestoPdErrorReductionFactor = 1.0 - 1e-4;

// Maximum number of successive soft pre-stage iterations before escalating to
// the full restoration switch (Ipopt max_soft_resto_iters, shipped default 10).
inline constexpr int kMaxSoftRestoIters = 10;

// =============================================================================
// FeasibilitySwitchRecovery — outermost recovery link; last-resort switch to
// feasibility restoration. See the file docstring.
// =============================================================================
class FeasibilitySwitchRecovery : public RecoveryChain {
  public:
    explicit FeasibilitySwitchRecovery(std::unique_ptr<RecoveryChain> inner)
        : inner_(std::move(inner)) {
        if (!inner_)
            throw std::invalid_argument(
                "FeasibilitySwitchRecovery: inner recovery chain must not be null");
    }

    Action on_step_rejected(IterateInfo &Citer, const std::vector<IterateInfo> &iters,
                            SolverContext &ctx, AcceptanceStrategy &acceptance,
                            GlobalizationMechanism &mechanism, PSIOPT::LineSearchModes lsmode,
                            double obj_scale, double mu, double prim_obj, double barr_obj,
                            Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, Eigen::VectorXd &XSL2,
                            Eigen::VectorXd &RHS, Eigen::VectorXd &RHS2, double &alpha,
                            double &alphap, double &alphad, int &soc_steps, int &resolved_depth,
                            int &watchdog_activations) override;

    // A genuinely accepted regular step means the ordinary optimality-phase
    // acceptance test recovered, so any soft pre-stage in progress has ended:
    // clear the successive-soft-iteration counter, then thread through to the
    // inner chain (mirroring WatchdogRecovery's decorator).
    void notify_step_accepted() override {
        soft_counter_ = 0;
        inner_->notify_step_accepted();
    }

    // μ-event / phase-change reset also clears the soft pre-stage counter (a
    // restoration entry or exit resets the pre-stage), then threads through.
    void reset() override {
        soft_counter_ = 0;
        inner_->reset();
    }

  private:
    friend class ::NestedLifecycleHarness;

    std::unique_ptr<RecoveryChain> inner_;

    // Number of successive soft pre-stage iterations taken (nested restoration
    // only). Incremented each time the ladder-exhausted rejection yields a soft
    // step; once it exceeds kMaxSoftRestoIters the link escalates to the full
    // restoration switch. Cleared by notify_step_accepted()/reset() (see above).
    // A mid-pre-stage accept-as-is (entry refused by guard/budget) does NOT
    // clear it — at worst the pre-stage escalates one episode early into the
    // full restoration backstop, which is the safe direction.
    int soft_counter_ = 0;
};

} // namespace tycho::solvers
