// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// FeasibilitySwitchRecovery — the outermost RecoveryChain link that converts a
// ladder-exhausted step rejection into a feasibility-restoration mode-switch.
// It is built (by rebuild_globalization_components) only when
// restoration_mode_ == proximal_switch, wrapping whatever inner chain the other
// settings produced (NoopRecovery, ChainedRecovery, or a WatchdogRecovery-
// decorated chain).
//
// Behavior: delegate the whole rejection to the inner chain first. If the inner
// chain RESOLVES the rejection (returns kRetry / kSwitchToFeasibility / kGiveUp)
// the outcome passes through untouched — the restoration switch is strictly a
// last resort, tried only when nothing else salvaged the step. The inner
// chain's kAcceptAsIs return is exactly today's ladder-exhaustion fallback (the
// surviving alpha is taken as-is); FeasibilitySwitchRecovery intercepts ONLY
// that case and, if a real restoration episode is warranted — the current point
// is not already near-feasible AND the per-phase entry budget is not exhausted
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
// Ownership: holds only the inner chain (per RecoveryChain's ownership rule).
// reset()/notify_step_accepted() thread straight through to it. Definitions
// live in src/solvers/psiopt_globalization.cpp.

#pragma once

#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "tycho/detail/solvers/globalization/recovery_chain.h"

namespace tycho::solvers {

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

    // Threaded straight through to the inner chain (this link holds no
    // per-accept state of its own), mirroring WatchdogRecovery's decorator.
    void notify_step_accepted() override { inner_->notify_step_accepted(); }

    void reset() override { inner_->reset(); }

  private:
    std::unique_ptr<RecoveryChain> inner_;
};

} // namespace tycho::solvers
