// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the globalization component extraction: this header is an
// interface-only placeholder until a feasibility-restoration strategy is
// implemented. No corresponding today's-behavior section: today's PSIOPT has
// no restoration machinery at all (settings_.max_feas_rest_ is an explicitly
// reserved, currently-unread field — psiopt.h Settings comment, "reserved —
// feasibility restoration, not currently implemented"); this interface has
// no existing behavior to preserve, unlike the other component headers.
//
// This file: interface declaration only. NO implementation exists or is
// planned until a feasibility-restoration strategy is implemented, which is
// expected to ship as a trio: a proximal feasibility mode-switch, a nested l1
// proximal restoration, and an elastic/penalty relaxation. The method shapes
// below are a best-effort placeholder consistent with that plan and with the
// AcceptanceStrategy / GlobalizationMechanism / BarrierGovernor /
// RecoveryChain shapes already established (SolverContext for solver access,
// ProgressMeasures for the (θ,f) pair, an explicit reset()) — they are NOT
// source-verified against a concrete implementation the way the other
// component headers are, and should be expected to change when feasibility
// restoration actually lands. Do not build recovery/governor/mechanism wiring
// against restoration entry/exit assumptions beyond "restoration is inert
// until a feasibility-restoration strategy exists" (RecoveryChain::Action::
// kSwitchToFeasibility has nowhere to dispatch to until then).

#pragma once

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/solver_context.h"

namespace tycho::solvers {

// =============================================================================
// RestorationStrategy — placeholder interface for the feasibility-restoration
// trio.
// =============================================================================
class RestorationStrategy {
  public:
    virtual ~RestorationStrategy() = default;

    // Enter restoration mode. `reference` is the (θ,f) pair of the point
    // restoration was entered from (the eventual exit test compares against
    // it: "θ <= 0.9*θ_entry"); ctx gives access to the live NLP/settings_ a
    // restoration sub-problem needs to construct itself.
    virtual void enter_restoration(const ProgressMeasures &reference, SolverContext &ctx) = 0;

    // Restoration-exit test. The eventual test additionally requires
    // filter-acceptability; that half of the test lives on whichever
    // AcceptanceStrategy is active, not here — this method is restoration's
    // own half of the combined exit condition.
    virtual bool should_exit_restoration(const ProgressMeasures &trial,
                                          const SolverContext &ctx) const = 0;

    // Leave restoration mode (equality multipliers reset by least squares,
    // bound multipliers reset above a 1e3 threshold).
    virtual void exit_restoration(SolverContext &ctx) = 0;

    // Whether restoration mode is currently active.
    virtual bool is_active() const = 0;

    // μ-event / phase-change reset hook, matching the other four
    // globalization interfaces.
    virtual void reset() = 0;
};

} // namespace tycho::solvers
