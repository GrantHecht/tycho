// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Part of the E2 G1 globalization extraction. Spec:
// docs/superpowers/specs/2026-07-16-e2-psiopt-globalization-design.md §3
// ("restoration.h — Interface only until G5") and §4 ("G5 — Restoration
// trio"). No corresponding dossier section: today's PSIOPT has no
// restoration machinery at all (settings_.max_feas_rest_ is an explicitly
// reserved, currently-unread field — psiopt.h Settings comment, "reserved —
// feasibility restoration, not currently implemented"); this interface has
// no existing behavior to preserve, unlike the other six Task 1 headers.
//
// G1 (this file): interface declaration only. NO implementation exists or is
// planned until G5 (spec §4), which is expected to ship as a trio (G5a/G5b):
// a proximal feasibility mode-switch, a nested l1 proximal restoration, and
// an elastic/penalty relaxation. The method shapes below are a best-effort
// placeholder consistent with that plan and with the AcceptanceStrategy /
// GlobalizationMechanism / BarrierGovernor / RecoveryChain shapes already
// established (SolverContext for solver access, ProgressMeasures for the
// (θ,f) pair, an explicit reset()) — they are NOT source-verified against a
// concrete implementation the way the other six headers are, and should be
// expected to change when G5 actually lands. Do not build G2-G4 wiring
// against restoration entry/exit assumptions beyond "restoration is inert
// until G5" (RecoveryChain::Action::kSwitchToFeasibility has nowhere to
// dispatch to until then).

#pragma once

#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/solver_context.h"

namespace tycho::solvers {

// =============================================================================
// RestorationStrategy — placeholder interface for the G5 restoration trio.
// =============================================================================
class RestorationStrategy {
  public:
    virtual ~RestorationStrategy() = default;

    // Enter restoration mode. `reference` is the (θ,f) pair of the point
    // restoration was entered from (spec §4 G5b's exit test compares against
    // it: "θ <= 0.9*θ_entry"); ctx gives access to the live NLP/settings a
    // restoration sub-problem needs to construct itself.
    virtual void enter_restoration(const ProgressMeasures &reference, SolverContext &ctx) = 0;

    // Restoration-exit test. G5b's actual test additionally requires
    // filter-acceptability (spec §4); that half of the test lives on
    // whichever AcceptanceStrategy is active, not here — this method is
    // restoration's own half of the combined exit condition.
    virtual bool should_exit_restoration(const ProgressMeasures &trial,
                                          const SolverContext &ctx) const = 0;

    // Leave restoration mode (spec §4 G5b: equality multipliers reset by
    // least squares, bound multipliers reset above a 1e3 threshold).
    virtual void exit_restoration(SolverContext &ctx) = 0;

    // Whether restoration mode is currently active.
    virtual bool is_active() const = 0;

    // μ-event / phase-change reset hook, matching the other four
    // globalization interfaces.
    virtual void reset() = 0;
};

} // namespace tycho::solvers
