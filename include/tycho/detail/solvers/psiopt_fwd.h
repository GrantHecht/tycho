// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Extracted ConvergenceFlags enum, severity ordering operator, and forward declaration from
//   psiopt.h
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
// =============================================================================

#pragma once
#include <compare>

namespace tycho {

/// Optimizer convergence status. Lives in tycho:: (not tycho::solvers) so callers
/// outside the solvers module can reference it directly. Placed in a forward-
/// declaration header so dependent modules can use ConvergenceFlags and forward-
/// declare PSIOPT without pulling in the full definition and its heavy transitive
/// includes.
enum class ConvergenceFlags {
    CONVERGED = 0,
    ACCEPTABLE = 1,
    NOTCONVERGED = 2,
    DIVERGING = 3,
};

// Severity ordering: CONVERGED < ACCEPTABLE < NOTCONVERGED < DIVERGING
constexpr auto operator<=>(ConvergenceFlags a, ConvergenceFlags b) {
    return static_cast<int>(a) <=> static_cast<int>(b);
}

} // namespace tycho

namespace tycho::solvers {
class PSIOPT;

// Step-acceptance strategy selector (PSIOPT::Settings::acceptance_strategy_).
// Declared here — rather than nested in PSIOPT like BarrierModes/LineSearchModes
// — so both PSIOPT::Settings (psiopt.h) and the acceptance components
// (globalization/modern_merit.h) can name it without a circular include.
// Style matches the nested mode enums: strongly-typed with explicit values.
//   classic_merit — the fused classic backtracking merit line search
//                   (ClassicMeritAcceptance); the bit-identical default.
//   merit         — the modernized merit family driven through the GENERIC
//                   AcceptanceStrategy path (ModernMeritAcceptance).
//   funnel        — the scalar-funnel-width strategy on the shared
//                   Wächter–Biegler switching skeleton (FunnelAcceptance).
//   filter        — the (θ, φ)-pair filter strategy on the shared
//                   Wächter–Biegler switching skeleton (FilterAcceptance).
enum class AcceptanceStrategies { classic_merit = 0, merit = 1, funnel = 2, filter = 3 };

// Penalty-parameter rule for the modernized merit family
// (PSIOPT::Settings::merit_penalty_rule_; read only when
// acceptance_strategy_ == merit).
//   wmno     — Waltz, Morales, Nocedal & Orban, Math. Program. 107 (2006),
//              §3.1: single penalty ν updated from the directional-derivative
//              condition (Eqs 3.5, 3.6).
//   flexible — Curtis & Nocedal, IMA J. Numer. Anal. 28(4) (2008): a penalty
//              INTERVAL [π_l, π_u]; a step is accepted if it reduces the merit
//              for at least one π in the interval (Eqs 2.1, 3.9, 3.10).
enum class MeritPenaltyRules { wmno = 0, flexible = 1 };

// Barrier-parameter governor selector (PSIOPT::Settings::barrier_governor_).
// Declared here alongside AcceptanceStrategies/MeritPenaltyRules — for the
// same reason: both PSIOPT::Settings (psiopt.h) and the governor components
// (globalization/classic_adaptive_governor.h, globalization/
// monitored_governor.h) need it without a circular include.
//   classic_adaptive — the classic PROBE/LOQO free-mode barrier update
//                      (ClassicAdaptiveGovernor); the bit-identical default.
//   monitored        — the free<->monotone monitored governor
//                      (MonitoredBarrierGovernor): a KKT-error monitor hands
//                      off to a Fiacco-McCormick monotone mode when free-mode
//                      progress stalls, then re-enters free mode once
//                      progress resumes. Composes a ClassicAdaptiveGovernor
//                      as its free-mode delegate, so any acceptance strategy
//                      may pair with it.
enum class BarrierGovernors { classic_adaptive = 0, monitored = 1 };

// Feasibility-restoration mode selector (PSIOPT::Settings::restoration_mode_).
// Declared here alongside the other strategy selectors — for the same reason:
// both PSIOPT::Settings (psiopt.h) and the restoration component
// (globalization/proximal_restoration.h) need it without a circular include.
//   off             — no feasibility restoration (default). The globalization
//                     failure path behaves exactly as it did before restoration
//                     existed: a ladder-exhausted rejection is taken as-is. No
//                     RestorationStrategy is constructed, so every restoration
//                     branch in the solver is provably dead.
//   proximal_switch — the proximal feasibility mode-switch
//                     (ProximalSwitchRestoration): on a ladder-exhausted
//                     rejection, keep the same barrier algorithm running but
//                     swap the true objective for a proximal term pulling the
//                     primals back toward the entry point, until infeasibility
//                     is sufficiently reduced. Composes with every acceptance
//                     strategy and barrier governor.
//   l1_nested       — the nested l1 elastic feasibility restoration
//                     (NestedL1Restoration, globalization/l1_restoration.h):
//                     on a ladder-exhausted rejection, solve the l1 elastic
//                     reformulation (Ipopt-lineage restoration NLP) as a
//                     CONDENSED in-place phase that reuses the outer barrier
//                     algorithm's KKT system, rather than switching the outer
//                     objective the way proximal_switch does. Every shipped
//                     acceptance strategy implements the restoration exit
//                     test, so l1_nested composes with every
//                     acceptance_strategy and barrier_governor exactly like
//                     proximal_switch (no matrix restrictions).
enum class RestorationModes { off = 0, proximal_switch = 1, l1_nested = 2 };

} // namespace tycho::solvers
