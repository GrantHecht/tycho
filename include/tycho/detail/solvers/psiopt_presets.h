// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Named PSIOPT configuration presets: five mechanism-named globalization
// configurations, each a pure Settings field assignment (no algorithm code is
// touched). Evidence of record for the non-classic presets is the globalization
// campaign's post-fixes evidence refresh,
// docs/dev/analysis/2026-07-e2-fixes-evidence-refresh.md (cell hashes cited
// per preset below); `classic` is the stock Settings{} baseline the campaign
// was measured against, pinned here as literals rather than read off a
// default-constructed Settings so this table keeps meaning even if a future
// change moves the struct's own defaults (see the default-drift regression
// test in tests/cpp/solvers/test_psiopt_presets.cpp).
//
// Every preset assigns exactly the same nine globalization fields (the full
// set PSIOPT::apply_preset() touches): acceptance_strategy_,
// merit_penalty_rule_, barrier_governor_, never_monotone_, restoration_mode_,
// inertia_mode_, max_soc_, ls_extended_iters_, watchdog_. No other Settings
// field (tolerances, iteration caps, QP parameters, ...) is read or written by
// a preset.
//
// kPSIOPTPresets is the single source of truth for the preset name set: both
// PSIOPT::apply_preset()'s dispatch/error-message and any downstream consumer
// that needs to enumerate or document the valid names (e.g. the Python
// binding's docstring) should read this table rather than re-deriving the
// list.

#pragma once

#include <array>
#include <string_view>

#include "tycho/detail/solvers/psiopt_fwd.h"

namespace tycho::solvers {

// The nine globalization fields a preset assigns, exactly mirroring the
// PSIOPT::Settings members of the same names.
struct PSIOPTPresetFields {
    AcceptanceStrategies acceptance_strategy_;
    MeritPenaltyRules merit_penalty_rule_;
    BarrierGovernors barrier_governor_;
    bool never_monotone_;
    RestorationModes restoration_mode_;
    InertiaModes inertia_mode_;
    int max_soc_;
    int ls_extended_iters_;
    bool watchdog_;
};

struct PSIOPTPresetEntry {
    std::string_view name_;
    PSIOPTPresetFields fields_;
};

// clang-format off
inline constexpr std::array<PSIOPTPresetEntry, 5> kPSIOPTPresets = {{
    // classic — the stock Settings{} baseline (bit-identical default path).
    // Pinned as literals (not read off Settings{}) so this preset keeps its
    // mechanism meaning independent of the struct's own defaults; the
    // default-drift tripwire test in test_psiopt_presets.cpp compares this
    // preset's result against a live default-constructed Settings and fails
    // loudly if the two ever diverge, forcing a conscious decision instead of
    // a silent behavior change.
    {"classic", PSIOPTPresetFields{
        /*acceptance_strategy_=*/AcceptanceStrategies::classic_merit,
        /*merit_penalty_rule_=*/MeritPenaltyRules::wmno,
        /*barrier_governor_=*/BarrierGovernors::classic_adaptive,
        /*never_monotone_=*/false,
        /*restoration_mode_=*/RestorationModes::off,
        /*inertia_mode_=*/InertiaModes::classic,
        /*max_soc_=*/0,
        /*ls_extended_iters_=*/0,
        /*watchdog_=*/false,
    }},
    // filter_l1 — filter acceptance + monitored governor + nested-l1
    // restoration. The original campaign's sole leader and one of the three
    // post-fixes co-leaders at 12/17 (evidence refresh, cell 62994231856d).
    // Worst example-suite tail: +609% (DionysusLowThrust) at +31% aggregate
    // iterations vs stock, median parity.
    {"filter_l1", PSIOPTPresetFields{
        /*acceptance_strategy_=*/AcceptanceStrategies::filter,
        /*merit_penalty_rule_=*/MeritPenaltyRules::wmno,
        /*barrier_governor_=*/BarrierGovernors::monitored,
        /*never_monotone_=*/false,
        /*restoration_mode_=*/RestorationModes::l1_nested,
        /*inertia_mode_=*/InertiaModes::classic,
        /*max_soc_=*/0,
        /*ls_extended_iters_=*/0,
        /*watchdog_=*/false,
    }},
    // soc_recovery_l1 — classic-merit acceptance + monitored governor +
    // proximal-regularization inertia + SOC(4) + recovery (extended
    // backtrack(2) + watchdog) + nested-l1 restoration. A post-fixes
    // co-leader at 12/17 (evidence refresh, cell 8417a47846c1); the
    // composite recovery axis expands to ls_extended_iters=2 + watchdog=true.
    // Flattest worst-case tail among the three co-leaders (+100%,
    // OptimalDocking) at the highest aggregate cost (+42%).
    {"soc_recovery_l1", PSIOPTPresetFields{
        /*acceptance_strategy_=*/AcceptanceStrategies::classic_merit,
        /*merit_penalty_rule_=*/MeritPenaltyRules::wmno,
        /*barrier_governor_=*/BarrierGovernors::monitored,
        /*never_monotone_=*/false,
        /*restoration_mode_=*/RestorationModes::l1_nested,
        /*inertia_mode_=*/InertiaModes::proximal_regularization,
        /*max_soc_=*/4,
        /*ls_extended_iters_=*/2,
        /*watchdog_=*/true,
    }},
    // soc_proximal — classic-merit acceptance + monitored governor +
    // proximal-regularization inertia + SOC(4) + proximal-switch restoration
    // (no l1 machinery). A post-fixes co-leader at 12/17 (evidence refresh,
    // cell 8d8397c915b2); the lowest aggregate example-suite cost of the
    // three co-leaders (+27%), worst tail +219% (MinimumTimeToClimb).
    {"soc_proximal", PSIOPTPresetFields{
        /*acceptance_strategy_=*/AcceptanceStrategies::classic_merit,
        /*merit_penalty_rule_=*/MeritPenaltyRules::wmno,
        /*barrier_governor_=*/BarrierGovernors::monitored,
        /*never_monotone_=*/false,
        /*restoration_mode_=*/RestorationModes::proximal_switch,
        /*inertia_mode_=*/InertiaModes::proximal_regularization,
        /*max_soc_=*/4,
        /*ls_extended_iters_=*/0,
        /*watchdog_=*/false,
    }},
    // merit_l1 — modernized merit acceptance (classic barrier governor, NOT
    // monitored) + nested-l1 restoration. The matched-call section of the
    // evidence refresh: 7+2 under module call shape, 8+2 under the matched
    // (single-optimize()) shape, with zermelo's wrong-basin guess converging
    // under the matched shape at iteration 40 to objective
    // 1.7009270229362865 (the Ipopt-agreement reference) — the call-shape
    // lever, not the acceptance mechanism, decides that problem's outcome.
    {"merit_l1", PSIOPTPresetFields{
        /*acceptance_strategy_=*/AcceptanceStrategies::merit,
        /*merit_penalty_rule_=*/MeritPenaltyRules::wmno,
        /*barrier_governor_=*/BarrierGovernors::classic_adaptive,
        /*never_monotone_=*/false,
        /*restoration_mode_=*/RestorationModes::l1_nested,
        /*inertia_mode_=*/InertiaModes::classic,
        /*max_soc_=*/0,
        /*ls_extended_iters_=*/0,
        /*watchdog_=*/false,
    }},
}};
// clang-format on

} // namespace tycho::solvers
