///////////////////////////////////////////////////////////////////////////////
// Unit tests for InteriorPointSolver::apply_preset() -- the five named mechanism configu-
// rations in detail/solvers/interior_point_solver_presets.h. Covers: the per-preset field
// truth table (all nine globalization fields, exact enum values); a
// deliberate default-drift tripwire pinning `classic` against a fresh
// default-constructed Settings{}; the unknown-name error (message lists every
// valid name); every preset passing Settings::validate(); every preset
// leaving non-globalization fields untouched; and a smoke solve (CONVERGED or
// ACCEPTABLE) for every preset on the shared Brachistochrone fixture.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/drivers/interior_point_solver_presets.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace tycho;
using tycho::solvers::AcceptanceStrategies;
using tycho::solvers::BarrierGovernors;
using tycho::solvers::InertiaModes;
using tycho::solvers::InteriorPointSolver;
using tycho::solvers::InteriorPointSolverPresetFields;
using tycho::solvers::kInteriorPointSolverPresets;
using tycho::solvers::MeritPenaltyRules;
using tycho::solvers::RestorationModes;
using TychoTest::make_brach_solver_phase;
using TychoTest::SolverTest;

namespace {

// Field-by-field comparison against an expected InteriorPointSolverPresetFields value.
// File-local name kept unique across tests/cpp/solvers/ (unity build merges
// these TUs; anonymous namespaces do not isolate across files in the same
// unity TU) -- prefix "PresetGate" is not used anywhere else in tests/cpp/.
void PresetGateExpectFieldsMatch(const InteriorPointSolver::Settings &s,
                                 const InteriorPointSolverPresetFields &f) {
    EXPECT_EQ(s.acceptance_strategy_, f.acceptance_strategy_);
    EXPECT_EQ(s.merit_penalty_rule_, f.merit_penalty_rule_);
    EXPECT_EQ(s.barrier_governor_, f.barrier_governor_);
    EXPECT_EQ(s.never_monotone_, f.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, f.restoration_mode_);
    EXPECT_EQ(s.inertia_mode_, f.inertia_mode_);
    EXPECT_EQ(s.max_soc_, f.max_soc_);
    EXPECT_EQ(s.ls_extended_iters_, f.ls_extended_iters_);
    EXPECT_EQ(s.watchdog_, f.watchdog_);
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Table sanity: the five names, in the declared order.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateTableHasExpectedNames) {
    ASSERT_EQ(kInteriorPointSolverPresets.size(), 5u);
    EXPECT_EQ(kInteriorPointSolverPresets[0].name_, "classic");
    EXPECT_EQ(kInteriorPointSolverPresets[1].name_, "filter_l1");
    EXPECT_EQ(kInteriorPointSolverPresets[2].name_, "soc_recovery_l1");
    EXPECT_EQ(kInteriorPointSolverPresets[3].name_, "soc_proximal");
    EXPECT_EQ(kInteriorPointSolverPresets[4].name_, "merit_l1");
}

///////////////////////////////////////////////////////////////////////////////
// Per-preset field truth table -- exact enum values transcribed from
// docs/dev/analysis/2026-07-e2-fixes-evidence-refresh.md's evidence tables
// (see the header's per-entry comments), independent of kInteriorPointSolverPresets so a
// typo in the header's literal table is caught here too.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateClassicFields) {
    InteriorPointSolver opt;
    opt.apply_preset("classic");
    const auto &s = opt.settings();
    EXPECT_EQ(s.acceptance_strategy_, AcceptanceStrategies::classic_merit);
    EXPECT_EQ(s.merit_penalty_rule_, MeritPenaltyRules::wmno);
    EXPECT_EQ(s.barrier_governor_, BarrierGovernors::classic_adaptive);
    EXPECT_FALSE(s.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, RestorationModes::off);
    EXPECT_EQ(s.inertia_mode_, InertiaModes::classic);
    EXPECT_EQ(s.max_soc_, 0);
    EXPECT_EQ(s.ls_extended_iters_, 0);
    EXPECT_FALSE(s.watchdog_);
}

TEST_F(SolverTest, PresetGateFilterL1Fields) {
    InteriorPointSolver opt;
    opt.apply_preset("filter_l1");
    const auto &s = opt.settings();
    EXPECT_EQ(s.acceptance_strategy_, AcceptanceStrategies::filter);
    EXPECT_EQ(s.merit_penalty_rule_, MeritPenaltyRules::wmno);
    EXPECT_EQ(s.barrier_governor_, BarrierGovernors::monitored);
    EXPECT_FALSE(s.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, RestorationModes::l1_nested);
    EXPECT_EQ(s.inertia_mode_, InertiaModes::classic);
    EXPECT_EQ(s.max_soc_, 0);
    EXPECT_EQ(s.ls_extended_iters_, 0);
    EXPECT_FALSE(s.watchdog_);
}

TEST_F(SolverTest, PresetGateSocRecoveryL1Fields) {
    InteriorPointSolver opt;
    opt.apply_preset("soc_recovery_l1");
    const auto &s = opt.settings();
    EXPECT_EQ(s.acceptance_strategy_, AcceptanceStrategies::classic_merit);
    EXPECT_EQ(s.merit_penalty_rule_, MeritPenaltyRules::wmno);
    EXPECT_EQ(s.barrier_governor_, BarrierGovernors::monitored);
    EXPECT_FALSE(s.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, RestorationModes::l1_nested);
    EXPECT_EQ(s.inertia_mode_, InertiaModes::proximal_regularization);
    EXPECT_EQ(s.max_soc_, 4);
    EXPECT_EQ(s.ls_extended_iters_, 2);
    EXPECT_TRUE(s.watchdog_);
}

TEST_F(SolverTest, PresetGateSocProximalFields) {
    InteriorPointSolver opt;
    opt.apply_preset("soc_proximal");
    const auto &s = opt.settings();
    EXPECT_EQ(s.acceptance_strategy_, AcceptanceStrategies::classic_merit);
    EXPECT_EQ(s.merit_penalty_rule_, MeritPenaltyRules::wmno);
    EXPECT_EQ(s.barrier_governor_, BarrierGovernors::monitored);
    EXPECT_FALSE(s.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, RestorationModes::proximal_switch);
    EXPECT_EQ(s.inertia_mode_, InertiaModes::proximal_regularization);
    EXPECT_EQ(s.max_soc_, 4);
    EXPECT_EQ(s.ls_extended_iters_, 0);
    EXPECT_FALSE(s.watchdog_);
}

TEST_F(SolverTest, PresetGateMeritL1Fields) {
    InteriorPointSolver opt;
    opt.apply_preset("merit_l1");
    const auto &s = opt.settings();
    EXPECT_EQ(s.acceptance_strategy_, AcceptanceStrategies::merit);
    EXPECT_EQ(s.merit_penalty_rule_, MeritPenaltyRules::wmno);
    EXPECT_EQ(s.barrier_governor_, BarrierGovernors::classic_adaptive);
    EXPECT_FALSE(s.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, RestorationModes::l1_nested);
    EXPECT_EQ(s.inertia_mode_, InertiaModes::classic);
    EXPECT_EQ(s.max_soc_, 0);
    EXPECT_EQ(s.ls_extended_iters_, 0);
    EXPECT_FALSE(s.watchdog_);
}

///////////////////////////////////////////////////////////////////////////////
// Default-drift tripwire: `classic` is pinned as literals in
// interior_point_solver_presets.h (mechanism semantics), not read off Settings{}'s own
// defaults. This test instead compares apply_preset("classic")'s result
// against a FRESH default-constructed Settings{}. If a future change ever
// moves one of Settings{}'s own defaults among these nine fields, THIS TEST
// FAILS -- forcing a conscious decision about whether `classic` should track
// the new default or keep its pinned mechanism meaning, rather than the
// preset silently drifting out of sync with what "classic" is documented to
// mean.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateClassicMatchesFreshDefaultSettings) {
    InteriorPointSolver opt;
    opt.apply_preset("classic");

    InteriorPointSolver::Settings fresh; // default-constructed -- the drift tripwire target
    InteriorPointSolverPresetFields fresh_as_fields{
        fresh.acceptance_strategy_,
        fresh.merit_penalty_rule_,
        fresh.barrier_governor_,
        fresh.never_monotone_,
        fresh.restoration_mode_,
        fresh.inertia_mode_,
        fresh.max_soc_,
        fresh.ls_extended_iters_,
        fresh.watchdog_,
    };
    PresetGateExpectFieldsMatch(opt.settings(), fresh_as_fields);
}

///////////////////////////////////////////////////////////////////////////////
// Unknown preset name.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateUnknownNameThrows) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.apply_preset("not_a_real_preset"), std::invalid_argument);
}

TEST_F(SolverTest, PresetGateUnknownNameMessageListsEveryValidName) {
    InteriorPointSolver opt;
    try {
        opt.apply_preset("not_a_real_preset");
        FAIL() << "expected apply_preset to throw std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        for (const auto &entry : kInteriorPointSolverPresets)
            EXPECT_NE(msg.find(entry.name_), std::string::npos)
                << "error message missing preset name: " << entry.name_;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Every preset passes Settings::validate().
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateEveryPresetPassesValidate) {
    for (const auto &entry : kInteriorPointSolverPresets) {
        InteriorPointSolver opt;
        opt.apply_preset(entry.name_);
        EXPECT_NO_THROW(opt.settings().validate()) << "preset: " << entry.name_;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Presets touch only the nine globalization fields -- every other field
// (sentineled here with max_iters_/econ_tol_) is untouched.
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateLeavesNonGlobalizationFieldsUntouched) {
    for (const auto &entry : kInteriorPointSolverPresets) {
        InteriorPointSolver opt;
        opt.settings().max_iters_ = 4321;
        opt.settings().econ_tol_ = 1.234e-9;

        opt.apply_preset(entry.name_);

        EXPECT_EQ(opt.settings().max_iters_, 4321) << "preset: " << entry.name_;
        EXPECT_DOUBLE_EQ(opt.settings().econ_tol_, 1.234e-9) << "preset: " << entry.name_;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Smoke solve: every preset must solve the small Brachistochrone problem to
// CONVERGED or ACCEPTABLE. set_qp_threads(1) for determinism; print_level(3)
// for silence (3 = silent -- see InteriorPointSolver::Settings::print_level_).
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PresetGateEveryPresetSolvesSmokeProblem) {
    for (const auto &entry : kInteriorPointSolverPresets) {
        auto phase = make_brach_solver_phase(16);
        InteriorPointSolver ipm;
        ipm.set_print_level(3);
        ipm.set_qp_threads(1);
        ipm.apply_preset(entry.name_);

        const auto status = phase->solve(&ipm, {.presolve = true}).flag_;
        EXPECT_LE(status, tycho::ConvergenceFlags::ACCEPTABLE)
            << "preset: " << entry.name_ << " did not converge/acceptable on the smoke problem";
    }
}
