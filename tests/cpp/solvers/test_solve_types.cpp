///////////////////////////////////////////////////////////////////////////////
// Solve-call value types: Mode, StageResult, PhaseResult, SolveResult.
//
// These are plain value types with no solver-engine behavior of their own, so
// this file pins only their own contract: the Mode string round trip and its
// refusal message, and SolveResult's final-stage-forwarding conveniences
// (objective()/iterations()/converged()/operator bool). Reaches only the
// solver library's own headers -- no tycho/tycho.h, no VectorFunction.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/solve_types.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using tycho::solvers::Mode;
using tycho::solvers::mode_from_string;
using tycho::solvers::PhaseResult;
using tycho::solvers::SolveResult;
using tycho::solvers::StageResult;

namespace {

StageResult solve_types_make_stage(tycho::ConvergenceFlags flag, double objective, int iterations) {
    StageResult stage;
    stage.flag_ = flag;
    stage.objective_ = objective;
    stage.iterations_ = iterations;
    return stage;
}

} // namespace

TEST(SolveTypes, ModeStringsRoundTripAndRefuse) {
    EXPECT_EQ(mode_from_string("optimal"), Mode::Optimal);
    EXPECT_EQ(mode_from_string("Feasible"), Mode::Feasible);
    EXPECT_THROW(mode_from_string("feasible_then_optimal"), std::invalid_argument);
    try {
        mode_from_string("both");
        FAIL();
    } catch (const std::invalid_argument &e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("both"), std::string::npos);
        EXPECT_NE(msg.find("optimal"), std::string::npos);
        EXPECT_NE(msg.find("feasible"), std::string::npos);
    }
}

TEST(SolveTypes, ModeToStringRoundTrips) {
    EXPECT_STREQ(tycho::solvers::to_string(Mode::Optimal), "Optimal");
    EXPECT_STREQ(tycho::solvers::to_string(Mode::Feasible), "Feasible");
}

TEST(SolveTypes, ConveniencesForwardToFinalStage) {
    SolveResult result;
    result.stages_.push_back(
        solve_types_make_stage(tycho::ConvergenceFlags::NOTCONVERGED, 1.0, 10));
    result.stages_.push_back(solve_types_make_stage(tycho::ConvergenceFlags::CONVERGED, 2.0, 20));
    result.flag_ = result.stages_.back().flag_;

    EXPECT_DOUBLE_EQ(result.objective(), 2.0);
    EXPECT_EQ(result.iterations(), 20);
    EXPECT_TRUE(result.converged());
    EXPECT_TRUE(static_cast<bool>(result));
}

TEST(SolveTypes, EmptyResultRefusesConveniences) {
    EXPECT_THROW(SolveResult{}.objective(), std::logic_error);
    EXPECT_THROW(SolveResult{}.iterations(), std::logic_error);
    EXPECT_THROW(SolveResult{}.final_stage(), std::logic_error);
}

TEST(SolveTypes, BoolCountsAcceptableAsConverged) {
    SolveResult converged_result;
    converged_result.flag_ = tycho::ConvergenceFlags::CONVERGED;
    EXPECT_TRUE(converged_result.converged());
    EXPECT_TRUE(static_cast<bool>(converged_result));

    SolveResult acceptable_result;
    acceptable_result.flag_ = tycho::ConvergenceFlags::ACCEPTABLE;
    EXPECT_TRUE(acceptable_result.converged());
    EXPECT_TRUE(static_cast<bool>(acceptable_result));

    SolveResult not_converged_result;
    not_converged_result.flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
    EXPECT_FALSE(not_converged_result.converged());
    EXPECT_FALSE(static_cast<bool>(not_converged_result));
}
