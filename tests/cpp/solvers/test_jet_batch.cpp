///////////////////////////////////////////////////////////////////////////////
// Jet batch runner tests
//
// jet_run() -- the entry point Jet::map dispatches to on each pool worker --
// clones the prototype engine set_jet_job() staged (via clone_prototype()),
// plus any staged presolve/polish auxiliary engine, solves the clones with
// the staged SolveOptions, and returns the deciding stage's flag;
// last_result() reflects the call afterward. A problem that never had
// set_jet_job() called on it refuses with std::logic_error, naming the
// missing call -- Jet::map's own plumbing (single problem list, single
// generator, a pool-saturating job count, multiple generators dispatched by
// index) still runs every job and correctly propagates the first job's
// exception to the caller (hven::Jet::map's own future-draining behavior),
// rather than hanging or losing the failure, so the four
// *RefusesWithNoJobStaged tests below exercise that plumbing without staging
// a job. JetRunUsesAClonedEnginePerProblem below is the real-dispatch case:
// a staged prototype never runs itself, only its settings are cloned into
// each per-job engine, and the staged SolveOptions (not just the shared
// engine's settings) reach each clone.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

#include <stdexcept>

using namespace tycho;
using TychoTest::BrachODE;
using TychoTest::make_brach_solver_phase;
using TychoTest::quiet_ipm;
using TychoTest::SolverTest;

// Jet lives in tycho::solvers; this file previously relied on the
// TychoTest -> tycho::solvers using-directive leak (fixed in
// solver_test_utils.h) to see it unqualified.
using tycho::solvers::BackendProblemBase;
using tycho::solvers::InteriorPointSolver;
using tycho::solvers::Jet;
using tycho::solvers::SolveOptions;
using tycho::solvers::SolveResult;

TEST_F(SolverTest, JetMapPrebuiltProblemsRefusesWithNoJobStaged) {
    std::vector<std::shared_ptr<ODEPhase<BrachODE>>> phases;
    for (int i = 0; i < 3; ++i) {
        phases.push_back(make_brach_solver_phase(16));
    }

    EXPECT_THROW(Jet::map(phases, false), std::logic_error);
}

TEST_F(SolverTest, JetMapSingleGeneratorRefusesWithNoJobStaged) {
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen = [](int n_segs) {
        return make_brach_solver_phase(n_segs);
    };

    std::vector<int> args = {16, 16};
    EXPECT_THROW(Jet::map(gen, args, false), std::logic_error);
}

TEST_F(SolverTest, JetMapSaturatedPoolRefusesWithNoJobStaged) {
    // Regression coverage retained from the pre-M5 deadlock test: with more
    // jobs than pool threads, every job's exception must still be collected
    // (and the first re-thrown) without the pool hanging.
    int nt = tycho::utils::get_num_threads();
    int num_jobs = std::max(nt + 2, 6); // more jobs than pool threads

    std::vector<std::shared_ptr<ODEPhase<BrachODE>>> phases;
    for (int i = 0; i < num_jobs; ++i) {
        phases.push_back(make_brach_solver_phase(16));
    }

    EXPECT_THROW(Jet::map(phases, false), std::logic_error);
}

TEST_F(SolverTest, JetMapMultiGeneratorRefusesWithNoJobStaged) {
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen16 = [](int) {
        return make_brach_solver_phase(16);
    };
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen32 = [](int) {
        return make_brach_solver_phase(32);
    };

    std::vector<std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)>> genfuncs = {gen16, gen32};
    std::vector<int> args = {0, 0, 0}; // dummy args
    Eigen::VectorXi genfidxes(3);
    genfidxes << 0, 1, 0; // gen16, gen32, gen16

    EXPECT_THROW(Jet::map(genfuncs, args, genfidxes, false), std::logic_error);
}

TEST_F(SolverTest, JetRunUsesAClonedEnginePerProblem) {
    auto phase1 = make_brach_solver_phase(16);
    auto phase2 = make_brach_solver_phase(16);

    // The prototype carries a marker setting (max_iters_ = 1) and is shared,
    // unmodified, as the job description for both problems.
    InteriorPointSolver prototype;
    quiet_ipm(prototype);
    prototype.settings().max_iters_ = 1;

    // phase1 stages the default (main-stage-only) options; phase2 stages
    // presolve=true, so its clone runs a Feasible stage ahead of the main
    // one. Different options per problem also pins that each problem's
    // clone is independently configured/solved rather than one clone's
    // result being shared across both -- not just settings-propagation from
    // one shared prototype, which the max_iters_ marker alone already
    // covers.
    phase1->set_jet_job(prototype, SolveOptions{});
    phase2->set_jet_job(prototype, SolveOptions{.presolve = true});

    std::vector<std::shared_ptr<BackendProblemBase>> jobs = {phase1, phase2};
    auto results = Jet::map(jobs, false);
    ASSERT_EQ(results.size(), 2u);

    // The prototype itself never ran: its own result stays exactly as cold
    // as it started (jet_run() clones settings, never dispatches through the
    // prototype instance itself).
    EXPECT_EQ(prototype.result().iter_num_, 0);

    for (const auto &r : results) {
        ASSERT_TRUE(r);
        const SolveResult &sr = r->last_result();
        ASSERT_FALSE(sr.stages_.empty());
        // Marker visible in every stage report: every engine that actually
        // ran was capped at the prototype's max_iters_ = 1.
        for (const auto &stage : sr.stages_) {
            EXPECT_LE(stage.iterations_, 1);
        }
    }

    // phase1 (no presolve staged) ran exactly the main stage; phase2
    // (presolve=true staged) ran a presolve stage ahead of it -- the staged
    // SolveOptions, not just the shared engine settings, reached jet_run()'s
    // clone.
    const SolveResult &result1 = results[0]->last_result();
    ASSERT_EQ(result1.stages_.size(), 1u);
    EXPECT_EQ(result1.stages_[0].role_, "main");

    const SolveResult &result2 = results[1]->last_result();
    ASSERT_EQ(result2.stages_.size(), 2u);
    EXPECT_EQ(result2.stages_[0].role_, "presolve");
    EXPECT_EQ(result2.stages_[1].role_, "main");
}
