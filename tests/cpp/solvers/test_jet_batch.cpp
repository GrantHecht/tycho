///////////////////////////////////////////////////////////////////////////////
// Jet batch runner tests
//
// jet_run() -- the entry point Jet::map dispatches to on each pool worker --
// clones the prototype engine set_jet_job() staged (via clone_prototype()),
// solves the clone with the staged SolveOptions, and returns the deciding
// stage's flag; last_result() reflects the call afterward. A problem that
// never had set_jet_job() called on it refuses with std::logic_error, naming
// the missing call -- Jet::map's own plumbing (single problem list, single
// generator, a pool-saturating job count, multiple generators dispatched by
// index) still runs every job and correctly propagates the first job's
// exception to the caller (hven::Jet::map's own future-draining behavior),
// rather than hanging or losing the failure, so the four
// *RefusesWithNoJobStaged tests below exercise that plumbing without staging
// a job. JetRunUsesAClonedEnginePerProblem below is the real-dispatch case:
// a staged prototype never runs itself, only its settings are cloned into
// each per-job engine.
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

    phase1->set_jet_job(prototype, SolveOptions{});
    phase2->set_jet_job(prototype, SolveOptions{});

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
        // Marker visible in the stage report: the engine that actually ran
        // was capped at the prototype's max_iters_ = 1.
        EXPECT_LE(sr.stages_.back().iterations_, 1);
    }
}
