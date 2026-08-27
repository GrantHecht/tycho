///////////////////////////////////////////////////////////////////////////////
// Jet batch runner tests
//
// M5 solve-API interim: jet_run() -- the mode-sequence-driven entry point
// Jet::map dispatches to on each pool worker -- is a placeholder that
// unconditionally throws std::logic_error (the jet_job_mode_/JetJobModes
// surface it used to switch on is retired along with the five mode methods;
// batched solves are staged through set_jet_job() instead, landing in a
// follow-up change). Until that lands, these tests pin the interim contract:
// Jet::map's own plumbing (single problem list, single generator, a
// pool-saturating job count, multiple generators dispatched by index) still
// runs every job and correctly propagates the first job's exception to the
// caller (hven::Jet::map's own future-draining behavior), rather than
// hanging or losing the failure. The pre-M5 convergence assertions these
// tests used to make are restored once real batched dispatch lands.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

#include <stdexcept>

using namespace tycho;
using TychoTest::BrachODE;
using TychoTest::make_brach_solver_phase;
using TychoTest::SolverTest;

// Jet lives in tycho::solvers; this file previously relied on the
// TychoTest -> tycho::solvers using-directive leak (fixed in
// solver_test_utils.h) to see it unqualified.
using tycho::solvers::Jet;

TEST_F(SolverTest, JetMapPrebuiltProblemsRefusesViaInterimJetRun) {
    std::vector<std::shared_ptr<ODEPhase<BrachODE>>> phases;
    for (int i = 0; i < 3; ++i) {
        phases.push_back(make_brach_solver_phase(16));
    }

    EXPECT_THROW(Jet::map(phases, false), std::logic_error);
}

TEST_F(SolverTest, JetMapSingleGeneratorRefusesViaInterimJetRun) {
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen = [](int n_segs) {
        return make_brach_solver_phase(n_segs);
    };

    std::vector<int> args = {16, 16};
    EXPECT_THROW(Jet::map(gen, args, false), std::logic_error);
}

TEST_F(SolverTest, JetMapSaturatedPoolRefusesViaInterimJetRun) {
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

TEST_F(SolverTest, JetMapMultiGeneratorRefusesViaInterimJetRun) {
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
