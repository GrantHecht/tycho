///////////////////////////////////////////////////////////////////////////////
// Jet batch runner tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(SolverTest, JetMapPrebuiltProblems) {
    // Create 3 identical Brachistochrone phases, solve via Jet::map
    std::vector<std::shared_ptr<ODEPhase<BrachODE>>> phases;
    for (int i = 0; i < 3; ++i) {
        auto p = make_brach_solver_phase(16);
        p->JetJobMode = OptimizationProblemBase::JetJobModes::SolveOptimize;
        phases.push_back(p);
    }

    auto results = Jet::map(phases, false);
    ASSERT_EQ(results.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        auto traj = results[i]->returnTraj();
        double tf = traj.back()[3];
        EXPECT_NEAR(tf, 1.8013, 0.02)
            << "Jet problem " << i << " did not converge to expected time";
    }
}

TEST_F(SolverTest, JetMapSingleGenerator) {
    // Single generator function that builds a Brach phase from a segment count
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen = [](int n_segs) {
        auto p = make_brach_solver_phase(n_segs);
        p->JetJobMode = OptimizationProblemBase::JetJobModes::SolveOptimize;
        return p;
    };

    std::vector<int> args = {16, 16};
    auto results = Jet::map(gen, args, false);
    ASSERT_EQ(results.size(), 2u);
    for (int i = 0; i < 2; ++i) {
        auto traj = results[i]->returnTraj();
        double tf = traj.back()[3];
        EXPECT_NEAR(tf, 1.8013, 0.02) << "Jet single-gen problem " << i << " did not converge";
    }
}

TEST_F(SolverTest, JetMapSaturatedPool) {
    // Regression test: Jet.map must not deadlock when num_jobs >= pool threads.
    // Root cause: parallel_task() in NLP eval methods was submitting work to the
    // global pool while already running on a pool worker, saturating all threads.
    int nt = tycho::get_num_threads();
    int num_jobs = std::max(nt + 2, 6); // more jobs than pool threads

    std::vector<std::shared_ptr<ODEPhase<BrachODE>>> phases;
    for (int i = 0; i < num_jobs; ++i) {
        auto p = make_brach_solver_phase(16);
        p->JetJobMode = OptimizationProblemBase::JetJobModes::SolveOptimize;
        phases.push_back(p);
    }

    auto results = Jet::map(phases, false);
    ASSERT_EQ(results.size(), static_cast<size_t>(num_jobs));
    for (int i = 0; i < num_jobs; ++i) {
        auto traj = results[i]->returnTraj();
        double tf = traj.back()[3];
        EXPECT_NEAR(tf, 1.8013, 0.02) << "Jet saturated-pool problem " << i << " did not converge";
    }
}

TEST_F(SolverTest, JetMapMultiGenerator) {
    // Two generators: different segment counts
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen16 = [](int) {
        auto p = make_brach_solver_phase(16);
        p->JetJobMode = OptimizationProblemBase::JetJobModes::SolveOptimize;
        return p;
    };
    std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)> gen32 = [](int) {
        auto p = make_brach_solver_phase(32);
        p->JetJobMode = OptimizationProblemBase::JetJobModes::SolveOptimize;
        return p;
    };

    std::vector<std::function<std::shared_ptr<ODEPhase<BrachODE>>(int)>> genfuncs = {gen16, gen32};
    std::vector<int> args = {0, 0, 0}; // dummy args
    Eigen::VectorXi genfidxes(3);
    genfidxes << 0, 1, 0; // gen16, gen32, gen16

    auto results = Jet::map(genfuncs, args, genfidxes, false);
    ASSERT_EQ(results.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        auto traj = results[i]->returnTraj();
        double tf = traj.back()[3];
        EXPECT_NEAR(tf, 1.8013, 0.02) << "Jet multi-gen problem " << i << " did not converge";
    }
}
