///////////////////////////////////////////////////////////////////////////////
// Solver tests
//
// Tests PSIOPT convergence, Jet batch runner, and NLP structure.
///////////////////////////////////////////////////////////////////////////////

#include "Solvers/Jet.h"
#include "Tycho.h"
#include "test_utils.h"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <memory>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class SolverTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// PSIOPT convergence
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, BrachistochroneEndToEnd) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf_val = 10.0, yf_val = 5.0;

    int n_pts = 100;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf_val - x0) * s;
        pt[1] = y0 + (yf_val - y0) * s;
        pt[2] = g * s * 1.0 * std::cos(1.0);
        pt[3] = t0 + 1.0 * s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, 32);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf_val, yf_val;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);
    phase->optimizer->PrintLevel = 0;

    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    auto result = phase->returnTraj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(SolverTest, BrachistochroneSolveOnly) {
    // Solve (feasibility) without optimization
    constexpr double g = 9.81;

    int n_pts = 50;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = 10.0 * s;
        pt[1] = 10.0 - 5.0 * s;
        pt[2] = g * s * 1.0 * std::cos(1.0);
        pt[3] = 1.0 * s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, 16);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << 0.0, 10.0, 0.0, 0.0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << 10.0, 5.0;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);
    phase->optimizer->PrintLevel = 0;

    // solve() only finds feasibility
    auto status = phase->solve();
    // Should converge (feasible) — Brachistochrone is well-posed
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
}

TEST_F(SolverTest, ConvergenceFlagOrdering) {
    // Verify the severity ordering of convergence flags
    EXPECT_LT(PSIOPT::ConvergenceFlags::CONVERGED, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    EXPECT_LT(PSIOPT::ConvergenceFlags::ACCEPTABLE, PSIOPT::ConvergenceFlags::NOTCONVERGED);
    EXPECT_LT(PSIOPT::ConvergenceFlags::NOTCONVERGED, PSIOPT::ConvergenceFlags::DIVERGING);
}

///////////////////////////////////////////////////////////////////////////////
// VectorFunction as NLP-level constraint/objective
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, ConstraintInterfaceFromScalar) {
    auto args = Arguments<3>();
    auto n = args.norm();
    GenericFunction<-1, 1> gf(n);
    ConstraintInterface ci(gf);
    EXPECT_EQ(ci.IRows(), 3);
    EXPECT_EQ(ci.ORows(), 1);
}

TEST_F(SolverTest, ObjectiveInterfaceFromScalar) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    GenericFunction<-1, 1> gf(sn);
    ObjectiveInterface oi(gf);
    EXPECT_EQ(oi.IRows(), 3);
    EXPECT_EQ(oi.ORows(), 1);
}

TEST_F(SolverTest, ConstraintInterfaceCopy) {
    auto args = Arguments<4>();
    auto f = 2.0 * args;
    GenericFunction<-1, -1> gf(f);
    ConstraintInterface ci1(gf);
    ConstraintInterface ci2(ci1);
    EXPECT_EQ(ci2.IRows(), 4);
    EXPECT_EQ(ci2.ORows(), 4);
}

///////////////////////////////////////////////////////////////////////////////
// Helper: build a Brachistochrone phase for solver/Jet tests
///////////////////////////////////////////////////////////////////////////////

static std::shared_ptr<ODEPhase<BrachODE>> make_brach_solver_phase(int n_segs = 16) {
    constexpr double g = 9.81;
    int n_pts = n_segs * 3 + 1;
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = 10.0 * s;
        pt[1] = 10.0 - 5.0 * s;
        pt[2] = g * s * 1.0 * std::cos(1.0);
        pt[3] = 1.0 * s;
        pt[4] = 1.0;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, n_segs);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << 0.0, 10.0, 0.0, 0.0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << 10.0, 5.0;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);
    phase->optimizer->PrintLevel = 0;

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Jet batch runner
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, JetMapPrebuiltProblems) {
    // Create 3 identical Brachistochrone phases, solve via Jet::map
    std::vector<std::shared_ptr<ODEPhase<BrachODE>>> phases;
    for (int i = 0; i < 3; ++i) {
        auto p = make_brach_solver_phase(16);
        p->JetJobMode = OptimizationProblemBase::JetJobModes::SolveOptimize;
        phases.push_back(p);
    }

    auto results = Jet::map(phases, 2, false);
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
    auto results = Jet::map(gen, args, 2, false);
    ASSERT_EQ(results.size(), 2u);
    for (int i = 0; i < 2; ++i) {
        auto traj = results[i]->returnTraj();
        double tf = traj.back()[3];
        EXPECT_NEAR(tf, 1.8013, 0.02) << "Jet single-gen problem " << i << " did not converge";
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

    auto results = Jet::map(genfuncs, args, genfidxes, 2, false);
    ASSERT_EQ(results.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        auto traj = results[i]->returnTraj();
        double tf = traj.back()[3];
        EXPECT_NEAR(tf, 1.8013, 0.02) << "Jet multi-gen problem " << i << " did not converge";
    }
}

///////////////////////////////////////////////////////////////////////////////
// NLP structure
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, NLPDimensionsConsistency) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->PrimalVars, 0);
    EXPECT_GT(nlp->EqualCons, 0);
    EXPECT_EQ(nlp->KKTdim, nlp->PrimalVars + nlp->EqualCons + 2 * nlp->InequalCons);
}

TEST_F(SolverTest, NLPSparsityNonEmpty) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->KKTcoeffRows.size(), 0);
    EXPECT_GT(nlp->KKTcoeffCols.size(), 0);
    EXPECT_EQ(nlp->numKKTElems, nlp->numUserKKTElems + nlp->numSolverKKTElems);
}

///////////////////////////////////////////////////////////////////////////////
// Additional PSIOPT tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(SolverTest, PrintLevelZeroSilent) {
    auto phase = make_brach_solver_phase(16);
    phase->optimizer->PrintLevel = 0;
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);
}
