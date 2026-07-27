///////////////////////////////////////////////////////////////////////////////
// Backend-neutral NLP solve dispatch (OptimizationProblemBase::nlp_solver_).
//
// Every solve/optimize entry point routes through a single dispatch seam so an
// alternative NLP solver backend can intercept the transcribed NLP. These tests
// run in the default build (no Ipopt linked): they lock the default backend to
// the built-in solver, verify that selecting the Ipopt backend without build
// support fails loudly rather than silently falling back, and pin the sentinel
// defaults of the run-info record. Solve-path tests for the Ipopt backend
// itself live in the build-gated Ipopt test group.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/ipopt_backend.h"
#include "tycho/detail/solvers/optimization_problem.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include <Eigen/Core>

namespace ts = tycho::solvers;

using tycho::solvers::OptimizationProblem;
using TychoTest::make_brach_solver_phase;

namespace {

// A well-conditioned equality NLP: min x^2 s.t. x - 1 = 0, optimum x = 1,
// objective 1 -- the same smallest problem the inertia-regularization parity
// test drives through the public solve path.
std::unique_ptr<OptimizationProblem> build_dispatch_test_nlp() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 0.0));
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
    }
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(0);
    return prob;
}

TEST(NlpSolverDispatch, DefaultsToPsiopt) {
    auto prob = build_dispatch_test_nlp();
    EXPECT_EQ(prob->nlp_solver_, ts::NLPSolvers::psiopt);
    auto flag = prob->optimize();
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(prob->optimizer_->result().obj_val_, 1.0, 1e-6);
}

TEST(NlpSolverDispatch, IpoptWithoutBuildSupportThrows) {
    if (ts::ipopt_backend::available()) {
        GTEST_SKIP() << "built with Ipopt support";
    }
    auto prob = build_dispatch_test_nlp();
    prob->nlp_solver_ = ts::NLPSolvers::ipopt;
    EXPECT_THROW(prob->optimize(), std::runtime_error);
}

// The phase entry points route through the same seam, so selecting the Ipopt
// backend without build support fails there too rather than silently solving
// with the built-in solver.
TEST(NlpSolverDispatch, PhaseIpoptWithoutBuildSupportThrows) {
    if (ts::ipopt_backend::available()) {
        GTEST_SKIP() << "built with Ipopt support";
    }
    auto phase = make_brach_solver_phase(4);
    phase->nlp_solver_ = ts::NLPSolvers::ipopt;
    EXPECT_THROW(phase->solve(), std::runtime_error);
}

// Mesh refinement consumes the built-in solver's constraint residuals at the
// solution, so it is rejected up front for any other backend -- before a solve
// is attempted -- instead of refining on stale data.
TEST(NlpSolverDispatch, AdaptiveMeshWithIpoptRejected) {
    if (ts::ipopt_backend::available()) {
        GTEST_SKIP() << "built with Ipopt support";
    }
    auto phase = make_brach_solver_phase(4);
    phase->set_adaptive_mesh(true);
    phase->print_mesh_info_ = false;
    phase->nlp_solver_ = ts::NLPSolvers::ipopt;
    EXPECT_THROW(phase->solve(), std::invalid_argument);
}

// Ipopt is not reliably re-entrant, so a Jet batch element that selects it is
// rejected at the jet_run entry point -- before jet_initialize() touches the
// problem and before any solve begins. The guard reads the backend enum, which
// exists in every build, so this holds with or without Ipopt linked.
TEST(NlpSolverDispatch, JetRunWithIpoptBackendRejected) {
    auto prob = build_dispatch_test_nlp();
    prob->set_jet_job_mode(ts::OptimizationProblemBase::JetJobModes::Optimize);
    prob->nlp_solver_ = ts::NLPSolvers::ipopt;
    EXPECT_THROW(prob->jet_run(), std::invalid_argument);
}

// The same batch element with the built-in backend runs to convergence, so the
// rejection above is attributable to the backend selection and not to the Jet
// entry point itself.
TEST(NlpSolverDispatch, JetRunWithPsioptBackendRuns) {
    auto prob = build_dispatch_test_nlp();
    prob->set_jet_job_mode(ts::OptimizationProblemBase::JetJobModes::Optimize);
    EXPECT_EQ(prob->jet_run(), tycho::ConvergenceFlags::CONVERGED);
}

TEST(NlpSolverDispatch, RunInfoDefaultsAreSentinels) {
    ts::IpoptRunInfo info;
    EXPECT_FALSE(info.ran_);
    EXPECT_TRUE(info.status_.empty());
    EXPECT_TRUE(info.normalized_.empty());
    EXPECT_EQ(info.converge_flag_, tycho::ConvergenceFlags::NOTCONVERGED);
    EXPECT_EQ(info.iterations_, -1);
    EXPECT_DOUBLE_EQ(info.objective_, 0.0);
    EXPECT_DOUBLE_EQ(info.constraint_violation_, -1.0);
    EXPECT_DOUBLE_EQ(info.wall_time_s_, -1.0);
}

} // namespace
