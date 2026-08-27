///////////////////////////////////////////////////////////////////////////////
// Backend-neutral engine dispatch (tycho::solvers::EngineRef).
//
// M5 solve-API: backend selection is no longer a property stored on the
// problem (the old BackendProblemBase::nlp_solver_/NLPSolvers enum); it is
// simply which engine (InteriorPointSolver or IpoptSolver) the caller passes
// to solve(). These tests run in the default build (no Ipopt linked): they
// pin that constructing an IpoptSolver engine without ENABLE_IPOPT fails
// loudly at construction time rather than silently falling back, that
// adaptive mesh refinement refuses an engine that reports no constraint
// residuals (Ipopt is such an engine) before touching the host, and the
// sentinel defaults of the run-info record. Solve-path tests for the Ipopt
// backend itself live in the build-gated Ipopt test group.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/engines.h"
#include "tycho/detail/solvers/nlp_backend.h"
#include "tycho/detail/solvers_vf/optimization_problem.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace ts = tycho::solvers;

using tycho::solvers::OptimizationProblem;
using TychoTest::make_brach_solver_phase;

namespace {

// A well-conditioned equality NLP: min x^2 s.t. x - 1 = 0, optimum x = 1,
// objective 1 -- the same smallest problem the inertia-regularization parity
// test drives through the public solve path.
std::unique_ptr<OptimizationProblem> build_ipopt_dispatch_nlp() {
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
    return prob;
}

TEST(NlpSolverDispatch, InteriorPointSolverEngineSolves) {
    auto prob = build_ipopt_dispatch_nlp();
    ts::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    auto flag = prob->solve(&ipm).flag_;
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(ipm.result().obj_val_, 1.0, 1e-6);
}

// IpoptSolver's constructor is the refusal point, not a per-problem dispatch
// gate: any code path that would construct one without ENABLE_IPOPT fails
// here, before there is a problem or an engine reference to dispatch on.
TEST(NlpSolverDispatch, ConstructingIpoptSolverWithoutBuildSupportThrows) {
    if (ts::ipopt_backend::available()) {
        GTEST_SKIP() << "built with Ipopt support";
    }
    EXPECT_THROW(
        {
            ts::IpoptSolver ipopt;
            (void)ipopt;
        },
        std::runtime_error);
}

// Mesh refinement consumes an engine's constraint residuals at the solution,
// so run_amr_loop refuses any engine that does not report them -- before a
// solve is attempted -- instead of refining on stale data. Ipopt is such an
// engine (fill_ipopt_stage leaves StageOutput::eq_cons_/iq_cons_ empty).
// Constructing an IpoptSolver requires ENABLE_IPOPT (see
// ConstructingIpoptSolverWithoutBuildSupportThrows above), so this test only
// runs in a build configured with it; the build-gated Ipopt test group
// carries the rest of the Ipopt-specific solve-path coverage.
TEST(NlpSolverDispatch, AdaptiveMeshWithIpoptRejected) {
    if (!ts::ipopt_backend::available()) {
        GTEST_SKIP() << "not built with Ipopt support";
    }
    auto phase = make_brach_solver_phase(4);
    phase->set_adaptive_mesh(true);
    phase->print_mesh_info_ = false;
    ts::IpoptSolver ipopt;
    EXPECT_THROW(phase->solve(&ipopt), std::invalid_argument);
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
