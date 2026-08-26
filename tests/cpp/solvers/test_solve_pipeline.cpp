///////////////////////////////////////////////////////////////////////////////
// The engine handle layer: EngineRef naming, the SqpSolver/Ipopt
// feasibility-mode refusals, IpoptSolver's absent-build refusal,
// clone_prototype's settings-only-no-runtime-state contract, the
// MakeConstraint internal-fixing-row guard, and two end-to-end SQP
// convergence probes (equality-only, and an active inequality) that
// actually drive SqpModelAdapter through a real solve.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/engines.h"
#include "tycho/detail/solvers_vf/optimization_problem.h"

#include <tycho/vector_functions.h>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace {

using tycho::solvers::EngineRef;
using tycho::solvers::FixedVariableTreatments;
using tycho::solvers::InteriorPointSolver;
using tycho::solvers::IpoptSolver;
using tycho::solvers::Mode;
using tycho::solvers::NonLinearProgram;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::SqpSolver;
using tycho::solvers::StageOutput;

// A well-conditioned equality NLP: min x^2 s.t. x - 1 = 0, optimum x = 1. The
// smallest problem that can be transcribed and handed to a real
// InteriorPointSolver, needed only by PrototypeCloneCopiesOptionsNotState
// below (the other three cases never touch a real NLP).
std::unique_ptr<OptimizationProblem> solve_pipeline_build_tiny_nlp() {
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

// An active-inequality NLP: min x^2 s.t. 2 - x <= 0 (i.e. x >= 2), optimum
// x = 2, objective 4, lambda_i = 4 (grad f + Ji^T lambda_i - z = 0 at a free
// variable: 2x + (-1)*lambda_i = 0 => lambda_i = 2x = 4). Exercises the half
// of SqpModelAdapter an equality-only probe leaves cold: jac_i_, lambda_i,
// and the cI(x) <= 0 sense.
std::unique_ptr<OptimizationProblem> solve_pipeline_build_inequality_nlp() {
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
        prob->add_inequal_con(GenericFunction<-1, -1>(2.0 - x),
                              (Eigen::VectorXi(1) << 0).finished());
    }
    return prob;
}

// min x^2 with x fixed at 3.0 and fixed_variable_treatment_ = MakeConstraint,
// so that optimize() installs one internal fixing row ("x - 3 = 0") on
// prob->nlp_ via NonLinearProgram::configure_variable_treatment -- the
// artifact SqpFeasibleRefusesByName's sibling guard below refuses on.
std::unique_ptr<OptimizationProblem> solve_pipeline_build_make_constraint_nlp() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 0.0));
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->add_variable_bound(0, 3.0, 3.0);
    prob->optimizer_->settings().fixed_variable_treatment_ =
        FixedVariableTreatments::MakeConstraint;
    return prob;
}

} // namespace

TEST(SolvePipeline, EngineRefNamesAllThree) {
    InteriorPointSolver ipm;
    SqpSolver sqp;
    // name() is static on both handle classes and never dereferences the
    // pointer, so a null IpoptSolver* names the engine correctly even in a
    // build without Ipopt support (where a live instance cannot exist).
    IpoptSolver *no_ipopt_instance = nullptr;

    EngineRef ipm_ref = &ipm;
    EngineRef sqp_ref = &sqp;
    EngineRef ipopt_ref = no_ipopt_instance;

    EXPECT_STREQ(tycho::solvers::engine_name(ipm_ref), "InteriorPointSolver");
    EXPECT_STREQ(tycho::solvers::engine_name(sqp_ref), "SqpSolver");
    EXPECT_STREQ(tycho::solvers::engine_name(ipopt_ref), "Ipopt");
}

TEST(SolvePipeline, IpoptHandleRefusesWhenAbsent) {
    if (tycho::solvers::ipopt_backend::available()) {
        GTEST_SKIP() << "built with Ipopt support";
    }
    EXPECT_THROW(IpoptSolver(), std::runtime_error);
}

TEST(SolvePipeline, SqpFeasibleRefusesByName) {
    SqpSolver sqp;
    EngineRef ref = &sqp;
    // The refusal is checked before nlp/x0 are touched, so a null nlp and an
    // empty x0 are fine here.
    std::shared_ptr<NonLinearProgram> no_nlp;
    Eigen::VectorXd no_x0;

    try {
        tycho::solvers::run_engine_stage(ref, Mode::Feasible, no_nlp, no_x0, nullptr);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_STREQ(e.what(),
                     "the SQP engine has no feasibility-only mode; use the interior-point "
                     "engine for mode=Feasible");
    }
}

TEST(SolvePipeline, PrototypeCloneCopiesOptionsNotState) {
    auto prob = solve_pipeline_build_tiny_nlp();
    prob->transcribe();
    ASSERT_TRUE(prob->nlp_);

    InteriorPointSolver original;
    original.set_print_level(3);
    original.set_nlp(prob->nlp_);
    original.settings().max_iters_ = 777;

    std::unique_ptr<InteriorPointSolver> clone = tycho::solvers::clone_prototype(original);
    ASSERT_TRUE(clone);
    EXPECT_EQ(clone->settings().max_iters_, 777);
    EXPECT_EQ(clone->result().iter_num_, 0);

    original.optimize(Eigen::VectorXd::Constant(1, 0.0));
    EXPECT_GT(original.result().iter_num_, 0);

    // The clone never had set_nlp called on it and never ran a solve: it
    // stays exactly as cold as it was right after construction.
    EXPECT_EQ(clone->result().iter_num_, 0);
}

TEST(SolvePipeline, IpoptFeasibleRefusesByName) {
    if (!tycho::solvers::ipopt_backend::available()) {
        GTEST_SKIP() << "not built with Ipopt support";
    }
    IpoptSolver ipopt;
    EngineRef ref = &ipopt;
    // Same shape as SqpFeasibleRefusesByName: the refusal is checked before
    // nlp/x0 are touched.
    std::shared_ptr<NonLinearProgram> no_nlp;
    Eigen::VectorXd no_x0;

    try {
        tycho::solvers::run_engine_stage(ref, Mode::Feasible, no_nlp, no_x0, nullptr);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        EXPECT_STREQ(e.what(),
                     "the Ipopt backend has no feasibility-only mode; use the interior-point "
                     "engine for mode=Feasible");
    }
}

TEST(SolvePipeline, SqpRefusesMakeConstraintInternalFixingRows) {
    auto prob = solve_pipeline_build_make_constraint_nlp();
    // Materializes the internal fixing row on prob->nlp_ via
    // configure_variable_treatment(MakeConstraint, ...), run from inside
    // optimize()'s own transcribe-then-solve path.
    prob->optimize();
    ASSERT_TRUE(prob->nlp_);
    ASSERT_GT(prob->nlp_->internal_fixed_constraints(), 0);

    SqpSolver sqp;
    EngineRef ref = &sqp;
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 3.0);
    EXPECT_THROW(tycho::solvers::run_engine_stage(ref, Mode::Optimal, prob->nlp_, x0, nullptr),
                 std::invalid_argument);
}

TEST(SolvePipeline, SqpOptimalEqualityConverges) {
    auto prob = solve_pipeline_build_tiny_nlp();
    prob->transcribe();
    ASSERT_TRUE(prob->nlp_);

    SqpSolver sqp;
    EngineRef ref = &sqp;
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 0.0);

    const StageOutput out =
        tycho::solvers::run_engine_stage(ref, Mode::Optimal, prob->nlp_, x0, nullptr);

    EXPECT_EQ(out.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(out.primal_.size(), 1);
    EXPECT_NEAR(out.primal_[0], 1.0, 1e-6);
    EXPECT_NEAR(out.report_.objective_, 1.0, 1e-6);
    // Stationarity at x=1: grad f + Je^T lambda_e - z = 0, z = 0 (no active
    // bound) => 2*1 + 1*lambda_e = 0 => lambda_e = -2.
    ASSERT_EQ(out.eq_lmults_.size(), 1);
    EXPECT_NEAR(out.eq_lmults_[0], -2.0, 1e-4);
    EXPECT_EQ(out.report_.engine_name_, "SqpSolver");
}

TEST(SolvePipeline, SqpOptimalActiveInequalityConverges) {
    auto prob = solve_pipeline_build_inequality_nlp();
    prob->transcribe();
    ASSERT_TRUE(prob->nlp_);

    SqpSolver sqp;
    EngineRef ref = &sqp;
    // Deliberately infeasible start (2 - 0 = 2 > 0): exercises the solver's
    // own path to the boundary, not just a feasible-start confirmation.
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 0.0);

    const StageOutput out =
        tycho::solvers::run_engine_stage(ref, Mode::Optimal, prob->nlp_, x0, nullptr);

    EXPECT_EQ(out.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(out.primal_.size(), 1);
    EXPECT_NEAR(out.primal_[0], 2.0, 1e-4);
    EXPECT_NEAR(out.report_.objective_, 4.0, 1e-3);
    // Stationarity at x=2: grad f + Ji^T lambda_i - z = 0, z = 0 (no variable
    // bound) => 2*2 + (-1)*lambda_i = 0 => lambda_i = 4, and lambda_i >= 0
    // (feasible sign) confirms the active side.
    ASSERT_EQ(out.iq_lmults_.size(), 1);
    EXPECT_NEAR(out.iq_lmults_[0], 4.0, 1e-3);
    EXPECT_GE(out.iq_lmults_[0], 0.0);
}
