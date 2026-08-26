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

// oc_test_utils.h (tests/cpp/optimal_control/) supplies make_brach_phase(),
// reused here (rather than duplicated) for the one AMR-through-solve() case
// below -- it needs a real Phase with a real mesh, which a bare VF problem
// cannot stand in for.
#include "oc_test_utils.h"

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
using tycho::solvers::SolveOptions;
using tycho::solvers::SolveResult;
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

///////////////////////////////////////////////////////////////////////////////
// The staged solve() pipeline on BackendProblemBase (M5 Task 4): the refusal
// matrix, presolve/main/polish stage sequencing, the warm-start stamp
// pre-check, the per-engine concurrency latch, and last_result() caching.
// Exercised entirely against the VF problem (OptimizationProblem) --
// Phase/OCP get the same hooks, but their own ctest slices (run through the
// OLD, untouched surface) are the coverage for those two.
///////////////////////////////////////////////////////////////////////////////

TEST(SolvePipeline, RefusalMatrixNamesBothParts) {
    auto prob = solve_pipeline_build_tiny_nlp();
    InteriorPointSolver main_engine;
    InteriorPointSolver polish_engine;
    EngineRef main_ref = &main_engine;
    EngineRef polish_ref = &polish_engine;

    // Row 1: polish= paired with mode=Feasible.
    {
        SolveOptions opts;
        opts.mode = Mode::Feasible;
        opts.polish = &polish_ref;
        try {
            prob->solve(main_ref, opts);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument &e) {
            EXPECT_STREQ(e.what(),
                         "polish= is an optimality refinement; it cannot follow mode=Feasible");
        }
    }
    // Row 2a: presolve=true paired with mode=Feasible.
    {
        SolveOptions opts;
        opts.mode = Mode::Feasible;
        opts.presolve = true;
        try {
            prob->solve(main_ref, opts);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument &e) {
            EXPECT_STREQ(e.what(),
                         "presolve= runs a feasibility stage; mode=Feasible already is one");
        }
    }
    // Row 2b: presolve_engine= (without presolve=true) paired with
    // mode=Feasible -- the same refusal, not the "implies presolve" path.
    {
        SolveOptions opts;
        opts.mode = Mode::Feasible;
        opts.presolve_engine = &polish_ref;
        try {
            prob->solve(main_ref, opts);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument &e) {
            EXPECT_STREQ(e.what(),
                         "presolve= runs a feasibility stage; mode=Feasible already is one");
        }
    }
}

TEST(SolvePipeline, StagesRunInOrderAndAllReport) {
    auto prob = solve_pipeline_build_tiny_nlp();
    InteriorPointSolver main_engine;
    InteriorPointSolver polish_engine;
    EngineRef main_ref = &main_engine;
    EngineRef polish_ref = &polish_engine;

    SolveOptions opts;
    opts.mode = Mode::Optimal;
    opts.presolve = true;
    opts.polish = &polish_ref;

    SolveResult result = prob->solve(main_ref, opts);

    ASSERT_EQ(result.stages_.size(), 3u);
    EXPECT_EQ(result.stages_[0].role_, "presolve");
    EXPECT_EQ(result.stages_[1].role_, "main");
    EXPECT_EQ(result.stages_[2].role_, "polish");
    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(prob->active_variables_[0], 1.0, 1e-6);
}

TEST(SolvePipeline, NonConvergenceIsAValueNotAThrow) {
    // The active-inequality fixture: an interior-point method needs several
    // barrier-continuation iterations to drive complementarity down even on
    // a tiny problem, so capping at one iteration reliably leaves it short.
    auto prob = solve_pipeline_build_inequality_nlp();
    InteriorPointSolver ipm;
    ipm.settings().max_iters_ = 1;
    EngineRef ref = &ipm;

    SolveResult result;
    ASSERT_NO_THROW(result = prob->solve(ref));
    EXPECT_EQ(result.flag_, tycho::ConvergenceFlags::NOTCONVERGED);
    ASSERT_FALSE(result.stages_.empty());
    EXPECT_FALSE(result.converged());
}

TEST(SolvePipeline, WarmStampMismatchRefusesNamingBothKeys) {
    // prob A: one variable, one equality row.
    auto prob_a = solve_pipeline_build_tiny_nlp();
    InteriorPointSolver engine_a;
    EngineRef ref_a = &engine_a;
    SolveResult result_a = prob_a->solve(ref_a);
    ASSERT_TRUE(result_a.converged());

    // prob B: one variable, one INEQUALITY row -- a differently-shaped
    // declaration, so its DeclarationKey differs from prob A's.
    auto prob_b = solve_pipeline_build_inequality_nlp();
    InteriorPointSolver engine_b;
    EngineRef ref_b = &engine_b;

    SolveOptions opts;
    opts.warm = &result_a.warm_;

    try {
        prob_b->solve(ref_b, opts);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("does not match"), std::string::npos);
        // Names both keys: two distinct "digest=" occurrences.
        const std::size_t first = what.find("digest=");
        ASSERT_NE(first, std::string::npos);
        const std::size_t second = what.find("digest=", first + 1);
        EXPECT_NE(second, std::string::npos);
    }
}

TEST(SolvePipeline, WarmSeededSolveMatchesStagedEngineRun) {
    // Source run: an ordinary solve, whose exported warm currency seeds the
    // comparison below.
    auto prob_source = solve_pipeline_build_tiny_nlp();
    InteriorPointSolver source_engine;
    EngineRef source_ref = &source_engine;
    SolveResult source_result = prob_source->solve(source_ref);
    ASSERT_TRUE(source_result.converged());

    // Pipeline path: a fresh problem/engine pair, warm-seeded through
    // solve()'s opts.warm.
    auto prob_pipeline = solve_pipeline_build_tiny_nlp();
    prob_pipeline->transcribe();
    InteriorPointSolver pipeline_engine;
    EngineRef pipeline_ref = &pipeline_engine;
    SolveOptions opts;
    opts.warm = &source_result.warm_;
    SolveResult pipeline_result = prob_pipeline->solve(pipeline_ref, opts);

    // Manual path: a fresh problem/engine pair, driven directly through
    // stage_warm_start + optimize -- the same two calls
    // run_engine_stage/fill_ipm_stage makes internally.
    auto prob_manual = solve_pipeline_build_tiny_nlp();
    prob_manual->transcribe();
    InteriorPointSolver manual_engine;
    manual_engine.set_nlp(prob_manual->nlp_);
    manual_engine.stage_warm_start(source_result.warm_);
    const Eigen::VectorXd x0 = prob_manual->active_variables_;
    const Eigen::VectorXd manual_primal = manual_engine.optimize(x0);

    ASSERT_EQ(pipeline_result.stages_.size(), 1u);
    EXPECT_EQ(pipeline_result.flag_, manual_engine.result().converge_flag_);
    EXPECT_NEAR(pipeline_result.stages_.back().objective_, manual_engine.result().obj_val_, 1e-12);
    ASSERT_EQ(prob_pipeline->active_variables_.size(), manual_primal.size());
    for (int i = 0; i < manual_primal.size(); ++i) {
        EXPECT_NEAR(prob_pipeline->active_variables_[i], manual_primal[i], 1e-12);
    }
}

TEST(SolvePipeline, ConcurrentEngineUseRefuses) {
    // A reentrant solve() call on the SAME engine, made from inside the
    // outer call's own accept_stage() hook -- deterministic and
    // single-threaded, unlike a real concurrent-thread race, but it exercises
    // exactly what the latch defends against: the same engine identity
    // inside two overlapping solve() calls at once (the outer call's
    // latches are still held; it has not returned yet).
    class ReentrantProblem : public OptimizationProblem {
      public:
        EngineRef *reentry_engine_ = nullptr;

      protected:
        void accept_stage(const StageOutput &out) override {
            OptimizationProblem::accept_stage(out);
            if (this->reentry_engine_ != nullptr) {
                EngineRef *saved = this->reentry_engine_;
                this->reentry_engine_ = nullptr;
                EXPECT_THROW(this->solve(*saved), std::invalid_argument);
            }
        }
    };

    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    ReentrantProblem prob;
    prob.set_vars(Eigen::VectorXd::Constant(1, 0.0));
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob.add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
        prob.add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
    }

    InteriorPointSolver ipm;
    EngineRef ref = &ipm;
    prob.reentry_engine_ = &ref;

    SolveResult result = prob.solve(ref);
    EXPECT_TRUE(result.converged());
}

TEST(SolvePipeline, LastResultCachesTheReturn) {
    auto prob = solve_pipeline_build_tiny_nlp();
    EXPECT_THROW(prob->last_result(), std::logic_error);

    InteriorPointSolver ipm;
    EngineRef ref = &ipm;
    SolveResult result = prob->solve(ref);

    const SolveResult &cached = prob->last_result();
    EXPECT_EQ(cached.flag_, result.flag_);
    ASSERT_EQ(cached.stages_.size(), result.stages_.size());
    EXPECT_EQ(cached.stages_.back().objective_, result.stages_.back().objective_);
}

///////////////////////////////////////////////////////////////////////////////
// Adaptive mesh refinement driven through the NEW solve() surface.
//
// Every other case above is a bare VF problem (OptimizationProblem), which
// never enables adaptive_mesh_enabled() -- run_adaptive_mesh() and the
// shared BackendProblemBase::run_amr_loop() it delegates to are otherwise
// verified by compilation only. This drives a real Phase, with a real mesh,
// through solve(engine, opts), and pins the one property a missing
// per-iteration re-transcription would break: the TRANSCRIBED problem's own
// dimension must grow along with the mesh, not stay frozen at the coarse
// start while num_defects_/active_traj_ silently move on without it.
///////////////////////////////////////////////////////////////////////////////

TEST(SolvePipeline, AdaptiveMeshRefinesThroughNewSolvePipeline) {
    // Coarse start (8 segments) + a tight mesh tolerance: the same
    // combination test_mesh_refinement.cpp's MeshRefinementIterates uses to
    // force at least one real refinement within a handful of iterations.
    auto phase = TychoTest::make_brach_phase(50, 8);
    phase->optimizer_->set_print_level(0);
    phase->print_mesh_info_ = false;
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-7);
    phase->set_max_mesh_iters(4);

    phase->transcribe();
    ASSERT_TRUE(phase->nlp_);
    const int coarse_primal_vars = phase->nlp_->primal_vars_;

    InteriorPointSolver ipm;
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.presolve = true; // mirrors the old solve_optimize() shape this mesh loop used

    SolveResult result = phase->solve(ref, opts);

    EXPECT_TRUE(result.converged());

    // The mesh loop actually iterated past iteration 0: a presolve stage,
    // then one main stage per mesh iteration (solve_only_first_ defaults
    // true, so presolve does not repeat).
    ASSERT_GE(result.stages_.size(), 3u);
    EXPECT_EQ(result.stages_.front().role_, "presolve");
    for (std::size_t i = 1; i < result.stages_.size(); ++i) {
        EXPECT_EQ(result.stages_[i].role_, "main");
    }
    // result.flag_ is the last appended stage's flag (also exercises the
    // pipeline's own cross-check between run_adaptive_mesh's return value
    // and result.stages_.back().flag_).
    EXPECT_EQ(result.flag_, result.stages_.back().flag_);

    // The property a missing per-iteration re-transcription breaks: the
    // TRANSCRIBED problem's dimension must have moved with the refined
    // mesh, not stayed pinned at the coarse start. Pre-fix, run_adaptive_mesh
    // never re-ran prepare_solve() between mesh iterations, so this would
    // still read coarse_primal_vars unchanged even though the mesh itself
    // (num_defects_/active_traj_) had already grown underneath it.
    ASSERT_TRUE(phase->nlp_);
    EXPECT_NE(phase->nlp_->primal_vars_, coarse_primal_vars);
}
