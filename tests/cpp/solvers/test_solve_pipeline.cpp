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
using tycho::solvers::PhaseResult;
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

// A single-variable equality NLP that genuinely diverges under Mode::Feasible:
// exp(x) - 5 = 0, started at x = 1000. exp(1000) overflows to +inf in double
// precision, so the very first iterate's constraint residual (and the KKT
// infeasibility measure built from it) is already non-finite -- hven's own
// divergence check (`!std::isfinite(...)`) fires immediately, and the
// diverging iterate's primal/multiplier blocks captured into the completed
// warm-start payload are themselves non-finite. Needed only by
// DivergingPresolveDoesNotThrowOnMainStageHandoff below, to reach that
// payload through the real solve() pipeline rather than fabricating a
// WarmStartData by hand (warm_or_null is file-local to
// solve_pipeline.cpp).
std::unique_ptr<OptimizationProblem> solve_pipeline_build_diverging_nlp() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 1000.0));
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x.exp() - 5.0),
                            (Eigen::VectorXi(1) << 0).finished());
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

// min x^2 with x fixed at 3.0. The caller solves this with an engine whose
// fixed_variable_treatment_ = MakeConstraint, so that the solve installs one
// internal fixing row ("x - 3 = 0") on prob->nlp_ via
// NonLinearProgram::configure_variable_treatment -- the artifact
// SqpFeasibleRefusesByName's sibling guard below refuses on.
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
    return prob;
}

// A test-local Phase subclass exposing a way to overwrite the protected
// post-solve multiplier state directly (active_eq_lmults_ is protected on
// ODEPhaseBase), mirroring oc_test_utils.h's BrachSwitchTestPhase pattern.
// Used only by PhaseResultIsSnapshotNotView below, to prove
// fill_phase_results() copied its slices out rather than aliasing back into
// this phase's own mutable state.
struct SolvePipelinePhaseResultSnapshotPhase : ODEPhase<TychoTest::LinearODE> {
    using ODEPhase<TychoTest::LinearODE>::ODEPhase;
    void clobber_eq_lmults_for_test() {
        this->active_eq_lmults_ =
            Eigen::VectorXd::Constant(this->active_eq_lmults_.size(), 12345.0);
    }
};

// Builds a SolvePipelinePhaseResultSnapshotPhase with the exact same linear-dynamics
// setup as TychoTest::make_linear_phase() (x0=0, v0=1, t in [0, 1], 2
// defects) -- duplicated rather than reused because make_linear_phase()
// returns the plain ODEPhase<LinearODE> type, not this test-local subclass.
std::shared_ptr<SolvePipelinePhaseResultSnapshotPhase> solve_pipeline_make_snapshot_phase() {
    constexpr double x0 = 0.0, v0 = 1.0, t0 = 0.0, tf = 1.0;
    constexpr int n_pts = 5;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        double t = t0 + (tf - t0) * s;
        Eigen::VectorXd pt(3);
        pt[0] = x0 + v0 * (t - t0);
        pt[1] = v0;
        pt[2] = t;
        traj.push_back(pt);
    }

    TychoTest::LinearODE ode;
    auto phase =
        std::make_shared<SolvePipelinePhaseResultSnapshotPhase>(ode, TranscriptionModes::LGL3, traj,
                                                                /*nsegs=*/2);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(3, 0, 2);
    Eigen::VectorXd front_val(3);
    front_val << x0, v0, t0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(1);
    back_idx << 2;
    Eigen::VectorXd back_val(1);
    back_val << tf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    return phase;
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
    InteriorPointSolver ipm;
    ipm.settings().fixed_variable_treatment_ = FixedVariableTreatments::MakeConstraint;
    // Materializes the internal fixing row on prob->nlp_ via
    // configure_variable_treatment(MakeConstraint, ...), run from inside
    // solve()'s own transcribe-then-solve path.
    prob->solve(&ipm);
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
// The staged solve() pipeline on BackendProblemBase: the refusal
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

// Regression test (review finding I-1): a presolve stage that genuinely
// diverges must not crash the hand-off to the main stage. Before the fix,
// warm_or_null() only checked the exported payload's primal_ for
// non-emptiness, so a diverged presolve's non-finite bound_lmults_ (and
// primal_) were staged onto the main stage's engine via stage_warm_start(),
// which throws std::invalid_argument out of hven's own block validation
// (InteriorPointSolver::validate_warm_start_blocks) -- turning a value the
// pipeline's own contract promises ("a non-convergent stage is a value ...
// never a thrown exception") into an uncaught exception instead. With the
// fix, a non-finite export is treated exactly like an empty one: the main
// stage runs unseeded and reports its own flag as a value.
TEST(SolvePipeline, DivergingPresolveDoesNotThrowOnMainStageHandoff) {
    auto prob = solve_pipeline_build_diverging_nlp();
    InteriorPointSolver ipm;
    EngineRef ref = &ipm;

    SolveOptions opts;
    opts.mode = Mode::Optimal;
    opts.presolve = true;

    SolveResult result;
    ASSERT_NO_THROW(result = prob->solve(ref, opts));

    ASSERT_EQ(result.stages_.size(), 2u);
    EXPECT_EQ(result.stages_[0].role_, "presolve");
    EXPECT_EQ(result.stages_[0].flag_, tycho::ConvergenceFlags::DIVERGING);
    EXPECT_EQ(result.stages_[1].role_, "main");
    EXPECT_FALSE(result.converged());
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
    phase->print_mesh_info_ = false;
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-7);
    phase->set_max_mesh_iters(4);

    phase->transcribe();
    ASSERT_TRUE(phase->nlp_);
    const int coarse_primal_vars = phase->nlp_->primal_vars_;

    InteriorPointSolver ipm;
    ipm.set_print_level(0);
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

///////////////////////////////////////////////////////////////////////////////
// Per-phase result slices: fill_phase_results() on
// ODEPhaseBase (one PhaseResult, index 0, spanning the whole declared
// problem) and OptimalControlProblemBase (one per phase, index-keyed,
// reusing the OCP's own per-phase offset bookkeeping), plus the snapshot
// (copy, not view) guarantee both overrides make.
///////////////////////////////////////////////////////////////////////////////

TEST(SolvePipeline, SinglePhaseSolveHasExactlyOnePhaseResult) {
    auto phase = TychoTest::make_linear_phase();

    InteriorPointSolver ipm;
    ipm.set_print_level(0);
    EngineRef ref = &ipm;
    SolveResult result = phase->solve(ref);

    ASSERT_EQ(result.phases_.size(), 1u);
    const PhaseResult &pr = result.phases_[0];

    EXPECT_EQ(pr.index_, 0);
    EXPECT_EQ(pr.var_start_, 0);
    EXPECT_EQ(pr.eq_start_, 0);
    EXPECT_EQ(pr.iq_start_, 0);
    EXPECT_EQ(pr.var_count_, result.warm_.primal_.size());
    EXPECT_EQ(pr.eq_count_, result.warm_.eq_lmults_.size());
    EXPECT_EQ(pr.iq_count_, result.warm_.iq_lmults_.size());

    // A single Phase IS the whole declared problem, so its slice must equal
    // the full declared-space vectors exactly, not merely be sized like them.
    ASSERT_EQ(pr.eq_lmults_.size(), result.warm_.eq_lmults_.size());
    EXPECT_TRUE((pr.eq_lmults_.array() == result.warm_.eq_lmults_.array()).all());
    ASSERT_EQ(pr.iq_lmults_.size(), result.warm_.iq_lmults_.size());
    EXPECT_TRUE((pr.iq_lmults_.array() == result.warm_.iq_lmults_.array()).all());
    ASSERT_EQ(pr.bound_lmults_.size(), result.warm_.bound_lmults_.size());
    EXPECT_TRUE((pr.bound_lmults_.array() == result.warm_.bound_lmults_.array()).all());
}

TEST(SolvePipeline, MultiPhaseSlicesTileTheWholeSpace) {
    // A genuinely heterogeneous 2-phase OCP: different mesh sizes (so the
    // two phases' ranges differ in WIDTH, not just position -- a same-size
    // pair cannot catch an offset bug that slices both phases from 0), a
    // real objective (delta time, from make_brach_phase), the control's
    // native box turned into an inequality-constraint pair at every path
    // node (control_bound_as_inequality=true), plus one more deterministic
    // inequality and one more deterministic variable bound per phase.
    //
    // The inequality is a hard pin (two rows forming an equality) on the
    // Front control, at an arbitrary constant strictly inside the outer
    // [-0.1, 2.0] box -- a pin away from the true unconstrained optimum
    // forces a genuinely nonzero multiplier there (the same mechanism
    // solve_pipeline_build_make_constraint_nlp above already relies on).
    //
    // The bound is the SAME pin mechanism applied at the Back control
    // instead, via a fixed (lower==upper) native variable bound -- but the
    // default MakeParameter fixed-variable treatment ELIMINATES a fixed
    // variable from the reduced problem entirely (non_linear_program.h),
    // which zero-fills its bound multiplier by construction regardless of
    // how "away from the optimum" the pin is. RelaxBounds keeps it a
    // genuine two-sided bounded variable instead, so its z is real; the
    // solving ENGINE's own settings are what matter here -- there is no
    // problem-owned optimizer any more for a stray per-phase setting to
    // shadow it with.
    //
    // Together this guarantees eq_lmults_/iq_lmults_/bound_lmults_ are all
    // nonzero and phase-distinct, rather than vacuously all-zero.
    auto phase0 =
        TychoTest::make_brach_phase(/*n_pts=*/50, /*n_defects=*/6, TranscriptionModes::LGL3,
                                    /*control_bound_as_inequality=*/true);
    auto phase1 =
        TychoTest::make_brach_phase(/*n_pts=*/50, /*n_defects=*/8, TranscriptionModes::LGL3,
                                    /*control_bound_as_inequality=*/true);

    Eigen::VectorXi theta_idx(1);
    theta_idx << 4;
    {
        auto args = Arguments<1>();
        auto theta = args.coeff<0>();
        phase0->add_inequal_con(PhaseRegionFlags::Front, StackedOutputs{1.5 - theta, theta - 1.5},
                                theta_idx, ScaleModes::AUTO);
    }
    {
        auto args = Arguments<1>();
        auto theta = args.coeff<0>();
        phase1->add_inequal_con(PhaseRegionFlags::Front, StackedOutputs{1.0 - theta, theta - 1.0},
                                theta_idx, ScaleModes::AUTO);
    }
    phase0->add_lu_var_bound(PhaseRegionFlags::Back, 4, 0.8, 0.8);
    phase1->add_lu_var_bound(PhaseRegionFlags::Back, 4, 0.3, 0.3);

    OptimalControlProblemBase ocp;
    ocp.add_phase(phase0);
    ocp.add_phase(phase1);

    InteriorPointSolver ipm;
    ipm.set_print_level(0);
    ipm.settings().fixed_variable_treatment_ = FixedVariableTreatments::RelaxBounds;
    EngineRef ref = &ipm;
    SolveResult result = ocp.solve(ref);
    EXPECT_TRUE(result.converged());

    ASSERT_EQ(result.phases_.size(), 2u);
    const PhaseResult &p0 = result.phases_[0];
    const PhaseResult &p1 = result.phases_[1];

    EXPECT_EQ(p0.index_, 0);
    EXPECT_EQ(p1.index_, 1);

    // Ranges are disjoint and cover the whole declared space exactly. The
    // two phases have different mesh sizes, so this also pins that the
    // offsets track each phase's OWN width, not a shared/guessed one.
    EXPECT_NE(p0.var_count_, p1.var_count_);
    EXPECT_EQ(p0.var_start_, 0);
    EXPECT_EQ(p1.var_start_, p0.var_count_);
    EXPECT_EQ(p0.var_count_ + p1.var_count_, result.warm_.primal_.size());

    EXPECT_EQ(p0.eq_start_, 0);
    EXPECT_EQ(p1.eq_start_, p0.eq_count_);
    EXPECT_EQ(p0.eq_count_ + p1.eq_count_, result.warm_.eq_lmults_.size());

    EXPECT_EQ(p0.iq_start_, 0);
    EXPECT_EQ(p1.iq_start_, p0.iq_count_);
    EXPECT_EQ(p0.iq_count_ + p1.iq_count_, result.warm_.iq_lmults_.size());

    // Guard on the fixture itself: none of the three declared-space blocks
    // is vacuously all-zero. A zero vector would make the concatenation
    // checks below pass under any slicing bug, including "both phases slice
    // from offset 0" -- the exact failure mode this test exists to catch.
    ASSERT_GT(result.warm_.eq_lmults_.size(), 0);
    ASSERT_GT(result.warm_.iq_lmults_.size(), 0);
    ASSERT_GT(result.warm_.bound_lmults_.size(), 0);
    EXPECT_FALSE((result.warm_.eq_lmults_.array() == 0.0).all());
    EXPECT_FALSE((result.warm_.iq_lmults_.array() == 0.0).all());
    EXPECT_FALSE((result.warm_.bound_lmults_.array() == 0.0).all());

    // Each phase's OWN slice carries genuine nonzero content too -- not just
    // the union of the two.
    EXPECT_FALSE((p0.eq_lmults_.array() == 0.0).all());
    EXPECT_FALSE((p1.eq_lmults_.array() == 0.0).all());
    EXPECT_FALSE((p0.iq_lmults_.array() == 0.0).all());
    EXPECT_FALSE((p1.iq_lmults_.array() == 0.0).all());
    EXPECT_FALSE((p0.bound_lmults_.array() == 0.0).all());
    EXPECT_FALSE((p1.bound_lmults_.array() == 0.0).all());

    // Concatenating the slices reproduces the full declared-space vectors
    // exactly, unconditionally on all three blocks (each is non-empty per
    // the fixture guard above).
    Eigen::VectorXd eq_cat(result.warm_.eq_lmults_.size());
    eq_cat << p0.eq_lmults_, p1.eq_lmults_;
    EXPECT_TRUE((eq_cat.array() == result.warm_.eq_lmults_.array()).all());

    Eigen::VectorXd iq_cat(result.warm_.iq_lmults_.size());
    iq_cat << p0.iq_lmults_, p1.iq_lmults_;
    EXPECT_TRUE((iq_cat.array() == result.warm_.iq_lmults_.array()).all());

    Eigen::VectorXd bound_cat(result.warm_.bound_lmults_.size());
    bound_cat << p0.bound_lmults_, p1.bound_lmults_;
    EXPECT_TRUE((bound_cat.array() == result.warm_.bound_lmults_.array()).all());
}

TEST(SolvePipeline, PhaseResultIsSnapshotNotView) {
    auto phase = solve_pipeline_make_snapshot_phase();

    InteriorPointSolver ipm;
    ipm.set_print_level(0);
    EngineRef ref = &ipm;
    SolveResult result = phase->solve(ref);

    ASSERT_EQ(result.phases_.size(), 1u);
    ASSERT_GT(result.phases_[0].eq_count_, 0);
    const Eigen::VectorXd before = result.phases_[0].eq_lmults_;

    // Mutate the phase's own post-solve state directly. If fill_phase_results
    // had captured a view (an Eigen::Ref/Map, or an alias into
    // active_eq_lmults_) instead of a value copy, this would reach back into
    // the already-returned SolveResult.
    phase->clobber_eq_lmults_for_test();

    ASSERT_EQ(result.phases_[0].eq_lmults_.size(), before.size());
    EXPECT_TRUE((result.phases_[0].eq_lmults_.array() == before.array()).all());
    EXPECT_FALSE((result.phases_[0].eq_lmults_.array() == 12345.0).all());
}
