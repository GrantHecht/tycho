///////////////////////////////////////////////////////////////////////////////
// InteriorPointSolver convergence tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>
#include <limits>

using namespace tycho;
using TychoTest::make_brach_solver_phase;
using TychoTest::SolverTest;

TEST_F(SolverTest, BrachistochroneEndToEnd) {
    auto phase = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm;
    auto status = phase->solve(&ipm, {.presolve = true}).flag_;
    EXPECT_EQ(status, tycho::ConvergenceFlags::CONVERGED);

    auto result = phase->return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(SolverTest, BrachistochroneSolveOnly) {
    auto phase = make_brach_solver_phase(16);
    tycho::solvers::InteriorPointSolver ipm;
    // solve() only finds feasibility
    auto status = phase->solve(&ipm, {.mode = tycho::solvers::Mode::Feasible}).flag_;
    // Should converge (feasible) -- Brachistochrone is well-posed.
    // ConvergenceFlags enum is ordered by severity: CONVERGED < ACCEPTABLE < NOTCONVERGED <
    // DIVERGING, so <= ACCEPTABLE accepts either CONVERGED or ACCEPTABLE.
    EXPECT_LE(status, tycho::ConvergenceFlags::ACCEPTABLE);
}

// The severity ordering of the convergence flags is a compile-time contract --
// every `status <= ACCEPTABLE` comparison in this suite and in the solver
// depends on it. Asserted where it is decided, not at run time.
static_assert(tycho::ConvergenceFlags::CONVERGED < tycho::ConvergenceFlags::ACCEPTABLE,
              "ConvergenceFlags must be ordered by severity: CONVERGED < ACCEPTABLE");
static_assert(tycho::ConvergenceFlags::ACCEPTABLE < tycho::ConvergenceFlags::NOTCONVERGED,
              "ConvergenceFlags must be ordered by severity: ACCEPTABLE < NOTCONVERGED");
static_assert(tycho::ConvergenceFlags::NOTCONVERGED < tycho::ConvergenceFlags::DIVERGING,
              "ConvergenceFlags must be ordered by severity: NOTCONVERGED < DIVERGING");

TEST_F(SolverTest, PrintLevelZeroConverges) {
    auto phase = make_brach_solver_phase(16);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(0);
    auto status = phase->solve(&ipm, {.presolve = true}).flag_;
    EXPECT_EQ(status, tycho::ConvergenceFlags::CONVERGED);
}

// =============================================================================
// Setter validation tests
// =============================================================================

TEST_F(SolverTest, ToleranceSetterRejectsInvalid) {
    InteriorPointSolver opt;
    // Negative
    EXPECT_THROW(opt.set_kkt_tol(-1.0), std::invalid_argument);
    // Zero
    EXPECT_THROW(opt.set_kkt_tol(0.0), std::invalid_argument);
    // NaN
    EXPECT_THROW(opt.set_kkt_tol(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    // Inf
    EXPECT_THROW(opt.set_kkt_tol(std::numeric_limits<double>::infinity()), std::invalid_argument);
    // Valid
    EXPECT_NO_THROW(opt.set_kkt_tol(1e-8));

    // One representative from each group
    EXPECT_THROW(opt.set_acc_econ_tol(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_acc_econ_tol(1e-4));
    EXPECT_THROW(opt.set_div_bar_tol(0.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_div_bar_tol(1e6));
}

TEST_F(SolverTest, MuSettersRejectInvalid) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_init_mu(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_init_mu(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_init_mu(0.01));
    EXPECT_THROW(opt.set_min_mu(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_max_mu(-1.0), std::invalid_argument);
}

TEST_F(SolverTest, QpThreadsRejectsInvalid) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_qp_threads(0), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_threads(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_threads(2));
}

TEST_F(SolverTest, ObjScaleRejectsZero) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_obj_scale(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_obj_scale(-1.0), std::invalid_argument); // negative flips dual signs
    EXPECT_NO_THROW(opt.set_obj_scale(2.0));
}

TEST_F(SolverTest, BoundFractionRejectsOutOfRange) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_bound_fraction(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_bound_fraction(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_bound_fraction(0.5));
}

TEST_F(SolverTest, AlphaRedRejectsBelowOne) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_alpha_red(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_alpha_red(2.0));
}

TEST_F(SolverTest, HpertParamsValidation) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_delta_h(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_incr_h(1.0), std::invalid_argument);
    EXPECT_THROW(opt.set_decr_h(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_decr_h(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_hpert_params(1e-5, 8.0, 0.333));
}

TEST_F(SolverTest, IntegerSetterValidation) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_max_iters(0), std::invalid_argument);
    EXPECT_THROW(opt.set_max_iters(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_max_iters(100));
    EXPECT_THROW(opt.set_max_acc_iters(0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_max_acc_iters(10));
    EXPECT_THROW(opt.set_max_ls_iters(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_max_ls_iters(0)); // 0 is valid (no line search)
    EXPECT_THROW(opt.set_print_level(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_print_level(0));
}

TEST_F(SolverTest, ObjScaleRejectsNonFinite) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_obj_scale(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(opt.set_obj_scale(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST_F(SolverTest, QpParamSetterValidation) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_qp_pivot_perturb(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_pivot_perturb(0));
    EXPECT_NO_THROW(opt.set_qp_pivot_perturb(13));
    EXPECT_THROW(opt.set_qp_ref_steps(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_ref_steps(0));
    EXPECT_THROW(opt.set_qp_par_solve(-1), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_par_solve(2), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_par_solve(0));
    EXPECT_NO_THROW(opt.set_qp_par_solve(1));
    // Pardiso weighted matching (iparm[12]) and MPS scaling (iparm[10]): 0/1
    // flags with the same range check. qp_scaling in particular is a live
    // performance knob (see the note on Settings::qp_scaling_), so its setter
    // must reject out-of-range values rather than pass them to Pardiso.
    EXPECT_THROW(opt.set_qp_matching(-1), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_matching(2), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_matching(0));
    EXPECT_NO_THROW(opt.set_qp_matching(1));
    EXPECT_THROW(opt.set_qp_scaling(-1), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_scaling(2), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_scaling(0));
    EXPECT_NO_THROW(opt.set_qp_scaling(1));
}

TEST_F(SolverTest, BoundPushNegSlackResetValidation) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_bound_push(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_bound_push(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_bound_push(1e-4));
    EXPECT_THROW(opt.set_neg_slack_reset(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_neg_slack_reset(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_neg_slack_reset(1e-12));
}

TEST_F(SolverTest, CompositeSetterDelegation) {
    InteriorPointSolver opt;
    opt.set_tols(1e-7, 2e-7, 3e-7, 4e-7);
    EXPECT_DOUBLE_EQ(opt.settings().kkt_tol_, 1e-7);
    EXPECT_DOUBLE_EQ(opt.settings().econ_tol_, 2e-7);
    EXPECT_DOUBLE_EQ(opt.settings().icon_tol_, 3e-7);
    EXPECT_DOUBLE_EQ(opt.settings().bar_tol_, 4e-7);

    opt.set_acc_tols(1e-4, 2e-4, 3e-4, 4e-4);
    EXPECT_DOUBLE_EQ(opt.settings().acc_kkt_tol_, 1e-4);
    EXPECT_DOUBLE_EQ(opt.settings().acc_econ_tol_, 2e-4);
    EXPECT_DOUBLE_EQ(opt.settings().acc_icon_tol_, 3e-4);
    EXPECT_DOUBLE_EQ(opt.settings().acc_bar_tol_, 4e-4);

    opt.set_all_max_iters(100, 20);
    EXPECT_EQ(opt.settings().max_iters_, 100);
    EXPECT_EQ(opt.settings().max_acc_iters_, 20);
}

TEST_F(SolverTest, CompositeSetterValidationPropagates) {
    InteriorPointSolver opt;
    EXPECT_THROW(opt.set_tols(-1, 1e-7, 1e-7, 1e-7), std::invalid_argument);
    EXPECT_THROW(opt.set_acc_tols(1e-4, -1, 1e-4, 1e-4), std::invalid_argument);
    EXPECT_THROW(opt.set_all_max_iters(0, 20), std::invalid_argument);
    EXPECT_THROW(opt.set_all_max_iters(100, 0), std::invalid_argument);
}

TEST_F(SolverTest, StringToEnumConverters) {
    EXPECT_EQ(InteriorPointSolver::strto_LineSearchMode("AUGLANG"),
              InteriorPointSolver::LineSearchModes::AUGLANG);
    EXPECT_THROW(InteriorPointSolver::strto_LineSearchMode("INVALID"), std::invalid_argument);
    EXPECT_EQ(InteriorPointSolver::strto_BarrierMode("LOQO"),
              InteriorPointSolver::BarrierModes::LOQO);
    EXPECT_THROW(InteriorPointSolver::strto_BarrierMode("NOPE"), std::invalid_argument);
    EXPECT_EQ(InteriorPointSolver::strto_OrderingMode("METIS"),
              InteriorPointSolver::QPOrderingModes::METIS);
    EXPECT_THROW(InteriorPointSolver::strto_OrderingMode("INVALID"), std::invalid_argument);
    EXPECT_EQ(InteriorPointSolver::strto_BestCriteriaMode("KKT"),
              InteriorPointSolver::BestCriteriaModes::KKT);
    EXPECT_THROW(InteriorPointSolver::strto_BestCriteriaMode("INVALID"), std::invalid_argument);
}

TEST_F(SolverTest, SettingsDefaultsRegression) {
    InteriorPointSolver::Settings s;
    EXPECT_EQ(s.max_iters_, 500);
    EXPECT_EQ(s.max_ls_iters_, 2);
    EXPECT_EQ(s.max_acc_iters_, 50);
    EXPECT_EQ(s.max_refac_, 15);
    EXPECT_DOUBLE_EQ(s.kkt_tol_, 1.0e-6);
    EXPECT_DOUBLE_EQ(s.econ_tol_, 1.0e-6);
    EXPECT_DOUBLE_EQ(s.icon_tol_, 1.0e-6);
    EXPECT_DOUBLE_EQ(s.bar_tol_, 1.0e-6);
    EXPECT_DOUBLE_EQ(s.init_mu_, 0.001);
    EXPECT_DOUBLE_EQ(s.obj_scale_, 1.0);
    EXPECT_DOUBLE_EQ(s.bound_fraction_, 0.99);
    EXPECT_EQ(s.print_level_, 0);

    // The fields that decide which algorithm actually runs. Every one of these
    // selects a shipped opt-in mechanism when flipped, so a changed default here
    // changes shipped behaviour for every user who never touched the setting --
    // exactly the change that must not land silently.
    EXPECT_EQ(s.acceptance_strategy_, tycho::solvers::AcceptanceStrategies::classic_merit);
    EXPECT_EQ(s.merit_penalty_rule_, tycho::solvers::MeritPenaltyRules::wmno);
    EXPECT_EQ(s.barrier_governor_, tycho::solvers::BarrierGovernors::classic_adaptive);
    EXPECT_FALSE(s.never_monotone_);
    EXPECT_EQ(s.restoration_mode_, tycho::solvers::RestorationModes::off);
    EXPECT_EQ(s.max_feas_rest_, 2);
    EXPECT_EQ(s.inertia_mode_, tycho::solvers::InertiaModes::classic);
    EXPECT_EQ(s.pd_step_strategy_, InteriorPointSolver::PDStepStrategies::PrimSlackEq_Iq);
    EXPECT_EQ(s.max_soc_, 0);           // SOC off
    EXPECT_EQ(s.ls_extended_iters_, 0); // extended backtracking off
    EXPECT_FALSE(s.watchdog_);          // watchdog off
}

// =============================================================================
// run_phase_sequence guard tests
// =============================================================================

TEST_F(SolverTest, OptimizeThrowsWithoutNlp) {
    InteriorPointSolver opt;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(10);
    EXPECT_THROW(opt.optimize(x), std::runtime_error);
}

TEST_F(SolverTest, OptimizeThrowsOnSizeMismatch) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false); // set up NLP without solving
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_nlp(phase->nlp_);
    Eigen::VectorXd wrong_size = Eigen::VectorXd::Zero(3);
    EXPECT_THROW(ipm.optimize(wrong_size), std::invalid_argument);
}

// =============================================================================
// Result accessor tests
// =============================================================================

TEST_F(SolverTest, BrachistochroneOptimizeSolve) {
    auto phase = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    // optimize_solve equivalence: OPT, then (conditionally, skipped when OPT
    // reported CONVERGED) a trailing Feasible-mode SOE run.
    auto status = phase->solve(&ipm).flag_;
    if (status != tycho::ConvergenceFlags::CONVERGED) {
        status = phase->solve(&ipm, {.mode = tycho::solvers::Mode::Feasible}).flag_;
    }
    EXPECT_LE(status, tycho::ConvergenceFlags::ACCEPTABLE);
}

TEST_F(SolverTest, ConditionalStepSkippedOnConvergence) {
    // This is the suite's only exact iteration-count equality across two
    // independent solves, and reproducing it needs BOTH pins: set_qp_threads(1)
    // alone is not enough, because the KKT assembly itself is still partitioned
    // across threads (num_partitions_ defaults to get_num_threads()*4), and
    // concurrent partition accumulation into shared slots drifts by ULPs run to
    // run (see the KKT-scatter test in test_kkt_canonical_lock.cpp, and
    // docs/dev/analysis/2026-07-pr9-pardiso-options.md for the Pardiso side).
    // set_num_partitions(1) forces single-partition assembly so the
    // accumulation order -- not just the factorization -- is deterministic.
    // optimize alone
    auto phase_opt = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm_opt;
    ipm_opt.set_print_level(3);
    ipm_opt.set_qp_threads(1);
    phase_opt->set_num_partitions(1);
    auto status_opt = phase_opt->solve(&ipm_opt).flag_;
    ASSERT_EQ(status_opt, tycho::ConvergenceFlags::CONVERGED);
    int opt_iters = ipm_opt.result().iter_num_;

    // The caller-side conditional retry that replaced the retired
    // optimize_solve() (see BrachistochroneOptimizeSolve above: a trailing
    // Feasible-mode solve, run only if the first call did not already
    // converge) must actually SKIP that trailing call once the first call
    // converges -- pinned here via an explicit second_call_ran flag rather
    // than an `if` whose branch this scenario alone can never prove was
    // live. The other branch (below) proves the flag is not just always
    // false by construction: it forces the first call short of convergence
    // and asserts the retry actually runs there.
    auto phase_skip = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm_skip;
    ipm_skip.set_print_level(3);
    ipm_skip.set_qp_threads(1);
    phase_skip->set_num_partitions(1);
    auto status_skip = phase_skip->solve(&ipm_skip).flag_;
    ASSERT_EQ(status_skip, tycho::ConvergenceFlags::CONVERGED);
    bool second_call_ran = false;
    if (status_skip != tycho::ConvergenceFlags::CONVERGED) {
        second_call_ran = true;
        status_skip = phase_skip->solve(&ipm_skip, {.mode = tycho::solvers::Mode::Feasible}).flag_;
    }
    EXPECT_FALSE(second_call_ran)
        << "the conditional retry must be skipped once the first call already converged";
    int skip_iters = ipm_skip.result().iter_num_;
    EXPECT_EQ(skip_iters, opt_iters)
        << "iteration count must match the optimize-only reference when the "
           "conditional retry is (correctly) skipped";

    // Now force the first call short of convergence, so the same conditional
    // structure takes its OTHER branch: the retry must actually run.
    auto phase_retry = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm_retry;
    ipm_retry.set_print_level(3);
    ipm_retry.set_qp_threads(1);
    ipm_retry.set_max_iters(3); // force NOTCONVERGED on the first call
    phase_retry->set_num_partitions(1);
    auto status_retry = phase_retry->solve(&ipm_retry).flag_;
    ASSERT_NE(status_retry, tycho::ConvergenceFlags::CONVERGED);
    bool retry_second_call_ran = false;
    if (status_retry != tycho::ConvergenceFlags::CONVERGED) {
        retry_second_call_ran = true;
        ipm_retry.set_max_iters(200); // the retry gets a real budget, unlike the starved first call
        status_retry =
            phase_retry->solve(&ipm_retry, {.mode = tycho::solvers::Mode::Feasible}).flag_;
    }
    EXPECT_TRUE(retry_second_call_ran)
        << "the conditional retry must actually run once the first call failed to converge";
    EXPECT_LE(status_retry, tycho::ConvergenceFlags::ACCEPTABLE);
}

TEST_F(SolverTest, BrachistochroneSolveOptimizeSolve) {
    auto phase = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    // solve_optimize_solve equivalence: a presolve+main call (SOE then OPT),
    // then (conditionally, skipped when OPT reported CONVERGED) a trailing
    // Feasible-mode SOE run.
    auto status = phase->solve(&ipm, {.presolve = true}).flag_;
    if (status != tycho::ConvergenceFlags::CONVERGED) {
        status = phase->solve(&ipm, {.mode = tycho::solvers::Mode::Feasible}).flag_;
    }
    EXPECT_LE(status, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_EQ(ipm.result().primals_.size(), phase->nlp_->primal_vars_);
}

TEST_F(SolverTest, ResultAccessorPopulatedAfterSolve) {
    auto phase = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm;
    auto status = phase->solve(&ipm, {.presolve = true}).flag_;
    EXPECT_EQ(status, tycho::ConvergenceFlags::CONVERGED);

    const auto &r = ipm.result();
    EXPECT_GT(r.iter_num_, 0);
    EXPECT_GT(r.obj_val_, 0.0);
    EXPECT_GT(r.total_time_, 0.0);
    EXPECT_EQ(r.primals_.size(), phase->nlp_->primal_vars_);

    EXPECT_GE(r.misc_time(), 0.0);

    // Cross-check: primals should produce the expected brachistochrone trajectory
    auto result = phase->return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(SolverTest, ResultResetBetweenCalls) {
    auto phase = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    phase->solve(&ipm, {.presolve = true});
    int first_iters = ipm.result().iter_num_;

    phase->solve(&ipm);
    const auto &r = ipm.result();
    int second_iters = r.iter_num_;

    // iter_num_ should reflect only the second call, not accumulated
    EXPECT_LT(second_iters, first_iters);

    // Additional fields should be valid and non-stale
    EXPECT_GT(r.obj_val_, 0.0);
    EXPECT_LE(r.converge_flag_, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GT(r.total_time_, 0.0);
    EXPECT_EQ(r.primals_.size(), phase->nlp_->primal_vars_);

    // Timing fields should reflect only the second call, not accumulated
    EXPECT_GT(r.func_time_, 0.0);
    EXPECT_GT(r.kkt_time_, 0.0);
    EXPECT_GE(r.misc_time(), 0.0);
}

// =============================================================================
// return_best and divergence tests
// =============================================================================

TEST_F(SolverTest, ReturnBestPreservesNonFinalIterate) {
    auto phase = make_brach_solver_phase(16);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    ipm.set_max_iters(3); // force NOTCONVERGED
    ipm.settings().return_best_ = true;
    ipm.settings().best_criteria_ = InteriorPointSolver::BestCriteriaModes::ECONS;

    auto status = phase->solve(&ipm).flag_;
    EXPECT_EQ(status, tycho::ConvergenceFlags::NOTCONVERGED);

    const auto &r = ipm.result();
    EXPECT_GT(r.primals_.size(), 0u);
    EXPECT_EQ(r.primals_.size(), phase->nlp_->primal_vars_);
    EXPECT_GT(r.obj_val_, 0.0);

    // Verify without return_best for comparison — both should produce valid primals
    auto phase2 = make_brach_solver_phase(16);
    tycho::solvers::InteriorPointSolver ipm2;
    ipm2.set_print_level(3);
    ipm2.set_max_iters(3);
    ipm2.settings().return_best_ = false;

    phase2->solve(&ipm2);
    EXPECT_EQ(ipm2.result().primals_.size(), r.primals_.size());
}

TEST_F(SolverTest, DivergenceEarlyExitInPhaseSequence) {
    auto phase = make_brach_solver_phase(16);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    // Set divergence tolerances extremely tight — solver triggers DIVERGING quickly.
    // Must also lower convergence and acceptable tols to satisfy
    // the conv <= acc <= div cross-field invariant.
    ipm.set_tols(1e-22, 1e-22, 1e-22, 1e-22);
    ipm.set_acc_tols(1e-21, 1e-21, 1e-21, 1e-21);
    ipm.set_div_tols(1e-20, 1e-20, 1e-20, 1e-20);
    auto status = phase->solve(&ipm, {.presolve = true}).flag_;
    EXPECT_EQ(status, tycho::ConvergenceFlags::DIVERGING);
}

// =============================================================================
// String-based mode setter tests
// =============================================================================

TEST_F(SolverTest, StringModeSetters) {
    InteriorPointSolver opt;

    opt.set_opt_ls_mode("LANG");
    EXPECT_EQ(opt.settings().opt_ls_mode_, InteriorPointSolver::LineSearchModes::LANG);
    opt.set_opt_ls_mode("AUGLANG");
    EXPECT_EQ(opt.settings().opt_ls_mode_, InteriorPointSolver::LineSearchModes::AUGLANG);
    opt.set_opt_ls_mode("L1");
    EXPECT_EQ(opt.settings().opt_ls_mode_, InteriorPointSolver::LineSearchModes::L1);
    opt.set_opt_ls_mode("NOLS");
    EXPECT_EQ(opt.settings().opt_ls_mode_, InteriorPointSolver::LineSearchModes::NOLS);

    opt.set_soe_ls_mode("L1");
    EXPECT_EQ(opt.settings().soe_ls_mode_, InteriorPointSolver::LineSearchModes::L1);

    opt.set_opt_bar_mode("PROBE");
    EXPECT_EQ(opt.settings().opt_bar_mode_, InteriorPointSolver::BarrierModes::PROBE);
    opt.set_opt_bar_mode("LOQO");
    EXPECT_EQ(opt.settings().opt_bar_mode_, InteriorPointSolver::BarrierModes::LOQO);

    opt.set_soe_bar_mode("PROBE");
    EXPECT_EQ(opt.settings().soe_bar_mode_, InteriorPointSolver::BarrierModes::PROBE);

    opt.set_qp_ordering_mode("MINDEG");
    EXPECT_EQ(opt.settings().qp_ord_, InteriorPointSolver::QPOrderingModes::MINDEG);
    opt.set_qp_ordering_mode("METIS");
    EXPECT_EQ(opt.settings().qp_ord_, InteriorPointSolver::QPOrderingModes::METIS);

    opt.set_best_criteria("ECons");
    EXPECT_EQ(opt.settings().best_criteria_, InteriorPointSolver::BestCriteriaModes::ECONS);
    opt.set_best_criteria("ICons");
    EXPECT_EQ(opt.settings().best_criteria_, InteriorPointSolver::BestCriteriaModes::ICONS);
    opt.set_best_criteria("KKT");
    EXPECT_EQ(opt.settings().best_criteria_, InteriorPointSolver::BestCriteriaModes::KKT);
    opt.set_best_criteria("Obj");
    EXPECT_EQ(opt.settings().best_criteria_, InteriorPointSolver::BestCriteriaModes::OBJ);
}

// =============================================================================
// Composite setter delegation tests
// =============================================================================

TEST_F(SolverTest, DivTolsCompositeDelegation) {
    InteriorPointSolver opt;
    opt.set_div_tols(1e10, 2e10, 3e10, 4e10);
    EXPECT_DOUBLE_EQ(opt.settings().div_kkt_tol_, 1e10);
    EXPECT_DOUBLE_EQ(opt.settings().div_econ_tol_, 2e10);
    EXPECT_DOUBLE_EQ(opt.settings().div_icon_tol_, 3e10);
    EXPECT_DOUBLE_EQ(opt.settings().div_bar_tol_, 4e10);
}

// =============================================================================
// Result population — multipliers and constraints
// =============================================================================

TEST_F(SolverTest, MultiplierAndConstraintResultPopulation) {
    auto phase = make_brach_solver_phase(32);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    auto status = phase->solve(&ipm, {.presolve = true}).flag_;
    ASSERT_EQ(status, tycho::ConvergenceFlags::CONVERGED);

    const auto &r = ipm.result();

    // Equality multipliers and constraints should be populated and correctly sized
    EXPECT_EQ(r.eq_lmults_.size(), phase->nlp_->equal_cons_);
    EXPECT_EQ(r.eq_cons_.size(), phase->nlp_->equal_cons_);

    // Inequality constraints (from lu bounds on the control variable)
    if (phase->nlp_->inequal_cons_ > 0) {
        EXPECT_EQ(r.iq_lmults_.size(), phase->nlp_->inequal_cons_);
        EXPECT_EQ(r.iq_cons_.size(), phase->nlp_->inequal_cons_);
    }
}

// =============================================================================
// Settings::validate() tests
// =============================================================================

TEST_F(SolverTest, SettingsValidateAcceptsDefaults) {
    InteriorPointSolver::Settings s;
    EXPECT_NO_THROW(s.validate());
}

TEST_F(SolverTest, SettingsValidateCatchesCrossFieldInvariants) {
    InteriorPointSolver::Settings s;

    // min_mu > max_mu
    s.min_mu_ = 200.0;
    s.max_mu_ = 100.0;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.min_mu_ = 1e-12;
    s.max_mu_ = 100.0;

    // init_mu outside [min_mu, max_mu]
    s.init_mu_ = 200.0;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.init_mu_ = 1e-14; // below min_mu
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.init_mu_ = 0.001;

    // convergence tol > acceptable tol (all four families)
    s.kkt_tol_ = 1.0;
    s.acc_kkt_tol_ = 0.01;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.kkt_tol_ = 1e-6;
    s.acc_kkt_tol_ = 1e-2;

    s.econ_tol_ = 1.0;
    s.acc_econ_tol_ = 0.01;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.econ_tol_ = 1e-6;
    s.acc_econ_tol_ = 1e-3;

    s.icon_tol_ = 1.0;
    s.acc_icon_tol_ = 0.01;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.icon_tol_ = 1e-6;
    s.acc_icon_tol_ = 1e-3;

    s.bar_tol_ = 1.0;
    s.acc_bar_tol_ = 0.01;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.bar_tol_ = 1e-6;
    s.acc_bar_tol_ = 1e-3;

    // acceptable tol > divergence tol
    s.acc_kkt_tol_ = 1e20;
    s.div_kkt_tol_ = 1e15;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.acc_kkt_tol_ = 1e-2;
    s.div_kkt_tol_ = 1e15;

    s.acc_econ_tol_ = 1e20;
    s.div_econ_tol_ = 1e15;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.acc_econ_tol_ = 1e-3;
    s.div_econ_tol_ = 1e15;

    s.acc_icon_tol_ = 1e20;
    s.div_icon_tol_ = 1e15;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.acc_icon_tol_ = 1e-3;
    s.div_icon_tol_ = 1e15;

    s.acc_bar_tol_ = 1e20;
    s.div_bar_tol_ = 1e15;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.acc_bar_tol_ = 1e-3;
    s.div_bar_tol_ = 1e15;

    // max_refac_ = 0 disables the perturbation ladder outright (the base
    // factorization exhausts immediately on wrong inertia, routed through the
    // recovery chain as SINGULAR_KKT -- see interior_point_solver.cpp's kkt_exhausted
    // handling), so it is valid; negative remains invalid.
    s.max_refac_ = 0;
    EXPECT_NO_THROW(s.validate());
    s.max_refac_ = -1;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.max_refac_ = 15;

    // Invalid per-field value
    s.bound_fraction_ = 5.0;
    EXPECT_THROW(s.validate(), std::invalid_argument);
}
