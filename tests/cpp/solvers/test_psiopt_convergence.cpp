///////////////////////////////////////////////////////////////////////////////
// PSIOPT convergence tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>
#include <limits>

using namespace tycho;
using namespace TychoTest;

TEST_F(SolverTest, BrachistochroneEndToEnd) {
    auto phase = make_brach_solver_phase(32);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    auto result = phase->return_traj();
    double tf = result.back()[3];
    EXPECT_NEAR(tf, 1.8013, 0.01);
}

TEST_F(SolverTest, BrachistochroneSolveOnly) {
    auto phase = make_brach_solver_phase(16);
    // solve() only finds feasibility
    auto status = phase->solve();
    // Should converge (feasible) -- Brachistochrone is well-posed.
    // ConvergenceFlags enum is ordered by severity: CONVERGED < ACCEPTABLE < NOTCONVERGED <
    // DIVERGING, so <= ACCEPTABLE accepts either CONVERGED or ACCEPTABLE.
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
}

TEST_F(SolverTest, ConvergenceFlagOrdering) {
    // Verify the severity ordering of convergence flags
    EXPECT_LT(PSIOPT::ConvergenceFlags::CONVERGED, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    EXPECT_LT(PSIOPT::ConvergenceFlags::ACCEPTABLE, PSIOPT::ConvergenceFlags::NOTCONVERGED);
    EXPECT_LT(PSIOPT::ConvergenceFlags::NOTCONVERGED, PSIOPT::ConvergenceFlags::DIVERGING);
}

TEST_F(SolverTest, PrintLevelZeroSilent) {
    auto phase = make_brach_solver_phase(16);
    phase->optimizer_->set_print_level(0);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);
}

// =============================================================================
// Setter validation tests
// =============================================================================

TEST_F(SolverTest, ToleranceSetterRejectsInvalid) {
    PSIOPT opt;
    // Negative
    EXPECT_THROW(opt.set_kkt_tol(-1.0), std::invalid_argument);
    // Zero
    EXPECT_THROW(opt.set_kkt_tol(0.0), std::invalid_argument);
    // NaN
    EXPECT_THROW(opt.set_kkt_tol(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    // Inf
    EXPECT_THROW(opt.set_kkt_tol(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    // Valid
    EXPECT_NO_THROW(opt.set_kkt_tol(1e-8));

    // One representative from each group
    EXPECT_THROW(opt.set_acc_econ_tol(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_acc_econ_tol(1e-4));
    EXPECT_THROW(opt.set_div_bar_tol(0.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_div_bar_tol(1e6));
}

TEST_F(SolverTest, MuSettersRejectInvalid) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_init_mu(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_init_mu(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_init_mu(0.01));
    EXPECT_THROW(opt.set_min_mu(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_max_mu(-1.0), std::invalid_argument);
}

TEST_F(SolverTest, QpThreadsRejectsInvalid) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_qp_threads(0), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_threads(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_threads(2));
}

TEST_F(SolverTest, ObjScaleRejectsZero) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_obj_scale(0.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_obj_scale(-1.0)); // negative = maximization
    EXPECT_NO_THROW(opt.set_obj_scale(2.0));
}

TEST_F(SolverTest, BoundFractionRejectsOutOfRange) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_bound_fraction(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_bound_fraction(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_bound_fraction(0.5));
}

TEST_F(SolverTest, AlphaRedRejectsBelowOne) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_alpha_red(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_alpha_red(2.0));
}

TEST_F(SolverTest, HpertParamsValidation) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_delta_h(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_incr_h(1.0), std::invalid_argument);
    EXPECT_THROW(opt.set_decr_h(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_decr_h(1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_hpert_params(1e-5, 8.0, 0.333));
}

TEST_F(SolverTest, IntegerSetterValidation) {
    PSIOPT opt;
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
    PSIOPT opt;
    EXPECT_THROW(opt.set_obj_scale(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(opt.set_obj_scale(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST_F(SolverTest, QpParamSetterValidation) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_qp_pivot_perturb(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_pivot_perturb(0));
    EXPECT_NO_THROW(opt.set_qp_pivot_perturb(13));
    EXPECT_THROW(opt.set_qp_ref_steps(-1), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_ref_steps(0));
    EXPECT_THROW(opt.set_qp_par_solve(-1), std::invalid_argument);
    EXPECT_THROW(opt.set_qp_par_solve(2), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_qp_par_solve(0));
    EXPECT_NO_THROW(opt.set_qp_par_solve(1));
}

TEST_F(SolverTest, BoundPushNegSlackResetValidation) {
    PSIOPT opt;
    EXPECT_THROW(opt.set_bound_push(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_bound_push(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_bound_push(1e-4));
    EXPECT_THROW(opt.set_neg_slack_reset(0.0), std::invalid_argument);
    EXPECT_THROW(opt.set_neg_slack_reset(-1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt.set_neg_slack_reset(1e-12));
}

TEST_F(SolverTest, CompositeSetterDelegation) {
    PSIOPT opt;
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
}

TEST_F(SolverTest, StringToEnumConverters) {
    EXPECT_EQ(PSIOPT::strto_LineSearchMode("AUGLANG"), PSIOPT::LineSearchModes::AUGLANG);
    EXPECT_THROW(PSIOPT::strto_LineSearchMode("INVALID"), std::invalid_argument);
    EXPECT_EQ(PSIOPT::strto_BarrierMode("LOQO"), PSIOPT::BarrierModes::LOQO);
    EXPECT_THROW(PSIOPT::strto_BarrierMode("NOPE"), std::invalid_argument);
    EXPECT_EQ(PSIOPT::strto_OrderingMode("METIS"), PSIOPT::QPOrderingModes::METIS);
    EXPECT_THROW(PSIOPT::strto_OrderingMode("INVALID"), std::invalid_argument);
    EXPECT_EQ(PSIOPT::strto_BestCriteriaMode("KKT"), PSIOPT::BestCriteriaModes::KKT);
    EXPECT_THROW(PSIOPT::strto_BestCriteriaMode("INVALID"), std::invalid_argument);
}

TEST_F(SolverTest, SettingsDefaultsRegression) {
    PSIOPT::Settings s;
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
}

// =============================================================================
// run_phase_sequence guard tests
// =============================================================================

TEST_F(SolverTest, OptimizeThrowsWithoutNlp) {
    PSIOPT opt;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(10);
    EXPECT_THROW(opt.optimize(x), std::runtime_error);
}

TEST_F(SolverTest, OptimizeThrowsOnSizeMismatch) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false); // set up NLP without solving
    Eigen::VectorXd wrong_size = Eigen::VectorXd::Zero(3);
    EXPECT_THROW(phase->optimizer_->optimize(wrong_size), std::invalid_argument);
}

// =============================================================================
// Result accessor tests
// =============================================================================

TEST_F(SolverTest, BrachistochroneOptimizeSolve) {
    auto phase = make_brach_solver_phase(32);
    phase->optimizer_->set_print_level(3);
    auto status = phase->optimize_solve();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
}

TEST_F(SolverTest, ConditionalStepSkippedOnConvergence) {
    // optimize alone
    auto phase_opt = make_brach_solver_phase(32);
    phase_opt->optimizer_->set_print_level(3);
    auto status_opt = phase_opt->optimize();
    ASSERT_EQ(status_opt, PSIOPT::ConvergenceFlags::CONVERGED);
    int opt_iters = phase_opt->optimizer_->result().iter_num_;

    // optimize_solve: the solve step is conditional, so if optimize converges,
    // the total iteration count should equal optimize-only.
    auto phase_os = make_brach_solver_phase(32);
    phase_os->optimizer_->set_print_level(3);
    auto status_os = phase_os->optimize_solve();
    EXPECT_LE(status_os, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    int os_iters = phase_os->optimizer_->result().iter_num_;

    EXPECT_EQ(os_iters, opt_iters)
        << "optimize_solve should skip the conditional solve when optimize converges";
}

TEST_F(SolverTest, BrachistochroneSolveOptimizeSolve) {
    auto phase = make_brach_solver_phase(32);
    phase->optimizer_->set_print_level(3);
    auto status = phase->solve_optimize_solve();
    EXPECT_LE(status, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    EXPECT_EQ(phase->optimizer_->result().primals_.size(), phase->nlp_->primal_vars_);
}

TEST_F(SolverTest, ResultAccessorPopulatedAfterSolve) {
    auto phase = make_brach_solver_phase(32);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    const auto &r = phase->optimizer_->result();
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
    phase->optimizer_->set_print_level(3);
    phase->solve_optimize();
    int first_iters = phase->optimizer_->result().iter_num_;

    phase->optimize();
    const auto &r = phase->optimizer_->result();
    int second_iters = r.iter_num_;

    // iter_num_ should reflect only the second call, not accumulated
    EXPECT_LT(second_iters, first_iters);

    // Additional fields should be valid and non-stale
    EXPECT_GT(r.obj_val_, 0.0);
    EXPECT_LE(r.converge_flag_, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GT(r.total_time_, 0.0);
    EXPECT_EQ(r.primals_.size(), phase->nlp_->primal_vars_);
}

// =============================================================================
// return_best and divergence tests
// =============================================================================

TEST_F(SolverTest, ReturnBestPreservesNonFinalIterate) {
    auto phase = make_brach_solver_phase(16);
    phase->optimizer_->set_print_level(3);
    phase->optimizer_->set_max_iters(3); // force NOTCONVERGED
    phase->optimizer_->settings().return_best_ = true;
    phase->optimizer_->settings().best_criteria_ = PSIOPT::BestCriteriaModes::ECONS;

    auto status = phase->optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::NOTCONVERGED);

    const auto &r = phase->optimizer_->result();
    EXPECT_GT(r.primals_.size(), 0u);
    EXPECT_EQ(r.primals_.size(), phase->nlp_->primal_vars_);
    EXPECT_GT(r.obj_val_, 0.0);

    // Verify without return_best for comparison — both should produce valid primals
    auto phase2 = make_brach_solver_phase(16);
    phase2->optimizer_->set_print_level(3);
    phase2->optimizer_->set_max_iters(3);
    phase2->optimizer_->settings().return_best_ = false;

    phase2->optimize();
    EXPECT_EQ(phase2->optimizer_->result().primals_.size(), r.primals_.size());
}

TEST_F(SolverTest, DivergenceEarlyExitInMultiPhase) {
    auto phase = make_brach_solver_phase(16);
    phase->optimizer_->set_print_level(3);
    // Set divergence tolerances extremely tight — solver triggers DIVERGING quickly
    phase->optimizer_->set_div_tols(1e-20, 1e-20, 1e-20, 1e-20);
    auto status = phase->solve_optimize();
    EXPECT_EQ(status, PSIOPT::ConvergenceFlags::DIVERGING);
}

// =============================================================================
// String-based mode setter tests
// =============================================================================

TEST_F(SolverTest, StringModeSetters) {
    PSIOPT opt;

    opt.set_opt_ls_mode("LANG");
    EXPECT_EQ(opt.settings().opt_ls_mode_, PSIOPT::LineSearchModes::LANG);
    opt.set_opt_ls_mode("AUGLANG");
    EXPECT_EQ(opt.settings().opt_ls_mode_, PSIOPT::LineSearchModes::AUGLANG);
    opt.set_opt_ls_mode("L1");
    EXPECT_EQ(opt.settings().opt_ls_mode_, PSIOPT::LineSearchModes::L1);
    opt.set_opt_ls_mode("NOLS");
    EXPECT_EQ(opt.settings().opt_ls_mode_, PSIOPT::LineSearchModes::NOLS);

    opt.set_soe_ls_mode("L1");
    EXPECT_EQ(opt.settings().soe_ls_mode_, PSIOPT::LineSearchModes::L1);

    opt.set_opt_bar_mode("PROBE");
    EXPECT_EQ(opt.settings().opt_bar_mode_, PSIOPT::BarrierModes::PROBE);
    opt.set_opt_bar_mode("LOQO");
    EXPECT_EQ(opt.settings().opt_bar_mode_, PSIOPT::BarrierModes::LOQO);

    opt.set_soe_bar_mode("PROBE");
    EXPECT_EQ(opt.settings().soe_bar_mode_, PSIOPT::BarrierModes::PROBE);

    opt.set_qp_ordering_mode("MINDEG");
    EXPECT_EQ(opt.settings().qp_ord_, PSIOPT::QPOrderingModes::MINDEG);
    opt.set_qp_ordering_mode("METIS");
    EXPECT_EQ(opt.settings().qp_ord_, PSIOPT::QPOrderingModes::METIS);

    opt.set_best_criteria("ECons");
    EXPECT_EQ(opt.settings().best_criteria_, PSIOPT::BestCriteriaModes::ECONS);
    opt.set_best_criteria("ICons");
    EXPECT_EQ(opt.settings().best_criteria_, PSIOPT::BestCriteriaModes::ICONS);
    opt.set_best_criteria("KKT");
    EXPECT_EQ(opt.settings().best_criteria_, PSIOPT::BestCriteriaModes::KKT);
    opt.set_best_criteria("Obj");
    EXPECT_EQ(opt.settings().best_criteria_, PSIOPT::BestCriteriaModes::OBJ);
}

// =============================================================================
// Composite setter delegation tests
// =============================================================================

TEST_F(SolverTest, UnaccTolsCompositeDelegation) {
    PSIOPT opt;
    opt.set_unacc_tols(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(opt.settings().unacc_kkt_tol_, 1.0);
    EXPECT_DOUBLE_EQ(opt.settings().unacc_econ_tol_, 2.0);
    EXPECT_DOUBLE_EQ(opt.settings().unacc_icon_tol_, 3.0);
    EXPECT_DOUBLE_EQ(opt.settings().unacc_bar_tol_, 4.0);
}

TEST_F(SolverTest, DivTolsCompositeDelegation) {
    PSIOPT opt;
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
    phase->optimizer_->set_print_level(3);
    auto status = phase->solve_optimize();
    ASSERT_EQ(status, PSIOPT::ConvergenceFlags::CONVERGED);

    const auto &r = phase->optimizer_->result();

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
    PSIOPT::Settings s;
    EXPECT_NO_THROW(s.validate());
}

TEST_F(SolverTest, SettingsValidateCatchesCrossFieldInvariants) {
    PSIOPT::Settings s;

    // min_mu > max_mu
    s.min_mu_ = 200.0;
    s.max_mu_ = 100.0;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.min_mu_ = 1e-12;
    s.max_mu_ = 100.0;

    // convergence tol > acceptable tol
    s.kkt_tol_ = 1.0;
    s.acc_kkt_tol_ = 0.01;
    EXPECT_THROW(s.validate(), std::invalid_argument);
    s.kkt_tol_ = 1e-6;
    s.acc_kkt_tol_ = 1e-2;

    // Invalid per-field value
    s.bound_fraction_ = 5.0;
    EXPECT_THROW(s.validate(), std::invalid_argument);
}
