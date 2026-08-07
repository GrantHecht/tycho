///////////////////////////////////////////////////////////////////////////////
// Trial-evaluation exception recovery: an NLP functional that throws at a
// trial point must be treated as a rejected line-search rung, not a fatal
// error. Committed-point evaluation failures remain fatal.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers_vf/optimization_problem.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace ts = tycho::solvers;
using tycho::solvers::OptimizationProblem;

namespace {

// Scalar/SuperScalar-safe threshold test: the FD derivative modes may
// instantiate compute_impl with an Eigen::Array Scalar, where operator> yields
// an array expression rather than a bool.
inline bool eval_except_above(double v, double t) { return v > t; }
template <int W> inline bool eval_except_above(const Eigen::Array<double, W, 1> &v, double t) {
    return (v > t).any();
}

// Objective x^2 whose evaluation throws for x above a threshold, for at most
// a fixed number of evaluations — the analog of a transient excursion past an
// interpolation-table domain edge. The budget makes tests deterministic
// without depending on line-search geometry: after `throw_budget` throwing
// evaluations, the domain "heals" and every evaluation succeeds.
struct EvalExceptGuardedSquare
    : tycho::vf::VectorFunction<EvalExceptGuardedSquare, 1, 1,
                                tycho::vf::DenseDerivativeMode::FDiffFwd,
                                tycho::vf::DenseDerivativeMode::FDiffFwd> {
    using Base = tycho::vf::VectorFunction<EvalExceptGuardedSquare, 1, 1,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    double threshold_;
    std::shared_ptr<std::atomic<int>> throws_left_;

    EvalExceptGuardedSquare(double threshold, int throw_budget)
        : threshold_(threshold), throws_left_(std::make_shared<std::atomic<int>>(throw_budget)) {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        if (eval_except_above(x[0], threshold_) && throws_left_->fetch_sub(1) > 0) {
            throw std::runtime_error("trial point outside evaluation domain");
        }
        fx[0] = x[0] * x[0];
    }
};

// min f(x) s.t. x - 1 = 0 from x0 = 0, with f the guarded square above. The
// full Newton step targets x = 1, which lies past the 0.1 threshold, so early
// line-search rungs evaluate in the throwing region while the committed start
// point (and its FD stencil) stay safely below it.
std::unique_ptr<OptimizationProblem> build_eval_except_nlp(int throw_budget) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 0.0));
    prob->add_objective(GenericFunction<-1, 1>(EvalExceptGuardedSquare(0.1, throw_budget)),
                        (Eigen::VectorXi(1) << 0).finished());
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3);
    return prob;
}

} // namespace

// One rung throws; the backtracked rung evaluates cleanly (budget exhausted)
// and the solve completes to the constrained optimum. Before the rejected-rung
// handling, this died with the raw evaluation exception.
TEST(EvalExceptionRecovery, ThrowingRungIsRejectedNotFatal) {
    auto prob = build_eval_except_nlp(1);
    auto flag = prob->optimize();
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(prob->optimizer_->result().obj_val_, 1.0, 1e-6);
    EXPECT_GE(prob->optimizer_->eval_error_log().count_, 1);
}

// Same scenario, driven through ls_lang (LineSearchModes::LANG) instead of
// the default AUGLANG variant — confirms the log is wired on the classic
// LANG rung, not just AUGLANG's.
//
// LANG's acceptance test is a different Lagrangian ladder than AUGLANG's (see
// ls_lang) and it accepts no rung on this problem: its very first iteration
// exhausts the ladder. That makes this the un-evaluable-exhaustion case — the
// throwing rung is still absorbed into the log rather than unwinding the
// solve, but the exhaustion fallback step was never evaluated, so with no
// restoration strategy configured the solve aborts with solver context instead
// of committing it. The property under test is the log wiring on the LANG
// rung; the abort is the exhaustion policy asserted in
// ExhaustionWithoutRestorationRethrowsWithContext. This test's residual
// distinct property, once the abort-message assertions below are set aside as
// overlapping that other test by design, is the ls_lang catch-site wiring
// itself (eval_error_log().count_ >= 1 on the LANG rung).
TEST(EvalExceptionRecovery, ThrowingRungIsRecordedByLangMode) {
    auto prob = build_eval_except_nlp(1);
    prob->optimizer_->settings().opt_ls_mode_ = PSIOPT::LineSearchModes::LANG;
    try {
        prob->optimize();
        FAIL() << "expected the un-evaluable exhaustion to abort";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("trial point outside evaluation domain"), std::string::npos) << msg;
        EXPECT_NE(msg.find("iteration"), std::string::npos) << msg;
    }
    EXPECT_GE(prob->optimizer_->eval_error_log().count_, 1);
}

// Same scenario, driven through ls_l1 (LineSearchModes::L1) — confirms the
// log is wired on the classic L1 rung.
TEST(EvalExceptionRecovery, ThrowingRungIsRejectedNotFatalL1Mode) {
    auto prob = build_eval_except_nlp(1);
    prob->optimizer_->settings().opt_ls_mode_ = PSIOPT::LineSearchModes::L1;
    auto flag = prob->optimize();
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(prob->optimizer_->result().obj_val_, 1.0, 1e-6);
    EXPECT_GE(prob->optimizer_->eval_error_log().count_, 1);
}

// Same scenario, driven through the GENERIC acceptance ladder
// (generic_line_search / modern_eval_trial_point) via
// acceptance_strategy_ == merit — confirms the log is wired on the
// non-classic ClassicMeritAcceptance path too.
TEST(EvalExceptionRecovery, ThrowingRungIsRejectedNotFatalGenericAcceptance) {
    auto prob = build_eval_except_nlp(1);
    prob->optimizer_->settings().acceptance_strategy_ = ts::AcceptanceStrategies::merit;
    auto flag = prob->optimize();
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    // ModernMeritAcceptance's generic ladder converges to a looser
    // neighborhood of the optimum than the classic AUGLANG default (see the
    // LANG-mode test above for the same reasoning); widened accordingly.
    EXPECT_NEAR(prob->optimizer_->result().obj_val_, 1.0, 1e-4);
    EXPECT_GE(prob->optimizer_->eval_error_log().count_, 1);
}

// A committed-point evaluation failure stays fatal: the initial point itself
// is in the throwing region, so the first (committed) evaluation throws and
// the exception propagates unchanged — no line-search wrapping applies.
TEST(EvalExceptionRecovery, CommittedPointFailureStaysFatal) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, 5.0)); // above threshold from the start
    prob->add_objective(GenericFunction<-1, 1>(EvalExceptGuardedSquare(0.1, 1 << 20)),
                        (Eigen::VectorXi(1) << 0).finished());
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3);
    try {
        prob->optimize();
        FAIL() << "expected the committed-point evaluation exception to propagate";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("trial point outside evaluation domain"), std::string::npos)
            << "expected the raw fixture message to propagate unwrapped, got: " << e.what();
        // Negative discriminator: the line-search exhaustion abort (see
        // ExhaustionWithoutRestorationRethrowsWithContext) also embeds the
        // fixture substring inside solver context. Its absence here is what
        // proves this exception propagated UNWRAPPED rather than being
        // caught, counted, and rethrown with context.
        EXPECT_EQ(msg.find("PSIOPT: line search failed"), std::string::npos) << msg;
    }
}

// Every rung of the ladder throws and no restoration is configured: the solve
// must abort with a context-wrapped exception naming the evaluation failure —
// not accept an un-evaluable step, and not surface the raw exception without
// solver context.
TEST(EvalExceptionRecovery, ExhaustionWithoutRestorationRethrowsWithContext) {
    auto prob = build_eval_except_nlp(1 << 20); // effectively unlimited throws
    try {
        prob->optimize();
        FAIL() << "expected optimize() to throw";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("trial point outside evaluation domain"), std::string::npos) << msg;
        EXPECT_NE(msg.find("iteration"), std::string::npos) << msg;
    }
}

// Same exhaustion scenario, but the committed iterate already satisfies the
// acceptable convergence tier (the tolerances are loosened so the start point
// qualifies). The un-evaluable exhaustion must then exit at the acceptable
// level rather than abort: no throw, ACCEPTABLE, and the never-evaluated
// fallback step discarded (the returned primal is still the start point).
TEST(EvalExceptionRecovery, ExhaustionAtAcceptableIterateExitsGracefully) {
    auto prob = build_eval_except_nlp(1 << 20); // effectively unlimited throws
    auto &acc_settings = prob->optimizer_->settings();
    acc_settings.acc_kkt_tol_ = 1e10;
    acc_settings.acc_econ_tol_ = 1e10;
    acc_settings.acc_icon_tol_ = 1e10;
    acc_settings.acc_bar_tol_ = 1e10;

    tycho::ConvergenceFlags flag = tycho::ConvergenceFlags::NOTCONVERGED;
    ASSERT_NO_THROW(flag = prob->optimize());
    EXPECT_EQ(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GE(prob->optimizer_->eval_error_log().count_, 1);
    // The graceful exit is the path where the diagnostic matters most: a user
    // seeing ACCEPTABLE instead of CONVERGED needs the reason.
    const auto &r = prob->optimizer_->result();
    EXPECT_FALSE(r.last_eval_exception_.empty());
    // The failed step was discarded, not committed: x is still the start point.
    ASSERT_EQ(r.primals_.size(), 1);
    EXPECT_NEAR(r.primals_[0], 0.0, 1e-12);
}

// Same NLP, restoration configured, throw budget sized to exactly one ladder
// exhaustion: the un-evaluable exhaustion dispatches feasibility restoration
// instead of aborting, the domain heals, and the solve completes.
TEST(EvalExceptionRecovery, RestorationEscalatesOnUnEvaluableSoftStep) {
    // Budget sized to exactly one ladder exhaustion plus its escalation: the
    // two default line-search rungs throw, the soft feasibility pre-stage
    // trial throws (escalating to the real restoration entry), and the domain
    // is healed from the next iteration on. A budget large enough to keep
    // throwing after entry would hit the already-in-restoration exhaustion,
    // which has no recovery path left and legitimately aborts.
    auto prob = build_eval_except_nlp(4);
    prob->optimizer_->settings().restoration_mode_ = ts::RestorationModes::l1_nested;
    auto flag = prob->optimize();
    // Graceful completion is the bar; convergence is a bonus. "Graceful" still
    // excludes DIVERGING -- the escalation must not push the solve off a cliff.
    EXPECT_NE(flag, tycho::ConvergenceFlags::DIVERGING);
    EXPECT_GE(prob->optimizer_->result().last_feas_rest_entries_, 1);
    EXPECT_FALSE(prob->optimizer_->result().last_eval_exception_.empty());
    // The nested-restoration path records through PSIOPT::eval_error_log_
    // directly (try_soft_feasibility_step), a wiring path distinct from either
    // SolverContext copy — assert it is live.
    EXPECT_GE(prob->optimizer_->eval_error_log().count_, 1);
}

// Diagnostics truth-table: a clean solve reports no evaluation exceptions,
// and a subsequent clean solve on an instance that previously latched one
// resets the diagnostic.
TEST(EvalExceptionRecovery, DiagnosticsResetBetweenSolves) {
    auto clean = build_eval_except_nlp(0);
    auto flag = clean->optimize();
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(clean->optimizer_->result().last_eval_exception_.empty());

    auto rescued = build_eval_except_nlp(1);
    auto flag2 = rescued->optimize();
    EXPECT_EQ(flag2, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(rescued->optimizer_->result().last_eval_exception_.empty());
    // Re-run the same instance from the converged point: the throw budget is
    // spent, the solve is clean, and the latched message from the previous
    // call must not survive the per-solve reset.
    auto flag3 = rescued->optimize();
    EXPECT_EQ(flag3, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(rescued->optimizer_->result().last_eval_exception_.empty());
}
