///////////////////////////////////////////////////////////////////////////////
// Trial-evaluation exception recovery: an NLP functional that throws at a
// trial point must be treated as a rejected line-search rung, not a fatal
// error. Committed-point evaluation failures remain fatal.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/optimization_problem.h"

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
    prob->optimizer_->set_print_level(0);
    return prob;
}

} // namespace

// One rung throws; the backtracked rung evaluates cleanly (budget exhausted)
// and the solve completes to the constrained optimum. Today this test dies
// with the raw evaluation exception.
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
TEST(EvalExceptionRecovery, ThrowingRungIsRejectedNotFatalLangMode) {
    auto prob = build_eval_except_nlp(1);
    prob->optimizer_->settings().opt_ls_mode_ = PSIOPT::LineSearchModes::LANG;
    auto flag = prob->optimize();
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    // LANG's acceptance test is a different Lagrangian ladder than AUGLANG's
    // (see ls_lang), so it converges to a looser neighborhood of the optimum
    // than the default-mode test above; the tolerance here is widened
    // accordingly. The property under test is the log wiring, not per-mode
    // convergence precision.
    EXPECT_NEAR(prob->optimizer_->result().obj_val_, 1.0, 1e-4);
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
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::merit;
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
    prob->optimizer_->set_print_level(0);
    try {
        prob->optimize();
        FAIL() << "expected the committed-point evaluation exception to propagate";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("trial point outside evaluation domain"),
                 std::string::npos)
            << "expected the raw fixture message to propagate unwrapped, got: " << e.what();
    }
}
