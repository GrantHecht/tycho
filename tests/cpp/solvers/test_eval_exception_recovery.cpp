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
    EXPECT_THROW(prob->optimize(), std::runtime_error);
}
