///////////////////////////////////////////////////////////////////////////////
// Cross-formulation solve check.
//
// The same NLP instance, posed once through OptimizationProblem + a
// VectorFunction expression tree and once through an NLPProblem subclass fed
// to NLPSolver, must converge to the same point from the same start. The two
// paths are not required to agree bit-for-bit -- the KKT element order
// differs between the VF-transcribed triplets and the hand-supplied
// structure/value triplets the NLP adapter packs, so factorization pivots
// (and the iterate history) can differ -- only the solutions need to agree to
// solver tolerance.
//
// Problem: min (x0-1)^2 + (x1-2)^2  s.t.  x0^2 + x1^2 = 4,  x0 >= 0.3.
// The closest point on the circle to (1, 2) is (2/sqrt5, 4/sqrt5), so the
// bound is declared but numerically slack at the optimum -- it still
// exercises the native variable-bound path identically on both sides.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers_vf/optimization_problem.h"
#include <hven/model/nlp_problem.h>
#include <hven/model/nlp_solver.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <Eigen/Core>

using tycho::ConstEigenRef;
using tycho::solvers::OptimizationProblem;
// The solver-neutral NLP interface belongs to the solver library and carries no
// backend selection, so it is named where it lives rather than through tycho's
// solver namespace -- which is exactly the arm of this cross-check: a problem
// stated directly against the library, against the same problem stated as a
// VectorFunction expression tree on tycho's side.
using hven::solvers::NLPProblem;
using hven::solvers::NLPSolver;

namespace {

constexpr double kNlpCrossCheckInf = std::numeric_limits<double>::infinity();

Eigen::VectorXd nlp_cross_check_start() { return (Eigen::VectorXd(2) << 1.0, 1.0).finished(); }

// f = (x0-1)^2 + (x1-2)^2, g = x0^2 + x1^2 pinned to 4 via gl == gu, and a
// native lower bound x0 >= 0.3 -- see the file banner for the problem.
struct NlpCrossCheckProblem : NLPProblem {
    int num_vars() const override { return 2; }
    int num_cons() const override { return 1; }
    int num_jac_nonzeros() const override { return 2; }
    int num_hess_nonzeros() const override { return 2; } // diagonal only: g is separable

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << 0.3, -kNlpCrossCheckInf;
        xu << kNlpCrossCheckInf, kNlpCrossCheckInf;
        gl << 4.0;
        gu << 4.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        f = a * a + b * b;
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] * x[0] + x[1] * x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0;
        c << 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1;
        c << 0, 1;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * x[0];
        v[1] = 2.0 * x[1];
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd> lambda,
                   Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor + 2.0 * lambda[0];
        v[1] = 2.0 * obj_factor + 2.0 * lambda[0];
    }
    std::string name() const override { return "NlpCrossCheckProblem"; }
};

// Same problem posed through OptimizationProblem + VectorFunction expressions.
// The variable bound is declared straight on the NonLinearProgram between
// transcribe() and the solve (there is no Phase here to route it through) --
// the same pattern test_interior_point_solver_native_bounds.cpp's restoration harness uses.
Eigen::VectorXd nlp_cross_check_solve_vf(tycho::ConvergenceFlags &flag_out) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars(nlp_cross_check_start());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(
            GenericFunction<-1, 1>((x0 - 1.0) * (x0 - 1.0) + (x1 - 2.0) * (x1 - 2.0)),
            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 4.0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    prob.transcribe();
    prob.nlp_->set_variable_bound(0, 0.3, kNlpCrossCheckInf);
    prob.nlp_->make_nlp(prob.nlp_->primal_vars_, prob.nlp_->user_equal_cons_,
                        prob.nlp_->inequal_cons_);
    ipm.set_nlp(prob.nlp_);

    const Eigen::VectorXd x = ipm.optimize(nlp_cross_check_start());
    flag_out = ipm.result().converge_flag_;
    return x;
}

} // namespace

TEST(NlpCrossFormulation, VfAndNlpProblemAgreeToTolerance) {
    tycho::ConvergenceFlags vf_flag;
    const Eigen::VectorXd x_vf = nlp_cross_check_solve_vf(vf_flag);
    ASSERT_EQ(vf_flag, tycho::ConvergenceFlags::CONVERGED);

    NLPSolver solver(std::make_shared<NlpCrossCheckProblem>());
    solver.optimizer_->set_print_level(3);
    const auto nlp_flag = solver.optimize(nlp_cross_check_start());
    ASSERT_EQ(nlp_flag, tycho::ConvergenceFlags::CONVERGED);
    const Eigen::VectorXd x_nlp = solver.return_x();

    ASSERT_EQ(x_vf.size(), x_nlp.size());
    EXPECT_LT((x_vf - x_nlp).lpNorm<Eigen::Infinity>(), 1e-8);
}
