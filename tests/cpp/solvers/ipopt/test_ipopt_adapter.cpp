///////////////////////////////////////////////////////////////////////////////
// Ipopt backend: TNLP adapter over the transcribed NLP.
//
// This whole group is only compiled in builds configured with Ipopt support.
// It pins the adapter's derivative extraction (slot maps and the multiplier
// sign convention) against finite differences, the structural contract it
// reports to Ipopt, and the end-to-end solve path through the backend-neutral
// dispatch seam.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/ipopt/tnlp_adapter.h"
#include "tycho/detail/solvers/ipopt_backend.h"
#include "tycho/detail/solvers/optimization_problem.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace ts = tycho::solvers;

using tycho::solvers::OptimizationProblem;
using tycho::solvers::TychoTNLP;
using TychoTest::make_brach_solver_phase;

namespace {

using IpIndex = Ipopt::Index;
using IpNumber = Ipopt::Number;

///////////////////////////////////////////////////////////////////////////////
// Shared test problem
//
//   min  (x0 - 1)^2 + (x1 - 2)^2
//   s.t. x0^2 + x1^2 - 4 = 0
//        x0 - x1        <= 0
//
// A nonlinear objective, one nonlinear equality, one linear inequality. The
// solution is the point of the circle of radius 2 closest to (1, 2), i.e.
// 2*(1, 2)/sqrt(5) ~ (0.894, 1.789), which satisfies the inequality strictly;
// the optimal objective is (sqrt(5) - 2)^2.
///////////////////////////////////////////////////////////////////////////////

std::unique_ptr<OptimizationProblem> build_ipopt_adapter_problem(double x0 = 0.5, double x1 = 1.7) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    auto prob = std::make_unique<OptimizationProblem>();

    Eigen::VectorXd vars(2);
    vars << x0, x1;
    prob->set_vars(vars);

    Eigen::VectorXi idx(2);
    idx << 0, 1;

    {
        auto args = Arguments<2>();
        auto a = args.coeff<0>();
        auto b = args.coeff<1>();
        prob->add_objective(GenericFunction<-1, 1>((a - 1.0) * (a - 1.0) + (b - 2.0) * (b - 2.0)),
                            idx);
    }
    {
        auto args = Arguments<2>();
        auto a = args.coeff<0>();
        auto b = args.coeff<1>();
        prob->add_equal_con(GenericFunction<-1, -1>(a * a + b * b - 4.0), idx);
    }
    {
        auto args = Arguments<2>();
        auto a = args.coeff<0>();
        auto b = args.coeff<1>();
        prob->add_inequal_con(GenericFunction<-1, -1>(a - b), idx);
    }

    prob->optimizer_->set_print_level(0);
    return prob;
}

/// Optimal objective of the shared test problem, (sqrt(5) - 2)^2.
double ipopt_adapter_optimal_objective() {
    const double d = std::sqrt(5.0) - 2.0;
    return d * d;
}

/// Transcribe the problem and wrap its NLP in an adapter positioned at @p x.
Ipopt::SmartPtr<TychoTNLP> make_adapter(OptimizationProblem &prob, const Eigen::VectorXd &x) {
    prob.transcribe();
    return Ipopt::SmartPtr<TychoTNLP>(new TychoTNLP(prob.nlp_, x, 1.0));
}

/// Dense Jacobian assembled from the adapter's triplet structure and values.
Eigen::MatrixXd adapter_jacobian(TychoTNLP &tnlp, const Eigen::VectorXd &x, IpIndex n, IpIndex m,
                                 IpIndex nnz) {
    std::vector<IpIndex> rows(static_cast<std::size_t>(nnz));
    std::vector<IpIndex> cols(static_cast<std::size_t>(nnz));
    std::vector<IpNumber> vals(static_cast<std::size_t>(nnz));

    EXPECT_TRUE(tnlp.eval_jac_g(n, nullptr, true, m, nnz, rows.data(), cols.data(), nullptr));
    EXPECT_TRUE(tnlp.eval_jac_g(n, x.data(), true, m, nnz, nullptr, nullptr, vals.data()));

    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(m, n);
    for (std::size_t i = 0; i < vals.size(); ++i) {
        jac(rows[i], cols[i]) += vals[i];
    }
    return jac;
}

/// Dense symmetric Hessian assembled from the adapter's triplet structure and
/// values (the triplets carry one triangle, so off-diagonal entries mirror).
Eigen::MatrixXd adapter_hessian(TychoTNLP &tnlp, const Eigen::VectorXd &x, double obj_factor,
                                const Eigen::VectorXd &lambda, IpIndex n, IpIndex m, IpIndex nnz) {
    std::vector<IpIndex> rows(static_cast<std::size_t>(nnz));
    std::vector<IpIndex> cols(static_cast<std::size_t>(nnz));
    std::vector<IpNumber> vals(static_cast<std::size_t>(nnz));

    EXPECT_TRUE(tnlp.eval_h(n, nullptr, true, obj_factor, m, nullptr, true, nnz, rows.data(),
                            cols.data(), nullptr));
    EXPECT_TRUE(tnlp.eval_h(n, x.data(), true, obj_factor, m, lambda.data(), true, nnz, nullptr,
                            nullptr, vals.data()));

    Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(n, n);
    for (std::size_t i = 0; i < vals.size(); ++i) {
        hess(rows[i], cols[i]) += vals[i];
        if (rows[i] != cols[i]) {
            hess(cols[i], rows[i]) += vals[i];
        }
    }
    return hess;
}

/// Gradient of the Lagrangian obj_factor*f + lambda^T g, assembled from the
/// adapter's first derivatives only — independent of the Hessian path under
/// test, so it is a valid reference for it.
Eigen::VectorXd lagrangian_gradient(TychoTNLP &tnlp, const Eigen::VectorXd &x, double obj_factor,
                                    const Eigen::VectorXd &lambda, IpIndex n, IpIndex m,
                                    IpIndex nnz_jac) {
    std::vector<IpNumber> grad(static_cast<std::size_t>(n));
    EXPECT_TRUE(tnlp.eval_grad_f(n, x.data(), true, grad.data()));

    Eigen::VectorXd out = obj_factor * Eigen::Map<Eigen::VectorXd>(grad.data(), n);
    out += adapter_jacobian(tnlp, x, n, m, nnz_jac).transpose() * lambda;
    return out;
}

///////////////////////////////////////////////////////////////////////////////
// Objective that refuses to evaluate outside its domain, for the
// evaluation-error path. The guard is written against both the plain and the
// packed scalar types the finite-difference derivative modes evaluate with.
///////////////////////////////////////////////////////////////////////////////

constexpr double kIpoptAdapterDomainLimit = 1.0e3;
constexpr const char *kIpoptAdapterDomainError = "objective evaluated outside its domain";

inline bool ipopt_adapter_out_of_domain(double v) { return v > kIpoptAdapterDomainLimit; }

template <class Derived>
inline bool ipopt_adapter_out_of_domain(const Eigen::ArrayBase<Derived> &v) {
    return (v > kIpoptAdapterDomainLimit).any();
}

struct IpoptAdapterGuardedObjective
    : tycho::vf::VectorFunction<IpoptAdapterGuardedObjective, 1, 1,
                                tycho::vf::DenseDerivativeMode::FDiffFwd,
                                tycho::vf::DenseDerivativeMode::FDiffFwd> {
    using Base = tycho::vf::VectorFunction<IpoptAdapterGuardedObjective, 1, 1,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        if (ipopt_adapter_out_of_domain(x[0])) {
            throw std::runtime_error(kIpoptAdapterDomainError);
        }
        fx[0] = x[0] * x[0];
    }
};

} // namespace

///////////////////////////////////////////////////////////////////////////////

TEST(IpoptBackend, Available) { EXPECT_TRUE(ts::ipopt_backend::available()); }

// Finite-difference cross-check of every derivative the adapter reports. This
// is what pins the KKT slot maps and the multiplier sign convention: the
// gradient and Jacobian references come from the function values, and the
// Hessian reference comes from the Lagrangian gradient assembled out of those
// (already-verified) first derivatives.
TEST(IpoptBackend, FdCrossCheck) {
    auto prob = build_ipopt_adapter_problem();
    Eigen::VectorXd x(2);
    x << 0.5, 1.7;
    auto tnlp = make_adapter(*prob, x);

    IpIndex n = 0;
    IpIndex m = 0;
    IpIndex nnz_jac = 0;
    IpIndex nnz_hess = 0;
    Ipopt::TNLP::IndexStyleEnum style = Ipopt::TNLP::FORTRAN_STYLE;
    ASSERT_TRUE(tnlp->get_nlp_info(n, m, nnz_jac, nnz_hess, style));
    ASSERT_EQ(n, 2);
    ASSERT_EQ(m, 2);

    const double h = 1.0e-5;
    const double tol = 1.0e-6;

    auto eval_f_at = [&](const Eigen::VectorXd &p) {
        IpNumber f = 0.0;
        EXPECT_TRUE(tnlp->eval_f(n, p.data(), true, f));
        return f;
    };
    auto eval_g_at = [&](const Eigen::VectorXd &p) {
        std::vector<IpNumber> g(static_cast<std::size_t>(m));
        EXPECT_TRUE(tnlp->eval_g(n, p.data(), true, m, g.data()));
        return Eigen::Map<Eigen::VectorXd>(g.data(), m).eval();
    };

    // Objective gradient vs central differences of the objective.
    std::vector<IpNumber> grad(static_cast<std::size_t>(n));
    ASSERT_TRUE(tnlp->eval_grad_f(n, x.data(), true, grad.data()));
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        xp[j] += h;
        xm[j] -= h;
        const double fd = (eval_f_at(xp) - eval_f_at(xm)) / (2.0 * h);
        EXPECT_NEAR(grad[j], fd, tol * std::max(1.0, std::abs(fd))) << "grad_f entry " << j;
    }

    // Constraint Jacobian vs central differences of the constraints.
    const Eigen::MatrixXd jac = adapter_jacobian(*tnlp, x, n, m, nnz_jac);
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        xp[j] += h;
        xm[j] -= h;
        const Eigen::VectorXd fd = (eval_g_at(xp) - eval_g_at(xm)) / (2.0 * h);
        for (int i = 0; i < m; ++i) {
            EXPECT_NEAR(jac(i, j), fd[i], tol * std::max(1.0, std::abs(fd[i])))
                << "jac_g entry (" << i << "," << j << ")";
        }
    }

    // Hessian of the Lagrangian vs central differences of the Lagrangian
    // gradient, at a deliberately asymmetric (obj_factor, lambda).
    const double obj_factor = 0.7;
    Eigen::VectorXd lambda(2);
    lambda << 0.3, -1.1;

    const Eigen::MatrixXd hess = adapter_hessian(*tnlp, x, obj_factor, lambda, n, m, nnz_hess);
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        xp[j] += h;
        xm[j] -= h;
        const Eigen::VectorXd fd =
            (lagrangian_gradient(*tnlp, xp, obj_factor, lambda, n, m, nnz_jac) -
             lagrangian_gradient(*tnlp, xm, obj_factor, lambda, n, m, nnz_jac)) /
            (2.0 * h);
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(hess(i, j), fd[i], tol * std::max(1.0, std::abs(fd[i])))
                << "hessian entry (" << i << "," << j << ")";
        }
    }

    // The same numbers analytically: the objective and the equality both have
    // Hessian 2*I and the inequality is linear, so the Lagrangian Hessian is
    // (0.7 + 0.3)*2*I. A flipped multiplier sign would give (0.7 - 0.3)*2*I.
    EXPECT_NEAR(hess(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(hess(1, 1), 2.0, 1e-9);
    EXPECT_NEAR(hess(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(hess(1, 0), 0.0, 1e-9);
}

// The structural contract handed to Ipopt: problem dimensions, the bounds that
// encode "no variable bounds, equalities then inequalities", and triplet
// structures that stay inside the primal block and the triangle Ipopt expects.
TEST(IpoptBackend, StructuralShape) {
    auto prob = build_ipopt_adapter_problem();
    Eigen::VectorXd x(2);
    x << 0.5, 1.7;
    auto tnlp = make_adapter(*prob, x);

    const int primal_vars = prob->nlp_->primal_vars_;
    const int equal_cons = prob->nlp_->equal_cons_;
    const int inequal_cons = prob->nlp_->inequal_cons_;

    IpIndex n = 0;
    IpIndex m = 0;
    IpIndex nnz_jac = 0;
    IpIndex nnz_hess = 0;
    Ipopt::TNLP::IndexStyleEnum style = Ipopt::TNLP::FORTRAN_STYLE;
    ASSERT_TRUE(tnlp->get_nlp_info(n, m, nnz_jac, nnz_hess, style));

    EXPECT_EQ(n, primal_vars);
    EXPECT_EQ(m, equal_cons + inequal_cons);
    EXPECT_EQ(style, Ipopt::TNLP::C_STYLE);
    EXPECT_GT(nnz_jac, 0);
    EXPECT_GT(nnz_hess, 0);

    // Bounds: variables unbounded, equality rows pinned to zero, inequality
    // rows bounded above by zero only.
    std::vector<IpNumber> x_l(static_cast<std::size_t>(n));
    std::vector<IpNumber> x_u(static_cast<std::size_t>(n));
    std::vector<IpNumber> g_l(static_cast<std::size_t>(m));
    std::vector<IpNumber> g_u(static_cast<std::size_t>(m));
    ASSERT_TRUE(tnlp->get_bounds_info(n, x_l.data(), x_u.data(), m, g_l.data(), g_u.data()));
    for (int i = 0; i < n; ++i) {
        EXPECT_LE(x_l[i], -1.0e19);
        EXPECT_GE(x_u[i], 1.0e19);
    }
    for (int i = 0; i < equal_cons; ++i) {
        EXPECT_DOUBLE_EQ(g_l[i], 0.0);
        EXPECT_DOUBLE_EQ(g_u[i], 0.0);
    }
    for (int i = 0; i < inequal_cons; ++i) {
        EXPECT_LE(g_l[equal_cons + i], -1.0e19);
        EXPECT_DOUBLE_EQ(g_u[equal_cons + i], 0.0);
    }

    // Starting point is the point the adapter was constructed with.
    std::vector<IpNumber> x_start(static_cast<std::size_t>(n));
    ASSERT_TRUE(tnlp->get_starting_point(n, true, x_start.data(), false, nullptr, nullptr, m, false,
                                         nullptr));
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(x_start[i], x[i]);
    }

    // Jacobian triplets: inside the constraint rows and the primal columns —
    // no slack column leaks out of the KKT matrix.
    std::vector<IpIndex> jac_rows(static_cast<std::size_t>(nnz_jac));
    std::vector<IpIndex> jac_cols(static_cast<std::size_t>(nnz_jac));
    ASSERT_TRUE(
        tnlp->eval_jac_g(n, nullptr, true, m, nnz_jac, jac_rows.data(), jac_cols.data(), nullptr));
    for (IpIndex i = 0; i < nnz_jac; ++i) {
        EXPECT_GE(jac_rows[i], 0);
        EXPECT_LT(jac_rows[i], m);
        EXPECT_GE(jac_cols[i], 0);
        EXPECT_LT(jac_cols[i], n) << "Jacobian triplet outside the primal block";
    }

    // Hessian triplets: primal block only, lower triangle (row >= col), which
    // is the triangle Ipopt documents for a symmetric matrix.
    std::vector<IpIndex> hess_rows(static_cast<std::size_t>(nnz_hess));
    std::vector<IpIndex> hess_cols(static_cast<std::size_t>(nnz_hess));
    ASSERT_TRUE(tnlp->eval_h(n, nullptr, true, 1.0, m, nullptr, true, nnz_hess, hess_rows.data(),
                             hess_cols.data(), nullptr));
    for (IpIndex i = 0; i < nnz_hess; ++i) {
        EXPECT_GE(hess_cols[i], 0);
        EXPECT_LT(hess_rows[i], n) << "Hessian triplet outside the primal block";
        EXPECT_GE(hess_rows[i], hess_cols[i]) << "Hessian triplet outside the lower triangle";
    }
}

// Both backends run the identical NLP through the same dispatch seam and land
// on the same solution.
TEST(IpoptBackend, ParityWithPsioptOnSmoothProblem) {
    auto psiopt_prob = build_ipopt_adapter_problem();
    const auto psiopt_flag = psiopt_prob->optimize();
    ASSERT_EQ(psiopt_flag, tycho::ConvergenceFlags::CONVERGED);
    const double psiopt_obj = psiopt_prob->optimizer_->result().obj_val_;
    const Eigen::VectorXd psiopt_vars = psiopt_prob->return_vars();

    auto ipopt_prob = build_ipopt_adapter_problem();
    ipopt_prob->nlp_solver_ = ts::NLPSolvers::ipopt;
    const auto ipopt_flag = ipopt_prob->optimize();
    EXPECT_EQ(ipopt_flag, tycho::ConvergenceFlags::CONVERGED);

    const Eigen::VectorXd ipopt_vars = ipopt_prob->return_vars();
    ASSERT_EQ(ipopt_vars.size(), psiopt_vars.size());
    for (int i = 0; i < ipopt_vars.size(); ++i) {
        EXPECT_NEAR(ipopt_vars[i], psiopt_vars[i], 1e-6) << "variable " << i;
    }

    EXPECT_NEAR(psiopt_obj, ipopt_adapter_optimal_objective(), 1e-6);
    EXPECT_NEAR(ipopt_prob->last_ipopt_result_.objective_, psiopt_obj, 1e-6);

    EXPECT_TRUE(ipopt_prob->last_ipopt_result_.ran_);
    EXPECT_EQ(ipopt_prob->last_ipopt_result_.normalized_, "converged");
    EXPECT_EQ(ipopt_prob->last_ipopt_result_.converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_GT(ipopt_prob->last_ipopt_result_.iterations_, 0);
    EXPECT_GE(ipopt_prob->last_ipopt_result_.wall_time_s_, 0.0);
    EXPECT_LT(ipopt_prob->last_ipopt_result_.constraint_violation_, 1e-6);

    // The built-in solver's run info is untouched by an Ipopt run.
    EXPECT_FALSE(psiopt_prob->last_ipopt_result_.ran_);
}

// A collocation phase solves through the same seam, and the solved variables
// flow back into the phase's trajectory.
TEST(IpoptBackend, PhaseProblemEndToEnd) {
    auto phase = make_brach_solver_phase(4);
    phase->nlp_solver_ = ts::NLPSolvers::ipopt;

    const auto flag = phase->optimize();
    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE)
        << "ipopt backend did not solve the collocation phase";
    EXPECT_TRUE(phase->last_ipopt_result_.ran_);
    EXPECT_GT(phase->last_ipopt_result_.iterations_, 0);

    // Trajectory extraction: the boundary conditions the phase was built with
    // must hold at the returned trajectory's endpoints.
    auto traj = phase->return_traj();
    ASSERT_FALSE(traj.empty());
    EXPECT_NEAR(traj.front()[0], 0.0, 1e-5);
    EXPECT_NEAR(traj.front()[1], 10.0, 1e-5);
    EXPECT_NEAR(traj.front()[2], 0.0, 1e-5);
    EXPECT_NEAR(traj.back()[0], 10.0, 1e-5);
    EXPECT_NEAR(traj.back()[1], 5.0, 1e-5);
    // Minimum-time brachistochrone between those endpoints: about 1.80 s on a
    // fine mesh, so a band that a coarse four-segment mesh comfortably lands in
    // while still excluding the unsolved initial guess (whose final time is
    // exactly 1.0 s).
    EXPECT_GT(traj.back()[3], 1.0);
    EXPECT_LT(traj.back()[3], 3.0);
}

// An evaluation error inside a callback is latched rather than unwound through
// Ipopt, and re-raised once the run returns.
TEST(IpoptBackend, EvalExceptionSurfacesAfterSolve) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars(Eigen::VectorXd::Constant(1, 2.0e3));
    prob.add_objective(GenericFunction<-1, 1>(IpoptAdapterGuardedObjective()),
                       (Eigen::VectorXi(1) << 0).finished());
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob.add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
    }
    prob.optimizer_->set_print_level(0);
    prob.nlp_solver_ = ts::NLPSolvers::ipopt;

    try {
        prob.optimize();
        FAIL() << "expected the latched evaluation error to be re-raised";
    } catch (const std::runtime_error &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(kIpoptAdapterDomainError), std::string::npos)
            << "message did not carry the evaluation error: " << what;
    }
}

// An option Ipopt does not accept is reported by name instead of being applied
// silently or ignored.
TEST(IpoptBackend, BadOptionRejected) {
    auto prob = build_ipopt_adapter_problem();
    prob->nlp_solver_ = ts::NLPSolvers::ipopt;
    prob->ipopt_options_ = {{"linear_solver", "no_such_solver"}};

    try {
        prob->optimize();
        FAIL() << "expected the rejected option to be reported";
    } catch (const std::runtime_error &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("linear_solver"), std::string::npos)
            << "message did not name the option: " << what;
    }
}
