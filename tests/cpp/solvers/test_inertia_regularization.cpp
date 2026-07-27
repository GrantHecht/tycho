///////////////////////////////////////////////////////////////////////////////
// Proximal primal-dual KKT regularization (PSIOPT inertia_mode_).
//
// The proximal_regularization inertia mode bakes a small persistent, decaying
// primal base shift ρ_k on the Hessian diagonal and an always-on barrier-scaled
// dual shift −δ_c on the constraint-row diagonals into the base KKT matrix each
// iteration; the classic inertia ladder still escalates on top when the base
// attempt has wrong inertia or is singular. classic (the default) is unchanged.
//
// Layer 1 truth-tables the pure helpers (globalization/inertia_regularization.h)
// — δ_c = 1e-8·μ^0.25 at named μ, and the ρ_k decay toward and clamp at the
// floor. Layer 2 drives small hand-built NLPs through the public solve path: a
// rank-deficient (duplicated-equality) KKT the dual shift is meant to make
// factorizable, a parity smoke on a well-conditioned problem under both modes,
// and the mode composed with a nested l1 restoration on an infeasible problem
// (the dual-shift suppression path, exercised end-to-end).
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/globalization/inertia_regularization.h"

#include <gtest/gtest.h>


#include <cmath>

#include <Eigen/Core>

using namespace tycho;
using namespace TychoTest;

using tycho::solvers::AcceptanceStrategies;
using tycho::solvers::dual_regularization;
using tycho::solvers::InertiaModes;
using tycho::solvers::kDualRegExponent;
using tycho::solvers::kDualRegScale;
using tycho::solvers::kProxRegFloor;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::prox_reg_decay;
using tycho::solvers::RestorationModes;

namespace {

// =============================================================================
// Layer 1 -- pure helper truth tables
// =============================================================================

TEST(InertiaRegularizationConstants, MatchReferenceDefaults) {
    // Locked to the Ipopt jacobian_regularization defaults and the
    // Cipolla–Gondzio floor; a change here is a deliberate policy change.
    EXPECT_DOUBLE_EQ(kProxRegFloor, 1.0e-10);
    EXPECT_DOUBLE_EQ(kDualRegScale, 1.0e-8);
    EXPECT_DOUBLE_EQ(kDualRegExponent, 0.25);
}

TEST(InertiaRegularizationDualShift, ExactValuesAtNamedMu) {
    // δ_c(μ) = 1e-8 · μ^0.25.
    // μ = 1 is exact (pow(1, 0.25) == 1), so it is pinned with DOUBLE_EQ. The
    // other two go through std::pow on a non-trivial exponent, which is not
    // correctly rounded; each tolerance is ~1e-12 relative to its own expected
    // value -- tight enough to catch a changed scale or exponent, loose enough
    // to absorb libm's last-place error.
    EXPECT_DOUBLE_EQ(dual_regularization(1.0), 1.0e-8);         // 1e-8 · 1
    EXPECT_NEAR(dual_regularization(1.0e-4), 1.0e-9, 1.0e-21);  // 1e-8 · 1e-1
    EXPECT_NEAR(dual_regularization(1.0e-8), 1.0e-10, 1.0e-22); // 1e-8 · 1e-2
}

TEST(InertiaRegularizationDualShift, ShrinksMonotonicallyWithMu) {
    // δ_c → 0 as μ → 0, so it can never mask non-convergence.
    EXPECT_GT(dual_regularization(1.0e-2), dual_regularization(1.0e-4));
    EXPECT_GT(dual_regularization(1.0e-4), dual_regularization(1.0e-8));
    EXPECT_GT(dual_regularization(1.0e-8), 0.0);
}

TEST(InertiaRegularizationPrimalDecay, DecaysAboveFloor) {
    // Above the floor the decay is exactly applied_total · decr. Both bounds are
    // ~1e-12 relative to their expected value, absorbing the one multiplication's
    // rounding (and any FMA contraction under the project's -ffast-math build)
    // without admitting a changed decay rule.
    EXPECT_NEAR(prox_reg_decay(3.0e-4, 0.5), 1.5e-4, 1.0e-16);
    EXPECT_NEAR(prox_reg_decay(1.0e-5, 0.333333), 1.0e-5 * 0.333333, 1.0e-18);
}

TEST(InertiaRegularizationPrimalDecay, ClampsAtFloor) {
    // Once the decayed value would fall below the floor, it is clamped to it.
    EXPECT_DOUBLE_EQ(prox_reg_decay(kProxRegFloor, 0.333333), kProxRegFloor);
    EXPECT_DOUBLE_EQ(prox_reg_decay(2.0e-10, 0.333333), kProxRegFloor); // 6.7e-11 < floor
    EXPECT_DOUBLE_EQ(prox_reg_decay(0.0, 0.5), kProxRegFloor);
}

TEST(InertiaRegularizationPrimalDecay, ConvergesToFloorFromHealthyStart) {
    // Iterating the decay from a healthy (floor) start stays pinned at the floor
    // -- the parity-by-construction property on well-conditioned problems.
    double rho = kProxRegFloor;
    for (int i = 0; i < 50; ++i)
        rho = prox_reg_decay(rho, 0.333333);
    EXPECT_DOUBLE_EQ(rho, kProxRegFloor);
}

TEST(InertiaRegularizationPrimalDecay, DecaysToFloorFromLargeStart) {
    // A large successful shift decays geometrically and lands on the floor.
    double rho = 1.0e-2;
    for (int i = 0; i < 100; ++i)
        rho = prox_reg_decay(rho, 0.333333);
    EXPECT_DOUBLE_EQ(rho, kProxRegFloor);
}

// =============================================================================
// Layer 2 -- through the public solve path
// =============================================================================

// A rank-deficient KKT: min x0^2 + x1^2 subject to x0 + x1 = 1 stated TWICE.
// The two identical equality rows make the constraint Jacobian rank-deficient,
// so the KKT matrix is singular. The unique primal optimum is x0 = x1 = 0.5,
// objective 0.5.
std::unique_ptr<OptimizationProblem> build_inertia_duplicated_equality_nlp() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars((Eigen::VectorXd(2) << 0.0, 0.0).finished());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob->add_objective(GenericFunction<-1, 1>(x0 * x0 + x1 * x1),
                            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    for (int rep = 0; rep < 2; ++rep) {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob->add_equal_con(GenericFunction<-1, -1>(x0 + x1 - 1.0),
                            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    prob->optimizer_->set_print_level(3);
    return prob;
}

// (a) The dual shift makes the rank-deficient system factorizable, so the mode
// drives the duplicated-equality problem to its unique optimum.
TEST(InertiaRegularizationSolve, ProximalRegularizationConvergesOnRankDeficientKkt) {
    auto prob = build_inertia_duplicated_equality_nlp();
    prob->optimizer_->settings().inertia_mode_ = InertiaModes::proximal_regularization;
    prob->optimizer_->set_max_iters(100);
    auto flag = prob->optimize();

    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    const auto &r = prob->optimizer_->result();
    // Solve-output tolerances, not arithmetic ones: the solver stops at
    // econ_tol_/kkt_tol_ = 1e-6 on a regularized (shifted) KKT system, so the
    // primals land within a few multiples of that tolerance of the analytic
    // optimum and the objective, being quadratic and stationary there, lands
    // an order tighter. Loose enough not to re-litigate the barrier schedule,
    // tight enough that converging to any other point fails.
    EXPECT_NEAR(r.obj_val_, 0.5, 1e-5);
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 0.5, 1e-4);
    EXPECT_NEAR(r.primals_[1], 0.5, 1e-4);
}

// (b) The SAME rank-deficient problem under classic. This documents the classic
// behavior only -- static (Pardiso) pivoting may rescue the singular
// factorization, so no failure is asserted; the point is that the mode above is
// what carries a principled regularization for this case.
TEST(InertiaRegularizationSolve, ClassicOnRankDeficientKktDocumented) {
    auto prob = build_inertia_duplicated_equality_nlp();
    prob->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    prob->optimizer_->set_max_iters(100);
    auto flag = prob->optimize();

    const auto &r = prob->optimizer_->result();
    ASSERT_EQ(r.primals_.size(), 2);
    // "Classic never converges to a WRONG point on this problem" is the whole
    // property, and it is asserted unconditionally: a converged flag obliges the
    // value checks, a non-converged flag is a permitted outcome. Written as
    // implications rather than an `if` block so no assertion can silently drop
    // out of the run on a toolchain where classic does not converge. The
    // tolerances are one decade looser than the proximal test's: this path
    // reaches the optimum through static pivot perturbation on a singular KKT
    // rather than a principled shift, so its terminal accuracy is weaker.
    const bool converged = (flag == tycho::ConvergenceFlags::CONVERGED);
    EXPECT_TRUE(!converged || std::abs(r.obj_val_ - 0.5) < 1e-4)
        << "classic converged to obj_val_ = " << r.obj_val_ << ", expected 0.5";
    EXPECT_TRUE(!converged || std::abs(r.primals_[0] - 0.5) < 1e-3)
        << "classic converged to x0 = " << r.primals_[0] << ", expected 0.5";
    EXPECT_TRUE(!converged || std::abs(r.primals_[1] - 0.5) < 1e-3)
        << "classic converged to x1 = " << r.primals_[1] << ", expected 0.5";
}

// A well-conditioned equality NLP: min x^2 s.t. x - 1 = 0, optimum x = 1,
// objective 1.
std::unique_ptr<OptimizationProblem> build_inertia_wellcond_nlp() {
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
    prob->optimizer_->set_print_level(3);
    return prob;
}

// (c) Parity smoke: a well-conditioned problem reaches the same objective under
// both modes within solver tolerance (on healthy problems ρ_k sits at the floor
// and δ_c is negligible, so the mode is a near-no-op).
TEST(InertiaRegularizationSolve, WellConditionedParityAcrossModes) {
    auto prob_classic = build_inertia_wellcond_nlp();
    prob_classic->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    auto flag_classic = prob_classic->optimize();
    ASSERT_EQ(flag_classic, tycho::ConvergenceFlags::CONVERGED);
    const double obj_classic = prob_classic->optimizer_->result().obj_val_;

    auto prob_prox = build_inertia_wellcond_nlp();
    prob_prox->optimizer_->settings().inertia_mode_ = InertiaModes::proximal_regularization;
    auto flag_prox = prob_prox->optimize();
    ASSERT_EQ(flag_prox, tycho::ConvergenceFlags::CONVERGED);
    const double obj_prox = prob_prox->optimizer_->result().obj_val_;

    EXPECT_NEAR(obj_classic, 1.0, 1e-6);
    EXPECT_NEAR(obj_prox, obj_classic, 1e-6);
}

// (d) Mode composed with nested l1 restoration on an infeasible problem. Forcing
// every step to be rejected (max_ls_iters == 0) drives the recovery ladder to
// exhaustion, entering the nested restoration phase; the mode's dual shift is
// suppressed while that phase is active (the elastic pivots own the
// constraint-row slots). The solve must still enter restoration and exit cleanly
// (never falsely converge on a jointly-infeasible problem).
TEST(InertiaRegularizationSolve, ProximalRegularizationWithL1NestedRestoration) {
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
    {
        // Contradicts x - 1 = 0: jointly infeasible, violation bounded below by 1.
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x + 1.0), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3);
    prob->optimizer_->settings().inertia_mode_ = InertiaModes::proximal_regularization;
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::merit;
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::l1_nested;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(80);
    auto flag = prob->optimize();

    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1);             // restoration entered
    EXPECT_NE(flag, tycho::ConvergenceFlags::CONVERGED); // never falsely converges
}

} // namespace
