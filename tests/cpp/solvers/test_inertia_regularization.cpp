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
// (the dual-shift suppression path, exercised end-to-end). Layer 2 also carries
// the composition sentinels for native variable bounds: a solution sitting ON a
// bound drives the condensed bound curvature on the primal diagonal enormous,
// and a healthy system in that regime must still be accepted on its own
// inertia.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/barrier_math.h"
#include "tycho/detail/solvers/globalization/inertia_regularization.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

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

// (b) The SAME rank-deficient problem under classic. The full Ipopt IC condition
// engages the on-demand dual regularization when the factorization reports rank
// deficiency, so classic converges here too. (On MKL the static pivot
// perturbation may mask the deficiency instead; either road must reach the
// unique optimum.)
TEST(InertiaRegularizationSolve, ClassicConvergesOnRankDeficientKkt) {
    auto prob = build_inertia_duplicated_equality_nlp();
    prob->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    prob->optimizer_->set_max_iters(100);
    auto flag = prob->optimize();

    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    const auto &r = prob->optimizer_->result();
    EXPECT_NEAR(r.obj_val_, 0.5, 1e-5);
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 0.5, 1e-4);
    EXPECT_NEAR(r.primals_[1], 0.5, 1e-4);
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

// A convex NLP with a natively bounded variable whose solution sits ON its
// upper bound:
//
//   min (x0 - 2)^2 + x1^2   s.t.   x0 + x1 = 1.5,   lower <= x0 <= upper.
//
// On the constraint manifold the objective minimizes at x0 = 1.75, decisively
// outside any box this file passes, so the upper bound is active at the
// solution: x* = (upper, 1.5 - upper), and at upper = 1 the analytic optimum is
// (1, 0.5) with objective 1.25. Stationarity there fixes the multipliers --
// 2*x1 + lambda = 0 gives lambda = -1, and 2*(x0 - 2) + lambda + z_U = 0 gives
// z_U = 3 -- so the bound multiplier stays O(1) while the barrier drives the
// distance to the bound toward zero, which is exactly the regime that makes the
// condensed curvature z_U / d blow up.
//
// The equality row is not decoration: it gives the KKT system a constraint
// block, so the inertia the factorization is required to report is
// (kkt_dim - 1, 1, 0) rather than a trivially all-positive one.
std::unique_ptr<OptimizationProblem> build_inertia_active_bound_nlp(double lower, double upper) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;
    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars((Eigen::VectorXd(2) << 0.5, 1.0).finished());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob->add_objective(GenericFunction<-1, 1>((x0 - 2.0) * (x0 - 2.0) + x1 * x1),
                            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob->add_equal_con(GenericFunction<-1, -1>(x0 + x1 - 1.5),
                            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    prob->optimizer_->set_print_level(3);
    prob->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    // Bound declarations are staged on the NonLinearProgram, which transcription
    // creates -- so the declaration has to follow it. make_nlp then
    // re-materializes the staged box (the row count it is handed is the user's
    // own, user_equal_cons_) and the solver is re-pointed at the rebuilt
    // program. optimize() below does not re-transcribe: transcribe() clears the
    // pending flag, so the bound survives into the solve.
    prob->transcribe();
    prob->nlp_->set_variable_bound(0, lower, upper);
    prob->nlp_->make_nlp(prob->nlp_->primal_vars_, prob->nlp_->user_equal_cons_,
                         prob->nlp_->inequal_cons_);
    prob->optimizer_->set_nlp(prob->nlp_);
    return prob;
}

// What the bound machinery leaves on the primal (1,1) diagonal at a point, and
// how close that point is to a bound. Computed with the shipped kernel rather
// than a parallel implementation, so the number reported here is the one the
// solver actually assembled.
struct InertiaBoundCurvature {
    double max_sigma_;
    double min_distance_;
};

InertiaBoundCurvature inertia_bound_curvature_at(const Eigen::VectorXd &x,
                                                 const tycho::solvers::BoundSet &b,
                                                 const tycho::solvers::BoundDualState &z) {
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(x.size());
    tycho::solvers::detail::accumulate_bound_sigma(x, b, z, sigma);
    double dmin = std::numeric_limits<double>::infinity();
    for (int k = 0; k < b.lower_idx_.size(); k++) {
        dmin = std::min(dmin, x[b.lower_idx_[k]] - b.lower_val_[k]);
    }
    for (int k = 0; k < b.upper_idx_.size(); k++) {
        dmin = std::min(dmin, b.upper_val_[k] - x[b.upper_idx_[k]]);
    }
    return {sigma.maxCoeff(), dmin};
}

// Floor the curvature has to clear for the sentinels below to be about anything.
// It is not a tuned constant: a converged solve on these problems reports a
// barrier error below bar_tol_ = 1e-6, that error is entirely the bound pair's
// complementarity z*d (no inequality rows means no slack pairs to dilute it),
// and z settles at 3 -- so sigma = z/d = z^2/(z*d) > 9/1e-6 = 9e6 follows from
// the convergence test itself. Measured values are far past it (order 1e12 on
// the wide box, 1e7 on the narrow one), and the Hessian entries the curvature
// lands beside are order 2.
constexpr double kInertiaLargeSigma = 1.0e6;

} // namespace

// InertiaRegularizationSolve_ClassicDegeneracyLatchTracksSingularity_Test reaches
// PSIOPT::dc_latched_ (private) via the friend declaration in psiopt.h, which
// befriends the GLOBAL-scope class ::InertiaRegularizationSolve_..._Test that
// gtest's TEST() macro generates. TEST() declares its fixture class as a member
// of whatever namespace textually encloses it -- inside the anonymous namespace
// above, that would be a DISTINCT (anonymous namespace)::..._Test entity that the
// friend declaration does not name, so the access would be denied at compile
// time despite the friend declaration being syntactically present (see the
// established precedent in test_recovery_dispatch_gate.cpp, which closes its
// anonymous namespace before its own friended RecoveryDispatchGate_* tests for
// the same reason). Hence this test sits between the two anonymous-namespace
// blocks, at true global scope; build_inertia_duplicated_equality_nlp() and
// build_inertia_wellcond_nlp() remain reachable here because unqualified lookup
// from an enclosing scope finds anonymous-namespace members via their implicit
// using-directive (that direction is unaffected by the issue above). The two
// bound-curvature sentinels that follow it are befriended and placed the same
// way, for the same reason, and reach bounds_ / bound_duals_ as well.

// The degeneracy latch (Ipopt hess/jac-degenerate adaptation, sticky per phase):
// set once delta_c is engaged for a singular factorization, so later iterations
// pre-apply it at the base attempt instead of re-discovering the singularity;
// never set on problems whose factorizations stay full-rank.
TEST(InertiaRegularizationSolve, ClassicDegeneracyLatchTracksSingularity) {
    auto degen = build_inertia_duplicated_equality_nlp();
    degen->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    degen->optimizer_->set_max_iters(100);
    (void)degen->optimize();
#ifdef USE_ACCELERATE_SPARSE
    // Accelerate reports inertia honestly (no Pardiso-style static pivot
    // perturbation), so the duplicated-equality problem's singular KKT is
    // observed as such and delta_c engages, setting the latch. On a
    // pivot-perturbing backend (MKL Pardiso, Windows/Linux) the masked
    // deficiency may report neigs == m with zeigs == 0 and never trip
    // SingularitySignal(), so this half of the assertion is platform-dependent
    // -- see the sibling ClassicConvergesOnRankDeficientKkt test's comment for
    // the same caveat. The healthy-problem EXPECT_FALSE below is NOT guarded:
    // a never-singular solve must never latch on any backend.
    EXPECT_TRUE(degen->optimizer_->dc_latched_)
        << "delta_c engaged on a rank-deficient problem must set the latch";
#endif

    auto healthy = build_inertia_wellcond_nlp();
    healthy->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    (void)healthy->optimize();
    EXPECT_FALSE(healthy->optimizer_->dc_latched_)
        << "a full-rank problem must never engage delta_c or the latch";
}

// Composition sentinel: native variable bounds against the inertia machinery.
//
// Eliminating the bound-multiplier rows condenses a curvature term z/d onto the
// primal (1,1) KKT diagonal for every bounded variable, where d is the distance
// to the bound. When the solution sits ON a bound the barrier drives d -> 0 as
// mu -> 0 while z stays O(1), so that diagonal entry grows without bound (up to
// the multiplier clamp) even though the system stays full rank and well posed.
// The full inertia condition accepts a factorization only at exactly
// (kkt_dim - m, m, 0), and treats anything else -- a reported rank deficiency,
// or too few negative eigenvalues -- as a singularity, engaging the on-demand
// dual shift and setting the sticky per-phase latch. A large curvature entry
// makes the corresponding small eigenvalue of the system small in the opposite
// sense, which is precisely the shape a factorization backend could mistake for
// a zero pivot.
//
// So: on a healthy, full-rank, bounded problem whose solution is on a bound,
// the solve must converge outright and the singularity signal must never have
// fired -- no engaged dual shift, no latch, no exhausted ladder. Neither half
// is backend-conditional: a full-rank system reported as singular is a bug on
// any backend.
TEST(InertiaRegularizationSolve, ActiveBoundCurvatureNeverTripsSingularitySignal) {
    auto prob = build_inertia_active_bound_nlp(/*lower=*/0.0, /*upper=*/1.0);
    const auto flag = prob->optimize();
    auto &opt = *prob->optimizer_;

    // Not ACCEPTABLE (that would mean the solve limped in on the relaxed
    // tolerances), and not SINGULAR_KKT (an exhausted ladder).
    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    const auto &r = opt.result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 1.0, 1.0e-5);
    EXPECT_NEAR(r.primals_[1], 0.5, 1.0e-5);
    EXPECT_NEAR(r.obj_val_, 1.25, 1.0e-5);

    // The constraint block really is part of the inertia count, and there are no
    // inequality rows -- so no slack pairs dilute the barrier account, and the
    // curvature the sentinel is about is the only one on the diagonal.
    EXPECT_EQ(opt.equal_cons_, 1);
    EXPECT_EQ(opt.inequal_cons_, 0);

    // The regime is asserted, not assumed: without a genuinely large curvature
    // this test would pass vacuously. Nothing is eliminated from this problem
    // (no variable is fixed), so the solution vector and the bound set index the
    // same space.
    ASSERT_NE(opt.bounds_, nullptr);
    const auto curv = inertia_bound_curvature_at(r.primals_, *opt.bounds_, opt.bound_duals_);
    EXPECT_GT(curv.max_sigma_, kInertiaLargeSigma)
        << "the bound curvature never got large, so this solve did not exercise "
           "the regime the sentinel is about (closest bound distance "
        << curv.min_distance_ << ")";

    EXPECT_FALSE(opt.dc_latched_)
        << "a full-rank bounded system whose solution sits on a bound must not "
           "read as singular (bound curvature "
        << curv.max_sigma_ << ", closest bound distance " << curv.min_distance_ << ")";
}

// The same problem with the box tightened around the solution, so the distance
// to a bound is small from the interior push onward rather than only at the end
// -- the curvature is large for essentially every factorization of the solve,
// not just the last few. Same assertions.
TEST(InertiaRegularizationSolve, NarrowBoxCurvatureNeverTripsSingularitySignal) {
    auto prob = build_inertia_active_bound_nlp(/*lower=*/0.99, /*upper=*/1.0);
    const auto flag = prob->optimize();
    auto &opt = *prob->optimizer_;

    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    const auto &r = opt.result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 1.0, 1.0e-5);
    EXPECT_NEAR(r.primals_[1], 0.5, 1.0e-5);
    EXPECT_NEAR(r.obj_val_, 1.25, 1.0e-5);

    ASSERT_NE(opt.bounds_, nullptr);
    const auto curv = inertia_bound_curvature_at(r.primals_, *opt.bounds_, opt.bound_duals_);
    EXPECT_GT(curv.max_sigma_, kInertiaLargeSigma)
        << "the bound curvature never got large, so this solve did not exercise "
           "the regime the sentinel is about (closest bound distance "
        << curv.min_distance_ << ")";

    EXPECT_FALSE(opt.dc_latched_)
        << "a full-rank bounded system solved inside a narrow box must not read "
           "as singular (bound curvature "
        << curv.max_sigma_ << ", closest bound distance " << curv.min_distance_ << ")";
}

namespace {

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
