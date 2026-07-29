///////////////////////////////////////////////////////////////////////////////
// Native primal variable-bound machinery in PSIOPT: the barrier kernels, the
// interior push and multiplier seeding, the bound-multiplier Newton direction
// and its committed update with the kappa_sigma safeguard, and the condensed
// sigma diagonal's arrival in the assembled KKT.
//
// There is no fraction-to-boundary leg for bounded variables yet, so nothing
// here drives a bounded solve end to end. Every test is either a pure kernel
// checked against a written-out formula or a single private helper driven
// through NativeBoundsHarness, which PSIOPT befriends (psiopt.h).
//
// The last group is the neutrality check the whole feature rests on: a problem
// that declares no variable bounds leaves the bound state empty, leaves the
// solver's bound pointer null, and assembles a KKT diagonal byte-identical to
// the one it assembled before any of this existed.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/barrier_math.h"
#include "tycho/detail/solvers/bound_set.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/optimization_problem.h"
#include "tycho/detail/solvers/psiopt.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>

using tycho::solvers::BoundDualState;
using tycho::solvers::BoundSet;
using tycho::solvers::FixedVariableTreatments;
using tycho::solvers::kBoundMultInitCap;
using tycho::solvers::kFreeModeClipMuCap;
using tycho::solvers::kKappaD;
using tycho::solvers::kKappaSigma;
using tycho::solvers::NonLinearProgram;
using tycho::solvers::PSIOPT;

namespace {

constexpr double kNativeBoundsInf = std::numeric_limits<double>::infinity();

// The bound set every kernel test below shares, over three variables:
//   variable 0 -- lower bound 1.0 only
//   variable 1 -- upper bound 8.0 only
//   variable 2 -- two-sided, [0.0, 1.0]
// A two-sided variable appears in BOTH lists, which is what makes the shared
// accumulation onto one gradient/diagonal entry worth checking. The damping
// indicators mark the one-sided entries (variables 0 and 1) and clear the
// two-sided ones, matching what the classifier records.
BoundSet native_bounds_three_var_set() {
    BoundSet b;
    b.lower_idx_ = (Eigen::VectorXi(2) << 0, 2).finished();
    b.lower_val_ = (Eigen::VectorXd(2) << 1.0, 0.0).finished();
    b.lower_damp_ = (Eigen::VectorXd(2) << 1.0, 0.0).finished();
    b.upper_idx_ = (Eigen::VectorXi(2) << 1, 2).finished();
    b.upper_val_ = (Eigen::VectorXd(2) << 8.0, 1.0).finished();
    b.upper_damp_ = (Eigen::VectorXd(2) << 1.0, 0.0).finished();
    return b;
}

// The same three variables made purely TWO-SIDED, so every damping indicator is
// zero. Used where the damping term would otherwise obscure the property under
// test.
BoundSet native_bounds_two_sided_set() {
    BoundSet b;
    b.lower_idx_ = (Eigen::VectorXi(2) << 0, 2).finished();
    b.lower_val_ = (Eigen::VectorXd(2) << 1.0, 0.0).finished();
    b.lower_damp_ = Eigen::VectorXd::Zero(2);
    b.upper_idx_ = (Eigen::VectorXi(2) << 0, 2).finished();
    b.upper_val_ = (Eigen::VectorXd(2) << 8.0, 1.0).finished();
    b.upper_damp_ = Eigen::VectorXd::Zero(2);
    return b;
}

// Multipliers index-aligned to the set above: z_lower_ to {var 0, var 2},
// z_upper_ to {var 1, var 2}.
BoundDualState native_bounds_three_var_duals() {
    BoundDualState z;
    z.z_lower_ = (Eigen::VectorXd(2) << 0.7, 1.5).finished();
    z.z_upper_ = (Eigen::VectorXd(2) << 0.25, 4.0).finished();
    z.dz_lower_ = Eigen::VectorXd::Zero(2);
    z.dz_upper_ = Eigen::VectorXd::Zero(2);
    return z;
}

// The iterate the kernel tests evaluate at: strictly interior to every bound
// above, as the interior push and the fraction-to-boundary rule guarantee.
Eigen::VectorXd native_bounds_interior_point() {
    return (Eigen::VectorXd(3) << 2.0, 5.0, 0.5).finished();
}

// Adds the objective term (x[index] - center)^2. Started away from its
// minimizer so neither the objective nor its gradient vanishes at the point the
// assembly tests evaluate.
void native_bounds_add_shifted_square(tycho::solvers::OptimizationProblem &prob, int index,
                                      double center) {
    auto args = tycho::vf::Arguments<1>();
    auto term = (args.coeff<0>() - center).squared_norm();
    prob.add_objective(tycho::vf::GenericFunction<-1, 1>(term),
                       (Eigen::VectorXi(1) << index).finished());
}

} // namespace

// A live PSIOPT over a small bound-free-by-default problem, with the private
// bound surface exposed. The bound set is installed by driving the NLP's own
// classification and then pointing the solver at the result -- exactly what
// run_phase_sequence() does on its configuration success path -- so the tests
// exercise the shipped helpers rather than a parallel implementation.
class NativeBoundsHarness {
  public:
    explicit NativeBoundsHarness(int num_vars) {
        prob_.set_vars(Eigen::VectorXd::Zero(num_vars));
        for (int i = 0; i < num_vars; i++) {
            native_bounds_add_shifted_square(prob_, i, 3.0);
        }
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
    }

    PSIOPT &solver() { return *solver_; }
    NonLinearProgram &nlp() { return *prob_.nlp_; }
    int pv() const { return solver_->primal_vars_; }
    int dim() const { return solver_->kkt_dim_; }

    // Declares a bound and re-materializes the staged declarations, mirroring
    // what a transcription would have staged. make_nlp reallocates the KKT
    // arrays, so the solver is re-pointed at the NLP afterwards.
    void declare_bound(int index, double lower, double upper) {
        prob_.nlp_->set_variable_bound(index, lower, upper);
        prob_.nlp_->make_nlp(prob_.nlp_->primal_vars_, prob_.nlp_->equal_cons_,
                             prob_.nlp_->inequal_cons_);
        solver_->set_nlp(prob_.nlp_);
    }

    // Classifies the declared bounds with NO relaxation (so the recorded values
    // are the declared ones and the hand calculations stay exact) and installs
    // the resulting set on the solver, the way the configuration success path
    // in run_phase_sequence() does.
    void configure_bounds() {
        prob_.nlp_->configure_variable_treatment(FixedVariableTreatments::MakeParameter, 0.0);
        solver_->refresh_nlp_dimensions();
        solver_->bounds_ =
            prob_.nlp_->variable_bound_set().any() ? &prob_.nlp_->variable_bound_set() : nullptr;
    }

    const BoundSet *installed_bounds() const { return solver_->bounds_; }
    void install_bounds(const BoundSet *b) { solver_->bounds_ = b; }
    BoundDualState &duals() { return solver_->bound_duals_; }

    // A SolverContext built the way every non-bound call site builds one, i.e.
    // without naming the two bound members. Constructed here because the members
    // it reads are private to PSIOPT.
    tycho::solvers::SolverContext bare_context() {
        return tycho::solvers::SolverContext{
            &this->nlp(),           solver_->kkt_sol_,    solver_->settings_,
            solver_->primal_vars_,  solver_->slack_vars_, solver_->equal_cons_,
            solver_->inequal_cons_, solver_->kkt_dim_,    solver_->stli_scratch_};
    }

    void push(Eigen::VectorXd &x, double mu0) { solver_->push_initial_point_interior(x, mu0); }
    void direction(const Eigen::VectorXd &x, const Eigen::VectorXd &dx, double mu) {
        solver_->compute_bound_dual_direction(x, dx, mu);
    }
    // `xsl_new` is a full compound iterate. The harness problems carry no
    // constraints, so its width is the primal width and the slack/multiplier
    // blocks the free-mu average would read are empty.
    void commit(double alphad, Eigen::VectorXd xsl_new, double mu, bool monotone_mu) {
        tycho::solvers::KKTVector v = solver_->kkt_view(xsl_new);
        solver_->apply_bound_dual_step(alphad, v, mu, monotone_mu);
    }

    // Puts the solver's primal-diagonal coefficients into the state every
    // evaluation leaves them in (each mode that writes them resets them after),
    // so the first assembly below starts from the same place a later one does.
    void clear_primal_diags() { prob_.nlp_->set_primal_diags(0.0); }

    // Runs one optimality-mode evaluation at `primals` and returns the assembled
    // KKT matrix's primal diagonal.
    Eigen::VectorXd assemble_primal_diagonal(const Eigen::VectorXd &primals) {
        Eigen::VectorXd XSL = Eigen::VectorXd::Zero(dim());
        XSL.head(pv()) = primals;
        Eigen::VectorXd GX = Eigen::VectorXd::Zero(pv());
        Eigen::VectorXd RHS = Eigen::VectorXd::Zero(dim());
        double val = 0.0;
        solver_->eval_nlp(PSIOPT::AlgorithmModes::OPT, 1.0, XSL, val, GX, RHS,
                          solver_->kkt_sol_.get_matrix(), 0.1);
        Eigen::VectorXd diag(pv());
        for (int i = 0; i < pv(); i++) {
            diag[i] = solver_->kkt_sol_.get_matrix().coeff(i, i);
        }
        return diag;
    }

  private:
    tycho::solvers::OptimizationProblem prob_;
    PSIOPT *solver_ = nullptr;
};

// --- Barrier kernels ------------------------------------------------------

TEST(NativeBounds, BarrierObjectiveIsTheLogSumOverBothLists) {
    const BoundSet b = native_bounds_three_var_set();
    const Eigen::VectorXd x = native_bounds_interior_point();
    const double mu = 0.1;

    // -mu*ln(d) per entry, plus kappa_d*mu*d on the two ONE-SIDED entries
    // (variable 0's lower bound, variable 1's upper bound) and nothing on the
    // two-sided ones. Grouped in the accumulation order the kernel uses.
    const double expected = (-mu * std::log(1.0) + kKappaD * mu * 1.0 * (2.0 - 1.0)) +
                            (-mu * std::log(0.5) + kKappaD * mu * 0.0 * (0.5 - 0.0)) +
                            (-mu * std::log(3.0) + kKappaD * mu * 1.0 * (8.0 - 5.0)) +
                            (-mu * std::log(0.5) + kKappaD * mu * 0.0 * (1.0 - 0.5));
    EXPECT_DOUBLE_EQ(tycho::solvers::detail::bound_barrier_objective(x, b, mu), expected);
}

TEST(NativeBounds, MuFormGradientAccumulatesTheBarrierDerivative) {
    const BoundSet b = native_bounds_three_var_set();
    const Eigen::VectorXd x = native_bounds_interior_point();
    const double mu = 0.1;

    Eigen::VectorXd gx = (Eigen::VectorXd(3) << 1.0, 2.0, 3.0).finished();
    tycho::solvers::detail::accumulate_bound_barrier_gradient(x, b, mu, gx);

    // One-sided entries also carry the damping derivative: +kappa_d*mu on a
    // lower-only bound, -kappa_d*mu on an upper-only one.
    EXPECT_DOUBLE_EQ(gx[0], 1.0 + (-mu / (2.0 - 1.0) + kKappaD * mu * 1.0));
    EXPECT_DOUBLE_EQ(gx[1], 2.0 + (mu / (8.0 - 5.0) - kKappaD * mu * 1.0));
    // The two-sided variable takes both terms onto the same entry, and no
    // damping on either.
    EXPECT_DOUBLE_EQ(gx[2], (3.0 + (-mu / (0.5 - 0.0) + kKappaD * mu * 0.0)) +
                                (mu / (1.0 - 0.5) - kKappaD * mu * 0.0));
}

TEST(NativeBounds, ZFormGradientAccumulatesTheMultipliersWithIpoptSigns) {
    const BoundSet b = native_bounds_three_var_set();
    const BoundDualState z = native_bounds_three_var_duals();

    Eigen::VectorXd gx = (Eigen::VectorXd(3) << 1.0, 2.0, 3.0).finished();
    tycho::solvers::detail::accumulate_bound_dual_terms(b, z, gx);

    // Dual infeasibility is grad f + J'lambda - z_L + z_U. No damping term:
    // variables 0 and 1 are one-sided, but the residual is undamped by
    // construction -- these expectations would not hold if it leaked in.
    EXPECT_DOUBLE_EQ(gx[0], 1.0 - 0.7);
    EXPECT_DOUBLE_EQ(gx[1], 2.0 + 0.25);
    EXPECT_DOUBLE_EQ(gx[2], 3.0 - 1.5 + 4.0);
}

TEST(NativeBounds, SigmaAccumulatesOntoAnExistingDiagonalBase) {
    const BoundSet b = native_bounds_three_var_set();
    const BoundDualState z = native_bounds_three_var_duals();
    const Eigen::VectorXd x = native_bounds_interior_point();

    // A non-zero base stands in for the restoration / SOE / proximal bases the
    // solver already writes to these slots: sigma composes, it does not replace.
    Eigen::VectorXd sigma = (Eigen::VectorXd(3) << 10.0, 20.0, 30.0).finished();
    tycho::solvers::detail::accumulate_bound_sigma(x, b, z, sigma);

    // Pure z/d, with no damping on the one-sided entries: the condensed
    // curvature is undamped by construction, and these expectations pin it.
    EXPECT_DOUBLE_EQ(sigma[0], 10.0 + 0.7 / (2.0 - 1.0));
    EXPECT_DOUBLE_EQ(sigma[1], 20.0 + 0.25 / (8.0 - 5.0));
    EXPECT_DOUBLE_EQ(sigma[2], 30.0 + 1.5 / (0.5 - 0.0) + 4.0 / (1.0 - 0.5));
}

// The mu-form and the z-form are NOT interchangeable, and this pins the one
// point at which they coincide. Getting the two seams the wrong way round is
// invisible on the central path and wrong everywhere else, so a test that only
// evaluated at a complementary point would not catch it -- hence both arms.
//
// Deliberately run on the two-sided set: the damping term lives in the mu-form
// only, so on a one-sided bound the two forms differ by kappa_d*mu even at
// perfect complementarity. That offset is real and is pinned by
// DampingAppliesToOneSidedBoundsInTheBarrierAccountOnly below; here it would
// only obscure the seam property under test.
TEST(NativeBounds, TheTwoGradientFormsAgreeOnlyOnTheCentralPath) {
    const BoundSet b = native_bounds_two_sided_set();
    const Eigen::VectorXd x = native_bounds_interior_point();
    const double mu = 0.1;

    BoundDualState central;
    central.z_lower_ = (Eigen::VectorXd(2) << mu / (2.0 - 1.0), mu / (0.5 - 0.0)).finished();
    central.z_upper_ = (Eigen::VectorXd(2) << mu / (8.0 - 2.0), mu / (1.0 - 0.5)).finished();

    Eigen::VectorXd mu_form = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd z_form = Eigen::VectorXd::Zero(3);
    tycho::solvers::detail::accumulate_bound_barrier_gradient(x, b, mu, mu_form);
    tycho::solvers::detail::accumulate_bound_dual_terms(b, central, z_form);
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(mu_form[i], z_form[i]);
    }

    // Off the central path they must differ, or the two seams would be the same
    // seam and the split would be untestable.
    BoundDualState off_central;
    off_central.z_lower_ = (Eigen::VectorXd(2) << 0.7, 1.5).finished();
    off_central.z_upper_ = (Eigen::VectorXd(2) << 0.25, 4.0).finished();
    Eigen::VectorXd off_path = Eigen::VectorXd::Zero(3);
    tycho::solvers::detail::accumulate_bound_dual_terms(b, off_central, off_path);
    EXPECT_NE(mu_form[0], off_path[0]);
}

// The kappa_d damping: present in phi_mu and in the mu-form gradient, on
// one-sided entries only, and absent from the residual and from sigma. Two
// variables, one bounded below only and one two-sided, so the presence and the
// absence are pinned side by side in a single hand calculation.
TEST(NativeBounds, DampingAppliesToOneSidedBoundsInTheBarrierAccountOnly) {
    BoundSet b;
    b.lower_idx_ = (Eigen::VectorXi(2) << 0, 1).finished();
    b.lower_val_ = (Eigen::VectorXd(2) << 1.0, 0.0).finished();
    b.lower_damp_ = (Eigen::VectorXd(2) << 1.0, 0.0).finished(); // var 0 one-sided, var 1 not
    b.upper_idx_ = (Eigen::VectorXi(1) << 1).finished();
    b.upper_val_ = (Eigen::VectorXd(1) << 4.0).finished();
    b.upper_damp_ = Eigen::VectorXd::Zero(1);

    const Eigen::VectorXd x = (Eigen::VectorXd(2) << 2.0, 1.0).finished();
    const double mu = 0.1;

    // phi_mu: the lower-only variable's log term gains kappa_d*mu*(x-l), the
    // two-sided variable's two log terms gain nothing.
    const double expected_phi = (-mu * std::log(2.0 - 1.0) + kKappaD * mu * (2.0 - 1.0)) +
                                (-mu * std::log(1.0 - 0.0)) + (-mu * std::log(4.0 - 1.0));
    EXPECT_DOUBLE_EQ(tycho::solvers::detail::bound_barrier_objective(x, b, mu), expected_phi);

    // The damping is what makes phi_mu grow as the lower-only variable runs
    // away, which is the whole point of it: without the term the log barrier
    // alone falls monotonically in that direction.
    const Eigen::VectorXd x_far = (Eigen::VectorXd(2) << 1.0e6, 1.0).finished();
    EXPECT_GT(tycho::solvers::detail::bound_barrier_objective(x_far, b, mu),
              tycho::solvers::detail::bound_barrier_objective(x, b, mu));

    // mu-form gradient: +kappa_d*mu on the one-sided entry only.
    Eigen::VectorXd gx = Eigen::VectorXd::Zero(2);
    tycho::solvers::detail::accumulate_bound_barrier_gradient(x, b, mu, gx);
    EXPECT_DOUBLE_EQ(gx[0], -mu / (2.0 - 1.0) + kKappaD * mu * 1.0);
    EXPECT_DOUBLE_EQ(gx[1], (-mu / (1.0 - 0.0) + kKappaD * mu * 0.0) +
                                (mu / (4.0 - 1.0) - kKappaD * mu * 0.0));

    // Residual and sigma: undamped, so the one-sided entry is pure -z_L and
    // pure z_L/d respectively.
    BoundDualState z;
    z.z_lower_ = (Eigen::VectorXd(2) << 0.7, 1.5).finished();
    z.z_upper_ = (Eigen::VectorXd(1) << 0.25).finished();

    Eigen::VectorXd resid = Eigen::VectorXd::Zero(2);
    tycho::solvers::detail::accumulate_bound_dual_terms(b, z, resid);
    EXPECT_DOUBLE_EQ(resid[0], -0.7);

    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(2);
    tycho::solvers::detail::accumulate_bound_sigma(x, b, z, sigma);
    EXPECT_DOUBLE_EQ(sigma[0], 0.7 / (2.0 - 1.0));
}

TEST(NativeBounds, EveryKernelIsANoOpOnAnEmptyBoundSet) {
    const BoundSet empty;
    const BoundDualState no_duals;
    const Eigen::VectorXd x = native_bounds_interior_point();

    EXPECT_FALSE(empty.any());
    EXPECT_DOUBLE_EQ(tycho::solvers::detail::bound_barrier_objective(x, empty, 0.1), 0.0);

    const Eigen::VectorXd before = (Eigen::VectorXd(3) << 1.0, 2.0, 3.0).finished();
    Eigen::VectorXd gx = before;
    tycho::solvers::detail::accumulate_bound_barrier_gradient(x, empty, 0.1, gx);
    tycho::solvers::detail::accumulate_bound_dual_terms(empty, no_duals, gx);
    tycho::solvers::detail::accumulate_bound_sigma(x, empty, no_duals, gx);
    for (int i = 0; i < 3; i++) {
        EXPECT_DOUBLE_EQ(gx[i], before[i]);
    }
}

// --- Interior push and multiplier seeding ---------------------------------

TEST(NativeBounds, PushProjectsAnOutsideGuessOntoTheKappaFormulas) {
    NativeBoundsHarness h(3);
    h.declare_bound(0, 0.0, 1.0);                // two-sided
    h.declare_bound(1, 2.0, kNativeBoundsInf);   // lower only
    h.declare_bound(2, -kNativeBoundsInf, -4.0); // upper only
    h.configure_bounds();
    ASSERT_NE(h.installed_bounds(), nullptr);

    // Chosen so the two components of the push are separable by hand: kappa1
    // decides the one-sided variables, and the interval cap never binds here.
    h.solver().settings().bound_push_ = 0.1;
    h.solver().settings().bound_interval_push_ = 0.25;

    // Below its lower bound, exactly ON its lower bound, and far above its
    // upper bound -- all three are projected, none is rejected.
    Eigen::VectorXd x = (Eigen::VectorXd(3) << -5.0, 2.0, 100.0).finished();
    h.push(x, 0.5);

    // p_L = min(0.1*max(1,|0|), 0.25*(1-0)) = 0.1
    EXPECT_DOUBLE_EQ(x[0], 0.0 + 0.1);
    // p_L = 0.1*max(1,|2|) = 0.2 (one-sided: no interval term)
    EXPECT_DOUBLE_EQ(x[1], 2.0 + 0.2);
    // p_U = 0.1*max(1,|-4|) = 0.4
    EXPECT_DOUBLE_EQ(x[2], -4.0 - 0.4);
}

TEST(NativeBounds, PushLeavesAStrictlyInteriorGuessAlone) {
    NativeBoundsHarness h(2);
    h.declare_bound(0, 0.0, 10.0);
    h.declare_bound(1, -1.0, 1.0);
    h.configure_bounds();
    h.solver().settings().bound_push_ = 0.1;
    h.solver().settings().bound_interval_push_ = 0.25;

    Eigen::VectorXd x = (Eigen::VectorXd(2) << 5.0, 0.0).finished();
    h.push(x, 0.5);
    EXPECT_DOUBLE_EQ(x[0], 5.0);
    EXPECT_DOUBLE_EQ(x[1], 0.0);
}

TEST(NativeBounds, TheIntervalTermCapsThePushOnANarrowTwoSidedBound) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 0.0, 1.0e-6);
    h.configure_bounds();
    h.solver().settings().bound_push_ = 0.1;
    h.solver().settings().bound_interval_push_ = 0.25;

    Eigen::VectorXd x = (Eigen::VectorXd(1) << -3.0).finished();
    h.push(x, 0.5);
    // kappa1 alone would push to 0.1, past the upper bound entirely; the
    // interval term caps it at 0.25*(1e-6 - 0) instead.
    EXPECT_DOUBLE_EQ(x[0], 0.25 * (1.0e-6 - 0.0));
    EXPECT_LT(x[0], 1.0e-6);
}

TEST(NativeBounds, MultipliersAreSeededFromTheBarrierParameterAndCapped) {
    NativeBoundsHarness h(2);
    h.declare_bound(0, 0.0, 1.0);    // two-sided, comfortably wide
    h.declare_bound(1, 0.0, 1.0e-6); // two-sided, narrow enough to cap
    h.configure_bounds();
    h.solver().settings().bound_push_ = 0.1;
    h.solver().settings().bound_interval_push_ = 0.25;

    const double mu0 = 0.5;
    Eigen::VectorXd x = (Eigen::VectorXd(2) << -5.0, -5.0).finished();
    h.push(x, mu0);

    // Variable 0 lands at 0.1, distances 0.1 (lower) and 0.9 (upper).
    ASSERT_EQ(h.duals().z_lower_.size(), 2);
    ASSERT_EQ(h.duals().z_upper_.size(), 2);
    EXPECT_DOUBLE_EQ(h.duals().z_lower_[0], mu0 / (0.1 - 0.0));
    EXPECT_DOUBLE_EQ(h.duals().z_upper_[0], mu0 / (1.0 - 0.1));

    // Variable 1's lower distance is 2.5e-7, so mu0/d is 2e6 -- far past the cap.
    EXPECT_DOUBLE_EQ(h.duals().z_lower_[1], kBoundMultInitCap);
    // Its upper distance is 7.5e-7, likewise capped.
    EXPECT_DOUBLE_EQ(h.duals().z_upper_[1], kBoundMultInitCap);

    // The step buffers are sized alongside and start at zero.
    ASSERT_EQ(h.duals().dz_lower_.size(), 2);
    ASSERT_EQ(h.duals().dz_upper_.size(), 2);
    EXPECT_DOUBLE_EQ(h.duals().dz_lower_[0], 0.0);
    EXPECT_DOUBLE_EQ(h.duals().dz_upper_[0], 0.0);
}

// --- Multiplier direction and committed update ----------------------------

TEST(NativeBounds, DualDirectionMatchesTheClosedFormWithMirroredSigns) {
    NativeBoundsHarness h(2);
    h.declare_bound(0, 1.0, kNativeBoundsInf);  // lower only
    h.declare_bound(1, -kNativeBoundsInf, 8.0); // upper only
    h.configure_bounds();
    ASSERT_NE(h.installed_bounds(), nullptr);

    h.duals().z_lower_ = (Eigen::VectorXd(1) << 0.7).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 0.25).finished();
    h.duals().dz_lower_ = Eigen::VectorXd::Zero(1);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(1);

    const Eigen::VectorXd x = (Eigen::VectorXd(2) << 2.0, 5.0).finished();
    const Eigen::VectorXd dx = (Eigen::VectorXd(2) << 0.5, -0.25).finished();
    const double mu = 0.1;
    h.direction(x, dx, mu);

    // dz_L = mu/(x-l) - z_L - (z_L/(x-l)) * dx
    const double dl = 2.0 - 1.0;
    EXPECT_DOUBLE_EQ(h.duals().dz_lower_[0], mu / dl - 0.7 - (0.7 / dl) * 0.5);
    // dz_U = mu/(u-x) - z_U + (z_U/(u-x)) * dx -- the upper distance shrinks as
    // x grows, so the curvature term enters with the opposite sign.
    const double du = 8.0 - 5.0;
    EXPECT_DOUBLE_EQ(h.duals().dz_upper_[0], mu / du - 0.25 + (0.25 / du) * (-0.25));
}

TEST(NativeBounds, CommitTakesTheDualFractionThenClipsFromAbove) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 1.0, kNativeBoundsInf);
    h.configure_bounds();

    const double mu = 0.1;
    const Eigen::VectorXd x_new = (Eigen::VectorXd(1) << 2.0).finished();
    const double d = 2.0 - 1.0;

    // A step that would land the multiplier far above kappa_sigma * mu / d.
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << 1.0e30).finished();
    h.duals().z_upper_ = Eigen::VectorXd::Zero(0);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(0);
    h.commit(0.5, x_new, mu, /*monotone_mu=*/true);
    EXPECT_DOUBLE_EQ(h.duals().z_lower_[0], kKappaSigma * mu / d);
}

TEST(NativeBounds, CommitClipsFromBelowToo) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 1.0, kNativeBoundsInf);
    h.configure_bounds();

    const double mu = 0.1;
    const Eigen::VectorXd x_new = (Eigen::VectorXd(1) << 2.0).finished();
    const double d = 2.0 - 1.0;

    // A step that would drive the multiplier through zero: the floor keeps it
    // strictly positive, which is what the barrier terms downstream require.
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << -1.0e30).finished();
    h.duals().z_upper_ = Eigen::VectorXd::Zero(0);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(0);
    h.commit(1.0, x_new, mu, /*monotone_mu=*/true);
    EXPECT_DOUBLE_EQ(h.duals().z_lower_[0], mu / (kKappaSigma * d));
    EXPECT_GT(h.duals().z_lower_[0], 0.0);
}

TEST(NativeBounds, CommitLeavesAnUnclippedStepExactlyWhereItLands) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 1.0, kNativeBoundsInf);
    h.configure_bounds();

    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << 0.4).finished();
    h.duals().z_upper_ = Eigen::VectorXd::Zero(0);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(0);
    h.commit(0.5, (Eigen::VectorXd(1) << 2.0).finished(), 0.1, /*monotone_mu=*/true);
    EXPECT_DOUBLE_EQ(h.duals().z_lower_[0], 1.0 + 0.5 * 0.4);
}

// Free-mu mode clamps against the average complementarity at the new point
// (capped), not against the barrier parameter -- so the SAME committed
// multiplier lands in two different places depending on which barrier schedule
// produced the step. One variable, lower bound only, no slacks, so the average
// is a single product and the whole thing is a hand calculation.
TEST(NativeBounds, TheClipBarrierParameterFollowsTheBarrierSchedule) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 1.0, kNativeBoundsInf);
    h.configure_bounds();

    const double mu = 0.1;
    const Eigen::VectorXd x_new = (Eigen::VectorXd(1) << 2.0).finished();
    const double d = 2.0 - 1.0;

    // A step landing the multiplier at 1e30, far above either ceiling.
    auto drive = [&](bool monotone) {
        h.duals().z_lower_ = (Eigen::VectorXd(1) << 0.0).finished();
        h.duals().dz_lower_ = (Eigen::VectorXd(1) << 1.0e30).finished();
        h.duals().z_upper_ = Eigen::VectorXd::Zero(0);
        h.duals().dz_upper_ = Eigen::VectorXd::Zero(0);
        h.commit(1.0, x_new, mu, monotone);
        return h.duals().z_lower_[0];
    };

    // Monotone: the ceiling is kappa_sigma * mu / d.
    EXPECT_DOUBLE_EQ(drive(true), kKappaSigma * mu / d);

    // Free: the average complementarity at the new point is z*d = 1e30, capped
    // at kFreeModeClipMuCap, so the ceiling is kappa_sigma * cap / d instead --
    // twelve orders of magnitude higher, and nothing like the monotone answer.
    EXPECT_DOUBLE_EQ(drive(false), kKappaSigma * kFreeModeClipMuCap / d);
}

// The fraction-to-boundary leg for bound multipliers does not exist yet, so a
// committed step can still drive one non-positive -- which would make the
// free-mu average complementarity non-positive and the clamp bracket
// meaningless. The fallback to the barrier parameter keeps the floor doing its
// job. Provably dead once that leg lands and every product is positive.
TEST(NativeBounds, FreeModeFallsBackToTheBarrierParameterOnANonPositiveAverage) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 1.0, kNativeBoundsInf);
    h.configure_bounds();

    const double mu = 0.1;
    const double d = 2.0 - 1.0;
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << -1.0e30).finished();
    h.duals().z_upper_ = Eigen::VectorXd::Zero(0);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(0);

    h.commit(1.0, (Eigen::VectorXd(1) << 2.0).finished(), mu, /*monotone_mu=*/false);
    EXPECT_DOUBLE_EQ(h.duals().z_lower_[0], mu / (kKappaSigma * d));
    EXPECT_GT(h.duals().z_lower_[0], 0.0);
}

// --- Settings validation ---------------------------------------------------

// settings() hands out a mutable reference and promises re-validation at solve
// entry. kappa2 at or above one half lets the lower and upper projections
// cross, landing the point on or outside a bound, whose barrier term then takes
// the log of a non-positive number with no diagnostic -- so the range is
// enforced rather than assumed.
TEST(NativeBounds, TheIntervalPushIsRangeCheckedByValidate) {
    PSIOPT opt;
    EXPECT_NO_THROW(opt.settings().validate());

    opt.settings().bound_interval_push_ = 0.5;
    EXPECT_THROW(opt.settings().validate(), std::invalid_argument);

    opt.settings().bound_interval_push_ = 1.0;
    EXPECT_THROW(opt.settings().validate(), std::invalid_argument);

    opt.settings().bound_interval_push_ = 0.0;
    EXPECT_THROW(opt.settings().validate(), std::invalid_argument);

    opt.settings().bound_interval_push_ = -1.0e-3;
    EXPECT_THROW(opt.settings().validate(), std::invalid_argument);

    opt.settings().bound_interval_push_ = 0.25;
    EXPECT_NO_THROW(opt.settings().validate());
}

// --- Assembly -------------------------------------------------------------

// Sigma reaches the (1,1) diagonal of the assembled KKT through the same
// primal-diagonal slots the restoration and SOE bases use -- no new rows, no
// change in dimension.
TEST(NativeBounds, SigmaLandsOnThePrimalDiagonalWithoutGrowingTheSystem) {
    NativeBoundsHarness h(2);
    h.declare_bound(0, 1.0, kNativeBoundsInf);
    h.declare_bound(1, -kNativeBoundsInf, 8.0);
    h.configure_bounds();
    ASSERT_NE(h.installed_bounds(), nullptr);

    const Eigen::VectorXd primals = (Eigen::VectorXd(2) << 2.0, 5.0).finished();
    const int dim_with_bounds = h.dim();

    // Baseline: the same evaluation with the bound pointer withdrawn, i.e. the
    // assembly this solver performed before native bounds existed.
    const BoundSet *installed = h.installed_bounds();
    h.install_bounds(nullptr);
    h.clear_primal_diags();
    const Eigen::VectorXd base_diag = h.assemble_primal_diagonal(primals);

    h.install_bounds(installed);
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 0.7).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 0.25).finished();
    h.duals().dz_lower_ = Eigen::VectorXd::Zero(1);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(1);
    const Eigen::VectorXd bound_diag = h.assemble_primal_diagonal(primals);

    // EXPECT_NEAR, not EXPECT_DOUBLE_EQ: the assertion is a difference of two
    // assembled sums, so it carries the rounding of (H + sigma) - H rather than
    // reproducing sigma bit for bit. The kernel's exact value is pinned by the
    // kernel tests above; what this one is about is that sigma reached the
    // matrix, on the right diagonal entry, at the right magnitude.
    EXPECT_EQ(h.dim(), dim_with_bounds);
    EXPECT_NEAR(bound_diag[0] - base_diag[0], 0.7 / (2.0 - 1.0), 1.0e-12);
    EXPECT_NEAR(bound_diag[1] - base_diag[1], 0.25 / (8.0 - 5.0), 1.0e-12);

    // Withdrawing the bounds again must reproduce the baseline exactly: the
    // bound-carrying evaluation has to leave the primal-diagonal coefficients
    // back at zero, or the next evaluation would inherit its sigma. No
    // clear_primal_diags() here -- that reset is the property under test.
    h.install_bounds(nullptr);
    const Eigen::VectorXd base_again = h.assemble_primal_diagonal(primals);
    EXPECT_DOUBLE_EQ(base_again[0], base_diag[0]);
    EXPECT_DOUBLE_EQ(base_again[1], base_diag[1]);
}

// --- Neutrality on a problem with no variable bounds ----------------------

TEST(NativeBounds, AProblemWithoutBoundsLeavesTheWholeBoundStateEmpty) {
    NativeBoundsHarness h(2);
    h.configure_bounds();

    EXPECT_FALSE(h.nlp().variable_bound_set().any());
    EXPECT_EQ(h.installed_bounds(), nullptr);

    // The push is the one entry point that runs unconditionally at solve entry;
    // with no bounds it must touch neither the guess nor the multiplier state.
    Eigen::VectorXd x = (Eigen::VectorXd(2) << -5.0, 100.0).finished();
    h.push(x, 0.5);
    EXPECT_DOUBLE_EQ(x[0], -5.0);
    EXPECT_DOUBLE_EQ(x[1], 100.0);
    EXPECT_EQ(h.duals().z_lower_.size(), 0);
    EXPECT_EQ(h.duals().z_upper_.size(), 0);
    EXPECT_EQ(h.duals().dz_lower_.size(), 0);
    EXPECT_EQ(h.duals().dz_upper_.size(), 0);

    // And the two step helpers are no-ops rather than out-of-range reads.
    h.direction(x, Eigen::VectorXd::Zero(2), 0.1);
    h.commit(1.0, x, 0.1, /*monotone_mu=*/true);
    EXPECT_EQ(h.duals().z_lower_.size(), 0);
}

// The invariant every globalization component's bound branch is guarded on: a
// default-constructed context carries no bound set, so those branches are
// unreachable unless PSIOPT deliberately installs one.
TEST(NativeBounds, ADefaultedSolverContextCarriesNoBoundState) {
    NativeBoundsHarness h(2);
    h.configure_bounds();

    tycho::solvers::SolverContext ctx = h.bare_context();
    EXPECT_EQ(ctx.bounds_, nullptr);
    EXPECT_EQ(ctx.bound_duals_, nullptr);
    EXPECT_EQ(ctx.restoration_, nullptr);
}
