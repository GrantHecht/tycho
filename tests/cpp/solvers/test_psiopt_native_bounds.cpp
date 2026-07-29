///////////////////////////////////////////////////////////////////////////////
// Native primal variable-bound machinery in PSIOPT: the barrier kernels, the
// interior push and multiplier seeding, the bound-multiplier Newton direction
// and its committed update with the kappa_sigma safeguard, and the condensed
// sigma diagonal's arrival in the assembled KKT.
//
// Then the step-length half: the fraction-to-boundary legs that keep a bounded
// variable and its multiplier strictly interior, the complementarity account
// the barrier parameter is driven by, and the first solves that run all of it
// end to end.
//
// Tests come in three kinds. Pure kernels are checked against a written-out
// formula. Single private helpers are driven through NativeBoundsHarness, which
// PSIOPT befriends (psiopt.h). End-to-end solves go through the public entry
// point and assert on where they land, what the multipliers say about the
// active bound, and -- through a late-callback probe -- that no iterate ever
// left the box on the way there.
//
// The last group is the neutrality check the whole feature rests on: a problem
// that declares no variable bounds leaves the bound state empty, leaves the
// solver's bound pointer null, and assembles a KKT diagonal byte-identical to
// the one it assembled before any of this existed.
///////////////////////////////////////////////////////////////////////////////

#include "optimal_control/oc_test_utils.h"
#include "solver_test_utils.h"

#include "tycho/detail/solvers/barrier_math.h"
#include "tycho/detail/solvers/bound_set.h"
#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/backtracking_line_search.h"
#include "tycho/detail/solvers/globalization/noop_recovery.h"
#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/soc.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/globalization/watchdog.h"
#include "tycho/detail/solvers/non_linear_program.h"
#include "tycho/detail/solvers/optimization_problem.h"
#include "tycho/detail/solvers/psiopt.h"
#include "tycho/detail/solvers/psiopt_presets.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

// The watchdog's arm / relaxed-trial / revert outcomes all return without
// delegating, so no acceptance strategy is reached on the path the revert test
// drives. Failing loudly if one is beats silently accepting a wrong dispatch.
class NativeBoundsUnusedAcceptance : public tycho::solvers::AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    bool is_iterate_acceptable(const tycho::solvers::ProgressMeasures &,
                               const tycho::solvers::ProgressMeasures &,
                               const tycho::solvers::ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "the watchdog's own outcomes must not reach an acceptance strategy";
        return false;
    }
    bool
    is_infeasibility_sufficiently_reduced(const tycho::solvers::ProgressMeasures &,
                                          const tycho::solvers::ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
};

// Accepts the first trial it is shown, without evaluating anything. Lets the
// SOC test observe which direction the correction loop COMMITS without the
// verdict depending on a merit function's opinion of it.
class NativeBoundsAcceptingLineSearch : public tycho::solvers::AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    double classic_line_search(PSIOPT::LineSearchModes, double, double, double, double,
                               Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                               Eigen::VectorXd &, Eigen::VectorXd &,
                               tycho::solvers::IterateInfo &citer,
                               const std::vector<tycho::solvers::IterateInfo> &) override {
        citer.ls_iters_ = 0;
        citer.accepted_ = true;
        return 1.0;
    }
    bool is_iterate_acceptable(const tycho::solvers::ProgressMeasures &,
                               const tycho::solvers::ProgressMeasures &,
                               const tycho::solvers::ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "the classic driving path must not reach the generic surface";
        return false;
    }
    bool
    is_infeasibility_sufficiently_reduced(const tycho::solvers::ProgressMeasures &,
                                          const tycho::solvers::ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
};

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
    // `center` shifts every variable's objective term; the end-to-end tests use
    // it to place the unconstrained minimizer outside the box so a bound is
    // genuinely active at the solution.
    explicit NativeBoundsHarness(int num_vars, double center = 3.0) {
        prob_.set_vars(Eigen::VectorXd::Zero(num_vars));
        for (int i = 0; i < num_vars; i++) {
            native_bounds_add_shifted_square(prob_, i, center);
        }
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
    }

    PSIOPT &solver() { return *solver_; }
    NonLinearProgram &nlp() { return *prob_.nlp_; }
    int pv() const { return solver_->primal_vars_; }
    int dim() const { return solver_->kkt_dim_; }
    int ec() const { return solver_->equal_cons_; }
    int ic() const { return solver_->inequal_cons_; }

    // Adds x[i0] + x[i1] == target BEFORE transcription is re-run, for the
    // equality-only end-to-end case.
    void add_sum_equality(int i0, int i1, double target) {
        auto args = tycho::vf::Arguments<2>();
        auto con = args.coeff<0>() + args.coeff<1>() - target;
        prob_.add_equal_con(tycho::vf::GenericFunction<-1, -1>(con),
                            (Eigen::VectorXi(2) << i0, i1).finished());
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
    }

    // A full solve through the public entry point: run_phase_sequence does its
    // own bound classification (at the shipped relax factor), the interior push
    // and the whole iteration, so this is the end-to-end path rather than the
    // hand-driven one the kernel tests use.
    Eigen::VectorXd solve(const Eigen::VectorXd &x0) { return solver_->optimize(x0); }
    tycho::ConvergenceFlags flag() const { return solver_->result().converge_flag_; }

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

    // The same state on a PSIOPT this harness does not own -- a phase's
    // optimizer, or one driving a problem built outside the harness.
    // NativeBoundsHarness is the file's befriended type (psiopt.h), so every
    // test here reaches the private bound state through it rather than each
    // fixture needing its own friendship.
    static BoundDualState &duals_of(PSIOPT &opt) { return opt.bound_duals_; }

    // A SolverContext carrying the bound state, the way alg_impl builds one --
    // what the step-length mechanism reads to find the bound legs.
    tycho::solvers::SolverContext live_context() {
        tycho::solvers::SolverContext ctx = this->bare_context();
        ctx.bounds_ = solver_->bounds_;
        ctx.bound_duals_ = &solver_->bound_duals_;
        return ctx;
    }

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
    // The nested soft pre-stage's primal-dual error at one point, with the base
    // stationarity block handed over explicitly.
    double pd_error(Eigen::VectorXd &xsl, Eigen::VectorXd &rhs, const Eigen::VectorXd &prim_base,
                    double mu) {
        tycho::solvers::KKTVector vx = solver_->kkt_view(xsl);
        tycho::solvers::KKTVector vr = solver_->kkt_view(rhs);
        return solver_->primal_dual_error(vx, vr, prim_base, mu);
    }

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

    // The damping is what makes phi_mu eventually grow as the lower-only
    // variable runs away: without the term the log barrier alone falls
    // monotonically in that direction. The growth is asymptotic — with
    // kappa_d = 1e-5 the linear term only overtakes -mu*ln(d) past
    // d ~ 1.4e6 — so the far point must sit well beyond the crossover.
    const Eigen::VectorXd x_far = (Eigen::VectorXd(2) << 1.0e8, 1.0).finished();
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

    // A step landing the multiplier at 1e-12 -- far below the 1e-11 floor, but
    // still strictly positive, which is now the only thing the
    // fraction-to-boundary rule permits. The floor lifts it back onto a bracket
    // the barrier terms downstream can use. (The step has to be written as
    // -(1 - 1e-12) rather than -1 + 1e-12: at this magnitude the second form
    // rounds to exactly -1 and lands z on exactly zero, which is the one value
    // the boundary rule forbids.)
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << -(1.0 - 1.0e-12)).finished();
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

// --- Fraction-to-boundary legs ---------------------------------------------

namespace {

// Runs the step-length mechanism's scaling on a hand-built direction and hands
// back the two fractions it chose. `dxsl` is scaled in place, exactly as the
// solver's own call scales DXSL.
void native_bounds_scale_step(NativeBoundsHarness &h, Eigen::VectorXd &xsl, Eigen::VectorXd &dxsl,
                              double bfrac, double &alphap, double &alphad) {
    tycho::solvers::BacktrackingLineSearch mech;
    tycho::solvers::SolverContext ctx = h.live_context();
    mech.max_primal_dual_step(xsl, dxsl, bfrac, alphap, alphad, ctx);
}

} // namespace

// tau applied to the distance to the bound, through the same kernel and with
// the same bfrac the slack block uses. One variable in [0, 1] at 0.5, stepping
// straight at the lower bound: the cap has to leave exactly (1-tau) of the
// distance behind.
TEST(NativeBounds, PrimalStepIsCappedByTheDistanceToTheLowerBound) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 0.0, 1.0);
    h.configure_bounds();
    ASSERT_NE(h.installed_bounds(), nullptr);
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = Eigen::VectorXd::Zero(1);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(1);

    const double tau = 0.99;
    Eigen::VectorXd xsl = (Eigen::VectorXd(1) << 0.5).finished();
    Eigen::VectorXd dxsl = (Eigen::VectorXd(1) << -1.0).finished();
    double alphap = 1.0;
    double alphad = 1.0;
    native_bounds_scale_step(h, xsl, dxsl, tau, alphap, alphad);

    // -tau*0.5 / -1.0
    EXPECT_DOUBLE_EQ(alphap, -tau * 0.5 / -1.0);
    // The direction is scaled in place by that fraction, and the point it lands
    // on keeps exactly (1-tau) of its original distance to the bound.
    EXPECT_DOUBLE_EQ(dxsl[0], -1.0 * alphap);
    EXPECT_GT(xsl[0] + dxsl[0], 0.0);
    EXPECT_NEAR(xsl[0] + dxsl[0], 0.5 * (1.0 - tau), 1.0e-15);
}

// The upper side mirrors it: the distance shrinks as x grows, so the gathered
// direction is -dx and a positive dx is what triggers the cap.
TEST(NativeBounds, PrimalStepIsCappedByTheDistanceToTheUpperBound) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 0.0, 1.0);
    h.configure_bounds();
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = Eigen::VectorXd::Zero(1);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(1);

    const double tau = 0.99;
    Eigen::VectorXd xsl = (Eigen::VectorXd(1) << 0.5).finished();
    Eigen::VectorXd dxsl = (Eigen::VectorXd(1) << 2.0).finished();
    double alphap = 1.0;
    double alphad = 1.0;
    native_bounds_scale_step(h, xsl, dxsl, tau, alphap, alphad);

    EXPECT_DOUBLE_EQ(alphap, -tau * 0.5 / -2.0);
    EXPECT_LT(xsl[0] + dxsl[0], 1.0);
    EXPECT_NEAR(1.0 - (xsl[0] + dxsl[0]), 0.5 * (1.0 - tau), 1.0e-15);
}

// The dual leg is the same rule on the multipliers themselves, and it lands in
// alphad rather than alphap -- so a multiplier heading for zero shortens the
// dual fraction without shortening the primal one.
TEST(NativeBounds, DualStepIsCappedByTheBoundMultipliers) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 0.0, kNativeBoundsInf);
    h.configure_bounds();
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << -2.0).finished();
    h.duals().z_upper_ = Eigen::VectorXd::Zero(0);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(0);

    const double tau = 0.99;
    // The primal moves away from its bound, so only the dual leg can bind.
    Eigen::VectorXd xsl = (Eigen::VectorXd(1) << 0.5).finished();
    Eigen::VectorXd dxsl = (Eigen::VectorXd(1) << 1.0).finished();
    double alphap = 1.0;
    double alphad = 1.0;
    native_bounds_scale_step(h, xsl, dxsl, tau, alphap, alphad);

    EXPECT_DOUBLE_EQ(alphap, 1.0);
    EXPECT_DOUBLE_EQ(alphad, -tau * 1.0 / -2.0);
    // z + alphad*dz keeps exactly (1-tau) of the multiplier.
    EXPECT_NEAR(1.0 + alphad * -2.0, 1.0 * (1.0 - tau), 1.0e-15);
}

// A direction that moves every bounded quantity inward caps nothing: the legs
// are a safeguard, not a throttle.
TEST(NativeBounds, AnInwardDirectionIsNotCappedByTheBoundLegs) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 0.0, 1.0);
    h.configure_bounds();
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = (Eigen::VectorXd(1) << 0.5).finished();
    h.duals().dz_upper_ = (Eigen::VectorXd(1) << 0.5).finished();

    Eigen::VectorXd xsl = (Eigen::VectorXd(1) << 0.5).finished();
    Eigen::VectorXd dxsl = (Eigen::VectorXd(1) << 0.0).finished();
    double alphap = 1.0;
    double alphad = 1.0;
    native_bounds_scale_step(h, xsl, dxsl, 0.99, alphap, alphad);

    EXPECT_DOUBLE_EQ(alphap, 1.0);
    EXPECT_DOUBLE_EQ(alphad, 1.0);
}

// --- Complementarity account ------------------------------------------------

// The bound pairs join the account as a count-weighted union, exactly the shape
// the nested-restoration elastics are folded in with. Hand-built over two
// bounds and no slacks, so the union IS the bound set and every number is
// checkable by inspection.
TEST(NativeBounds, BoundPairsJoinTheComplementarityAccountByCount) {
    const BoundSet b = native_bounds_three_var_set();
    const BoundDualState z = native_bounds_three_var_duals();
    const Eigen::VectorXd x = native_bounds_interior_point();

    // Pairs: (2-1)*0.7 = 0.7, (0.5-0)*1.5 = 0.75, (8-5)*0.25 = 0.75,
    //        (1-0.5)*4.0 = 2.0.
    double avg = 0.0;
    double mn = 0.0;
    double mx = 0.0;
    tycho::solvers::detail::augment_bound_complementarity(x, b, z, 0, avg, mn, mx);
    EXPECT_DOUBLE_EQ(mn, 0.7);
    EXPECT_DOUBLE_EQ(mx, 2.0);
    EXPECT_DOUBLE_EQ(avg, (0.7 + 0.75 + 0.75 + 2.0) / 4.0);
}

// With slack pairs already reduced, the union average is count-weighted against
// them and the base aggregates are NOT re-reduced -- which is what keeps the
// slack reduction's ordering (and therefore mu) untouched.
TEST(NativeBounds, TheUnionAverageIsWeightedAgainstTheSlackPairCount) {
    const BoundSet b = native_bounds_three_var_set();
    const BoundDualState z = native_bounds_three_var_duals();
    const Eigen::VectorXd x = native_bounds_interior_point();

    // Two slack pairs already reduced to an average of 3.0, min 1.0, max 5.0.
    double avg = 3.0;
    double mn = 1.0;
    double mx = 5.0;
    tycho::solvers::detail::augment_bound_complementarity(x, b, z, 2, avg, mn, mx);

    const double bsum = 0.7 + 0.75 + 0.75 + 2.0;
    EXPECT_DOUBLE_EQ(avg, (3.0 * 2.0 + bsum) / 6.0);
    EXPECT_DOUBLE_EQ(mn, 0.7); // union min: the bound side is smaller
    EXPECT_DOUBLE_EQ(mx, 5.0); // union max: the slack side is larger
}

TEST(NativeBounds, AnEmptyBoundSetLeavesTheComplementarityAggregatesAlone) {
    const BoundSet empty;
    const BoundDualState no_duals;
    double avg = 3.0;
    double mn = 1.0;
    double mx = 5.0;
    tycho::solvers::detail::augment_bound_complementarity(native_bounds_interior_point(), empty,
                                                          no_duals, 2, avg, mn, mx);
    EXPECT_DOUBLE_EQ(avg, 3.0);
    EXPECT_DOUBLE_EQ(mn, 1.0);
    EXPECT_DOUBLE_EQ(mx, 5.0);
}

// --- End-to-end solves ------------------------------------------------------

namespace {

constexpr const char *kNativeBoundsEnvVar = "TYCHO_DEV_NATIVE_BOUNDS";

// Sets the bring-up switch for the lifetime of the guard and restores the
// previous state after. The switch is read fresh at every declaration site, so
// it has to be held across phase construction, not across the solve.
class NativeBoundsSwitchGuard {
  public:
    explicit NativeBoundsSwitchGuard(bool on) {
        const char *prev = std::getenv(kNativeBoundsEnvVar);
        had_previous_ = prev != nullptr;
        if (had_previous_)
            previous_ = prev;
        set(on ? "1" : "0");
    }
    ~NativeBoundsSwitchGuard() {
        if (had_previous_)
            set(previous_.c_str());
        else
            unset();
    }

    NativeBoundsSwitchGuard(const NativeBoundsSwitchGuard &) = delete;
    NativeBoundsSwitchGuard &operator=(const NativeBoundsSwitchGuard &) = delete;

  private:
    static void set(const char *value) {
#ifdef _WIN32
        _putenv_s(kNativeBoundsEnvVar, value);
#else
        setenv(kNativeBoundsEnvVar, value, 1);
#endif
    }
    static void unset() {
#ifdef _WIN32
        _putenv_s(kNativeBoundsEnvVar, "");
#else
        unsetenv(kNativeBoundsEnvVar);
#endif
    }

    bool had_previous_ = false;
    std::string previous_;
};

// Watches every committed iterate for a bound violation. The whole point of the
// fraction-to-boundary legs is that this never fires: a barrier term evaluated
// at a point on or outside its bound is a log of a non-positive number, so an
// iterate that leaves the box does not merely degrade the solve, it ends it.
struct NativeBoundsInteriorProbe {
    // How the probe finds the bound set, resolved AT CALLBACK TIME rather than
    // captured up front. A phase installs a FRESH NonLinearProgram at every
    // transcription, and for a phase the transcription runs INSIDE the solve --
    // so a pointer taken before the solve is null when nothing has been
    // transcribed yet, and stale afterwards even when something has. Both of
    // those are segfaults in a callback, and both were.
    std::function<const NonLinearProgram *()> resolve;

    int iterates = 0;
    int violations = 0;
    int unresolved = 0;
    int bounds_seen = 0;
    double min_distance = std::numeric_limits<double>::infinity();
    double last_kkt_inf = -1.0;
    double last_barr_inf = -1.0;

    // Optional dual-side watch. The primal box is only half of "the bounds stay
    // hard": a multiplier that reached zero would mean the dual
    // fraction-to-boundary leg (or the kappa_sigma clip behind it) stopped
    // running, and every sigma = z/d built from it would be wrong. Left null by
    // callers that only care about the primal side.
    const BoundDualState *duals = nullptr;
    int multipliers_seen = 0;
    double min_multiplier = std::numeric_limits<double>::infinity();

    // Optional per-iterate history, for the caller that compares one committed
    // iterate against the one before it. Off by default: a phase problem's
    // primal block is thousands of entries wide and no other caller reads it.
    bool record_history = false;
    std::vector<Eigen::VectorXd> primal_history;
    std::vector<Eigen::VectorXd> multiplier_history;

    // Installs itself as the solver's late callback. The callback sees the
    // iterate in the solver's REDUCED space, which is the space the bound set's
    // indices are already recorded in -- so the two index each other directly,
    // with no mapping.
    //
    // What it measures is distance to the RELAXED bound, which is what BoundSet
    // stores: the classification widens every declared bound by
    // bound_relax_factor * max(1, |bound|). That is the right box for this
    // assertion -- it is the one every barrier term divides by, so a violation
    // here is a non-finite barrier and the end of the solve. It is NOT a
    // statement that the user's declared bound was satisfied, which is a
    // strictly tighter claim this probe does not make.
    void install(PSIOPT &opt) {
        opt.set_late_callback([this](const tycho::solvers::IterateInfo &iter,
                                     tycho::ConstEigenRef<Eigen::VectorXd> xsl,
                                     tycho::ConstEigenRef<Eigen::VectorXd>) {
            this->iterates++;
            this->last_kkt_inf = iter.kkt_inf_;
            this->last_barr_inf = iter.barr_inf_;

            const NonLinearProgram *nlp = this->resolve ? this->resolve() : nullptr;
            if (nlp == nullptr) {
                // Counted rather than dereferenced: every caller asserts this is
                // zero, so an unresolvable NLP fails the test instead of ending
                // the process.
                this->unresolved++;
                return 0;
            }

            const BoundSet &b = nlp->variable_bound_set();
            this->bounds_seen =
                static_cast<int>(b.lower_idx_.size()) + static_cast<int>(b.upper_idx_.size());
            for (int k = 0; k < b.lower_idx_.size(); k++) {
                const double d = xsl[b.lower_idx_[k]] - b.lower_val_[k];
                this->min_distance = std::min(this->min_distance, d);
                if (!(d > 0.0))
                    this->violations++;
            }
            for (int k = 0; k < b.upper_idx_.size(); k++) {
                const double d = b.upper_val_[k] - xsl[b.upper_idx_[k]];
                this->min_distance = std::min(this->min_distance, d);
                if (!(d > 0.0))
                    this->violations++;
            }

            if (this->duals != nullptr) {
                const Eigen::VectorXd &zl = this->duals->z_lower_;
                const Eigen::VectorXd &zu = this->duals->z_upper_;
                this->multipliers_seen = static_cast<int>(zl.size()) + static_cast<int>(zu.size());
                for (int k = 0; k < zl.size(); k++)
                    this->min_multiplier = std::min(this->min_multiplier, zl[k]);
                for (int k = 0; k < zu.size(); k++)
                    this->min_multiplier = std::min(this->min_multiplier, zu[k]);
                if (this->record_history) {
                    Eigen::VectorXd z(zl.size() + zu.size());
                    z.head(zl.size()) = zl;
                    z.tail(zu.size()) = zu;
                    this->multiplier_history.push_back(z);
                }
            }
            if (this->record_history)
                this->primal_history.emplace_back(xsl.head(nlp->primal_vars_));
            return 0;
        });
    }

    // The assertions every caller makes. Bundled because the emptiness check is
    // the one that stops the rest from being vacuous: min_distance starts at
    // +infinity and violations at zero, so on an empty bound set the interior
    // claims pass without measuring anything.
    void expect_every_iterate_interior() const {
        EXPECT_GT(this->iterates, 0) << "no iterate was observed";
        EXPECT_EQ(this->unresolved, 0) << "the bound set could not be resolved at callback time";
        EXPECT_GT(this->bounds_seen, 0) << "the bound set was empty; the interior checks below "
                                           "would pass without measuring anything";
        EXPECT_EQ(this->violations, 0);
        EXPECT_GT(this->min_distance, 0.0);
    }

    // The dual half, for the callers that set `duals`. Same emptiness guard as
    // above: min_multiplier starts at +infinity, so without the count check the
    // positivity claim passes on a solver carrying no multipliers at all.
    void expect_every_bound_multiplier_positive() const {
        EXPECT_GT(this->multipliers_seen, 0) << "no bound multiplier was observed";
        EXPECT_GT(this->min_multiplier, 0.0);
    }
};

// Resolves through a phase, which replaces its NonLinearProgram on every
// transcription -- including the one its own solve performs. The shared_ptr is
// captured by value so the resolver cannot dangle if handed a temporary.
std::function<const NonLinearProgram *()> native_bounds_phase_resolver(auto phase) {
    return [phase]() -> const NonLinearProgram * { return phase->nlp_.get(); };
}

// Builds the standard brachistochrone phase with the bring-up switch held in
// the requested state across the declaration sites -- which is the only moment
// the switch is read. Returning from inside the guard's scope keeps the phase
// alive past it.
auto native_bounds_build_brach(bool native) {
    NativeBoundsSwitchGuard guard(native);
    return TychoTest::make_brach_phase(100, 32);
}

} // namespace

// (a) The box-constrained QP. min (x-2)^2 subject to 0 <= x <= 1 -- no
// constraint rows at all, so every barrier term in the problem is a bound term
// and the whole chain (push, sigma, complementarity, mu schedule, both
// fraction-to-boundary legs, the z-form residual) has to work with no slack
// block to lean on. The minimizer sits outside the box, so the upper bound is
// active and its multiplier is the whole story at the solution.
TEST(NativeBounds, BoxConstrainedQpConvergesToItsActiveBound) {
    NativeBoundsHarness h(1, /*center=*/2.0);
    h.declare_bound(0, 0.0, 1.0);

    NativeBoundsInteriorProbe probe;
    probe.resolve = [&h]() -> const NonLinearProgram * { return &h.nlp(); };
    probe.install(h.solver());

    // Started ON the lower bound, so the interior push has to move it before the
    // first evaluation can even be taken.
    const Eigen::VectorXd sol = h.solve((Eigen::VectorXd(1) << 0.0).finished());

    EXPECT_EQ(h.flag(), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(sol[0], 1.0, 1.0e-5);

    // Stationarity at the solution is 2(x-2) - z_L + z_U = 0 with the lower
    // bound inactive, so z_U -> 2 and z_L -> 0.
    ASSERT_EQ(h.duals().z_upper_.size(), 1);
    ASSERT_EQ(h.duals().z_lower_.size(), 1);
    EXPECT_NEAR(h.duals().z_upper_[0], 2.0, 1.0e-3);
    EXPECT_NEAR(h.duals().z_lower_[0], 0.0, 1.0e-3);

    // The convergence verdict is taken on honest residuals, asserted rather than
    // inferred from the flag. The barrier error is the one worth naming: on a
    // problem with no slack pairs it is ENTIRELY the bound pairs' complementarity,
    // so a barrier error that both moved (it is not the -1 sentinel the probe
    // starts at) and landed inside tolerance is direct evidence that the bound
    // pairs reached the complementarity account -- the vacuous-zero failure the
    // guard disjunct exists to prevent would satisfy the tolerance too, but only
    // by never having been computed.
    // Strictly positive: an uncomputed barr_inf_ defaults to exactly 0.0, and
    // the real barrier error is a max over strictly positive products.
    EXPECT_GT(probe.last_barr_inf, 0.0);
    EXPECT_LT(probe.last_barr_inf, h.solver().settings().bar_tol_);
    EXPECT_GE(probe.last_kkt_inf, 0.0);
    EXPECT_LT(probe.last_kkt_inf, h.solver().settings().kkt_tol_);
    probe.expect_every_iterate_interior();
}

// (d) Equality constraints and a bounded variable, but NO inequality
// constraints -- the combination that exercises the guard disjuncts. Without
// them the step is never fraction-to-boundary scaled (compute_step's guard),
// the barrier parameter is never updated (alg_impl's governor guard), and the
// barrier residual is never reported (fill_residual_info's guard).
//
// min (x0-3)^2 + (x1-3)^2 s.t. x0 + x1 = 4, 0 <= x0 <= 1. On the constraint
// manifold the objective minimizes at x0 = 2, so the upper bound is active and
// the solution is (1, 3).
TEST(NativeBounds, EqualityOnlyProblemWithABoundedVariableSolves) {
    NativeBoundsHarness h(2);
    h.add_sum_equality(0, 1, 4.0);
    h.declare_bound(0, 0.0, 1.0);

    NativeBoundsInteriorProbe probe;
    probe.resolve = [&h]() -> const NonLinearProgram * { return &h.nlp(); };
    probe.install(h.solver());

    const Eigen::VectorXd sol = h.solve((Eigen::VectorXd(2) << 0.5, 3.5).finished());

    EXPECT_EQ(h.flag(), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(sol[0], 1.0, 1.0e-5);
    EXPECT_NEAR(sol[1], 3.0, 1.0e-5);
    EXPECT_EQ(h.ic(), 0);
    EXPECT_GT(h.ec(), 0);
    probe.expect_every_iterate_interior();
}

// (b) A real phase problem with a control path bound, solved natively. The
// bound is inactive at the brachistochrone optimum, which is the harder case
// for the barrier: it has to keep every iterate strictly inside a box it never
// needs to touch, without dragging the solve off the answer.
TEST(NativeBounds, PhaseWithAControlPathBoundSolvesNatively) {
    auto phase = native_bounds_build_brach(true);
    phase->optimizer_->set_print_level(3);

    NativeBoundsInteriorProbe probe;
    probe.resolve = native_bounds_phase_resolver(phase);
    probe.install(*phase->optimizer_);

    // A phase has no NonLinearProgram until it is transcribed, and its solve is
    // what transcribes it -- so nothing may read phase->nlp_ before this line.
    const auto status = phase->solve_optimize();
    EXPECT_EQ(status, tycho::ConvergenceFlags::CONVERGED);

    // The control bound was recorded natively, so no inequality rows carry it.
    ASSERT_TRUE(phase->nlp_->has_variable_bounds());

    const auto traj = phase->return_traj();
    EXPECT_NEAR(traj.back()[3], 1.8013, 0.01);

    // Every committed iterate stayed strictly inside the relaxed control box --
    // the box every barrier term in the problem divides by.
    probe.expect_every_iterate_interior();
}

// (c) The same problem both ways on one build. Both converge to the same
// trajectory, and the native formulation's KKT system is smaller by exactly two
// dimensions per lowered inequality row -- one constraint row and one slack
// column each, which is the bookkeeping the native path exists to avoid.
TEST(NativeBounds, NativeAndLoweredControlBoundsAgreeAndTheNativeKktIsSmaller) {
    auto native = native_bounds_build_brach(true);
    auto lowered = native_bounds_build_brach(false);
    native->optimizer_->set_print_level(3);
    lowered->optimizer_->set_print_level(3);

    ASSERT_EQ(native->solve_optimize(), tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(lowered->solve_optimize(), tycho::ConvergenceFlags::CONVERGED);

    EXPECT_NEAR(native->return_traj().back()[3], lowered->return_traj().back()[3], 1.0e-3);

    // The lowered formulation carries the bound as inequality rows; the native
    // one carries none of them.
    const int lowered_iq = lowered->nlp_->inequal_cons_;
    const int native_iq = native->nlp_->inequal_cons_;
    EXPECT_GT(lowered_iq - native_iq, 0);
    EXPECT_EQ(lowered->nlp_->kkt_dim_ - native->nlp_->kkt_dim_, 2 * (lowered_iq - native_iq));
    EXPECT_TRUE(native->nlp_->has_variable_bounds());
    EXPECT_FALSE(lowered->nlp_->has_variable_bounds());
}

// The recovery links reach step scaling through the SAME shared entry point the
// main path does, so a corrected direction inherits the bound legs with no code
// of its own -- but only because SOC's own guard was extended to ask about
// bounds as well as slacks. Driven directly against SocRecovery, with the
// correction forced to overshoot, so the test fails if either half regresses.
//
// The differential is what makes it able to fail: the same correction is run
// once with the bound set installed and once with it withdrawn. With the legs
// the committed step lands strictly inside; without them it leaves the box.
TEST(NativeBounds, CorrectedDirectionsInheritTheBoundFractionToBoundary) {
    NativeBoundsHarness h(2);
    h.add_sum_equality(0, 1, 4.0);
    h.declare_bound(0, 0.0, 1.0);
    // A converged solve leaves a live factorization for the correction's
    // back-substitution (the second-to-last iterate's factors -- convergence
    // breaks above the factorization) and an iterate sitting hard against the
    // upper bound, which is the only place a correction can overshoot one.
    const Eigen::VectorXd sol = h.solve((Eigen::VectorXd(2) << 0.5, 3.5).finished());
    ASSERT_EQ(h.flag(), tycho::ConvergenceFlags::CONVERGED);
    ASSERT_NE(h.installed_bounds(), nullptr);

    // SOC's own cap is read through the context, which holds the solver's
    // settings by const reference -- so it is set on the solver.
    h.solver().settings().max_soc_ = 4;
    tycho::solvers::SolverContext ctx = h.live_context();
    const BoundSet &b = *h.installed_bounds();
    ASSERT_EQ(b.upper_idx_.size(), 1);
    const int bounded = b.upper_idx_[0];
    const double upper = b.upper_val_[0];

    NativeBoundsAcceptingLineSearch acceptance;
    tycho::solvers::BacktrackingLineSearch mechanism;
    tycho::solvers::SocRecovery soc;

    // Runs one SOC dispatch and returns the distance the committed step leaves
    // between the bounded variable and its upper bound. A huge constraint
    // residual is what forces the correction to overshoot: the corrected
    // right-hand side carries it, so the back-substituted direction is large.
    auto committed_distance_to_bound = [&](bool with_bounds) {
        ctx.bounds_ = with_bounds ? &b : nullptr;

        Eigen::VectorXd xsl = Eigen::VectorXd::Zero(h.dim());
        xsl.head(h.pv()) = sol;
        Eigen::VectorXd dxsl = Eigen::VectorXd::Zero(h.dim());
        Eigen::VectorXd xsl2 = Eigen::VectorXd::Zero(h.dim());
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(h.dim());
        Eigen::VectorXd rhs2 = Eigen::VectorXd::Zero(h.dim());
        // Negative residual: the corrected direction is the NEGATED
        // back-substitution, so its sign follows -r and only a negative r
        // pushes the bounded variable further INTO its active upper bound.
        rhs.tail(h.ec() + h.ic()).setConstant(-1.0e3);

        tycho::solvers::IterateInfo citer;
        citer.first_rejection_iter_ = 0;         // SOC triggers on a FIRST-trial rejection
        citer.theta_at_first_rejection_ = 1.0e9; // ... whose trial was no better

        std::vector<tycho::solvers::IterateInfo> iters;
        double alpha = 1.0;
        double alphap = 1.0;
        double alphad = 1.0;
        int soc_steps = 0;
        int resolved_depth = tycho::solvers::kRecoveryDepthUnresolved;
        int activations = 0;

        const auto action = soc.on_step_rejected(
            citer, iters, ctx, acceptance, mechanism, PSIOPT::LineSearchModes::AUGLANG,
            /*obj_scale=*/1.0, /*mu=*/0.1, /*prim_obj=*/0.0, /*barr_obj=*/0.0, xsl, dxsl, xsl2, rhs,
            rhs2, alpha, alphap, alphad, soc_steps, resolved_depth, activations);

        EXPECT_EQ(action, tycho::solvers::RecoveryChain::Action::kRetry);
        EXPECT_GT(soc_steps, 0) << "the correction never ran, so nothing was scaled";
        return upper - (xsl[bounded] + alpha * dxsl[bounded]);
    };

    // With the legs: the corrected step is capped short of the bound.
    const double with_legs = committed_distance_to_bound(true);
    EXPECT_GT(with_legs, 0.0);

    // Without them: the same correction walks out of the box, which is what the
    // legs and SOC's guard disjunct together prevent.
    const double without_legs = committed_distance_to_bound(false);
    EXPECT_LE(without_legs, 0.0)
        << "the correction did not overshoot even unscaled, so the comparison proves nothing";
}

// --- Riders carried from the previous review --------------------------------

// The watchdog's revert discards a whole trajectory, and the bound multipliers
// are part of that trajectory: Sigma = z/d and the z-form residual are both
// built from them, so multipliers belonging to the discarded path would
// describe an iterate that no longer exists. Driven straight through the
// decorator's state machine -- arm, one relaxed trial, then window exhaustion --
// with the bound duals mutated in between, so the restore is observed rather
// than inferred.
TEST(NativeBounds, WatchdogRevertRestoresTheBoundMultipliersWithTheIterate) {
    TychoTest::InertSolverContext inert;
    inert.primal_vars_ = 1;
    inert.kkt_dim_ = 1;

    BoundSet bounds;
    bounds.lower_idx_ = (Eigen::VectorXi(1) << 0).finished();
    bounds.lower_val_ = (Eigen::VectorXd(1) << 0.0).finished();
    bounds.lower_damp_ = (Eigen::VectorXd(1) << 1.0).finished();

    BoundDualState duals;
    duals.z_lower_ = (Eigen::VectorXd(1) << 4.0).finished();
    duals.dz_lower_ = Eigen::VectorXd::Zero(1);

    tycho::solvers::SolverContext ctx = inert.ctx();
    ctx.bounds_ = &bounds;
    ctx.bound_duals_ = &duals;

    tycho::solvers::WatchdogRecovery watchdog(std::make_unique<tycho::solvers::NoopRecovery>());
    NativeBoundsUnusedAcceptance acceptance;
    tycho::solvers::BacktrackingLineSearch mechanism;

    Eigen::VectorXd v(1);
    int resolved_depth = tycho::solvers::kRecoveryDepthUnresolved;
    int activations = 0;
    double alpha = 1.0;

    auto drive = [&](double merit) {
        tycho::solvers::IterateInfo citer;
        std::vector<tycho::solvers::IterateInfo> iters;
        double alphap = 1.0;
        double alphad = 1.0;
        int soc_steps = 0;
        return watchdog.on_step_rejected(citer, iters, ctx, acceptance, mechanism,
                                         PSIOPT::LineSearchModes::AUGLANG, /*obj_scale=*/1.0,
                                         /*mu=*/1.0, merit, /*barr_obj=*/0.0, v, v, v, v, v, alpha,
                                         alphap, alphad, soc_steps, resolved_depth, activations);
    };

    // Arm. The snapshot of x and of z is taken here.
    v[0] = 7.0;
    for (int i = 0; i < tycho::solvers::kWatchdogShortenedIterTrigger; ++i)
        drive(10.0);
    ASSERT_EQ(activations, 1);

    // The watchdog window now explores: x moves and the multipliers move with
    // it, exactly as the solver's own commit would move them.
    v[0] = 99.0;
    duals.z_lower_[0] = 1234.0;
    drive(10.0); // relaxed trial, no progress

    // Window exhausted -- revert. Both halves of the iterate come back.
    const auto action = drive(10.0);
    EXPECT_EQ(action, tycho::solvers::RecoveryChain::Action::kRetry);
    EXPECT_DOUBLE_EQ(v[0], 7.0);
    ASSERT_EQ(duals.z_lower_.size(), 1);
    EXPECT_DOUBLE_EQ(duals.z_lower_[0], 4.0);
}

// The nested soft feasibility pre-stage compares a primal-dual error at two
// points, and its verdict is only meaningful if both are measured the same way.
// The live right-hand side's primal block is staged in the mu-form for part of
// each iteration, so the measurement takes the base block explicitly. This pins
// that the answer depends on the base block alone: staging the vector and
// passing the snapshot has to give the identical number, or the comparison
// acquires a direction and the pre-stage stops escalating when it should.
TEST(NativeBounds, TheSoftStepErrorDependsOnlyOnTheBaseStationarityBlock) {
    NativeBoundsHarness h(1);
    h.declare_bound(0, 0.0, 4.0);
    h.configure_bounds();
    ASSERT_NE(h.installed_bounds(), nullptr);
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 0.7).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 0.3).finished();
    h.duals().dz_lower_ = Eigen::VectorXd::Zero(1);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(1);

    const double mu = 0.1;
    Eigen::VectorXd xsl = (Eigen::VectorXd(1) << 1.0).finished();

    // Base staging: the primal block IS grad f + J'lambda.
    Eigen::VectorXd rhs_base = (Eigen::VectorXd(1) << 0.25).finished();
    const Eigen::VectorXd snapshot = rhs_base;
    const double from_base = h.pd_error(xsl, rhs_base, snapshot, mu);

    // Newton staging: the same point, with the mu-form bound terms installed on
    // the primal block the way alg_impl's bracket installs them. The base block
    // is handed over separately, as the bracket's snapshot hands it over.
    Eigen::VectorXd rhs_staged = snapshot;
    tycho::solvers::detail::accumulate_bound_barrier_gradient(xsl, *h.installed_bounds(), mu,
                                                              rhs_staged);
    ASSERT_NE(rhs_staged[0], snapshot[0]); // the staging really did change it
    const double from_staged = h.pd_error(xsl, rhs_staged, snapshot, mu);

    EXPECT_DOUBLE_EQ(from_staged, from_base);
}

// The l1 nested restoration mode and a bound set in the same solve. Before the
// fraction-to-boundary legs existed this combination could not be run at all --
// a bounded variable walked out of its box and the barrier terms went
// non-finite -- so the pre-stage's symmetry could only be argued. It runs now.
TEST(NativeBounds, NestedRestorationComposesWithVariableBounds) {
    NativeBoundsHarness h(2);
    h.add_sum_equality(0, 1, 4.0);
    h.declare_bound(0, 0.0, 1.0);
    h.solver().settings().restoration_mode_ = tycho::solvers::RestorationModes::l1_nested;

    NativeBoundsInteriorProbe probe;
    probe.resolve = [&h]() -> const NonLinearProgram * { return &h.nlp(); };
    probe.install(h.solver());

    const Eigen::VectorXd sol = h.solve((Eigen::VectorXd(2) << 0.5, 3.5).finished());

    EXPECT_EQ(h.flag(), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(sol[0], 1.0, 1.0e-5);
    EXPECT_NEAR(sol[1], 3.0, 1.0e-5);
    probe.expect_every_iterate_interior();
}

///////////////////////////////////////////////////////////////////////////////
// Globalization-mechanism matrix
//
// The four groups below are about the machinery that decides WHETHER a step is
// taken, rather than the barrier arithmetic that produced it. In order: that
// the infeasibility measure every acceptance test reads is untouched by the
// bound account; that a real restoration episode keeps both sides of every
// bound hard and takes exactly one multiplier step per committed iterate; that
// each of the five shipped preset configurations drives a natively bounded
// problem to an answer; and that the un-evaluable-trial machinery still
// composes on a bounded problem.
///////////////////////////////////////////////////////////////////////////////

// --- Theta purity -----------------------------------------------------------

namespace {

// Records the (theta, f, aux) triple the generic acceptance ladder builds and
// accepts the first trial it is shown, so exactly one triple is recorded per
// call. drives_classic_path() is false: that is what routes
// run_acceptance_backtrack into the generic loop, the only place in the solver
// where a ProgressMeasures is built from a live trial point.
class NativeBoundsCapturingAcceptance : public tycho::solvers::AcceptanceStrategy {
  public:
    tycho::solvers::ProgressMeasures current_{};
    tycho::solvers::ProgressMeasures trial_{};
    int calls_ = 0;

    bool drives_classic_path() const override { return false; }
    bool is_iterate_acceptable(const tycho::solvers::ProgressMeasures &current,
                               const tycho::solvers::ProgressMeasures &trial,
                               const tycho::solvers::ProgressMeasures &, double, double) override {
        current_ = current;
        trial_ = trial;
        calls_++;
        return true;
    }
    bool
    is_infeasibility_sufficiently_reduced(const tycho::solvers::ProgressMeasures &,
                                          const tycho::solvers::ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
};

} // namespace

// Native bounds are not inequality rows, so they never reach the KKT constraint
// block -- and theta is that block's L1 norm and nothing else. That exclusion is
// structural rather than defended by a guard, which is exactly why it is worth
// pinning: the filter, the funnel, the feasibility-stall detector and the
// restoration-entry gate all read this one number, and a barrier term leaking
// into it would silently retune every one of them.
//
// The differential is what makes the claim falsifiable. The SAME point is
// measured twice, once with the bound set installed and once with it withdrawn.
// theta has to come out identical; the barrier auxiliary has to come out
// different, by exactly the bound barrier objective -- otherwise the first
// equality would only be saying that nothing about the two runs differed.
TEST(NativeBounds, TheInfeasibilityMeasureExcludesTheBoundBarrierAccount) {
    NativeBoundsHarness h(2);
    h.add_sum_equality(0, 1, 4.0);
    h.declare_bound(0, 0.0, 1.0);
    h.configure_bounds();
    ASSERT_NE(h.installed_bounds(), nullptr);
    // The multiplier state is not read on this path -- the acceptance backtrack
    // runs no fraction-to-boundary scaling -- but it is sized here so the
    // context handed over is the one alg_impl would hand over.
    h.duals().z_lower_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().z_upper_ = (Eigen::VectorXd(1) << 1.0).finished();
    h.duals().dz_lower_ = Eigen::VectorXd::Zero(1);
    h.duals().dz_upper_ = Eigen::VectorXd::Zero(1);

    const double mu = 0.1;
    const int pv = h.pv();
    ASSERT_EQ(h.ic(), 0); // no slack barrier terms, so the auxiliary IS the bound account

    Eigen::VectorXd xsl = Eigen::VectorXd::Zero(h.dim());
    xsl.head(pv) = (Eigen::VectorXd(2) << 0.4, 2.0).finished();
    // Lands the trial at x0 = 0.6, strictly interior to [0, 1] on both sides,
    // and off the equality manifold so the trial theta is not zero either.
    Eigen::VectorXd dxsl = Eigen::VectorXd::Zero(h.dim());
    dxsl.head(pv) = (Eigen::VectorXd(2) << 0.2, 0.5).finished();
    Eigen::VectorXd xsl2 = Eigen::VectorXd::Zero(h.dim());
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(h.dim());
    Eigen::VectorXd rhs2 = Eigen::VectorXd::Zero(h.dim());
    // A non-zero constraint block, so the current-point measure is not a vacuous
    // zero on either arm.
    rhs.tail(h.ec() + h.ic()).setConstant(0.5);

    tycho::solvers::SolverContext ctx = h.live_context();
    tycho::solvers::BacktrackingLineSearch mechanism;
    const std::vector<tycho::solvers::IterateInfo> iters;

    auto measure = [&](bool with_bounds) {
        ctx.bounds_ = with_bounds ? h.installed_bounds() : nullptr;
        NativeBoundsCapturingAcceptance acceptance;
        tycho::solvers::IterateInfo citer;
        mechanism.run_acceptance_backtrack(PSIOPT::LineSearchModes::AUGLANG, /*obj_scale=*/1.0, mu,
                                           /*prim_obj=*/0.0, /*barr_obj=*/0.0, xsl, dxsl, xsl2, rhs,
                                           rhs2, acceptance, citer, iters, ctx);
        EXPECT_EQ(acceptance.calls_, 1) << "the ladder did not evaluate exactly one trial";
        return acceptance;
    };

    const NativeBoundsCapturingAcceptance with_bounds = measure(true);
    const NativeBoundsCapturingAcceptance without_bounds = measure(false);

    // The current-point measure IS the constraint block's L1 norm -- the very
    // expression the restoration-entry gate and the stall detector read off
    // their own RHS -- and the bound set does not enter it.
    const double cons_l1 = rhs.tail(h.ec() + h.ic()).template lpNorm<1>();
    EXPECT_GT(cons_l1, 0.0);
    EXPECT_DOUBLE_EQ(with_bounds.current_.infeasibility, cons_l1);
    EXPECT_DOUBLE_EQ(without_bounds.current_.infeasibility, cons_l1);

    // And the trial measure, at a point where a bound is genuinely being
    // approached: same number with the bounds and without them.
    EXPECT_GT(with_bounds.trial_.infeasibility, 0.0);
    EXPECT_DOUBLE_EQ(with_bounds.trial_.infeasibility, without_bounds.trial_.infeasibility);

    // The bound account went somewhere -- into the auxiliary slot, which is held
    // outside the (theta, f) pair on purpose. Without this the equality above
    // would be consistent with the bound set having been ignored entirely.
    const Eigen::VectorXd x_trial = xsl.head(pv) + dxsl.head(pv);
    const double bound_phi =
        tycho::solvers::detail::bound_barrier_objective(x_trial, *h.installed_bounds(), mu);
    EXPECT_NE(bound_phi, 0.0);
    EXPECT_DOUBLE_EQ(with_bounds.trial_.auxiliary - without_bounds.trial_.auxiliary, bound_phi);
    EXPECT_DOUBLE_EQ(without_bounds.trial_.auxiliary, 0.0);
}

// --- Restoration keeps the bounds hard --------------------------------------

namespace {

// A bounded problem that genuinely enters feasibility restoration:
//
//   min (x0-3)^2 + (x1-3)^2   s.t.   x0^2 + x1^2 = 1,   0.9 <= x0 <= 0.99
//
// started at (0.95, 5.0). The equality is violated by about 25 at the start and
// the box pins x0 into a sliver of the circle, so the line search exhausts its
// ladder and the restoration switch fires -- several times -- before the solve
// settles. It does settle: the answer is the circle point at the box's lower
// edge, (0.9, sqrt(0.19)), so a bound is active at the solution and the
// restoration episodes are entered AND left.
//
// The bound is declared straight on the NonLinearProgram rather than through
// the dev switch, the same way NativeBoundsHarness declares one: the switch
// only governs where an ODEPhase's declarations go, and this is not a phase.
class NativeBoundsRestorationHarness {
  public:
    explicit NativeBoundsRestorationHarness(tycho::solvers::RestorationModes mode) {
        prob_.set_vars(this->guess());
        {
            auto args = tycho::vf::Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_objective(tycho::vf::GenericFunction<-1, 1>((x0 - 3.0) * (x0 - 3.0) +
                                                                  (x1 - 3.0) * (x1 - 3.0)),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        {
            auto args = tycho::vf::Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_equal_con(tycho::vf::GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 1.0),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        prob_.optimizer_->set_print_level(3);
        prob_.optimizer_->set_qp_threads(1);
        prob_.transcribe();
        prob_.nlp_->set_variable_bound(0, 0.9, 0.99);
        prob_.nlp_->make_nlp(prob_.nlp_->primal_vars_, prob_.nlp_->equal_cons_,
                             prob_.nlp_->inequal_cons_);
        prob_.optimizer_->set_nlp(prob_.nlp_);
        prob_.optimizer_->settings().restoration_mode_ = mode;
        solver_ = prob_.optimizer_.get();
    }

    PSIOPT &solver() { return *solver_; }
    const NonLinearProgram *nlp() const { return prob_.nlp_.get(); }
    static Eigen::VectorXd guess() { return (Eigen::VectorXd(2) << 0.95, 5.0).finished(); }
    Eigen::VectorXd solve() { return solver_->optimize(this->guess()); }

  private:
    tycho::solvers::OptimizationProblem prob_;
    PSIOPT *solver_ = nullptr;
};

// The shared body of the two restoration-episode tests. The elastic (or
// proximal) relaxation applies to the CONSTRAINT ROWS; the bounds are not rows
// and are not relaxed, so the barrier terms and both fraction-to-boundary legs
// stay live for the whole episode -- entry, the iterations in mode, and the
// return. A bound that went slack anywhere in there would put a barrier term at
// the log of a non-positive number, so the primal watch below is not a quality
// check, it is the difference between an episode that runs and one that does
// not.
void native_bounds_expect_restoration_keeps_bounds_hard(tycho::solvers::RestorationModes mode) {
    NativeBoundsRestorationHarness h(mode);

    NativeBoundsInteriorProbe probe;
    probe.resolve = [&h]() -> const NonLinearProgram * { return h.nlp(); };
    probe.duals = &NativeBoundsHarness::duals_of(h.solver());
    probe.install(h.solver());

    const Eigen::VectorXd sol = h.solve();

    // Settled on the circle at the box's lower edge, so the episode was left
    // behind rather than merely survived.
    EXPECT_EQ(h.solver().result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(sol[0], 0.9, 1.0e-5);
    EXPECT_NEAR(sol[1], std::sqrt(1.0 - 0.81), 1.0e-4);

    // Restoration really was entered, and iterations really were spent in it --
    // without both, everything below is a claim about a solve that never went
    // near the machinery under test.
    EXPECT_GT(h.solver().result().last_feas_rest_entries_, 0)
        << "the solve never entered restoration; the episode assertions are vacuous";
    EXPECT_GT(h.solver().result().last_feas_rest_iters_, 0);
    EXPECT_GT(
        h.solver().result().recovery_depth_histogram_[tycho::solvers::kRecoveryDepthRestoration],
        0);

    // Both sides stayed hard: every committed iterate strictly inside the box,
    // and every bound multiplier strictly positive.
    probe.expect_every_iterate_interior();
    probe.expect_every_bound_multiplier_positive();
}

} // namespace

TEST(NativeBounds, ANestedRestorationEpisodeKeepsTheBoundsHardThroughout) {
    native_bounds_expect_restoration_keeps_bounds_hard(tycho::solvers::RestorationModes::l1_nested);
}

TEST(NativeBounds, AProximalRestorationEpisodeKeepsTheBoundsHardThroughout) {
    native_bounds_expect_restoration_keeps_bounds_hard(
        tycho::solvers::RestorationModes::proximal_switch);
}

// The bound multipliers move at exactly one place in the solver -- the line
// after XSL += alpha*DXSL -- and every recovery outcome, restoration entry and
// re-center included, reaches the iteration's end through it. A restoration
// dispatch discards its step by setting alpha to zero, so on that iteration the
// commit moves neither x nor z; the multiplier clip re-projects whatever z
// already was, against an x that did not move, and is therefore a no-op.
//
// That gives a directly observable statement of "exactly once": at every
// iterate whose primal block is bit-identical to the previous one -- which is
// what a zero alpha produces, and a restoration-entering solve produces several
// of them -- the bound multipliers must be bit-identical too. A second
// multiplier update on the restoration path, applied anywhere between the two
// callbacks, would move z while x stood still and this comparison would catch
// it.
TEST(NativeBounds, ADiscardedRestorationStepMovesNeitherTheIterateNorItsMultipliers) {
    NativeBoundsRestorationHarness h(tycho::solvers::RestorationModes::l1_nested);

    NativeBoundsInteriorProbe probe;
    probe.resolve = [&h]() -> const NonLinearProgram * { return h.nlp(); };
    probe.duals = &NativeBoundsHarness::duals_of(h.solver());
    probe.record_history = true;
    probe.install(h.solver());

    h.solve();
    ASSERT_EQ(h.solver().result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_GT(h.solver().result().last_feas_rest_entries_, 0);
    ASSERT_EQ(probe.primal_history.size(), probe.multiplier_history.size());
    ASSERT_GT(probe.primal_history.size(), 1u);

    int held_still = 0;
    for (std::size_t i = 1; i < probe.primal_history.size(); i++) {
        if (probe.primal_history[i] != probe.primal_history[i - 1])
            continue;
        held_still++;
        ASSERT_EQ(probe.multiplier_history[i].size(), probe.multiplier_history[i - 1].size());
        for (int k = 0; k < probe.multiplier_history[i].size(); k++) {
            EXPECT_DOUBLE_EQ(probe.multiplier_history[i][k], probe.multiplier_history[i - 1][k])
                << "bound multiplier " << k << " moved across iterate " << i
                << ", whose primal block did not";
        }
    }
    EXPECT_GT(held_still, 0) << "no iteration discarded its step, so nothing was compared";
}

// --- The five shipped preset configurations ---------------------------------

namespace {

// One preset, one natively bounded phase problem, solved end to end. The
// brachistochrone phase is the file's established native-bounds fixture: its
// control path bound is recorded natively (dev switch ON) rather than lowered
// into inequality rows, so every acceptance/governor/recovery combination a
// preset selects is driving a solve whose only barrier terms on that variable
// are bound terms.
//
// A failure here is a gap in the site enumeration behind the bound legs and the
// bound barrier account, not a preset problem: the presets touch nine Settings
// fields and no algorithm code.
void native_bounds_expect_preset_solves(const char *preset) {
    auto phase = native_bounds_build_brach(true);
    phase->optimizer_->set_print_level(3);
    phase->optimizer_->set_qp_threads(1);
    phase->optimizer_->apply_preset(preset);

    NativeBoundsInteriorProbe probe;
    probe.resolve = native_bounds_phase_resolver(phase);
    probe.install(*phase->optimizer_);

    const auto status = phase->solve_optimize();
    EXPECT_LE(status, tycho::ConvergenceFlags::ACCEPTABLE) << "preset: " << preset;

    // The bound is carried natively under every one of them -- a preset that
    // silently fell back to lowered rows would make the rest of this vacuous.
    ASSERT_TRUE(phase->nlp_->has_variable_bounds()) << "preset: " << preset;
    EXPECT_NEAR(phase->return_traj().back()[3], 1.8013, 0.01) << "preset: " << preset;
    probe.expect_every_iterate_interior();
}

} // namespace

TEST(NativeBounds, TheClassicPresetSolvesANativelyBoundedPhase) {
    native_bounds_expect_preset_solves("classic");
}

TEST(NativeBounds, TheFilterL1PresetSolvesANativelyBoundedPhase) {
    native_bounds_expect_preset_solves("filter_l1");
}

TEST(NativeBounds, TheSocRecoveryL1PresetSolvesANativelyBoundedPhase) {
    native_bounds_expect_preset_solves("soc_recovery_l1");
}

TEST(NativeBounds, TheSocProximalPresetSolvesANativelyBoundedPhase) {
    native_bounds_expect_preset_solves("soc_proximal");
}

TEST(NativeBounds, TheMeritL1PresetSolvesANativelyBoundedPhase) {
    native_bounds_expect_preset_solves("merit_l1");
}

// The five tests above are the whole shipped preset table, and this is what
// keeps that true: a sixth preset added to kPSIOPTPresets fails here instead of
// quietly going uncovered by the matrix.
TEST(NativeBounds, TheMechanismMatrixCoversEveryShippedPreset) {
    const std::vector<std::string> covered = {"classic", "filter_l1", "soc_recovery_l1",
                                              "soc_proximal", "merit_l1"};
    ASSERT_EQ(tycho::solvers::kPSIOPTPresets.size(), covered.size());
    for (const auto &entry : tycho::solvers::kPSIOPTPresets) {
        EXPECT_NE(std::find(covered.begin(), covered.end(), std::string(entry.name_)),
                  covered.end())
            << "preset " << entry.name_ << " has no matrix test";
    }
}

// --- Un-evaluable trials on a bounded problem -------------------------------

namespace {

// Scalar/SuperScalar-safe threshold test: the FD derivative modes may
// instantiate compute_impl with an Eigen::Array Scalar, where operator> yields
// an array expression rather than a bool.
inline bool native_bounds_above(double v, double t) { return v > t; }
template <int W> inline bool native_bounds_above(const Eigen::Array<double, W, 1> &v, double t) {
    return (v > t).any();
}

// Objective x^2 whose evaluation throws for x above a threshold, for at most a
// fixed number of evaluations, after which the domain "heals". Same shape as
// the fixture in test_eval_exception_recovery.cpp; carried here under this
// file's own prefix because the unity build merges the two translation units
// and an anonymous namespace does not separate them.
struct NativeBoundsGuardedSquare
    : tycho::vf::VectorFunction<NativeBoundsGuardedSquare, 1, 1,
                                tycho::vf::DenseDerivativeMode::FDiffFwd,
                                tycho::vf::DenseDerivativeMode::FDiffFwd> {
    using Base = tycho::vf::VectorFunction<NativeBoundsGuardedSquare, 1, 1,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd,
                                           tycho::vf::DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    double threshold_;
    std::shared_ptr<std::atomic<int>> throws_left_;

    NativeBoundsGuardedSquare(double threshold, int throw_budget)
        : threshold_(threshold), throws_left_(std::make_shared<std::atomic<int>>(throw_budget)) {}

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        if (native_bounds_above(x[0], threshold_) && throws_left_->fetch_sub(1) > 0) {
            throw std::runtime_error("trial point outside evaluation domain");
        }
        fx[0] = x[0] * x[0];
    }
};

} // namespace

// The fraction-to-boundary legs guarantee a trial point is inside the box
// before it is ever evaluated, so the bound-adjacent un-evaluable trial -- the
// classic reason an interior-point trial throws -- largely stops happening once
// bounds are native. What that does NOT do is retire the machinery: a function
// can be un-evaluable somewhere the box says nothing about.
//
// This is that case. min f(x) s.t. x = 1 with -5 <= x <= 5 declared natively,
// f throwing once for x above 0.1. The full Newton step targets x = 1, which is
// comfortably inside the box and hence not something any fraction-to-boundary
// rule will shorten, and past the threshold -- so the throwing trial is reached
// through a rung the bound legs deliberately allowed. It must come back as a
// rejected rung, and the solve must go on to the constrained optimum with every
// iterate still inside the box.
TEST(NativeBounds, AThrowingTrialTheBoundLegsCannotPreventIsStillJustARejectedRung) {
    tycho::solvers::OptimizationProblem prob;
    prob.set_vars(Eigen::VectorXd::Constant(1, 0.0));
    prob.add_objective(tycho::vf::GenericFunction<-1, 1>(NativeBoundsGuardedSquare(0.1, 1)),
                       (Eigen::VectorXi(1) << 0).finished());
    {
        auto args = tycho::vf::Arguments<1>();
        auto x = args.coeff<0>();
        prob.add_equal_con(tycho::vf::GenericFunction<-1, -1>(x - 1.0),
                           (Eigen::VectorXi(1) << 0).finished());
    }
    prob.optimizer_->set_print_level(3);
    prob.transcribe();
    prob.nlp_->set_variable_bound(0, -5.0, 5.0);
    prob.nlp_->make_nlp(prob.nlp_->primal_vars_, prob.nlp_->equal_cons_, prob.nlp_->inequal_cons_);
    prob.optimizer_->set_nlp(prob.nlp_);

    NativeBoundsInteriorProbe probe;
    probe.resolve = [&prob]() -> const NonLinearProgram * { return prob.nlp_.get(); };
    probe.install(*prob.optimizer_);

    const Eigen::VectorXd sol = prob.optimizer_->optimize(Eigen::VectorXd::Constant(1, 0.0));

    EXPECT_EQ(prob.optimizer_->result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(sol[0], 1.0, 1.0e-6);
    // The throw happened and was absorbed, rather than the threshold never
    // having been crossed at all.
    EXPECT_GE(prob.optimizer_->eval_error_log().count_, 1);
    EXPECT_TRUE(prob.nlp_->has_variable_bounds());
    probe.expect_every_iterate_interior();
}
