///////////////////////////////////////////////////////////////////////////////
// Unit tests for the second-order correction (SOC) recovery link.
//
// SOC is an opt-in RecoveryChain link (Settings::max_soc_ > 0) that, after the
// line search rejects a step on its first trial, re-solves the KKT system on the
// live factorization with a corrected constraint right-hand side and re-tries
// the acceptance test on the corrected step (Wächter & Biegler 2006, §2.4).
//
// The correctness-critical numeric path (KKT back-substitution + constraint
// evaluation) is exercised end-to-end by the solver corpus; here we truth-table
// the policy that governs it, which is factored into pure, machinery-free
// pieces:
//   - soc_should_trigger()  — when a rejection warrants a correction,
//   - soc_should_continue() — when to keep correcting vs. give up,
//   - soc_run_loop()        — the counter/verdict orchestration, driven with a
//                             scripted correction primitive (no real solve),
// plus the early-exit guards of SocRecovery::on_step_rejected (disabled cap and
// a non-triggering rejection both decline without touching the solve pieces),
// and the fraction-to-boundary rule the corrected direction must obey before it
// is committed (last section).
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/globalization/backtracking_line_search.h>
#include <hven/detail/globalization/globalization_mechanism.h>
#include <hven/detail/globalization/recovery_chain.h>
#include <hven/detail/globalization/restoration.h>
#include <hven/detail/globalization/soc.h>
#include <hven/detail/interior/iterate_info.h>

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::BacktrackingLineSearch;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kRecoveryDepthUnresolved;
using tycho::solvers::kSocRecommendedMaxCorrections;
using tycho::solvers::kSocViolationDecrease;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::InteriorPointSolver;
using tycho::solvers::RecoveryChain;
using tycho::solvers::RestorationStrategy;
using tycho::solvers::soc_run_loop;
using tycho::solvers::SocCorrectionOutcome;
using tycho::solvers::SocRecovery;
using tycho::solvers::soc_should_continue;
using tycho::solvers::soc_should_trigger;
using tycho::solvers::SolverContext;
using TychoTest::InertSolverContext;

using Action = RecoveryChain::Action;

// Builds an IterateInfo carrying just the two trigger signals SOC reads.
IterateInfo soc_make_iter(int first_rejection_iter, double theta_at_first_rejection) {
    IterateInfo it;
    it.first_rejection_iter_ = first_rejection_iter;
    it.theta_at_first_rejection_ = theta_at_first_rejection;
    return it;
}

// ---------------------------------------------------------------------------
// Trigger truth table (soc_should_trigger).
// ---------------------------------------------------------------------------
TEST(SocTrigger, TruthTable) {
    const double current = 1.0;

    // First trial rejected AND its violation did not improve on the current
    // iterate (theta >= current): fire.
    EXPECT_TRUE(soc_should_trigger(soc_make_iter(/*first=*/0, /*theta=*/2.0), current));
    // Equal fires.
    EXPECT_TRUE(soc_should_trigger(soc_make_iter(/*first=*/0, /*theta=*/1.0), current));

    // First trial rejected but its violation improved (theta < current): the
    // ordinary backtrack is already making progress, so no correction.
    EXPECT_FALSE(soc_should_trigger(soc_make_iter(/*first=*/0, /*theta=*/0.5), current));

    // Theta unavailable (< 0: LANG variant, or no rejection recorded): the
    // reduction test cannot be made, so conservatively do not fire.
    EXPECT_FALSE(soc_should_trigger(soc_make_iter(/*first=*/0, /*theta=*/-1.0), current));

    // The rejection was not the FIRST trial (a later backtrack): no correction.
    EXPECT_FALSE(soc_should_trigger(soc_make_iter(/*first=*/1, /*theta=*/2.0), current));

    // Step was accepted with no rejection (first_rejection_iter_ stays -1): the
    // recovery hook never even reaches SOC, and the predicate agrees.
    EXPECT_FALSE(soc_should_trigger(soc_make_iter(/*first=*/-1, /*theta=*/-1.0), current));
}

// ---------------------------------------------------------------------------
// Termination truth table (soc_should_continue).
// ---------------------------------------------------------------------------
TEST(SocTermination, TruthTable) {
    const int max_soc = 4;

    // Violation dropped by at least the kappa factor and the cap is not hit:
    // keep correcting.
    EXPECT_TRUE(soc_should_continue(/*trial=*/0.5, /*prev=*/1.0, /*done=*/1, max_soc));
    EXPECT_TRUE(soc_should_continue(/*trial=*/kSocViolationDecrease, /*prev=*/1.0,
                                    /*done=*/1, max_soc)); // exactly kappa still continues

    // Violation stagnated (dropped slower than the kappa factor): give up.
    EXPECT_FALSE(soc_should_continue(/*trial=*/0.995, /*prev=*/1.0, /*done=*/1, max_soc));

    // Correction cap reached: give up regardless of the violation.
    EXPECT_FALSE(soc_should_continue(/*trial=*/0.01, /*prev=*/1.0, /*done=*/max_soc, max_soc));
}

// ---------------------------------------------------------------------------
// Loop orchestration + counter (soc_run_loop) with a scripted primitive.
// ---------------------------------------------------------------------------

// Accept on the first correction: kRetry, exactly one back-substitution counted
// (accumulated onto the incoming counter value).
TEST(SocLoop, AcceptsFirstCorrection) {
    int soc_steps = 10; // pre-existing count: the loop accumulates.
    const Action action = soc_run_loop(
        /*first_trial_violation=*/1.0, /*max_soc=*/4, soc_steps, [](int, double) {
            return SocCorrectionOutcome{/*accepted=*/true, /*trial_violation=*/0.0};
        });
    EXPECT_EQ(action, Action::kRetry);
    EXPECT_EQ(soc_steps, 11);
}

// Violation keeps dropping, then a later correction is accepted: kRetry after
// counting every attempt.
TEST(SocLoop, DecreasesThenAccepts) {
    int soc_steps = 0;
    int calls = 0;
    const Action action = soc_run_loop(
        /*first_trial_violation=*/1.0, /*max_soc=*/4, soc_steps, [&](int, double prev) {
            ++calls;
            if (calls < 3)
                return SocCorrectionOutcome{false, 0.5 * prev}; // still improving
            return SocCorrectionOutcome{true, 0.0};             // accepted on the 3rd
        });
    EXPECT_EQ(action, Action::kRetry);
    EXPECT_EQ(soc_steps, 3);
}

// Violation stagnates after the first correction (drop slower than kappa): the
// loop stops and reverts (kAcceptAsIs) after a single attempt.
TEST(SocLoop, StagnationStops) {
    int soc_steps = 0;
    const Action action =
        soc_run_loop(/*first_trial_violation=*/1.0, /*max_soc=*/4, soc_steps,
                     [](int, double prev) { return SocCorrectionOutcome{false, 0.995 * prev}; });
    EXPECT_EQ(action, Action::kAcceptAsIs);
    EXPECT_EQ(soc_steps, 1);
}

// Violation keeps improving but never gets accepted: the cap stops the loop
// after exactly max_soc attempts (kAcceptAsIs).
TEST(SocLoop, CapStops) {
    int soc_steps = 0;
    const int max_soc = 4;
    const Action action =
        soc_run_loop(/*first_trial_violation=*/1.0, max_soc, soc_steps,
                     [](int, double prev) { return SocCorrectionOutcome{false, 0.5 * prev}; });
    EXPECT_EQ(action, Action::kAcceptAsIs);
    EXPECT_EQ(soc_steps, max_soc);
}

// ---------------------------------------------------------------------------
// SocRecovery early-exit guards (no solve machinery touched).
// ---------------------------------------------------------------------------

// Inert AcceptanceStrategy / GlobalizationMechanism: SocRecovery's early-exit
// paths return before ever calling these, so their bodies must never run. The
// acceptance double is reused by the fraction-to-boundary section below, where
// SOC reads only drives_classic_path() (to pick the norm its trigger compares
// in) and the mechanism intercepts the corrected step's re-test — so the verdict
// surface stays unreached there too.
class SocUnusedAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "the acceptance verdict surface must not be reached here";
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
};

class SocUnusedMechanism : public GlobalizationMechanism {
  public:
    double compute_step(InteriorPointSolver::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
                        Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                        AcceptanceStrategy &, double &, double &, IterateInfo &,
                        const std::vector<IterateInfo> &, SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached on an early-exit path";
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached on an early-exit path";
    }
    void reset() override {}
};

// Drives SocRecovery::on_step_rejected with a context whose dimensions are all
// zero (so the constraint block is empty and the KKT solver is never touched)
// and the given Settings/IterateInfo. Returns the Action and reports the SOC
// counter through `soc_steps`.
Action drive_soc(InteriorPointSolver::Settings &settings, IterateInfo &citer, int &soc_steps) {
    SocUnusedAcceptance acceptance;
    SocUnusedMechanism mechanism;
    InertSolverContext inert;
    inert.settings_ = settings;
    SolverContext ctx = inert.ctx();
    const std::vector<IterateInfo> iters;
    Eigen::VectorXd XSL, DXSL, XSL2, RHS, RHS2; // empty (ncons == 0)
    double alpha = 1.0, alphap = 1.0, alphad = 1.0;
    int resolved_depth = kRecoveryDepthUnresolved;
    int watchdog_activations = 0;
    return SocRecovery{}.on_step_rejected(
        citer, iters, ctx, acceptance, mechanism, InteriorPointSolver::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0,
        0.0, XSL, DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps, resolved_depth,
        watchdog_activations);
}

// max_soc_ == 0 (off): decline immediately, no corrections.
TEST(SocRecoveryGuards, DisabledDeclines) {
    InteriorPointSolver::Settings settings;
    settings.max_soc_ = 0;
    IterateInfo citer = soc_make_iter(/*first=*/0, /*theta=*/2.0); // would otherwise trigger
    int soc_steps = 0;
    EXPECT_EQ(drive_soc(settings, citer, soc_steps), Action::kAcceptAsIs);
    EXPECT_EQ(soc_steps, 0);
}

// Enabled but the rejection does not satisfy the trigger (not the first trial):
// decline before touching the solve machinery.
TEST(SocRecoveryGuards, NonTriggeringRejectionDeclines) {
    InteriorPointSolver::Settings settings;
    settings.max_soc_ = kSocRecommendedMaxCorrections;
    IterateInfo citer = soc_make_iter(/*first=*/1, /*theta=*/2.0); // not the first trial
    int soc_steps = 0;
    EXPECT_EQ(drive_soc(settings, citer, soc_steps), Action::kAcceptAsIs);
    EXPECT_EQ(soc_steps, 0);
}

// ---------------------------------------------------------------------------
// Fraction-to-boundary rule for a corrected direction.
//
// The main step path scales the search direction to the fraction-to-boundary
// limit when there are inequality slacks OR when a NESTED restoration strategy
// is active: the nested l1 reformulation condenses elastic bound variables out
// of the KKT system, and those carry their own positivity caps even on a problem
// with no inequality constraints at all. A second-order correction produces a
// NEW direction off the same factorization and commits it in place of the
// rejected one, so it is subject to the same rule.
//
// The tests below pin that rule on the shape where the elastic caps are the ONLY
// limits in play — an equality-only problem (inequal_cons_ == 0), where
// max_step_to_boundary loops over zero slack/bound rows and therefore returns
// 1.0. If the corrected direction is not elastic-scaled, the committed step is
// the raw back-substitution result and alphap/alphad keep their incoming values;
// if it is, both carry the caps the restoration strategy reports.
// ---------------------------------------------------------------------------

// Elastic caps reported by the nested-restoration double. Distinct from each
// other and from the 1.0 an equality-only problem's slack/bound caps produce, so
// the scaled and unscaled outcomes are separated by every observable below.
constexpr double kSocElasticPrimalAlpha = 0.25;
constexpr double kSocElasticDualAlpha = 0.5;

// The corrected direction the correction solve produces on the harness below,
// BEFORE any fraction-to-boundary scaling: dxsl_soc = -solve(rhs_soc) with an
// identity factorization and rhs_soc = [-1.0, c_soc], c_soc = -1.0 + (-0.5).
constexpr double kSocElasticUnscaledPrimalStep = 1.0;
constexpr double kSocElasticUnscaledEqMultStep = 1.5;

// Nested-restoration double with directly controllable active/nested flags and
// fixed elastic caps. Only the flags and the two boundary-alpha accessors are
// reached on this path; the rest of the surface satisfies the interface.
class SocElasticRestoration : public RestorationStrategy {
  public:
    void enter_restoration(const ProgressMeasures &, const Eigen::Ref<const Eigen::VectorXd> &,
                           double) override {
        active_ = true;
    }
    void exit_restoration() override { active_ = false; }
    bool is_active() const override { return active_; }
    void reset() override { active_ = false; }
    double proximal_objective(const Eigen::Ref<const Eigen::VectorXd> &) const override {
        return 0.0;
    }
    void add_proximal_gradient(const Eigen::Ref<const Eigen::VectorXd> &,
                               Eigen::Ref<Eigen::VectorXd>) const override {}
    const Eigen::VectorXd &proximal_diagonal() const override { return diag_; }
    const ProgressMeasures &reference() const override { return ref_; }
    void note_iteration() override {}
    bool is_nested() const override { return nested_; }

    double primal_boundary_alpha(double tau) const override {
        last_tau_ = tau;
        return kSocElasticPrimalAlpha;
    }
    double dual_boundary_alpha(double tau) const override {
        last_tau_ = tau;
        return kSocElasticDualAlpha;
    }

    bool active_ = false;
    bool nested_ = false;
    // The fraction the caps were requested at, recorded to confirm the corrected
    // step asks for the same bound_fraction_ the main step path asks for.
    mutable double last_tau_ = -1.0;

  private:
    Eigen::VectorXd diag_;
    ProgressMeasures ref_;
};

// Mechanism spy over the REAL fraction-to-boundary scaling: max_primal_dual_step
// counts its calls and delegates to BacktrackingLineSearch, so the elastic caps
// are applied exactly as the main step path applies them. The acceptance
// backtrack is short-circuited to an immediate accept, so the corrected
// direction is committed without dragging a merit/filter strategy in.
class SocElasticMechanism : public BacktrackingLineSearch {
  public:
    void max_primal_dual_step(Eigen::VectorXd &XSL, Eigen::VectorXd &DXSL, double bfrac,
                              double &alphap, double &alphad, const SolverContext &ctx) override {
        ++scale_calls_;
        BacktrackingLineSearch::max_primal_dual_step(XSL, DXSL, bfrac, alphap, alphad, ctx);
    }

    double run_acceptance_backtrack(InteriorPointSolver::LineSearchModes, double, double, double, double,
                                    Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                                    Eigen::VectorXd &, Eigen::VectorXd &, AcceptanceStrategy &,
                                    IterateInfo &Citer, const std::vector<IterateInfo> &,
                                    SolverContext &) override {
        ++backtrack_calls_;
        Citer.accepted_ = true;
        return 1.0;
    }

    int scale_calls_ = 0;
    int backtrack_calls_ = 0;
};

// Equality-only NLP (minimize x^2 subject to x - 1 = 0) plus the storage a
// SolverContext borrows: one primal, no slacks, one equality, no inequalities.
// The correction path needs a live NLP (it re-evaluates the constraint block at
// the trial point) and a factored KKT solver (the correction is one
// back-substitution on it); the factorization here is the identity, so the
// back-substitution returns rhs_soc unchanged and the corrected direction is
// exactly -rhs_soc — any scaling applied on top of it is directly readable off
// the committed step.
class SocElasticHarness {
  public:
    SocElasticHarness() {
        using tycho::vf::Arguments;
        using tycho::vf::GenericFunction;
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob_.set_vars(Eigen::VectorXd::Zero(1));
        prob_.add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
        prob_.add_equal_con(GenericFunction<-1, -1>(x - 1.0), (Eigen::VectorXi(1) << 0).finished());
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();

        inert_.primal_vars_ = prob_.nlp_->primal_vars_;
        inert_.slack_vars_ = prob_.nlp_->slack_vars_;
        inert_.equal_cons_ = prob_.nlp_->equal_cons_;
        inert_.inequal_cons_ = prob_.nlp_->inequal_cons_;
        inert_.kkt_dim_ = prob_.nlp_->kkt_dim_;
        inert_.settings_.max_soc_ = kSocRecommendedMaxCorrections;
        inert_.restoration_ = &restoration_;

        // The KKT shape every expectation below is written against: [x | y],
        // with no slack rows and no inequality rows.
        EXPECT_EQ(inert_.primal_vars_, 1);
        EXPECT_EQ(inert_.slack_vars_, 0);
        EXPECT_EQ(inert_.equal_cons_, 1);
        EXPECT_EQ(inert_.inequal_cons_, 0);
        EXPECT_EQ(inert_.kkt_dim_, 2);

        Eigen::SparseMatrix<double, Eigen::RowMajor> kkt(inert_.kkt_dim_, inert_.kkt_dim_);
        kkt.setIdentity();
        kkt.makeCompressed();
// The factorization owns its assembly buffer, so the matrix is written
        // through it rather than handed to compute(); backend configuration
        // travels with the factor's options and needs nothing here.
        inert_.kkt_solver_.matrix() = kkt;
        inert_.kkt_solver_.compute();
    }

    // A live context over this harness's storage, with the transcribed NLP
    // patched in (InertSolverContext leaves nlp_ null by default).
    SolverContext ctx() {
        SolverContext c = inert_.ctx();
        c.nlp_ = prob_.nlp_.get();
        return c;
    }

    // Detaches the restoration double entirely (restoration off).
    void clear_restoration() { inert_.restoration_ = nullptr; }

    SocElasticRestoration &restoration() { return restoration_; }
    const InteriorPointSolver::Settings &settings() const { return inert_.settings_; }
    Eigen::ComputationInfo factorization_info() { return inert_.kkt_solver_.info(); }
    int kkt_dim() const { return inert_.kkt_dim_; }

  private:
    OptimizationProblem prob_;
    InertSolverContext inert_;
    SocElasticRestoration restoration_;
};

// What one driven correction round committed.
struct SocElasticOutcome {
    Action action;
    Eigen::VectorXd committed_dxsl;
    double alphap;
    double alphad;
    int scale_calls;
};

// The working set the two drivers below share: XSL = (x, y) = (0, 0), a rejected
// direction of 0.5 on the primal, and RHS = (grad, c) = (-1, -1). The trigger
// signals are seeded so SOC fires (the first trial was rejected at a violation
// of 2.0, no better than the current iterate's RHS-tail measure of 1.0).
struct SocElasticWorkingSet {
    explicit SocElasticWorkingSet(int dim)
        : XSL(Eigen::VectorXd::Zero(dim)), DXSL(Eigen::VectorXd::Zero(dim)),
          XSL2(Eigen::VectorXd::Zero(dim)), RHS(Eigen::VectorXd::Constant(dim, -1.0)),
          RHS2(Eigen::VectorXd::Zero(dim)), citer(soc_make_iter(/*first=*/0, /*theta=*/2.0)) {
        DXSL[0] = 0.5;
    }

    Eigen::VectorXd XSL, DXSL, XSL2, RHS, RHS2;
    IterateInfo citer;
    const std::vector<IterateInfo> iters;
    double alpha = 1.0;
    double alphap = 1.0;
    double alphad = 1.0;
};

// Drives one SOC correction round on the harness and reports what was committed.
SocElasticOutcome run_soc_elastic_correction(SocElasticHarness &h, SocElasticMechanism &mechanism) {
    SocUnusedAcceptance acceptance;
    SolverContext ctx = h.ctx();
    SocElasticWorkingSet w(h.kkt_dim());
    int soc_steps = 0;
    int resolved_depth = kRecoveryDepthUnresolved;
    int watchdog_activations = 0;

    const Action action = SocRecovery{}.on_step_rejected(
        w.citer, w.iters, ctx, acceptance, mechanism, InteriorPointSolver::LineSearchModes::AUGLANG, 1.0, 1e-3,
        0.0, 0.0, w.XSL, w.DXSL, w.XSL2, w.RHS, w.RHS2, w.alpha, w.alphap, w.alphad, soc_steps,
        resolved_depth, watchdog_activations);
    return SocElasticOutcome{action, w.DXSL, w.alphap, w.alphad, mechanism.scale_calls_};
}

// Reference behavior: on the SAME equality-only context, the MAIN step path
// scales its direction to the elastic caps under an active nested restoration.
// This is the rule a corrected direction has to match.
TEST(SocElasticFractionToBoundary, MainStepPathScalesToTheElasticCaps) {
    SocElasticHarness h;
    ASSERT_EQ(h.factorization_info(), Eigen::Success);
    h.restoration().active_ = true;
    h.restoration().nested_ = true;

    SocElasticMechanism mechanism;
    SolverContext ctx = h.ctx();
    SocElasticWorkingSet w(h.kkt_dim());
    w.DXSL << kSocElasticUnscaledPrimalStep, kSocElasticUnscaledEqMultStep;
    SocUnusedAcceptance acceptance;

    mechanism.compute_step(InteriorPointSolver::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0, 0.0, w.XSL, w.DXSL,
                           w.XSL2, w.RHS, w.RHS2, acceptance, w.alphap, w.alphad, w.citer, w.iters,
                           ctx);

    EXPECT_EQ(mechanism.scale_calls_, 1);
    EXPECT_DOUBLE_EQ(w.alphap, kSocElasticPrimalAlpha);
    EXPECT_DOUBLE_EQ(w.alphad, kSocElasticDualAlpha);
    EXPECT_NEAR(w.DXSL[0], kSocElasticPrimalAlpha * kSocElasticUnscaledPrimalStep, 1e-12);
    EXPECT_NEAR(w.DXSL[1], kSocElasticPrimalAlpha * kSocElasticUnscaledEqMultStep, 1e-12);
}

// The property under test: with no inequality constraints and an ACTIVE NESTED
// restoration, the corrected direction is committed elastic-scaled — primal and
// equality-multiplier blocks cut to the primal cap (the default
// PrimSlackEq_Iq strategy drives both off the primal step length) and
// alphap/alphad reporting the caps rather than their incoming 1.0.
TEST(SocElasticFractionToBoundary, CorrectedStepIsScaledUnderActiveNestedRestoration) {
    SocElasticHarness h;
    ASSERT_EQ(h.factorization_info(), Eigen::Success);
    h.restoration().active_ = true;
    h.restoration().nested_ = true;

    SocElasticMechanism mechanism;
    const SocElasticOutcome out = run_soc_elastic_correction(h, mechanism);

    ASSERT_EQ(out.action, Action::kRetry); // a correction was accepted and committed
    ASSERT_EQ(mechanism.backtrack_calls_, 1);
    EXPECT_EQ(out.scale_calls, 1);
    EXPECT_DOUBLE_EQ(h.restoration().last_tau_, h.settings().bound_fraction_);

    EXPECT_DOUBLE_EQ(out.alphap, kSocElasticPrimalAlpha);
    EXPECT_DOUBLE_EQ(out.alphad, kSocElasticDualAlpha);
    EXPECT_NEAR(out.committed_dxsl[0], kSocElasticPrimalAlpha * kSocElasticUnscaledPrimalStep,
                1e-12);
    EXPECT_NEAR(out.committed_dxsl[1], kSocElasticPrimalAlpha * kSocElasticUnscaledEqMultStep,
                1e-12);
}

// The other rows of the same truth table. With no restoration attached, an
// inactive nested strategy, or an active NON-nested (proximal-switch) strategy
// there are no elastic bounds to respect on an equality-only problem, so the
// corrected direction is committed as solved and alphap/alphad keep their
// incoming values — the same three cases on which the main step path also
// declines to scale.
void expect_unscaled_correction(SocElasticHarness &h) {
    SocElasticMechanism mechanism;
    const SocElasticOutcome out = run_soc_elastic_correction(h, mechanism);

    ASSERT_EQ(out.action, Action::kRetry);
    ASSERT_EQ(mechanism.backtrack_calls_, 1);
    EXPECT_EQ(out.scale_calls, 0);
    EXPECT_DOUBLE_EQ(out.alphap, 1.0);
    EXPECT_DOUBLE_EQ(out.alphad, 1.0);
    EXPECT_NEAR(out.committed_dxsl[0], kSocElasticUnscaledPrimalStep, 1e-12);
    EXPECT_NEAR(out.committed_dxsl[1], kSocElasticUnscaledEqMultStep, 1e-12);
}

TEST(SocElasticFractionToBoundary, CorrectedStepIsUnscaledWithoutRestoration) {
    SocElasticHarness h;
    ASSERT_EQ(h.factorization_info(), Eigen::Success);
    h.clear_restoration();
    expect_unscaled_correction(h);
}

TEST(SocElasticFractionToBoundary, CorrectedStepIsUnscaledWhenNestedRestorationIsInactive) {
    SocElasticHarness h;
    ASSERT_EQ(h.factorization_info(), Eigen::Success);
    h.restoration().active_ = false;
    h.restoration().nested_ = true;
    expect_unscaled_correction(h);
}

TEST(SocElasticFractionToBoundary, CorrectedStepIsUnscaledWhenRestorationIsNotNested) {
    SocElasticHarness h;
    ASSERT_EQ(h.factorization_info(), Eigen::Success);
    h.restoration().active_ = true;
    h.restoration().nested_ = false;
    expect_unscaled_correction(h);
}

} // namespace
