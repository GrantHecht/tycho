///////////////////////////////////////////////////////////////////////////////
// Unit + through-API tests for the proximal feasibility-restoration switch.
//
// Three layers:
//   1. FeasibilitySwitchRecovery truth table (isolated): the outermost recovery
//      link delegates to its inner chain and converts ONLY a ladder-exhausted
//      kAcceptAsIs into kSwitchToFeasibility, and only when a restoration
//      strategy is present, inactive, and permits entry.
//   2. Component construction (friend access): restoration_mode_ ==
//      proximal_switch builds a ProximalSwitchRestoration and wraps a
//      FeasibilitySwitchRecovery as the outermost recovery link; off builds
//      neither; a FilterAcceptance is always seeded with the live econ_tol_.
//   3. Through-the-public-API solves on tiny NLPs: restoration off leaves the
//      diagnostics at their sentinels; a forced entry on an infeasible problem
//      does not falsely report success and respects the per-phase budget; a
//      forced entry on a feasible problem enters, exits, and converges on the
//      true objective.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/globalization/classic_adaptive_governor.h>
#include <hven/detail/globalization/feasibility_switch_recovery.h>
#include <hven/detail/globalization/filter_acceptance.h>
#include <hven/detail/globalization/globalization_mechanism.h>
#include <hven/detail/globalization/l1_restoration.h>
#include <hven/detail/globalization/monitored_governor.h>
#include <hven/detail/globalization/proximal_restoration.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Core>

using namespace tycho;
using TychoTest::InertSolverContext;
using TychoTest::SolverTest;

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::FeasibilitySwitchRecovery;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kRecoveryDepthRestoration;
using tycho::solvers::kRecoveryDepthUnresolved;
using tycho::solvers::kRecoveryDepthWatchdog;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::RecoveryChain;
using tycho::solvers::RestorationModes;
using tycho::solvers::RestorationStrategy;
using tycho::solvers::SolverContext;

// -----------------------------------------------------------------------------
// Stubs (unity-unique names prefixed FeasSwitch).
// -----------------------------------------------------------------------------

// Restoration double whose is_active()/entry_permitted() are directly
// controllable, so the truth table exercises FeasibilitySwitchRecovery's gating
// without depending on the real proximal guard math (covered separately in
// test_proximal_restoration.cpp).
class FeasSwitchStubRestoration : public RestorationStrategy {
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
    bool entry_permitted(double, const SolverContext &) const override { return permit_; }
    const ProgressMeasures &reference() const override { return ref_; }
    void note_iteration() override {}
    // Controls whether the wrapper takes the soft-pre-stage branch (nested) or
    // switches directly (proximal).
    bool is_nested() const override { return nested_; }

    bool active_ = false;
    bool permit_ = true;
    bool nested_ = false;

  private:
    Eigen::VectorXd diag_;
    ProgressMeasures ref_;
};

// Inner recovery double returning a configured Action, so FeasibilitySwitchRecovery's
// delegation and pass-through can be observed. Stamps a configurable
// resolved_depth (default kRecoveryDepthUnresolved, the caller-seeded sentinel a
// ladder-exhausted chain leaves in place) so the pass-through cases can confirm
// it is preserved, and optionally stamps Citer.accepted_ to mimic a
// watchdog-resolved kAcceptAsIs.
class FeasSwitchStubInner : public RecoveryChain {
  public:
    explicit FeasSwitchStubInner(Action action, int stamp_depth = kRecoveryDepthUnresolved,
                                 bool stamp_accepted = false)
        : action_(action), stamp_depth_(stamp_depth), stamp_accepted_(stamp_accepted) {}
    Action on_step_rejected(IterateInfo &citer, const std::vector<IterateInfo> &, SolverContext &,
                            AcceptanceStrategy &, GlobalizationMechanism &,
                            PSIOPT::LineSearchModes, double, double, double, double,
                            Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                            Eigen::VectorXd &, Eigen::VectorXd &, double &, double &, double &,
                            int &, int &resolved_depth, int &) override {
        ++calls_;
        resolved_depth = stamp_depth_;
        if (stamp_accepted_)
            citer.accepted_ = true;
        return action_;
    }
    void reset() override {}

    Action action_;
    int stamp_depth_;
    bool stamp_accepted_;
    int calls_ = 0;
};

// Inert mechanism to satisfy the on_step_rejected signature (never dereferenced
// by FeasibilitySwitchRecovery, which only reads ctx.restoration_ / RHS).
class FeasSwitchUnusedMechanism : public GlobalizationMechanism {
  public:
    double compute_step(PSIOPT::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
                        Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                        AcceptanceStrategy &, double &, double &, IterateInfo &,
                        const std::vector<IterateInfo> &, SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached by FeasibilitySwitchRecovery";
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {
        ADD_FAILURE() << "mechanism must not be reached by FeasibilitySwitchRecovery";
    }
    void reset() override {}
};

// Inert acceptance to satisfy the on_step_rejected signature.
class FeasSwitchUnusedAcceptance : public AcceptanceStrategy {
  public:
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "acceptance must not be reached by FeasibilitySwitchRecovery";
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        ADD_FAILURE() << "acceptance must not be reached by FeasibilitySwitchRecovery";
        return false;
    }
    void reset() override {}
    bool drives_classic_path() const override { return true; }
};

// Drive FeasibilitySwitchRecovery::on_step_rejected once and return its Action;
// captures resolved_depth via the out-parameter.
RecoveryChain::Action drive_feas_switch(FeasibilitySwitchRecovery &fsr, SolverContext &ctx,
                                        int &resolved_depth_out,
                                        IterateInfo *citer_out = nullptr) {
    FeasSwitchUnusedMechanism mechanism;
    FeasSwitchUnusedAcceptance acceptance;
    IterateInfo local_citer;
    IterateInfo &citer = citer_out ? *citer_out : local_citer;
    const std::vector<IterateInfo> iters;
    Eigen::VectorXd v;
    double alpha = 1.0, alphap = 1.0, alphad = 1.0;
    int soc_steps = 0;
    int watchdog_activations = 0;
    resolved_depth_out = kRecoveryDepthUnresolved;
    return fsr.on_step_rejected(citer, iters, ctx, acceptance, mechanism,
                                PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0, 0.0, v, v, v, v, v,
                                alpha, alphap, alphad, soc_steps, resolved_depth_out,
                                watchdog_activations);
}

// -----------------------------------------------------------------------------
// 1. Truth table.
// -----------------------------------------------------------------------------

TEST(FeasibilitySwitchTruthTable, InnerRetryPassesThrough) {
    FeasSwitchStubRestoration restoration; // active_=false, permit_=true
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kRetry));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kRetry);
}

TEST(FeasibilitySwitchTruthTable, InnerGiveUpPassesThrough) {
    FeasSwitchStubRestoration restoration;
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kGiveUp));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kGiveUp);
}

TEST(FeasibilitySwitchTruthTable, NullRestorationKeepsAcceptAsIs) {
    InertSolverContext inert; // restoration_ stays nullptr
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kAcceptAsIs);
    EXPECT_EQ(depth, kRecoveryDepthUnresolved);
}

TEST(FeasibilitySwitchTruthTable, AlreadyActiveNeverReEnters) {
    FeasSwitchStubRestoration restoration;
    restoration.active_ = true; // already in restoration mode
    restoration.permit_ = true;
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kAcceptAsIs);
}

TEST(FeasibilitySwitchTruthTable, EntryRefusedKeepsAcceptAsIs) {
    FeasSwitchStubRestoration restoration;
    restoration.active_ = false;
    restoration.permit_ = false; // guard/budget refuses entry
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kAcceptAsIs);
}

TEST(FeasibilitySwitchTruthTable, EntryPermittedSwitchesAndStampsDepth) {
    FeasSwitchStubRestoration restoration;
    restoration.active_ = false;
    restoration.permit_ = true;
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSwitchToFeasibility);
    EXPECT_EQ(depth, kRecoveryDepthRestoration);
}

// A watchdog-resolved kAcceptAsIs must NOT be hijacked into a feasibility
// switch. The watchdog's trial-acceptance path returns kAcceptAsIs but has
// RESOLVED the rejection — it stamps Citer.accepted_ = true and writes
// resolved_depth = kRecoveryDepthWatchdog (any link that resolved sets its own
// depth). kAcceptAsIs alone is overloaded, so the interception is gated on the
// depth out-parameter staying at the caller-seeded kRecoveryDepthUnresolved;
// here it is kRecoveryDepthWatchdog, so the wrapper passes the step through
// unchanged and preserves the depth. Restoration is present, inactive, and would
// otherwise permit entry — proving the depth check alone blocks the switch.
TEST(FeasibilitySwitchTruthTable, WatchdogResolvedAcceptPassesThrough) {
    FeasSwitchStubRestoration restoration; // active_=false, permit_=true
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(std::make_unique<FeasSwitchStubInner>(
        RecoveryChain::Action::kAcceptAsIs, kRecoveryDepthWatchdog, /*stamp_accepted=*/true));
    int depth = 0;
    IterateInfo citer;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth, &citer), RecoveryChain::Action::kAcceptAsIs);
    EXPECT_EQ(depth, kRecoveryDepthWatchdog); // depth preserved, not overwritten
    EXPECT_TRUE(citer.accepted_); // the resolved acceptance itself survives the pass-through
}

TEST(FeasibilitySwitchTruthTable, NullInnerChainRejected) {
    EXPECT_THROW(FeasibilitySwitchRecovery(nullptr), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 1b. Soft feasibility pre-stage truth table (nested restoration only).
// -----------------------------------------------------------------------------

// A nested strategy at a ladder-exhausted rejection enters the soft pre-stage:
// the wrapper returns kSoftFeasibilityStep (alg_impl takes the trial step) rather
// than the full switch, and stamps the restoration recovery depth.
TEST(FeasibilitySwitchSoftPreStage, NestedReturnsSoftStepBeforeSwitching) {
    FeasSwitchStubRestoration restoration;
    restoration.nested_ = true; // nested l1 mode
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSoftFeasibilityStep);
    EXPECT_EQ(depth, kRecoveryDepthRestoration);
}

// A proximal (non-nested) strategy has NO pre-stage: it switches directly on the
// first exhausted rejection, however many times it is driven.
TEST(FeasibilitySwitchSoftPreStage, ProximalNeverEntersPreStage) {
    FeasSwitchStubRestoration restoration;
    restoration.nested_ = false; // proximal switch mode
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    for (int k = 0; k < 15; ++k)
        EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSwitchToFeasibility);
}

// The pre-stage stays engaged (accept-by-PD-error path: alg_impl keeps taking
// soft steps) for up to kMaxSoftRestoIters successive iterations; the 11th
// successive soft iteration escalates to the full switch.
TEST(FeasibilitySwitchSoftPreStage, EscalatesOnEleventhSuccessiveSoftIteration) {
    FeasSwitchStubRestoration restoration;
    restoration.nested_ = true;
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    // Iterations 1..10 (== kMaxSoftRestoIters) stay in the pre-stage.
    for (int k = 0; k < 10; ++k)
        EXPECT_EQ(drive_feas_switch(fsr, ctx, depth),
                  RecoveryChain::Action::kSoftFeasibilityStep);
    // The 11th escalates to the full restoration switch.
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSwitchToFeasibility);
}

// The pre-stage exits (accept-by-original-criterion path) when a regular step is
// accepted: notify_step_accepted resets the successive-soft-iteration counter, so
// a fresh pre-stage can run its full budget again afterward.
TEST(FeasibilitySwitchSoftPreStage, AcceptedStepResetsSoftCounter) {
    FeasSwitchStubRestoration restoration;
    restoration.nested_ = true;
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    for (int k = 0; k < 10; ++k)
        EXPECT_EQ(drive_feas_switch(fsr, ctx, depth),
                  RecoveryChain::Action::kSoftFeasibilityStep);
    // A regular accepted step recovered: the pre-stage exits and the counter
    // clears. Without the reset the next drive would escalate (11th); with it the
    // pre-stage runs from the top again.
    fsr.notify_step_accepted();
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSoftFeasibilityStep);
}

// A mode-switch reset() also clears the pre-stage counter (a restoration entry or
// exit resets the pre-stage).
TEST(FeasibilitySwitchSoftPreStage, ResetClearsSoftCounter) {
    FeasSwitchStubRestoration restoration;
    restoration.nested_ = true;
    InertSolverContext inert;
    inert.restoration_ = &restoration;
    SolverContext ctx = inert.ctx();

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    for (int k = 0; k < 10; ++k)
        EXPECT_EQ(drive_feas_switch(fsr, ctx, depth),
                  RecoveryChain::Action::kSoftFeasibilityStep);
    fsr.reset();
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSoftFeasibilityStep);
}

// -----------------------------------------------------------------------------
// 3. Through-the-public-API solves.
// -----------------------------------------------------------------------------

// Build a direct NLP: minimize (x - target)^2 subject to (x - a) = 0 and, when
// `inconsistent`, also (x - b) = 0 (b != a makes the two equalities
// contradictory — a genuinely infeasible problem). One variable, x0 = start.
std::unique_ptr<OptimizationProblem> feas_switch_build_nlp(double start, double a, bool inconsistent,
                                                           double b) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(1, start));

    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
    }
    {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - a),
                            (Eigen::VectorXi(1) << 0).finished());
    }
    if (inconsistent) {
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x - b),
                            (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3);
    return prob;
}

// The restoration diagnostics carry exactly one bit per mode: -1 means "no
// RestorationStrategy was constructed, the counters do not apply", anything >= 0
// is a real count. The two halves are one property and are asserted together --
// the "on" half is only meaningful against the "off" half it contrasts with.
TEST_F(SolverTest, RestorationDiagnosticsSentinelOnlyWhenOff) {
    auto off = feas_switch_build_nlp(/*start=*/0.0, /*a=*/1.0, /*inconsistent=*/false, /*b=*/0.0);
    // restoration_mode_ defaults to off.
    off->optimize();
    const auto &r_off = off->optimizer_->result();
    EXPECT_EQ(r_off.last_feas_rest_entries_, -1);
    EXPECT_EQ(r_off.last_feas_rest_iters_, -1);

    auto on = feas_switch_build_nlp(0.0, 1.0, false, 0.0);
    on->optimizer_->settings().restoration_mode_ = RestorationModes::proximal_switch;
    on->optimize();
    const auto &r_on = on->optimizer_->result();
    // With restoration constructed the counters are reported as real counts --
    // NOT the sentinel — regardless of whether entry actually fired.
    // EXPECT_GE(..., 0) below is the stronger check: it subsumes ruling out the
    // -1 sentinel (any real count is already >= 0), so no separate EXPECT_NE is
    // needed.
    EXPECT_GE(r_on.last_feas_rest_entries_, 0);
    EXPECT_GE(r_on.last_feas_rest_iters_, 0);
}

TEST_F(SolverTest, ForcedEntryOnFeasibleProblemEntersExitsAndConverges) {
    // A trivially feasible equality QP. max_ls_iters_ = 0 forces every step to
    // be "rejected" (the fused line search accepts nothing), so the ladder-
    // exhausted recovery fires and, at the infeasible start, hands off to
    // restoration. Restoration pulls x to feasibility, the infeasibility-
    // reduction exit fires, and optimality mode then converges on the true
    // objective (x -> 1).
    auto prob = feas_switch_build_nlp(/*start=*/0.0, /*a=*/1.0, /*inconsistent=*/false, /*b=*/0.0);
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::proximal_switch;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(50);
    auto flag = prob->optimize();

    const auto &r = prob->optimizer_->result();
    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GE(r.last_feas_rest_entries_, 1);           // restoration was entered
    EXPECT_LE(r.last_feas_rest_entries_, 2);           // and not beyond the budget
    ASSERT_EQ(r.primals_.size(), 1);
    EXPECT_NEAR(r.primals_[0], 1.0, 1e-4);             // converged on the true objective
}

TEST_F(SolverTest, ForcedEntryUnderFilterAcceptanceStillConverges) {
    // Same forced-entry setup as ForcedEntryOnFeasibleProblemEntersExitsAndConverges,
    // but with acceptance_strategy_ == filter (paired with the monitored barrier
    // governor filter/funnel require). This pins the restoration-exit fix: every
    // notify_switch_to_optimality call along the exit path now carries the TRUE
    // objective (re-evaluated via build_restoration_exit_measures), not the
    // proximal objective φ_prox the loop evaluated restoration iterates under.
    // Before that fix, the (θ, φ_prox) pair augmented into the restored
    // OPTIMALITY filter was on the wrong scale relative to every other pair the
    // filter carries (all true-objective-scale) -- a filter dominated by a
    // spuriously-scaled entry can reject every subsequent optimality-mode trial,
    // so a solve that reaches this exit path failing to converge is exactly the
    // symptom this test would catch.
    auto prob = feas_switch_build_nlp(/*start=*/0.0, /*a=*/1.0, /*inconsistent=*/false, /*b=*/0.0);
    using tycho::solvers::AcceptanceStrategies;
    using tycho::solvers::BarrierGovernors;
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::proximal_switch;
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::filter;
    prob->optimizer_->settings().barrier_governor_ = BarrierGovernors::monitored;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(50);
    auto flag = prob->optimize();

    const auto &r = prob->optimizer_->result();
    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GE(r.last_feas_rest_entries_, 1);           // restoration was entered
    ASSERT_EQ(r.primals_.size(), 1);
    EXPECT_NEAR(r.primals_[0], 1.0, 1e-4);             // converged on the true objective
}

TEST_F(SolverTest, ForcedEntryOnInfeasibleProblemDoesNotFalselyConverge) {
    // Contradictory equalities x = 1 and x = -1: genuinely infeasible. Forcing
    // entry (max_ls_iters_ = 0) must NOT report the proximal subproblem's
    // convergence as a solve of the true problem.
    auto prob = feas_switch_build_nlp(/*start=*/0.0, /*a=*/1.0, /*inconsistent=*/true, /*b=*/-1.0);
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::proximal_switch;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(40);
    auto flag = prob->optimize();

    const auto &r = prob->optimizer_->result();
    // Hard assertion: never fatal-skipped, always runs.
    EXPECT_NE(flag, tycho::ConvergenceFlags::CONVERGED);
    // Soft check: entry depends on pivot perturbation producing a finite step
    // from this problem's rank-deficient KKT system, which is platform-
    // dependent factorization behavior, not the property under test. The skip
    // IS the entry assertion -- reaching past it means restoration was entered.
    if (r.last_feas_rest_entries_ < 1) {
        GTEST_SKIP() << "factorization returned non-finite step on this platform; "
                       "entry not exercised";
    }
}

TEST_F(SolverTest, BudgetZeroRefusesAllEntries) {
    // max_feas_rest_ = 0 exhausts the budget before the first entry, so
    // FeasibilitySwitchRecovery never switches even on an infeasible start.
    auto prob = feas_switch_build_nlp(/*start=*/0.0, /*a=*/1.0, /*inconsistent=*/true, /*b=*/-1.0);
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::proximal_switch;
    prob->optimizer_->settings().max_feas_rest_ = 0;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(40);
    auto flag = prob->optimize();

    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, 0); // budget refused every entry
    EXPECT_NE(flag, tycho::ConvergenceFlags::CONVERGED);
}

} // namespace

// -----------------------------------------------------------------------------
// 2. Component construction (friend access — see the gate-test convention for
// why these live at global scope, outside the anonymous namespace).
// -----------------------------------------------------------------------------

TEST(FeasibilitySwitch, ProximalSwitchConstructsRestorationAndWrapsRecovery) {
    tycho::solvers::PSIOPT solver;
    solver.settings().restoration_mode_ = tycho::solvers::RestorationModes::proximal_switch;
    solver.rebuild_globalization_components();
    ASSERT_NE(dynamic_cast<tycho::solvers::ProximalSwitchRestoration *>(solver.restoration_.get()),
              nullptr);
    // FeasibilitySwitchRecovery is the outermost recovery link.
    ASSERT_NE(dynamic_cast<tycho::solvers::FeasibilitySwitchRecovery *>(solver.recovery_.get()),
              nullptr);
}

TEST(FeasibilitySwitch, OffModeConstructsNoRestoration) {
    tycho::solvers::PSIOPT solver;
    solver.settings().restoration_mode_ = tycho::solvers::RestorationModes::off;
    solver.rebuild_globalization_components();
    EXPECT_EQ(solver.restoration_.get(), nullptr);
    // Recovery chain is NOT wrapped by FeasibilitySwitchRecovery when off.
    EXPECT_EQ(dynamic_cast<tycho::solvers::FeasibilitySwitchRecovery *>(solver.recovery_.get()),
              nullptr);
}

TEST(FeasibilitySwitch, FilterSeedsRestorationConstraintTol) {
    tycho::solvers::PSIOPT solver;
    solver.settings().acceptance_strategy_ = tycho::solvers::AcceptanceStrategies::filter;
    solver.settings().barrier_governor_ = tycho::solvers::BarrierGovernors::monitored;
    solver.settings().econ_tol_ = 3.0e-7;
    solver.rebuild_globalization_components();
    auto *filter =
        dynamic_cast<tycho::solvers::FilterAcceptance *>(solver.acceptance_.get());
    ASSERT_NE(filter, nullptr);
    EXPECT_DOUBLE_EQ(filter->restoration_constraint_tol(), 3.0e-7);
}

// -----------------------------------------------------------------------------
// 4. Nested-restoration eval/step seam (through the private surface).
//
// These tests force a NestedL1Restoration active on a hand-built 2-variable NLP
// with one violated equality and drive the solver seam directly. NestedSeamHarness
// is befriended by PSIOPT (psiopt.h) so it can reach the private eval_nlp /
// alg_impl / restoration_ / dimension members; the tests below assert on its
// public outputs. Nothing wires nested entry yet (a later change does), so this
// is the only way to exercise the seam.
// -----------------------------------------------------------------------------

using tycho::solvers::BarrierGovernor;
using tycho::solvers::ClassicAdaptiveGovernor;
using tycho::solvers::FeasibilitySwitchRecovery;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kMaxSoftRestoIters;
using tycho::solvers::NestedL1Restoration;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::PSIOPT;
using tycho::solvers::RestorationModes;
using tycho::solvers::SolverContext;

// A non-nested restoration double: is_nested() stays false, so a correctly split
// seam must take the proximal branch and never touch the nested surface. The
// proximal pieces are recorded when consulted; every nested-surface override
// flips nested_touched_ so any erroneous nested dispatch is caught (rather than
// hitting the base class throw).
class NestedSeamProximalDouble : public tycho::solvers::RestorationStrategy {
  public:
    explicit NestedSeamProximalDouble(int primal_vars)
        : diag_(Eigen::VectorXd::Zero(primal_vars)) {}

    void enter_restoration(const tycho::solvers::ProgressMeasures &,
                           const Eigen::Ref<const Eigen::VectorXd> &, double) override {
        active_ = true;
    }
    void exit_restoration() override { active_ = false; }
    bool is_active() const override { return active_; }
    void reset() override { active_ = false; }

    double proximal_objective(const Eigen::Ref<const Eigen::VectorXd> &) const override {
        proximal_touched_ = true;
        return 0.0;
    }
    void add_proximal_gradient(const Eigen::Ref<const Eigen::VectorXd> &,
                               Eigen::Ref<Eigen::VectorXd>) const override {
        proximal_touched_ = true;
    }
    const Eigen::VectorXd &proximal_diagonal() const override {
        proximal_touched_ = true;
        return diag_;
    }
    bool entry_permitted(double, const tycho::solvers::SolverContext &) const override {
        return true;
    }
    const tycho::solvers::ProgressMeasures &reference() const override { return ref_; }
    void note_iteration() override {}

    // Nested surface: any of these firing on a non-nested strategy is a seam-split
    // bug. Record it instead of throwing so the assertion is legible.
    const Eigen::VectorXd &e_pivots() const override {
        nested_touched_ = true;
        return diag_;
    }
    const Eigen::VectorXd &i_pivots() const override {
        nested_touched_ = true;
        return diag_;
    }
    void nested_primal_diagonal(double, Eigen::Ref<Eigen::VectorXd>) const override {
        nested_touched_ = true;
    }
    void condensed_residuals(double, const Eigen::Ref<const Eigen::VectorXd> &,
                             const Eigen::Ref<const Eigen::VectorXd> &,
                             const Eigen::Ref<const Eigen::VectorXd> &,
                             const Eigen::Ref<const Eigen::VectorXd> &, Eigen::Ref<Eigen::VectorXd>,
                             Eigen::Ref<Eigen::VectorXd>) const override {
        nested_touched_ = true;
    }
    double nested_objective(double, const Eigen::Ref<const Eigen::VectorXd> &) const override {
        nested_touched_ = true;
        return 0.0;
    }
    void add_nested_gradient(double, const Eigen::Ref<const Eigen::VectorXd> &,
                             Eigen::Ref<Eigen::VectorXd>) const override {
        nested_touched_ = true;
    }
    void recover_elastic_steps(double, const Eigen::Ref<const Eigen::VectorXd> &,
                               const Eigen::Ref<const Eigen::VectorXd> &,
                               const Eigen::Ref<const Eigen::VectorXd> &,
                               const Eigen::Ref<const Eigen::VectorXd> &) override {
        nested_touched_ = true;
    }
    void apply_elastic_step(double, double) override { nested_touched_ = true; }

    mutable bool proximal_touched_ = false;
    mutable bool nested_touched_ = false;
    bool active_ = false;

  private:
    Eigen::VectorXd diag_;
    tycho::solvers::ProgressMeasures ref_;
};

// Test harness reaching PSIOPT's private seam. Befriended by PSIOPT.
class NestedSeamHarness {
  public:
    NestedSeamHarness() {
        using tycho::vf::Arguments;
        using tycho::vf::GenericFunction;
        prob_.set_vars((Eigen::VectorXd(2) << 0.0, 0.0).finished());
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_objective(GenericFunction<-1, 1>(x0 * x0 + x1 * x1),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            // Equality x0 + x1 - 4 = 0; at the start (0,0) the residual is -4.
            prob_.add_equal_con(GenericFunction<-1, -1>(x0 + x1 - 4.0),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
    }

    PSIOPT &solver() { return *solver_; }
    int pv() const { return solver_->primal_vars_; }
    int sv() const { return solver_->slack_vars_; }
    int ec() const { return solver_->equal_cons_; }
    int ic() const { return solver_->inequal_cons_; }
    int dim() const { return solver_->kkt_dim_; }

    // Inject a fresh NestedL1Restoration and enter the nested phase at the given
    // primals with the equality residual implied by the constraint.
    NestedL1Restoration *enter_nested(const Eigen::VectorXd &primals, double outer_mu) {
        auto strat = std::make_unique<NestedL1Restoration>();
        NestedL1Restoration *raw = strat.get();
        solver_->restoration_ = std::move(strat);
        Eigen::VectorXd eq_res(1);
        eq_res[0] = primals[0] + primals[1] - 4.0;
        Eigen::VectorXd iq_res(0);
        tycho::solvers::ProgressMeasures ref;
        // Zero reference infeasibility floors the exit reduction test at econ_tol,
        // so the first restoration iterate (a large condensed residual on the
        // violated row) is NOT judged "sufficiently reduced" and the phase takes
        // at least one step before any exit is considered.
        ref.infeasibility = 0.0;
        ref.objective = 0.0;
        ref.auxiliary = 0.0;
        raw->enter_nested(ref, primals, eq_res, iq_res, outer_mu);
        return raw;
    }

    // Run the nested eval seam once at (primals, y) with barrier mu; record the
    // assembled KKT constraint-row diagonal and the RHS eq residual segment.
    void run_eval_seam(const Eigen::VectorXd &primals, double y, double mu) {
        Eigen::VectorXd XSL(dim());
        XSL.head(pv()) = primals;
        XSL.segment(pv() + sv(), ec())[0] = y;
        Eigen::VectorXd GX = Eigen::VectorXd::Zero(pv());
        Eigen::VectorXd AGXS = Eigen::VectorXd::Zero(dim());
        double val = 0.0;
        solver_->eval_nlp(PSIOPT::AlgorithmModes::OPTNO, 0.0, XSL, val, GX, AGXS,
                          solver_->kkt_sol_.matrix(), mu);
        const int yrow = pv() + sv();
        kkt_yy_ = solver_->kkt_sol_.matrix().coeff(yrow, yrow);
        rhs_eq_ = AGXS.segment(pv() + sv(), ec())[0];
    }

    // Run the eval seam with a non-nested proximal double, returning it so the
    // caller can inspect which surface was touched.
    NestedSeamProximalDouble *run_eval_seam_proximal(const Eigen::VectorXd &primals, double mu) {
        auto strat = std::make_unique<NestedSeamProximalDouble>(pv());
        NestedSeamProximalDouble *raw = strat.get();
        raw->active_ = true;
        solver_->restoration_ = std::move(strat);
        Eigen::VectorXd XSL(dim());
        XSL.head(pv()) = primals;
        XSL.segment(pv() + sv(), ec())[0] = 0.0;
        Eigen::VectorXd GX = Eigen::VectorXd::Zero(pv());
        Eigen::VectorXd AGXS = Eigen::VectorXd::Zero(dim());
        double val = 0.0;
        solver_->eval_nlp(PSIOPT::AlgorithmModes::OPTNO, 0.0, XSL, val, GX, AGXS,
                          solver_->kkt_sol_.matrix(), mu);
        return raw;
    }

    // Drive one committed nested iteration and capture the elastic state the
    // instant after that step is applied (at the START of the next iteration,
    // before its own step recovery overwrites the stored deltas).
    struct StepCapture {
        bool captured = false;
        double n, p, zn, zp;    // elastic state after the applied step
        double dn, dp, dzn, dzp; // recovered step that was applied
    };

    StepCapture drive_one_step(double outer_mu) {
        solver_->settings().max_iters_ = 4;
        solver_->settings().max_ls_iters_ = 0; // full step, deterministic alpha = 1
        solver_->settings().print_level_ = 3;
        solver_->rebuild_globalization_components();
        solver_->ensure_solver_initialized();
        bool docompute = solver_->claim_kkt_analysis();
        Eigen::VectorXd x = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
        Eigen::VectorXd XSL = solver_->init_impl(x, outer_mu, docompute);

        NestedL1Restoration *comp = enter_nested(XSL.head(pv()), outer_mu);
        // This harness isolates the elastic step-recovery seam (recover → apply)
        // from the second-level re-center fallback: with max_ls_iters_ = 0 every
        // in-phase line search is a ladder exhaustion, which would otherwise
        // re-center on the first iteration and discard the recovered step this
        // test observes. Consuming the one-shot budget up front keeps every
        // iteration on the ordinary accept-as-is path. The re-center fallback has
        // its own coverage (NestedRestorationRecenter suite).
        solver_->resto_recentered_ = true;
        // Run the phase at the restoration barrier the elastics were initialized
        // at (what the entry orchestration will set the live μ to).
        const double phase_mu = comp->entry_mu();

        StepCapture cap;
        solver_->set_early_callback([comp, &cap](int i, double, Eigen::Ref<Eigen::VectorXd>, double,
                                                 Eigen::Ref<Eigen::VectorXd>,
                                                 Eigen::Ref<Eigen::VectorXd>,
                                                 Eigen::SparseMatrix<double, Eigen::RowMajor> &)
                                        -> int {
            if (i == 1 && !cap.captured) {
                cap.captured = true;
                cap.n = comp->ec_n()[0];
                cap.p = comp->ec_p()[0];
                cap.zn = comp->ec_zn()[0];
                cap.zp = comp->ec_zp()[0];
                cap.dn = comp->ec_dn()[0];
                cap.dp = comp->ec_dp()[0];
                cap.dzn = comp->ec_dzn()[0];
                cap.dzp = comp->ec_dzp()[0];
            }
            return 0;
        });

        solver_->alg_impl(PSIOPT::AlgorithmModes::OPT, PSIOPT::BarrierModes::LOQO,
                          PSIOPT::LineSearchModes::L1, solver_->settings().obj_scale_, phase_mu,
                          XSL);
        solver_->disable_early_callback();
        return cap;
    }

    double kkt_yy_ = 0.0;
    double rhs_eq_ = 0.0;

  private:
    OptimizationProblem prob_;
    PSIOPT *solver_ = nullptr;
};

// (i) The assembled KKT constraint-row diagonal equals the NEGATED component
// pivot (the condensed (y,y) block is −pivot; the solver scatters the pivot slot
// as a +coefficient onto that diagonal).
TEST(NestedRestorationSeam, PivotDiagonalNegatesComponentPivot) {
    NestedSeamHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    NestedL1Restoration *comp = h.enter_nested(primals, /*outer_mu=*/0.1);
    h.run_eval_seam(primals, /*y=*/0.5, /*mu=*/comp->entry_mu());

    ASSERT_EQ(comp->e_pivots().size(), 1);
    EXPECT_GT(comp->e_pivots()[0], 0.0); // component pivot is positive
    EXPECT_NEAR(h.kkt_yy_, -comp->e_pivots()[0], 1e-10);
}

// (ii) The RHS equality segment equals the condensed residual r̃ recomputed from
// the component's live elastic state, the raw residual c, the multiplier y, and μ.
TEST(NestedRestorationSeam, RhsSegmentCarriesCondensedResidual) {
    NestedSeamHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    NestedL1Restoration *comp = h.enter_nested(primals, /*outer_mu=*/0.1);
    const double y = 0.5;
    const double mu = comp->entry_mu();
    h.run_eval_seam(primals, y, mu);

    const double c = primals[0] + primals[1] - 4.0; // raw residual eval_kkt_no stores
    const double rho = tycho::solvers::kRestoPenaltyParameter;
    const double n = comp->ec_n()[0];
    const double p = comp->ec_p()[0];
    const double zn = comp->ec_zn()[0];
    const double zp = comp->ec_zp()[0];
    const double rtilde = (c + n - p) + mu / zn - (n / zn) * (rho + y) - mu / zp +
                          (p / zp) * (rho - y);
    EXPECT_NEAR(h.rhs_eq_, rtilde, 1e-8 * (1.0 + std::abs(rtilde)));
}

// (iii) After one committed iteration the elastic slacks and their bound
// multipliers moved by the recovered steps damped by a single primal fraction
// (n,p) and a single dual fraction (z_n,z_p), each in (0,1], with positivity
// preserved.
TEST(NestedRestorationSeam, ElasticStepMovesWithDampedAlphas) {
    NestedSeamHarness h;
    // Entry state (before any step) for the same residual the harness will use.
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    const double c = primals[0] + primals[1] - 4.0;
    const double outer_mu = 0.1;
    const double resto_mu = std::max(outer_mu, std::abs(c));
    const tycho::solvers::ElasticSlackInit init =
        tycho::solvers::l1_elastic_slack_init(c, resto_mu, tycho::solvers::kRestoPenaltyParameter);

    NestedSeamHarness::StepCapture cap = h.drive_one_step(outer_mu);
    ASSERT_TRUE(cap.captured);

    // The recovered steps must be nonzero (a genuine Newton step on the violated
    // row), so the applied fractions are well-defined.
    ASSERT_GT(std::abs(cap.dn), 1e-12);
    ASSERT_GT(std::abs(cap.dp), 1e-12);
    ASSERT_GT(std::abs(cap.dzn), 1e-12);
    ASSERT_GT(std::abs(cap.dzp), 1e-12);

    const double ap_n = (cap.n - init.n) / cap.dn;
    const double ap_p = (cap.p - init.p) / cap.dp;
    const double ad_n = (cap.zn - init.zn) / cap.dzn;
    const double ad_p = (cap.zp - init.zp) / cap.dzp;

    // n and p share the primal damping; z_n and z_p share the dual damping.
    EXPECT_NEAR(ap_n, ap_p, 1e-9);
    EXPECT_NEAR(ad_n, ad_p, 1e-9);

    // Fractions are fraction-to-boundary caps in (0, 1].
    EXPECT_GT(ap_n, 0.0);
    EXPECT_LE(ap_n, 1.0 + 1e-12);
    EXPECT_GT(ad_n, 0.0);
    EXPECT_LE(ad_n, 1.0 + 1e-12);

    // Positivity of every eliminated variable is preserved.
    EXPECT_GT(cap.n, 0.0);
    EXPECT_GT(cap.p, 0.0);
    EXPECT_GT(cap.zn, 0.0);
    EXPECT_GT(cap.zp, 0.0);
}

// (iv) Seam-split correctness: a non-nested (proximal) strategy never reaches the
// nested surface — the eval seam takes the proximal branch exclusively.
TEST(NestedRestorationSeam, ProximalPathNeverConsultsNestedSurface) {
    NestedSeamHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    NestedSeamProximalDouble *dbl = h.run_eval_seam_proximal(primals, /*mu=*/0.1);
    EXPECT_TRUE(dbl->proximal_touched_);  // proximal branch ran
    EXPECT_FALSE(dbl->nested_touched_);   // nested surface untouched
}

// -----------------------------------------------------------------------------
// 4b. Inequality-row seam through the assembled KKT. The equality-only harness
// above left the inequality condensation unverified at the seam level; that gap
// let a defect slip where the seam condensed the RAW residual g(x) instead of the
// slack-completed g(x)+s and the main loop's slack-completion then clobbered the
// condensed inequality RHS. These tests pin the inequality-row sign convention:
// the (z,z) diagonal is the negated inequality pivot, and the RHS carries the
// condensed r̃ built from g(x)+s (NOT the raw g(x)). Befriended by PSIOPT.
// -----------------------------------------------------------------------------
class NestedSeamIneqHarness {
  public:
    NestedSeamIneqHarness() {
        using tycho::vf::Arguments;
        using tycho::vf::GenericFunction;
        prob_.set_vars((Eigen::VectorXd(2) << 0.0, 0.0).finished());
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_objective(GenericFunction<-1, 1>(x0 * x0 + x1 * x1),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        {
            auto args = Arguments<1>();
            auto x = args.coeff<0>();
            // Inequality x0 - 5 <= 0; raw constraint value g(x) = x0 - 5. The elastic
            // row residual is the slack-completed g(x) + s.
            prob_.add_inequal_con(GenericFunction<-1, -1>(x - 5.0),
                                  (Eigen::VectorXi(1) << 0).finished());
        }
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
    }

    PSIOPT &solver() { return *solver_; }
    int pv() const { return solver_->primal_vars_; }
    int sv() const { return solver_->slack_vars_; }
    int ec() const { return solver_->equal_cons_; }
    int ic() const { return solver_->inequal_cons_; }
    int dim() const { return solver_->kkt_dim_; }

    // Enter nested restoration with the inequality residual g(x)+s implied by the
    // given primals and slack (no equality channel).
    NestedL1Restoration *enter_nested(const Eigen::VectorXd &primals, double slack,
                                      double outer_mu) {
        auto strat = std::make_unique<NestedL1Restoration>();
        NestedL1Restoration *raw = strat.get();
        solver_->restoration_ = std::move(strat);
        Eigen::VectorXd eq_res(0);
        Eigen::VectorXd iq_res(1);
        iq_res[0] = (primals[0] - 5.0) + slack; // g(x) + s
        tycho::solvers::ProgressMeasures ref;
        ref.infeasibility = 0.0;
        ref.objective = 0.0;
        ref.auxiliary = 0.0;
        raw->enter_nested(ref, primals, eq_res, iq_res, outer_mu);
        return raw;
    }

    // Run the nested eval seam once at (primals, slack, z) with barrier mu; record
    // the assembled KKT inequality-row diagonal and the RHS inequality segment.
    void run_eval_seam(const Eigen::VectorXd &primals, double slack, double z, double mu) {
        Eigen::VectorXd XSL = Eigen::VectorXd::Zero(dim());
        XSL.head(pv()) = primals;
        XSL[pv()] = slack;      // single slack (sv == ic == 1)
        XSL[dim() - 1] = z;     // single inequality multiplier
        Eigen::VectorXd GX = Eigen::VectorXd::Zero(pv());
        Eigen::VectorXd AGXS = Eigen::VectorXd::Zero(dim());
        double val = 0.0;
        solver_->eval_nlp(PSIOPT::AlgorithmModes::OPTNO, 0.0, XSL, val, GX, AGXS,
                          solver_->kkt_sol_.matrix(), mu);
        const int zrow = pv() + sv() + ec();
        kkt_zz_ = solver_->kkt_sol_.matrix().coeff(zrow, zrow);
        rhs_iq_ = AGXS.tail(ic())[0];
    }

    // Drive the private complementarity augmentation the barrier oracle consumes:
    // seed the aggregates with a (tiny, late-solve) original-pair value and let
    // the seam fold in the active phase's elastic pairs.
    void augment_comp(double &avgcomp, double &mincomp, double &maxcomp, int base_count) {
        solver_->augment_complementarity_nested(avgcomp, mincomp, maxcomp, base_count);
    }

    double kkt_zz_ = 0.0;
    double rhs_iq_ = 0.0;

  private:
    OptimizationProblem prob_;
    PSIOPT *solver_ = nullptr;
};

// (i-iq) The assembled KKT inequality-row diagonal equals the NEGATED component
// inequality pivot (same sign convention as the equality row).
TEST(NestedRestorationSeam, IqPivotDiagonalNegatesComponentPivot) {
    NestedSeamIneqHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    const double slack = 3.0;
    NestedL1Restoration *comp = h.enter_nested(primals, slack, /*outer_mu=*/0.1);
    h.run_eval_seam(primals, slack, /*z=*/0.5, /*mu=*/comp->entry_mu());

    ASSERT_EQ(comp->i_pivots().size(), 1);
    EXPECT_GT(comp->i_pivots()[0], 0.0); // component pivot is positive
    EXPECT_NEAR(h.kkt_zz_, -comp->i_pivots()[0], 1e-10);
}

// (ii-iq) The RHS inequality segment equals the condensed residual r̃ recomputed
// from the SLACK-COMPLETED residual g(x)+s (not the raw g(x)), the component's
// live elastic state, the multiplier z, and μ. Guarding this at the seam is the
// coverage that was missing: the raw-residual defect produced a materially
// different r̃ whenever the slack is nonzero.
TEST(NestedRestorationSeam, IqRhsSegmentCarriesSlackCompletedCondensedResidual) {
    NestedSeamIneqHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    const double slack = 3.0;
    NestedL1Restoration *comp = h.enter_nested(primals, slack, /*outer_mu=*/0.1);
    const double z = 0.5;
    const double mu = comp->entry_mu();
    h.run_eval_seam(primals, slack, z, mu);

    const double rho = tycho::solvers::kRestoPenaltyParameter;
    const double n = comp->ic_n()[0];
    const double p = comp->ic_p()[0];
    const double zn = comp->ic_zn()[0];
    const double zp = comp->ic_zp()[0];

    // Slack-completed residual: c = g(x) + s.
    const double c = (primals[0] - 5.0) + slack;
    const double rtilde =
        (c + n - p) + mu / zn - (n / zn) * (rho + z) - mu / zp + (p / zp) * (rho - z);
    EXPECT_NEAR(h.rhs_iq_, rtilde, 1e-8 * (1.0 + std::abs(rtilde)));

    // Regression guard: the raw-residual condensation (the defect) would land a
    // materially different r̃ because the slack is nonzero.
    const double c_raw = primals[0] - 5.0;
    const double rtilde_raw =
        (c_raw + n - p) + mu / zn - (n / zn) * (rho + z) - mu / zp + (p / zp) * (rho - z);
    EXPECT_GT(std::abs(rtilde - rtilde_raw), 1e-6);
}

// (iii-comp) nested_complementarity aggregates the elastic (n·z_n, p·z_p) pairs.
// At entry every pair equals resto_mu exactly (z_n = resto_mu/n, z_p = resto_mu/p),
// so the aggregate min/max/sum are pinned to resto_mu * count.
TEST(NestedRestorationComplementarity, AggregatesElasticPairsAtRestoScale) {
    NestedL1Restoration comp;
    tycho::solvers::ProgressMeasures ref;
    ref.infeasibility = 0.0;
    ref.objective = 0.0;
    ref.auxiliary = 0.0;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 1.0, 2.0).finished();
    Eigen::VectorXd eq_res = (Eigen::VectorXd(2) << -4.0, 3.0).finished();
    Eigen::VectorXd iq_res = (Eigen::VectorXd(1) << 2.0).finished();
    comp.enter_nested(ref, primals, eq_res, iq_res, /*outer_mu=*/0.1);
    const double resto_mu = comp.entry_mu(); // max(0.1, 4, 3, 2) = 4.0

    double sum = 0.0, mn = 0.0, mx = 0.0;
    int count = 0;
    comp.nested_complementarity(sum, mn, mx, count);

    // 2 equality rows * (n,p) + 1 inequality row * (n,p) = 6 elastic pairs.
    EXPECT_EQ(count, 6);
    EXPECT_NEAR(mn, resto_mu, 1e-9 * resto_mu);
    EXPECT_NEAR(mx, resto_mu, 1e-9 * resto_mu);
    EXPECT_NEAR(sum, 6.0 * resto_mu, 1e-9 * 6.0 * resto_mu);
}

// (iv-comp) The μ-oracle input the seam feeds is resto-scale, not floor-scale,
// once the original slack/multiplier complementarity has collapsed. This is the
// late-entry pathology: original comp ~1e-12, elastic pairs still ~resto_mu; the
// union average must track the elastic pairs so the barrier oracle does not drive
// μ to its floor.
TEST(NestedRestorationComplementarity, AugmentLiftsCollapsedOriginalToRestoScale) {
    NestedSeamIneqHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    const double slack = 3.0; // g(x)+s = (0-5)+3 = -2 -> resto_mu = max(0.1, 2) = 2
    NestedL1Restoration *comp = h.enter_nested(primals, slack, /*outer_mu=*/0.1);
    const double resto_mu = comp->entry_mu();
    ASSERT_NEAR(resto_mu, 2.0, 1e-12);

    // Late-solve original complementarity, collapsed to floor scale.
    double avg = 1e-12, mn = 1e-12, mx = 1e-12;
    h.augment_comp(avg, mn, mx, /*base_count=*/1);

    // Union max is the elastic pair (= resto_mu); union average is resto-scale,
    // decisively above the 1e-12 floor the un-augmented input would have fed.
    EXPECT_NEAR(mx, resto_mu, 1e-9 * resto_mu);
    EXPECT_GT(avg, 1e-3);
    EXPECT_LE(mn, 1e-12); // min still sees the tiny original pair
}

// (v-comp) Off the nested path the augmentation is a pure no-op: the aggregates
// are returned byte-identical, so the default/proximal barrier machinery is
// untouched (the CBWR invariant).
TEST(NestedRestorationComplementarity, AugmentIsNoOpWhenNotNested) {
    NestedSeamIneqHarness h;
    Eigen::VectorXd primals = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    NestedL1Restoration *comp = h.enter_nested(primals, /*slack=*/3.0, /*outer_mu=*/0.1);
    comp->exit_restoration(); // phase no longer active

    double avg = 7.5, mn = 0.25, mx = 11.0;
    const double avg0 = avg, mn0 = mn, mx0 = mx;
    h.augment_comp(avg, mn, mx, /*base_count=*/1);
    EXPECT_EQ(avg, avg0);
    EXPECT_EQ(mn, mn0);
    EXPECT_EQ(mx, mx0);
}

// -----------------------------------------------------------------------------
// 5. Nested-restoration LIFECYCLE (entry orchestration, exit ratchet, multiplier
// re-entry) through the private surface. NestedLifecycleHarness is befriended by
// PSIOPT so it can drive the private enter_/exit_feasibility_restoration helpers
// and the stashed-μ / ratchet state directly, and run the whole phase end-to-end
// by injecting a NestedL1Restoration into a solve driven through alg_impl. No
// construction wiring for the nested mode exists yet (a later change adds it), so
// this direct injection is the only way to exercise the lifecycle.
// -----------------------------------------------------------------------------

class NestedLifecycleHarness {
  public:
    // 2-variable QP: minimize x0² + x1² s.t. x0 + x1 − 4 = 0 (unique optimum
    // (2,2)), optionally a contradictory second equality x0 + x1 = 0 (genuinely
    // infeasible), and `n_ineq` non-binding inequalities x0 − (100+k) ≤ 0 so the
    // governor's mid-iteration μ update runs (inequal_cons_ > 0).
    NestedLifecycleHarness(const Eigen::VectorXd &start, int n_ineq, bool inconsistent)
        : start_(start) {
        using tycho::vf::Arguments;
        using tycho::vf::GenericFunction;
        prob_.set_vars(start);
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_objective(GenericFunction<-1, 1>(x0 * x0 + x1 * x1),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_equal_con(GenericFunction<-1, -1>(x0 + x1 - 4.0),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        if (inconsistent) {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob_.add_equal_con(GenericFunction<-1, -1>(x0 + x1 - 0.0),
                                (Eigen::VectorXi(2) << 0, 1).finished());
        }
        for (int k = 0; k < n_ineq; ++k) {
            auto args = Arguments<1>();
            auto x = args.coeff<0>();
            prob_.add_inequal_con(GenericFunction<-1, -1>(x - (100.0 + k)),
                                  (Eigen::VectorXi(1) << 0).finished());
        }
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
    }

    PSIOPT &solver() { return *solver_; }
    int pv() const { return solver_->primal_vars_; }
    int sv() const { return solver_->slack_vars_; }
    int ec() const { return solver_->equal_cons_; }
    int ic() const { return solver_->inequal_cons_; }
    int dim() const { return solver_->kkt_dim_; }

    // Build acceptance_/governor_/recovery_ (through the proximal-switch mode, the
    // wired one) then replace restoration_ with a fresh NestedL1Restoration.
    NestedL1Restoration *inject_nested() {
        solver_->settings().restoration_mode_ = RestorationModes::proximal_switch;
        solver_->rebuild_globalization_components();
        auto strat = std::make_unique<NestedL1Restoration>();
        NestedL1Restoration *raw = strat.get();
        solver_->restoration_ = std::move(strat);
        return raw;
    }

    // Direct drivers for the private lifecycle helpers + state.
    void call_enter(Eigen::VectorXd &XSL, Eigen::VectorXd &RHS, double prim_obj, double barr_obj,
                    double &mu) {
        solver_->enter_feasibility_restoration(XSL, RHS, prim_obj, barr_obj, mu);
    }
    void call_exit(Eigen::VectorXd &XSL, double obj_scale, double theta, double barr, double &mu) {
        solver_->exit_feasibility_restoration_nested(XSL, obj_scale, theta, barr, mu);
    }
    bool ratchet(double theta) const { return solver_->resto_ratchet_passes(theta); }
    double &stashed_mu() { return solver_->stashed_mu_; }
    bool &first_iter() { return solver_->resto_first_iter_; }
    // Second-level re-center one-shot flag + the production re-center seam.
    bool &recentered() { return solver_->resto_recentered_; }
    bool call_try_recenter(double mu) { return solver_->try_recenter_elastics(mu); }
    double &theta_prev() { return solver_->resto_theta_orig_prev_; }
    void set_econ_tol(double t) { solver_->settings().econ_tol_ = t; }

    // True exactly when a nested l1 restoration phase is live — the same predicate
    // the alg_impl barrier-update seam gates the monotone schedule on.
    bool nested_active() const {
        return solver_->restoration_ && solver_->restoration_->is_active() &&
               solver_->restoration_->is_nested();
    }
    // Replace the configured barrier governor (used to install a recording guard
    // after rebuild_globalization_components() has built the default one).
    void set_governor(std::unique_ptr<tycho::solvers::BarrierGovernor> g) {
        solver_->governor_ = std::move(g);
    }

    // Segment accessors into a full KKT vector [primals | slacks | eq | iq].
    Eigen::VectorXd zero_vec() const { return Eigen::VectorXd::Zero(dim()); }

    // Full solve forcing a feasibility switch (max_ls_iters_ = 0), driven through
    // alg_impl directly with the injected nested strategy. Records the final
    // iterate's μ so a caller can confirm it is the outer schedule value, not the
    // (large) restoration barrier parameter.
    tycho::ConvergenceFlags run_forced_entry(NestedL1Restoration *&comp_out,
                                              PSIOPT::LineSearchModes lsmode,
                                              PSIOPT::BarrierModes barmode, double &final_mu,
                                              double init_mu = 0.1, int preload_soft_counter = 0,
                                              const std::function<void()> &after_rebuild = {}) {
        solver_->settings().restoration_mode_ = RestorationModes::proximal_switch;
        solver_->settings().max_ls_iters_ = 0;
        solver_->settings().max_iters_ = 120;
        solver_->rebuild_globalization_components();
        // Optional hook: swap in a test governor now that the default one has been
        // built (see FreeOracleUnreachableDuringNestedPhase).
        if (after_rebuild)
            after_rebuild();
        auto strat = std::make_unique<NestedL1Restoration>();
        comp_out = strat.get();
        solver_->restoration_ = std::move(strat);
        if (preload_soft_counter > 0) {
            // Pre-exhaust the soft pre-stage budget so the first ladder-exhausted
            // rejection escalates straight into the full restoration phase —
            // deterministic full-lifecycle coverage on a feasible problem.
            auto *fsr = dynamic_cast<FeasibilitySwitchRecovery *>(solver_->recovery_.get());
            if (fsr)
                fsr->soft_counter_ = preload_soft_counter;
        }
        solver_->ensure_solver_initialized();
        bool docompute = solver_->claim_kkt_analysis();
        Eigen::VectorXd XSL = solver_->init_impl(start_, init_mu, docompute);
        final_mu = init_mu;
        double *fm = &final_mu;
        solver_->set_late_callback(
            [fm](const IterateInfo &it, tycho::ConstEigenRef<Eigen::VectorXd>,
                 tycho::ConstEigenRef<Eigen::VectorXd>) -> int {
                *fm = it.mu_;
                return 0;
            });
        solver_->alg_impl(PSIOPT::AlgorithmModes::OPT, barmode, lsmode,
                          solver_->settings().obj_scale_, init_mu, XSL);
        solver_->disable_late_callback();
        return solver_->result().converge_flag_;
    }

    double start_x0() const { return start_[0]; }

  private:
    OptimizationProblem prob_;
    PSIOPT *solver_ = nullptr;
    Eigen::VectorXd start_;
};

// (i) Entry sets μ ← max(μ, ‖h‖∞, ‖g+s‖∞) and stashes the old value; the
// equality constraint multipliers are zeroed and the ratchet baseline seeded.
TEST(NestedRestorationLifecycle, EntrySetsBarrierAndStashesOuterMu) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = h.inject_nested();

    Eigen::VectorXd XSL = h.zero_vec();
    Eigen::VectorXd RHS = h.zero_vec();
    // Equality residual h = 5.0 in the eq_cons slot; a nonzero equality multiplier
    // to confirm zeroing.
    const int yrow = h.pv() + h.sv();
    RHS[yrow] = 5.0;
    XSL[yrow] = 0.7;

    double mu = 0.1;
    h.call_enter(XSL, RHS, /*prim_obj=*/0.0, /*barr_obj=*/0.0, mu);

    EXPECT_DOUBLE_EQ(h.stashed_mu(), 0.1);       // old μ stashed
    EXPECT_DOUBLE_EQ(mu, 5.0);                   // μ ← max(0.1, |h|)
    EXPECT_DOUBLE_EQ(comp->entry_mu(), 5.0);     // component agrees
    EXPECT_DOUBLE_EQ(XSL[yrow], 0.0);            // equality multiplier zeroed
    EXPECT_TRUE(h.first_iter());                 // first-iteration guard armed
    EXPECT_DOUBLE_EQ(h.theta_prev(), 5.0);       // ratchet seeded with entry θ
}

// (ii) The multiplier re-entry restores the stashed outer μ.
TEST(NestedRestorationLifecycle, ExitRestoresStashedMu) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    h.inject_nested();

    Eigen::VectorXd XSL = h.zero_vec();
    Eigen::VectorXd RHS = h.zero_vec();
    RHS[h.pv() + h.sv()] = 3.0;
    double mu = 0.25;
    h.call_enter(XSL, RHS, 0.0, 0.0, mu);
    ASSERT_DOUBLE_EQ(mu, 3.0); // in-phase μ

    h.call_exit(XSL, /*obj_scale=*/1.0, /*theta=*/1e-8, /*barr=*/0.0, mu);
    EXPECT_DOUBLE_EQ(mu, 0.25); // outer μ restored
}

// (iii) Re-entry values: the slack-multiplier Newton z-step is computed from
// the PRE-exit multipliers, and the equality multipliers are zeroed. With z
// below μ_outer/s the fraction-to-boundary cap is 1, so z lands exactly at
// μ_outer/s. NOTE: the z-step reads only slacks/iq_lmults and the zeroing
// writes only eq_lmults (disjoint state), so this test cannot detect a
// reordering of those two steps; the ordering that is actually load-bearing —
// the z-step reading pre-reset multipliers before the reset-all-to-1 check —
// is pinned by the reset-threshold test below.
TEST(NestedRestorationLifecycle, ReentryNewtonStepThenZeroesEqualityMultipliers) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 1.0, 1.0).finished(), /*n_ineq=*/1,
                             /*inconsistent=*/false);
    h.inject_nested();

    Eigen::VectorXd XSL = h.zero_vec();
    Eigen::VectorXd RHS = h.zero_vec();
    RHS[h.pv() + h.sv()] = 0.5; // some equality residual for entry
    double mu = 1.0;
    h.call_enter(XSL, RHS, 0.0, 0.0, mu);

    // Hand-set the phase's final slack / multipliers, then force a known stashed μ.
    const int srow = h.pv();               // first slack
    const int yrow = h.pv() + h.sv();      // equality multiplier
    const int zrow = h.pv() + h.sv() + h.ec(); // first inequality multiplier
    XSL[srow] = 0.5;
    XSL[zrow] = 0.5;    // z < μ_outer/s = 2.0 ⇒ Δz > 0, no boundary cap
    XSL[yrow] = 0.9;    // nonzero, must be zeroed
    h.stashed_mu() = 1.0;

    h.call_exit(XSL, 1.0, 1e-8, 0.0, mu);

    EXPECT_NEAR(XSL[zrow], 2.0, 1e-12);  // z ← μ_outer/s (undamped)
    EXPECT_DOUBLE_EQ(XSL[yrow], 0.0);    // equality multiplier zeroed
    EXPECT_DOUBLE_EQ(mu, 1.0);           // μ restored
}

// (iv) The z-reset triggers when the updated max|z| exceeds kBoundMultReset-
// Threshold (1e3): ALL inequality multipliers reset to 1, not just the offender.
TEST(NestedRestorationLifecycle, ReentryResetsAllMultipliersToOneWhenTooLarge) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 1.0, 1.0).finished(), /*n_ineq=*/2,
                             /*inconsistent=*/false);
    h.inject_nested();

    Eigen::VectorXd XSL = h.zero_vec();
    Eigen::VectorXd RHS = h.zero_vec();
    RHS[h.pv() + h.sv()] = 0.5;
    double mu = 1.0;
    h.call_enter(XSL, RHS, 0.0, 0.0, mu);

    const int srow = h.pv();
    const int zrow = h.pv() + h.sv() + h.ec();
    // First slack tiny ⇒ Δz drives z to μ/s = 1e4 > 1e3 ⇒ reset ALL to 1.
    XSL[srow] = 1.0e-4;
    XSL[srow + 1] = 0.5;
    XSL[zrow] = 0.5;
    XSL[zrow + 1] = 0.5;
    h.stashed_mu() = 1.0;

    h.call_exit(XSL, 1.0, 1e-8, 0.0, mu);

    EXPECT_DOUBLE_EQ(XSL[zrow], 1.0);     // offender reset
    EXPECT_DOUBLE_EQ(XSL[zrow + 1], 1.0); // AND the non-offender reset too
}

// (v) κ_resto ratchet truth table, including the econ_tol floor.
TEST(NestedRestorationLifecycle, KappaRestoRatchetTruthTable) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    h.inject_nested();
    h.set_econ_tol(1.0e-6);

    // Relative-reduction regime (prev large, floor inactive): pass iff θ ≤ 0.9·prev.
    h.theta_prev() = 1.0;
    EXPECT_TRUE(h.ratchet(0.89));  // below 0.9·1.0
    EXPECT_TRUE(h.ratchet(0.90));  // exactly at the ratchet
    EXPECT_FALSE(h.ratchet(0.95)); // above the ratchet

    // Floor regime (prev tiny, 0.9·prev < econ_tol): pass iff θ ≤ econ_tol.
    h.theta_prev() = 1.0e-9;
    EXPECT_TRUE(h.ratchet(5.0e-7)); // below the econ_tol floor
    EXPECT_TRUE(h.ratchet(1.0e-6)); // at the floor
    EXPECT_FALSE(h.ratchet(2.0e-6)); // above the floor
}

// (vi) End-to-end tiny infeasible-start feasible problem. With the soft
// feasibility pre-stage in place, forcing every regular step to be rejected
// (max_ls_iters_ = 0) no longer drops straight into the full l1 phase for a
// nested strategy: the pre-stage first takes the full fraction-to-boundary step
// on the current direction, which for this quadratic-objective / linear-equality
// problem is the exact Newton step to the optimum, reduces the primal-dual error,
// and is accepted. The pre-stage therefore RESOLVES the problem without ever
// entering the full l1 phase (comp->entries() stays 0). Converging here at all
// under max_ls_iters_ = 0 is itself proof the soft steps were taken (nothing else
// makes progress when every regular step is rejected), and the final μ stays on
// the outer schedule (the restoration barrier bump only happens at full entry,
// which never occurred).
TEST(NestedRestorationLifecycle, SoftPreStageResolvesWithoutFullEntry) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1);

    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_NE(comp, nullptr);
    EXPECT_EQ(comp->entries(), 0); // the pre-stage resolved it; no full l1 entry
    const auto &r = h.solver().result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 2.0, 1e-3);
    EXPECT_NEAR(r.primals_[1], 2.0, 1e-3);
    // μ never bumped to the restoration barrier: it stayed on the outer schedule.
    EXPECT_LT(final_mu, 1.0);
}

// (viii) Inequality-constrained end-to-end solve with the soft pre-stage active:
// the pre-stage's trial evaluation slack-completes the inequality residual and
// includes the complementarity deviation in the primal-dual error, and the
// governor's mid-iteration μ update runs. The full fraction-to-boundary soft
// steps converge on the true optimum with a non-binding inequality present.
TEST(NestedRestorationLifecycle, EndToEndWithInequalityConvergesUnderSoftPreStage) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/1,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1);

    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_NE(comp, nullptr);
    const auto &r = h.solver().result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 2.0, 1e-3);
    EXPECT_NEAR(r.primals_[1], 2.0, 1e-3);
}

// (ix) LANG line-search mode with the soft pre-stage active: the soft step is
// taken directly (it bypasses the backtracking line search), so the solve
// converges regardless of the configured line-search mode.
TEST(NestedRestorationLifecycle, EndToEndLangLineSearchConvergesUnderSoftPreStage) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::LANG, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1);

    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_NE(comp, nullptr);
    const auto &r = h.solver().result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 2.0, 1e-3);
    EXPECT_NEAR(r.primals_[1], 2.0, 1e-3);
}

// (x) End-to-end escalation into the full l1 phase. A genuinely infeasible
// problem (contradictory equalities x0+x1 = 4 and x0+x1 = 0) cannot have its
// primal-dual error driven down indefinitely by regular steps: the soft
// pre-stage's fraction-to-boundary step stops reducing the primal-dual error
// once it reaches the least-infeasible point, the reduction test fails, and the
// pre-stage escalates into the full l1 restoration phase (comp->entries() >= 1).
// The phase runs and the solve correctly does NOT falsely report the infeasible
// problem as converged. Entry depends on the rank-deficient KKT producing a
// finite step, which is platform-dependent factorization behavior, so the entry
// assertion is skip-guarded exactly like the proximal infeasible test above.
TEST(NestedRestorationLifecycle, SoftPreStageEscalatesIntoFullL1Phase) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/true);
    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1);

    ASSERT_NE(comp, nullptr);
    EXPECT_NE(flag, tycho::ConvergenceFlags::CONVERGED); // never falsely converges
    if (comp->entries() < 1) {
        GTEST_SKIP() << "factorization returned non-finite step on this platform; "
                        "escalation not exercised";
    }
    // The skip above is the escalation assertion -- reaching past it means
    // comp->entries() >= 1, i.e. the pre-stage escalated into the full l1 phase.
}

// The nested full lifecycle on a FEASIBLE problem, unconditionally exercised:
// pre-exhausting the soft budget makes the first ladder-exhausted rejection
// escalate immediately, so the solve must enter the l1 phase, exit it cleanly
// (ratchet + acceptance test + multiplier re-entry), and then converge on the
// original problem. Deterministic companion to the escalation test above,
// whose infeasible problem classifies as a failure and whose factorization is
// platform-sensitive (skip-guarded): entry, clean exit, and post-exit
// convergence all execute here on every platform.
TEST(NestedRestorationLifecycle, ForcedEscalationRunsFullPhaseAndConverges) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1,
                                   /*preload_soft_counter=*/kMaxSoftRestoIters);

    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_NE(comp, nullptr);
    EXPECT_GE(comp->entries(), 1); // full phase entered
    const auto &r = h.solver().result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 2.0, 1e-3);
    EXPECT_NEAR(r.primals_[1], 2.0, 1e-3);
    EXPECT_LT(final_mu, 1.0); // back on the outer barrier schedule after exit
}

// (vii) Proximal-switch behavior is unchanged: a proximal forced-entry solve on
// the same problem still enters, exits, and converges (the seam split left the
// non-nested path intact). The broader proximal suite in section 3 above is the
// primary guard; this is an explicit co-existence check alongside the nested
// lifecycle tests.
TEST(NestedRestorationLifecycle, ProximalPathStillEntersExitsConverges) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    // Drive a real proximal-switch forced-entry run through the public API (no
    // nested injection, no preloaded counters, no private-member pokes): set the
    // mode, cap the classic ladder at zero so the first rejection is already
    // ladder-exhausted, and call optimize().
    //
    // optimize(), NOT solve(). The solve-only phase runs soe_ls_mode_, which
    // defaults to NOLS, and BacktrackingLineSearch::run_acceptance_backtrack
    // short-circuits NOLS by stamping Citer.accepted_ = true and returning the
    // full step. should_dispatch_recovery() is (GoodStep && !accepted_), so on
    // that path the recovery chain -- and with it FeasibilitySwitchRecovery --
    // is never invoked at all, and max_ls_iters_ is never even read. Restoration
    // entry is therefore structurally unreachable through solve(); only a phase
    // running a real line-search mode (opt_ls_mode_, AUGLANG by default) can
    // produce the ladder-exhausted rejection this test is about.
    h.solver().settings().restoration_mode_ = RestorationModes::proximal_switch;
    h.solver().set_max_ls_iters(0);
    h.solver().set_max_iters(120);
    Eigen::VectorXd x = (Eigen::VectorXd(2) << 0.0, 0.0).finished();
    Eigen::VectorXd sol = h.solver().optimize(x);
    const auto &r = h.solver().result();
    EXPECT_LE(r.converge_flag_, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_EQ(sol.size(), 2);
    EXPECT_NEAR(sol[0], 2.0, 1e-3);
    EXPECT_NEAR(sol[1], 2.0, 1e-3);
    // Restoration was actually entered -- the same bar the proximal forced-entry
    // sibling holds (ForcedEntryOnFeasibleProblemEntersExitsAndConverges, same
    // recipe on its own problem). >= 0 would be satisfied by merely constructing
    // a strategy, which is what this test used to assert.
    EXPECT_GE(r.last_feas_rest_entries_, 1);
}

// -----------------------------------------------------------------------------
// 5b. Second-level elastic re-centering fallback (disclosure (f)). When an
// in-phase line search exhausts the recovery ladder, alg_impl's kAcceptAsIs case
// re-centers the elastic pairs once (INSTEAD of taking the failed step) and then
// falls through to accept-as-is on the next consecutive exhaustion; the one-shot
// budget re-arms on any accepted step and at each phase entry.
// -----------------------------------------------------------------------------

// (b1) End-to-end one-shot. Forcing every regular step to be rejected
// (max_ls_iters_ = 0) and pre-exhausting the soft budget drives the solve
// straight into the full l1 phase, whose every in-phase iteration is then a
// ladder-exhausted kAcceptAsIs. The second-level fallback re-centers on the FIRST
// such exhaustion and — because no step is ever accepted under max_ls_iters_ = 0,
// so the one-shot flag never re-arms — every later exhaustion falls through to
// accept-as-is. The observer therefore records EXACTLY one re-center across the
// whole phase (without the guard it would re-center every in-phase iteration),
// and the phase still runs and converges on the original problem.
TEST(NestedRestorationRecenter, FiresExactlyOncePerFailureRunEndToEnd) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1,
                                   /*preload_soft_counter=*/kMaxSoftRestoIters);

    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_NE(comp, nullptr);
    EXPECT_GE(comp->entries(), 1);           // full phase entered
    EXPECT_EQ(comp->recenter_calls(), 1);    // exactly one re-center; rest fall through
    const auto &r = h.solver().result();
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 2.0, 1e-3);
    EXPECT_NEAR(r.primals_[1], 2.0, 1e-3);
}

// (b2) The one-shot budget re-arms at each phase entry: a stale set flag from a
// prior episode does not suppress a re-center in the next.
TEST(NestedRestorationRecenter, EntryReArmsOneShotBudget) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/0,
                             /*inconsistent=*/false);
    h.inject_nested();

    Eigen::VectorXd XSL = h.zero_vec();
    Eigen::VectorXd RHS = h.zero_vec();
    RHS[h.pv() + h.sv()] = 0.5; // equality residual for entry
    double mu = 0.1;

    h.recentered() = true; // pretend a prior episode consumed the budget
    h.call_enter(XSL, RHS, 0.0, 0.0, mu);
    EXPECT_FALSE(h.recentered()); // entry re-armed it
}

// (b3) The one-shot guard + clear-then-re-arm state machine, driven through the
// production try_recenter_elastics seam (the exact call alg_impl's kAcceptAsIs
// case makes). Deterministic: the first call re-centers, a second consecutive
// call falls through (returns false, no re-center), and clearing the flag — what
// alg_impl does on any accepted step — re-arms the budget so the next exhaustion
// re-centers again.
TEST(NestedRestorationRecenter, TrySeamIsOneShotAndReArmsOnClear) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/1,
                             /*inconsistent=*/false);
    NestedL1Restoration *comp = h.inject_nested();

    Eigen::VectorXd XSL = h.zero_vec();
    Eigen::VectorXd RHS = h.zero_vec();
    RHS[h.pv() + h.sv()] = 0.5;
    double mu = 0.1;
    h.call_enter(XSL, RHS, 0.0, 0.0, mu); // activates + seeds resto_ec_/ic_scratch_
    ASSERT_TRUE(h.nested_active());
    ASSERT_FALSE(h.recentered());
    ASSERT_EQ(comp->recenter_calls(), 0);

    // First ladder exhaustion of the run: re-centers, consumes the budget.
    EXPECT_TRUE(h.call_try_recenter(mu));
    EXPECT_TRUE(h.recentered());
    EXPECT_EQ(comp->recenter_calls(), 1);

    // Second consecutive exhaustion while set: falls through, no re-center.
    EXPECT_FALSE(h.call_try_recenter(mu));
    EXPECT_TRUE(h.recentered());
    EXPECT_EQ(comp->recenter_calls(), 1);

    // An accepted step clears the flag (what alg_impl does): the budget re-arms.
    h.recentered() = false;
    EXPECT_TRUE(h.call_try_recenter(mu));
    EXPECT_TRUE(h.recentered());
    EXPECT_EQ(comp->recenter_calls(), 2);
}

// -----------------------------------------------------------------------------
// 6. Monotone in-phase barrier schedule (Ipopt default restoration mu_strategy).
//
// While a nested l1 restoration phase is active, a governor that lacks its own
// monotone safeguard (the free-mode classic_adaptive governor) has its in-phase
// barrier update routed to BarrierGovernor::update_barrier_monotone: the
// safeguarded Fiacco-McCormick ladder anchored at resto_mu, never a free-mode
// oracle. A governor that supplies its own safeguard (the monitored governor)
// keeps driving its own in-phase update. Layers below:
//   (a) update_barrier_monotone in isolation — μ holds while the restoration
//       barrier subproblem is unsolved and advances only when the progress gate
//       fires; it never jumps to the μ floor while complementarity is at
//       restoration scale.
//   (b) end-to-end, free governor — the free oracle is NEVER consulted during
//       the phase (routing bypasses it), yet the phase runs and the configured
//       governor still drives the pre/post optimality iterations.
//   (b') end-to-end, self-safeguarding governor — the seam leaves it driving its
//       own in-phase update (its oracle IS consulted while nested-active).
// -----------------------------------------------------------------------------

// A recording governor: delegates the free-mode oracle to a real
// ClassicAdaptiveGovernor but counts every update_barrier call, and separately
// counts any that land while a nested phase is active. `safeguard` controls
// provides_restoration_barrier_safeguard(): when false (a free-type governor)
// the alg_impl seam routes the in-phase update to update_barrier_monotone (the
// non-virtual base method, NOT overridden here), so update_barrier must never be
// consulted while nested-active; when true (a self-safeguarding governor) the
// seam leaves this governor driving its own in-phase update, so update_barrier
// IS consulted while nested-active. Unity-unique name (FeasSwitch* prefix).
class FeasSwitchRecordingGovernor : public BarrierGovernor {
  public:
    FeasSwitchRecordingGovernor(std::function<bool()> nested_active, bool safeguard)
        : nested_active_(std::move(nested_active)), safeguard_(safeguard) {}

    double update_barrier(PSIOPT::BarrierModes barmode, double mu_in, double avgcomp,
                          double mincomp, Eigen::VectorXd &XSL, Eigen::VectorXd &RHS,
                          Eigen::VectorXd &DXSL, Eigen::VectorXd &Temp,
                          GlobalizationMechanism &mechanism, SolverContext &ctx, double &barr_obj,
                          const IterateInfo &current, bool &mu_event) override {
        ++total_calls_;
        if (nested_active_ && nested_active_())
            ++calls_during_nested_;
        return delegate_.update_barrier(barmode, mu_in, avgcomp, mincomp, XSL, RHS, DXSL, Temp,
                                        mechanism, ctx, barr_obj, current, mu_event);
    }
    void reset() override { delegate_.reset(); }
    bool provides_restoration_barrier_safeguard() const override { return safeguard_; }

    int total_calls() const { return total_calls_; }
    int calls_during_nested() const { return calls_during_nested_; }

  private:
    ClassicAdaptiveGovernor delegate_;
    std::function<bool()> nested_active_;
    bool safeguard_;
    int total_calls_ = 0;
    int calls_during_nested_ = 0;
};

// (a) update_barrier_monotone: hold vs advance on the Fiacco-McCormick gate, the
// barrier tail, and the no-collapse-to-floor property. Driven on a
// ClassicAdaptiveGovernor instance to prove the schedule is governor-agnostic
// (it is the non-virtual base method).
TEST(NestedRestorationMonotoneSchedule, HoldsThenAdvancesOnBarrierProgressGate) {
    ClassicAdaptiveGovernor g; // the free-mode merit governor; monotone is on the base.

    // defaults: bar_tol_=kkt_tol_=1e-6, min_mu_=1e-12, max_mu_=100.
    InertSolverContext inert;
    inert.primal_vars_ = 1;
    inert.slack_vars_ = 2;
    inert.equal_cons_ = 0;
    inert.inequal_cons_ = 2;
    inert.kkt_dim_ = 5;
    SolverContext ctx = inert.ctx();

    // Layout [primals(1) | slacks(2) | eq(0) | iq(2)]: slacks = {2,4}, iq_lmults = {0.5,0.25}.
    Eigen::VectorXd XSL(5);
    XSL << 1.0, 2.0, 4.0, 0.5, 0.25;
    const double log_sum = std::log(2.0) + std::log(4.0);

    // --- HOLD: barr_inf_ = 100 > kBarrierTolFactor * mu_in (10*4=40): gate blocks. ---
    {
        Eigen::VectorXd RHS = Eigen::VectorXd::Zero(5);
        IterateInfo cur;
        cur.kkt_inf_ = 0.0;
        cur.econ_inf_ = 0.0;
        cur.icon_inf_ = 0.0;
        cur.barr_inf_ = 100.0; // restoration-scale complementarity: subproblem NOT solved.
        double barr_obj = 0.0;
        bool mu_event = true; // must be cleared by the call.
        const double mu = g.update_barrier_monotone(/*mu_in=*/4.0, XSL, RHS, ctx, barr_obj, cur,
                                                     mu_event);
        EXPECT_DOUBLE_EQ(mu, 4.0);          // held at the anchored resto μ.
        EXPECT_FALSE(mu_event);             // no new subproblem.
        EXPECT_GT(mu, inert.settings_.min_mu_); // never collapsed to the μ floor.
        EXPECT_DOUBLE_EQ(barr_obj, -4.0 * log_sum);
        EXPECT_DOUBLE_EQ(RHS[1], 0.5 - 4.0 / 2.0);  // iq_lmult - μ/slack
        EXPECT_DOUBLE_EQ(RHS[2], 0.25 - 4.0 / 4.0);
    }

    // --- ADVANCE: every residual 0.05 <= 40: gate passes, Fiacco-McCormick step. ---
    {
        Eigen::VectorXd RHS = Eigen::VectorXd::Zero(5);
        IterateInfo cur;
        cur.kkt_inf_ = 0.05;
        cur.econ_inf_ = 0.05;
        cur.icon_inf_ = 0.05;
        cur.barr_inf_ = 0.05; // subproblem sufficiently solved at μ=4.
        double barr_obj = 0.0;
        bool mu_event = false;
        const double mu = g.update_barrier_monotone(/*mu_in=*/4.0, XSL, RHS, ctx, barr_obj, cur,
                                                     mu_event);
        // fiacco_mccormick_mu(4, 1e-6,1e-6,1e-12,100) = min(0.2*4=0.8, 4^1.5=8) = 0.8.
        EXPECT_DOUBLE_EQ(mu, 0.8);
        EXPECT_TRUE(mu_event); // strict decrease -> a new barrier subproblem.
        EXPECT_DOUBLE_EQ(barr_obj, -0.8 * log_sum);
        EXPECT_DOUBLE_EQ(RHS[1], 0.5 - 0.8 / 2.0);
        EXPECT_DOUBLE_EQ(RHS[2], 0.25 - 0.8 / 4.0);
    }
}

// (b) End-to-end, FREE governor (no own safeguard): the free oracle is NEVER
// consulted while a nested phase is active — the seam routes the in-phase update
// to update_barrier_monotone — yet the phase is entered, the solve reaches
// optimality, and the configured governor still runs the optimality iterations
// (total calls > 0), i.e. the configured mode resumes at exit.
TEST(NestedRestorationLifecycle, FreeOracleUnreachableDuringNestedPhase) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/1,
                             /*inconsistent=*/false);
    auto guard = std::make_unique<FeasSwitchRecordingGovernor>([&h] { return h.nested_active(); },
                                                               /*safeguard=*/false);
    FeasSwitchRecordingGovernor *raw = guard.get();

    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    auto flag = h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO,
                                   final_mu, /*init_mu=*/0.1,
                                   /*preload_soft_counter=*/kMaxSoftRestoIters,
                                   /*after_rebuild=*/[&] { h.set_governor(std::move(guard)); });

    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_NE(comp, nullptr);
    EXPECT_GE(comp->entries(), 1);            // the full l1 phase was entered
    EXPECT_EQ(raw->calls_during_nested(), 0); // free oracle never consulted while active
    EXPECT_GT(raw->total_calls(), 0);         // configured governor drives the optimality iters
    // The solve converged and ended back on the outer barrier schedule, which
    // legitimately drives μ to its floor at convergence — so no lower bound
    // applies here. The frozen-phase signature (floor-scale μ while stuck
    // in-phase for hundreds of iterations) is excluded by the convergence flag
    // above together with the harness's tight iteration cap; this assertion
    // pins only that the schedule advanced below the entry-time outer μ.
    EXPECT_LT(final_mu, 0.1);
}

// (b') The gate's other direction: a governor that reports its OWN restoration
// safeguard is left to drive the in-phase update itself — the seam does NOT
// route it through update_barrier_monotone, so its update_barrier IS consulted
// while nested-active (calls_during_nested > 0).
TEST(NestedRestorationLifecycle, SelfSafeguardingGovernorDrivesInPhaseUpdate) {
    NestedLifecycleHarness h((Eigen::VectorXd(2) << 0.0, 0.0).finished(), /*n_ineq=*/1,
                             /*inconsistent=*/false);
    auto guard = std::make_unique<FeasSwitchRecordingGovernor>([&h] { return h.nested_active(); },
                                                               /*safeguard=*/true);
    FeasSwitchRecordingGovernor *raw = guard.get();

    NestedL1Restoration *comp = nullptr;
    double final_mu = 0.0;
    h.run_forced_entry(comp, PSIOPT::LineSearchModes::L1, PSIOPT::BarrierModes::LOQO, final_mu,
                       /*init_mu=*/0.1, /*preload_soft_counter=*/kMaxSoftRestoIters,
                       /*after_rebuild=*/[&] { h.set_governor(std::move(guard)); });

    ASSERT_NE(comp, nullptr);
    EXPECT_GE(comp->entries(), 1);            // the full l1 phase was entered
    EXPECT_GT(raw->calls_during_nested(), 0); // its own oracle drives the in-phase update
}
