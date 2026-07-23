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

#include "tycho/detail/solvers/globalization/feasibility_switch_recovery.h"
#include "tycho/detail/solvers/globalization/filter_acceptance.h"
#include "tycho/detail/solvers/globalization/proximal_restoration.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <Eigen/Core>

using namespace tycho;
using namespace TychoTest;

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::FeasibilitySwitchRecovery;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::kRecoveryDepthRestoration;
using tycho::solvers::kRecoveryDepthUnresolved;
using tycho::solvers::KktSolverType;
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

    bool active_ = false;
    bool permit_ = true;

  private:
    Eigen::VectorXd diag_;
    ProgressMeasures ref_;
};

// Inner recovery double returning a configured Action, so FeasibilitySwitchRecovery's
// delegation and pass-through can be observed. Stamps a marker resolved_depth so
// the pass-through case can confirm it is preserved.
class FeasSwitchStubInner : public RecoveryChain {
  public:
    explicit FeasSwitchStubInner(Action action) : action_(action) {}
    Action on_step_rejected(IterateInfo &, const std::vector<IterateInfo> &, SolverContext &,
                            AcceptanceStrategy &, GlobalizationMechanism &,
                            PSIOPT::LineSearchModes, double, double, double, double,
                            Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                            Eigen::VectorXd &, Eigen::VectorXd &, double &, double &, double &,
                            int &, int &resolved_depth, int &) override {
        ++calls_;
        resolved_depth = kRecoveryDepthUnresolved;
        return action_;
    }
    void reset() override {}

    Action action_;
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
        return 1.0;
    }
    void max_primal_dual_step(Eigen::VectorXd &, Eigen::VectorXd &, double, double &, double &,
                              const SolverContext &) override {}
    void reset() override {}
};

// Inert acceptance to satisfy the on_step_rejected signature.
class FeasSwitchUnusedAcceptance : public AcceptanceStrategy {
  public:
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}
    bool drives_classic_path() const override { return true; }
};

// Minimal SolverContext for the recovery signature (zero dims, so the RHS tail
// is empty and the constraint violation is 0 — the stub ignores it anyway).
SolverContext feas_switch_context(KktSolverType &solver, PSIOPT::Settings &settings, int &zero,
                                  Eigen::VectorXd &scratch, const RestorationStrategy *restoration) {
    SolverContext ctx{nullptr, solver,  settings, zero,    zero,    zero,
                      zero,    zero,    scratch,  scratch, scratch, scratch};
    ctx.restoration_ = restoration;
    return ctx;
}

// Drive FeasibilitySwitchRecovery::on_step_rejected once and return its Action;
// captures resolved_depth via the out-parameter.
RecoveryChain::Action drive_feas_switch(FeasibilitySwitchRecovery &fsr, SolverContext &ctx,
                                        int &resolved_depth_out) {
    FeasSwitchUnusedMechanism mechanism;
    FeasSwitchUnusedAcceptance acceptance;
    IterateInfo citer;
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
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    FeasSwitchStubRestoration restoration; // active_=false, permit_=true
    SolverContext ctx = feas_switch_context(solver, settings, zero, scratch, &restoration);

    FeasibilitySwitchRecovery fsr(std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kRetry));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kRetry);
}

TEST(FeasibilitySwitchTruthTable, InnerGiveUpPassesThrough) {
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    FeasSwitchStubRestoration restoration;
    SolverContext ctx = feas_switch_context(solver, settings, zero, scratch, &restoration);

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kGiveUp));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kGiveUp);
}

TEST(FeasibilitySwitchTruthTable, NullRestorationKeepsAcceptAsIs) {
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    SolverContext ctx = feas_switch_context(solver, settings, zero, scratch, /*restoration=*/nullptr);

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kAcceptAsIs);
    EXPECT_EQ(depth, kRecoveryDepthUnresolved);
}

TEST(FeasibilitySwitchTruthTable, AlreadyActiveNeverReEnters) {
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    FeasSwitchStubRestoration restoration;
    restoration.active_ = true; // already in restoration mode
    restoration.permit_ = true;
    SolverContext ctx = feas_switch_context(solver, settings, zero, scratch, &restoration);

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kAcceptAsIs);
}

TEST(FeasibilitySwitchTruthTable, EntryRefusedKeepsAcceptAsIs) {
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    FeasSwitchStubRestoration restoration;
    restoration.active_ = false;
    restoration.permit_ = false; // guard/budget refuses entry
    SolverContext ctx = feas_switch_context(solver, settings, zero, scratch, &restoration);

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kAcceptAsIs);
}

TEST(FeasibilitySwitchTruthTable, EntryPermittedSwitchesAndStampsDepth) {
    KktSolverType solver;
    PSIOPT::Settings settings;
    int zero = 0;
    Eigen::VectorXd scratch;
    FeasSwitchStubRestoration restoration;
    restoration.active_ = false;
    restoration.permit_ = true;
    SolverContext ctx = feas_switch_context(solver, settings, zero, scratch, &restoration);

    FeasibilitySwitchRecovery fsr(
        std::make_unique<FeasSwitchStubInner>(RecoveryChain::Action::kAcceptAsIs));
    int depth = 0;
    EXPECT_EQ(drive_feas_switch(fsr, ctx, depth), RecoveryChain::Action::kSwitchToFeasibility);
    EXPECT_EQ(depth, kRecoveryDepthRestoration);
}

TEST(FeasibilitySwitchTruthTable, NullInnerChainRejected) {
    EXPECT_THROW(FeasibilitySwitchRecovery(nullptr), std::invalid_argument);
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

TEST_F(SolverTest, RestorationOffLeavesDiagnosticsSentinel) {
    auto prob = feas_switch_build_nlp(/*start=*/0.0, /*a=*/1.0, /*inconsistent=*/false, /*b=*/0.0);
    // restoration_mode_ defaults to off.
    prob->optimize();
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, -1);
    EXPECT_EQ(r.last_feas_rest_iters_, -1);
}

TEST_F(SolverTest, RestorationOnReportsDiagnostics) {
    auto prob = feas_switch_build_nlp(0.0, 1.0, false, 0.0);
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::proximal_switch;
    prob->optimize();
    const auto &r = prob->optimizer_->result();
    // With restoration constructed, the diagnostics are reported (not the -1
    // sentinel) — entries/iters are >= 0 regardless of whether entry fired.
    EXPECT_GE(r.last_feas_rest_entries_, 0);
    EXPECT_GE(r.last_feas_rest_iters_, 0);
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
    EXPECT_LE(flag, PSIOPT::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GE(r.last_feas_rest_entries_, 1);           // restoration was entered
    EXPECT_LE(r.last_feas_rest_entries_, 2);           // and not beyond the budget
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
    EXPECT_NE(flag, PSIOPT::ConvergenceFlags::CONVERGED);
    EXPECT_GE(r.last_feas_rest_entries_, 1); // restoration was entered
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
    EXPECT_NE(flag, PSIOPT::ConvergenceFlags::CONVERGED);
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
