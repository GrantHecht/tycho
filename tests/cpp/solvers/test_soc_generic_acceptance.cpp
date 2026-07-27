///////////////////////////////////////////////////////////////////////////////
// Unit + through-API tests for the SOC and extended-backtracking recovery links
// under the GENERIC-path acceptance strategies (modern merit / filter / funnel).
//
// Both links re-drive the acceptance backtrack through
// GlobalizationMechanism::run_acceptance_backtrack, which dispatches to the
// fused classic merit test (classic path) or to
// AcceptanceStrategy::is_iterate_acceptable via generic_line_search (generic
// path). That routing is what lets a corrected (SOC) or further-scaled
// (extended) step be tested against the SAME acceptance criteria the ordinary
// step faced, on every acceptance strategy rather than only classic merit.
//
// The tests here cover:
//   1. Routing — run_acceptance_backtrack drives is_iterate_acceptable on a
//      generic strategy (accept and reject) and forwards to classic_line_search
//      on a classic strategy. Deterministic; no KKT factorization needed (the
//      generic driving path only evaluates trial points via nlp_->eval_occ).
//   2. Construction — merit+SOC and filter+SOC build a ChainedRecovery over a
//      SocRecovery plus the matching generic acceptance strategy.
//   3. Through-API composition — filter+SOC (monitored) and merit+SOC solves
//      converge (the former combination previously validate()-rejected), and an
//      l1-nested restoration solve with SOC enabled exercises the SOC link
//      inside the in-phase recovery chain end-to-end.
//
// The classic-path SOC policy truth-tables live in test_soc.cpp (untouched); the
// end-to-end evidence that a SOC correction FIRES and is accepted/rejected via a
// generic strategy on a Maratos-class problem is carried by the solver corpus
// scorecards (the second-order-correction configurations of the corpus sweep).
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/backtracking_line_search.h"
#include "tycho/detail/solvers/globalization/filter_acceptance.h"
#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/l1_restoration.h"
#include "tycho/detail/solvers/globalization/modern_merit.h"
#include "tycho/detail/solvers/globalization/monitored_governor.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/soc.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/globalization/watchdog.h"
#include "tycho/detail/solvers/iterate_info.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <Eigen/Core>

using tycho::solvers::AcceptanceStrategies;
using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::BacktrackingLineSearch;
using tycho::solvers::BarrierGovernors;
using tycho::solvers::ChainedRecovery;
using tycho::solvers::FilterAcceptance;
using tycho::solvers::IterateInfo;
using tycho::solvers::ModernMeritAcceptance;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::RecoveryChain;
using tycho::solvers::RestorationModes;
using tycho::solvers::SolverContext;

namespace {

// Generic-path spy: drives_classic_path() == false, so run_acceptance_backtrack
// must route to generic_line_search -> is_iterate_acceptable. classic_line_search
// is intentionally NOT overridden — the base throws, so any misrouting to the
// classic entry point surfaces as a thrown logic_error rather than a silent pass.
class GenericSpyAcceptance : public AcceptanceStrategy {
  public:
    explicit GenericSpyAcceptance(bool verdict) : verdict_(verdict) {}
    bool drives_classic_path() const override { return false; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        ++accept_calls_;
        return verdict_;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}

    int accept_calls_ = 0;

  private:
    bool verdict_;
};

// Classic-path spy: drives_classic_path() == true, so run_acceptance_backtrack
// must forward to classic_line_search (which stamps the verdict, as the real
// merit variants do). is_iterate_acceptable firing here would be a misroute.
class ClassicSpyAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return true; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        ADD_FAILURE() << "classic path must not reach is_iterate_acceptable";
        return false;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return false;
    }
    void reset() override {}

    double classic_line_search(PSIOPT::LineSearchModes, double, double, double, double,
                               Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &,
                               Eigen::VectorXd &, Eigen::VectorXd &, IterateInfo &Citer,
                               const std::vector<IterateInfo> &) override {
        ++classic_calls_;
        Citer.accepted_ = true;
        return 1.0;
    }

    int classic_calls_ = 0;
};

// Build a direct equality NLP: minimize x^2 subject to x - a = 0 (one variable,
// x0 = start). Converges to x = a under any acceptance strategy.
std::unique_ptr<OptimizationProblem> build_soc_nlp(double start, double a) {
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
        prob->add_equal_con(GenericFunction<-1, -1>(x - a), (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->set_print_level(3);
    return prob;
}

} // namespace

// Befriended harness (see the SocGenericHarness friend declaration in psiopt.h):
// a transcribed one-variable, one-inequality NLP (so there is one slack + barrier
// term) whose private nlp_/kkt_sol_/dims/scratch back a live SolverContext, and
// whose private acceptance_/recovery_ are reachable for the construction
// assertions. Lives at global scope (outside the anonymous namespace) because a
// friend declaration in a production header cannot name an anonymous-namespace
// class — the same convention the other seam harnesses in this suite follow.
class SocGenericHarness {
  public:
    SocGenericHarness() {
        using tycho::vf::Arguments;
        using tycho::vf::GenericFunction;
        prob_.set_vars((Eigen::VectorXd(1) << 0.5).finished());
        {
            auto args = Arguments<1>();
            auto x = args.coeff<0>();
            prob_.add_objective(GenericFunction<-1, 1>(x * x), (Eigen::VectorXi(1) << 0).finished());
        }
        {
            auto args = Arguments<1>();
            auto x = args.coeff<0>();
            // Inequality x - 2 (strictly satisfied at x = 0.5): one slack row.
            prob_.add_inequal_con(GenericFunction<-1, -1>(x - 2.0),
                                  (Eigen::VectorXi(1) << 0).finished());
        }
        prob_.optimizer_->set_print_level(3);
        prob_.transcribe();
        solver_ = prob_.optimizer_.get();
        solver_->rebuild_globalization_components();
    }

    tycho::solvers::PSIOPT &solver() { return *solver_; }
    void rebuild() { solver_->rebuild_globalization_components(); }
    tycho::solvers::RecoveryChain *recovery() { return solver_->recovery_.get(); }
    tycho::solvers::AcceptanceStrategy *acceptance() { return solver_->acceptance_.get(); }

    int pv() const { return solver_->primal_vars_; }
    int sv() const { return solver_->slack_vars_; }
    int dim() const { return solver_->kkt_dim_; }

    tycho::solvers::SolverContext make_ctx() {
        return tycho::solvers::SolverContext{
            solver_->nlp_.get(),        solver_->kkt_sol_,          solver_->settings_,
            solver_->primal_vars_,      solver_->slack_vars_,       solver_->equal_cons_,
            solver_->inequal_cons_,     solver_->kkt_dim_,          solver_->stli_scratch_,
            solver_->hp_scratch_,       solver_->best_xsl_scratch_, solver_->best_rhs_scratch_,
            solver_->restoration_.get()};
    }

  private:
    tycho::solvers::OptimizationProblem prob_;
    tycho::solvers::PSIOPT *solver_;
};

namespace {

// ---------------------------------------------------------------------------
// 1. Routing: the acceptance-backtrack seam dispatches per strategy.
// ---------------------------------------------------------------------------

TEST(SocGenericAcceptanceRouting, GenericStrategyReTestGoesThroughStrategySurfaceAccept) {
    SocGenericHarness h;
    BacktrackingLineSearch mechanism;
    SolverContext ctx = h.make_ctx();

    Eigen::VectorXd XSL = Eigen::VectorXd::Zero(h.dim());
    XSL[0] = 0.5;                // primal
    XSL[h.pv()] = 1.0;           // slack > 0
    XSL[h.pv() + h.sv()] = 0.1;  // inequality multiplier
    Eigen::VectorXd DXSL = Eigen::VectorXd::Constant(h.dim(), 0.001);
    Eigen::VectorXd XSL2 = Eigen::VectorXd::Zero(h.dim());
    Eigen::VectorXd RHS = Eigen::VectorXd::Constant(h.dim(), 0.1);
    Eigen::VectorXd RHS2 = Eigen::VectorXd::Zero(h.dim());

    GenericSpyAcceptance acceptance(/*verdict=*/true);
    IterateInfo citer;
    const std::vector<IterateInfo> iters;

    const double alpha = mechanism.run_acceptance_backtrack(
        PSIOPT::LineSearchModes::L1, 1.0, 1e-2, 0.0, 0.0, XSL, DXSL, XSL2, RHS, RHS2, acceptance,
        citer, iters, ctx);

    // The corrected/extended re-test verdict came from the generic strategy
    // surface (is_iterate_acceptable), never the classic path.
    EXPECT_GE(acceptance.accept_calls_, 1);
    EXPECT_TRUE(citer.accepted_);
    EXPECT_GT(alpha, 0.0);
}

TEST(SocGenericAcceptanceRouting, GenericStrategyReTestGoesThroughStrategySurfaceReject) {
    SocGenericHarness h;
    BacktrackingLineSearch mechanism;
    SolverContext ctx = h.make_ctx();

    Eigen::VectorXd XSL = Eigen::VectorXd::Zero(h.dim());
    XSL[0] = 0.5;
    XSL[h.pv()] = 1.0;
    XSL[h.pv() + h.sv()] = 0.1;
    Eigen::VectorXd DXSL = Eigen::VectorXd::Constant(h.dim(), 0.001);
    Eigen::VectorXd XSL2 = Eigen::VectorXd::Zero(h.dim());
    Eigen::VectorXd RHS = Eigen::VectorXd::Constant(h.dim(), 0.1);
    Eigen::VectorXd RHS2 = Eigen::VectorXd::Zero(h.dim());

    GenericSpyAcceptance acceptance(/*verdict=*/false);
    IterateInfo citer;
    const std::vector<IterateInfo> iters;

    mechanism.run_acceptance_backtrack(PSIOPT::LineSearchModes::L1, 1.0, 1e-2, 0.0, 0.0, XSL, DXSL,
                                       XSL2, RHS, RHS2, acceptance, citer, iters, ctx);

    // Every backtrack rung consulted the strategy surface and all rejected.
    EXPECT_GE(acceptance.accept_calls_, 1);
    EXPECT_FALSE(citer.accepted_);
}

TEST(SocGenericAcceptanceRouting, ClassicStrategyReTestForwardsToClassicLineSearch) {
    SocGenericHarness h;
    BacktrackingLineSearch mechanism;
    SolverContext ctx = h.make_ctx();

    Eigen::VectorXd v = Eigen::VectorXd::Zero(h.dim());
    Eigen::VectorXd XSL = v, DXSL = v, XSL2 = v, RHS = v, RHS2 = v;

    ClassicSpyAcceptance acceptance;
    IterateInfo citer;
    const std::vector<IterateInfo> iters;

    const double alpha = mechanism.run_acceptance_backtrack(
        PSIOPT::LineSearchModes::L1, 1.0, 1e-2, 0.0, 0.0, XSL, DXSL, XSL2, RHS, RHS2, acceptance,
        citer, iters, ctx);

    EXPECT_EQ(acceptance.classic_calls_, 1);
    EXPECT_TRUE(citer.accepted_);
    EXPECT_DOUBLE_EQ(alpha, 1.0);
}

// ---------------------------------------------------------------------------
// 2. Construction: SOC composes with the generic acceptance strategies.
// ---------------------------------------------------------------------------

TEST(SocGenericAcceptanceRouting, MeritPlusSocBuildsChainedRecoveryOverSocAndModernMerit) {
    SocGenericHarness h;
    h.solver().settings().acceptance_strategy_ = AcceptanceStrategies::merit;
    h.solver().settings().max_soc_ = 4;
    h.rebuild();

    EXPECT_NE(dynamic_cast<ModernMeritAcceptance *>(h.acceptance()), nullptr);
    EXPECT_FALSE(h.acceptance()->drives_classic_path());
    EXPECT_NE(dynamic_cast<ChainedRecovery *>(h.recovery()), nullptr);
}

TEST(SocGenericAcceptanceRouting, FilterPlusSocBuildsChainedRecoveryOverSocAndFilter) {
    SocGenericHarness h;
    h.solver().settings().acceptance_strategy_ = AcceptanceStrategies::filter;
    h.solver().settings().barrier_governor_ = BarrierGovernors::monitored;
    h.solver().settings().max_soc_ = 4;
    h.rebuild();

    EXPECT_NE(dynamic_cast<FilterAcceptance *>(h.acceptance()), nullptr);
    EXPECT_FALSE(h.acceptance()->drives_classic_path());
    EXPECT_NE(dynamic_cast<ChainedRecovery *>(h.recovery()), nullptr);
}

// ---------------------------------------------------------------------------
// 3. Through-API composition.
// ---------------------------------------------------------------------------

TEST(SocGenericAcceptanceIntegration, FilterMonitoredWithSocSolves) {
    auto prob = build_soc_nlp(/*start=*/0.0, /*a=*/1.0);
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::filter;
    prob->optimizer_->settings().barrier_governor_ = BarrierGovernors::monitored;
    prob->optimizer_->settings().max_soc_ = 4;
    prob->optimizer_->set_max_iters(80);
    auto flag = prob->optimize();
    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    const auto &r = prob->optimizer_->result();
    ASSERT_EQ(r.primals_.size(), 1);
    EXPECT_NEAR(r.primals_[0], 1.0, 1e-4);
}

TEST(SocGenericAcceptanceIntegration, MeritWithSocAndExtendedBacktrackSolves) {
    auto prob = build_soc_nlp(/*start=*/0.0, /*a=*/1.0);
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::merit;
    prob->optimizer_->settings().max_soc_ = 4;
    prob->optimizer_->settings().ls_extended_iters_ = 2;
    prob->optimizer_->set_max_iters(80);
    auto flag = prob->optimize();
    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    const auto &r = prob->optimizer_->result();
    ASSERT_EQ(r.primals_.size(), 1);
    EXPECT_NEAR(r.primals_[0], 1.0, 1e-4);
}

// In-phase composition: with l1-nested restoration on and SOC enabled, the SOC
// link sits inside the in-phase recovery chain (ChainedRecovery under
// FeasibilitySwitchRecovery). Forcing every step to be rejected
// (max_ls_iters == 0) drives the recovery chain — hence SocRecovery — on every
// iteration, including while the nested restoration phase is active, and the
// solve still reaches the true objective through restoration. This pins that
// SOC composes with the phase's trial machinery without breaking the in-phase
// path (the corrected re-test routes through the same generic acceptance surface
// the phase uses; see soc.h's interaction-rule (b)).
// On a feasible problem the correction rescues the rejected step before the
// recovery ladder can exhaust, so restoration is never entered at all — the
// correction pre-empting the (more expensive) feasibility phase is the desired
// composition on feasible problems, pinned here.
TEST(SocGenericAcceptanceIntegration, SocRescuePreemptsRestorationOnFeasibleProblem) {
    auto prob = build_soc_nlp(/*start=*/0.0, /*a=*/1.0);
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::merit;
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::l1_nested;
    prob->optimizer_->settings().max_soc_ = 4;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(80);
    auto flag = prob->optimize();
    EXPECT_LE(flag, tycho::ConvergenceFlags::ACCEPTABLE);
    const auto &r = prob->optimizer_->result();
    EXPECT_EQ(r.last_feas_rest_entries_, 0); // corrections rescued every rejection
    ASSERT_EQ(r.primals_.size(), 1);
    EXPECT_NEAR(r.primals_[0], 1.0, 1e-4);
}

// On a genuinely infeasible problem no correction can cure the rejection, the
// ladder (including the correction link) exhausts, and the restoration phase
// must still engage and run with the correction link active in-chain — the
// in-phase composition this suite exists to cover.
TEST(SocGenericAcceptanceIntegration, L1NestedRestorationWithSocComposesInPhase) {
    auto prob = build_soc_nlp(/*start=*/0.0, /*a=*/1.0);
    {
        using tycho::vf::Arguments;
        using tycho::vf::GenericFunction;
        auto args = Arguments<1>();
        auto x = args.coeff<0>();
        // Contradicts x - 1 = 0: jointly infeasible, violation bounded below by 1.
        prob->add_equal_con(GenericFunction<-1, -1>(x + 1.0),
                            (Eigen::VectorXi(1) << 0).finished());
    }
    prob->optimizer_->settings().acceptance_strategy_ = AcceptanceStrategies::merit;
    prob->optimizer_->settings().restoration_mode_ = RestorationModes::l1_nested;
    prob->optimizer_->settings().max_soc_ = 4;
    prob->optimizer_->set_max_ls_iters(0);
    prob->optimizer_->set_max_iters(80);
    auto flag = prob->optimize();
    const auto &r = prob->optimizer_->result();
    EXPECT_GE(r.last_feas_rest_entries_, 1); // restoration entered with SOC in the chain
    EXPECT_NE(flag, tycho::ConvergenceFlags::CONVERGED); // never falsely converges
}

} // namespace
