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
// a non-triggering rejection both decline without touching the solve pieces).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/globalization_mechanism.h"
#include "tycho/detail/solvers/globalization/recovery_chain.h"
#include "tycho/detail/solvers/globalization/soc.h"
#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/iterate_info.h"

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::GlobalizationMechanism;
using tycho::solvers::IterateInfo;
using tycho::solvers::KktSolverType;
using tycho::solvers::kSocRecommendedMaxCorrections;
using tycho::solvers::kSocViolationDecrease;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::RecoveryChain;
using tycho::solvers::soc_run_loop;
using tycho::solvers::SocCorrectionOutcome;
using tycho::solvers::SocRecovery;
using tycho::solvers::soc_should_continue;
using tycho::solvers::soc_should_trigger;
using tycho::solvers::SolverContext;

using Action = RecoveryChain::Action;

// Builds an IterateInfo carrying just the two trigger signals SOC reads.
IterateInfo make_iter(int first_rejection_iter, double theta_at_first_rejection) {
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
    EXPECT_TRUE(soc_should_trigger(make_iter(/*first=*/0, /*theta=*/2.0), current));
    EXPECT_TRUE(soc_should_trigger(make_iter(/*first=*/0, /*theta=*/1.0), current)); // equal fires

    // First trial rejected but its violation improved (theta < current): the
    // ordinary backtrack is already making progress, so no correction.
    EXPECT_FALSE(soc_should_trigger(make_iter(/*first=*/0, /*theta=*/0.5), current));

    // Theta unavailable (< 0: LANG variant, or no rejection recorded): the
    // reduction test cannot be made, so conservatively do not fire.
    EXPECT_FALSE(soc_should_trigger(make_iter(/*first=*/0, /*theta=*/-1.0), current));

    // The rejection was not the FIRST trial (a later backtrack): no correction.
    EXPECT_FALSE(soc_should_trigger(make_iter(/*first=*/1, /*theta=*/2.0), current));

    // Step was accepted with no rejection (first_rejection_iter_ stays -1): the
    // recovery hook never even reaches SOC, and the predicate agrees.
    EXPECT_FALSE(soc_should_trigger(make_iter(/*first=*/-1, /*theta=*/-1.0), current));
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
// paths return before ever calling these, so their bodies must never run.
class UnusedAcceptance : public AcceptanceStrategy {
  public:
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double) override {
        ADD_FAILURE() << "acceptance must not be reached on an early-exit path";
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
    double compute_step(PSIOPT::LineSearchModes, double, double, double, double, Eigen::VectorXd &,
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
Action drive_soc(PSIOPT::Settings &settings, IterateInfo &citer, int &soc_steps) {
    KktSolverType solver;
    UnusedAcceptance acceptance;
    SocUnusedMechanism mechanism;
    int zero = 0;
    Eigen::VectorXd scratch; // empty: dims are all zero
    SolverContext ctx{nullptr, solver,  settings, zero,    zero,    zero,
                      zero,    zero,    scratch,  scratch, scratch, scratch};
    const std::vector<IterateInfo> iters;
    Eigen::VectorXd XSL, DXSL, XSL2, RHS, RHS2; // empty (ncons == 0)
    double alpha = 1.0, alphap = 1.0, alphad = 1.0;
    return SocRecovery{}.on_step_rejected(
        citer, iters, ctx, acceptance, mechanism, PSIOPT::LineSearchModes::AUGLANG, 1.0, 1e-3, 0.0,
        0.0, XSL, DXSL, XSL2, RHS, RHS2, alpha, alphap, alphad, soc_steps);
}

// max_soc_ == 0 (off): decline immediately, no corrections.
TEST(SocRecoveryGuards, DisabledDeclines) {
    PSIOPT::Settings settings;
    settings.max_soc_ = 0;
    IterateInfo citer = make_iter(/*first=*/0, /*theta=*/2.0); // would otherwise trigger
    int soc_steps = 0;
    EXPECT_EQ(drive_soc(settings, citer, soc_steps), Action::kAcceptAsIs);
    EXPECT_EQ(soc_steps, 0);
}

// Enabled but the rejection does not satisfy the trigger (not the first trial):
// decline before touching the solve machinery.
TEST(SocRecoveryGuards, NonTriggeringRejectionDeclines) {
    PSIOPT::Settings settings;
    settings.max_soc_ = kSocRecommendedMaxCorrections;
    IterateInfo citer = make_iter(/*first=*/1, /*theta=*/2.0); // not the first trial
    int soc_steps = 0;
    EXPECT_EQ(drive_soc(settings, citer, soc_steps), Action::kAcceptAsIs);
    EXPECT_EQ(soc_steps, 0);
}

} // namespace
