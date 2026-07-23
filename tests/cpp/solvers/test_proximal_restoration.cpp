///////////////////////////////////////////////////////////////////////////////
// Unit tests for ProximalSwitchRestoration — the proximal feasibility
// mode-switch (first of the feasibility-restoration trio).
//
// No solver wiring exists for this component yet (see restoration.h /
// proximal_restoration.h file docstrings), so every test here drives the
// class directly: enter_restoration() / exit_restoration() / is_active() /
// reset(), the evaluation surface (proximal_objective /
// add_proximal_gradient / proximal_diagonal), entry_permitted(), and
// append_diagnostics().
//
// Every scenario's arithmetic is hand-computed in the comments from the
// rules documented in proximal_restoration.h:
//   zeta = kRestoProximityWeight * sqrt(mu), kRestoProximityWeight = 1.0
//   d_i  = min(1, 1/|x_R_i|)
//   diagonal_i = zeta * d_i^2
//   P(x) = 0.5 * sum_i diagonal_i * (x_i - x_R_i)^2
//   dP/dx_i = diagonal_i * (x_i - x_R_i)
//   entry_permitted refuses iff violation <= kNearFeasibleGuardFactor * econ_tol_
//                    OR entries so far >= max_feas_rest_
//
// UNITY RULE: the unity build defeats anonymous namespaces for ODR, so every
// file-local helper here is prefixed ProxResto* to stay globally unique
// across tests/cpp/ (grep-confirmed no other "ProxResto" symbol exists).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/proximal_restoration.h"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using tycho::solvers::kNearFeasibleGuardFactor;
using tycho::solvers::kRestoProximityWeight;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::ProximalSwitchRestoration;
using tycho::solvers::PSIOPT;
using tycho::solvers::SolverContext;

// Builds a minimal all-zero-dimension SolverContext with the given settings --
// entry_permitted only reads ctx.settings_.econ_tol_/max_feas_rest_, so the
// rest of the aggregate is inert (same pattern as
// test_monitored_governor.cpp's MonGovDelegation fixture).
SolverContext ProxRestoContext(PSIOPT::Settings &settings, tycho::solvers::KktSolverType &solver,
                               Eigen::VectorXd &scratch, int &zero) {
    return SolverContext{nullptr, solver,  settings, zero,    zero,    zero,
                         zero,    zero,    scratch,  scratch, scratch, scratch};
}

// -----------------------------------------------------------------------------
// zeta: frozen at entry mu, not re-derived from a later "live" mu.
// -----------------------------------------------------------------------------

TEST(ProxRestoZeta, SetOnceFromEntryMu) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 2.0;
    const ProgressMeasures ref{0.5, 1.0, 0.0};

    r.enter_restoration(ref, x0, /*mu=*/0.01);
    EXPECT_DOUBLE_EQ(r.zeta(), kRestoProximityWeight * std::sqrt(0.01)); // 0.1
}

TEST(ProxRestoZeta, FrozenAcrossIterationsNotRecomputedFromLiveMu) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 3.0;
    const ProgressMeasures ref{0.5, 1.0, 0.0};
    r.enter_restoration(ref, x0, /*mu=*/0.01);
    ASSERT_DOUBLE_EQ(r.zeta(), 0.1);

    // note_iteration() takes no mu argument at all -- there is no channel for
    // a later, live mu to reach zeta_ through it. Repeated calls (simulating
    // iterations elapsing while restoration stays active) must leave zeta_
    // untouched.
    r.note_iteration();
    r.note_iteration();
    r.note_iteration();
    EXPECT_DOUBLE_EQ(r.zeta(), 0.1);
}

TEST(ProxRestoZeta, FreshEntryRecomputesIndependentlyNotBlended) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 3.0;
    const ProgressMeasures ref{0.5, 1.0, 0.0};

    r.enter_restoration(ref, x0, /*mu=*/0.01);
    ASSERT_DOUBLE_EQ(r.zeta(), 0.1);

    // A second, later episode with a different mu overwrites zeta_ with an
    // INDEPENDENTLY-derived value (sqrt(0.04) = 0.2) -- not, say, an average
    // of 0.1 and 0.2 -- confirming "set once at switch time" per episode.
    r.enter_restoration(ref, x0, /*mu=*/0.04);
    EXPECT_DOUBLE_EQ(r.zeta(), 0.2);
}

// -----------------------------------------------------------------------------
// d_i per-coordinate scaling branches.
// -----------------------------------------------------------------------------

TEST(ProxRestoScaling, BelowOneMagnitudeGetsUnitScaling) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 0.5; // |x_R| = 0.5 < 1 -> d = min(1, 1/0.5=2) = 1.
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, /*mu=*/1.0);
    ASSERT_EQ(r.scaling().size(), 1);
    EXPECT_DOUBLE_EQ(r.scaling()[0], 1.0);
}

TEST(ProxRestoScaling, AboveOneMagnitudeGetsInverseScaling) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 4.0; // |x_R| = 4 > 1 -> d = min(1, 1/4=0.25) = 0.25.
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, /*mu=*/1.0);
    ASSERT_EQ(r.scaling().size(), 1);
    EXPECT_DOUBLE_EQ(r.scaling()[0], 0.25);
}

TEST(ProxRestoScaling, NegativeAboveOneMagnitudeUsesAbsoluteValue) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << -8.0; // |x_R| = 8 > 1 -> d = min(1, 1/8=0.125) = 0.125.
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, /*mu=*/1.0);
    EXPECT_DOUBLE_EQ(r.scaling()[0], 0.125);
}

TEST(ProxRestoScaling, ZeroCenterLandsInUnitScalingBranchNoDivideByZeroTrap) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 0.0; // 1/0 = +inf, min(1, +inf) = 1 -- falls in the "< 1" branch.
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, /*mu=*/1.0);
    EXPECT_DOUBLE_EQ(r.scaling()[0], 1.0);
}

TEST(ProxRestoScaling, UnitMagnitudeBoundaryGetsUnitScaling) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 1.0; // |x_R| == 1 -> 1/1 = 1 -> min(1,1) = 1 (either branch agrees).
    const ProgressMeasures ref{0.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, /*mu=*/1.0);
    EXPECT_DOUBLE_EQ(r.scaling()[0], 1.0);
}

// -----------------------------------------------------------------------------
// Hand-computed 3-vector value/gradient/diagonal.
//
// x_R = [0.5, 2.0, 4.0] -> d = [1, 0.5, 0.25] (all exact powers of two, so the
// arithmetic below is exact in double precision -- no rounding slop needed).
// mu = 1.0 -> zeta = 1.0. diagonal_i = zeta*d_i^2 = [1, 0.25, 0.0625].
// x = [1.5, 3.0, 5.0] -> delta = x - x_R = [1, 1, 1].
// P(x) = 0.5 * sum(diagonal_i * 1^2) = 0.5*(1 + 0.25 + 0.0625) = 0.65625.
// grad_i = diagonal_i * 1 = diagonal_i = [1, 0.25, 0.0625].
// -----------------------------------------------------------------------------

class ProxRestoThreeVectorFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        Eigen::VectorXd x_r(3);
        x_r << 0.5, 2.0, 4.0;
        const ProgressMeasures ref{0.0, 0.0, 0.0};
        r.enter_restoration(ref, x_r, /*mu=*/1.0);
    }

    ProximalSwitchRestoration r;
};

TEST_F(ProxRestoThreeVectorFixture, DiagonalHandComputed) {
    ASSERT_EQ(r.proximal_diagonal().size(), 3);
    EXPECT_DOUBLE_EQ(r.proximal_diagonal()[0], 1.0);
    EXPECT_DOUBLE_EQ(r.proximal_diagonal()[1], 0.25);
    EXPECT_DOUBLE_EQ(r.proximal_diagonal()[2], 0.0625);
}

TEST_F(ProxRestoThreeVectorFixture, ObjectiveHandComputed) {
    Eigen::VectorXd x(3);
    x << 1.5, 3.0, 5.0;
    EXPECT_DOUBLE_EQ(r.proximal_objective(x), 0.65625);
}

TEST_F(ProxRestoThreeVectorFixture, GradientAccumulatesIntoExistingGradOut) {
    Eigen::VectorXd x(3);
    x << 1.5, 3.0, 5.0;

    // Starts from a nonzero gradient to prove add_proximal_gradient ACCUMULATES
    // (+=) rather than overwriting -- the caller's existing objective gradient
    // must survive.
    Eigen::VectorXd grad(3);
    grad << 10.0, 20.0, 30.0;
    r.add_proximal_gradient(x, grad);
    EXPECT_DOUBLE_EQ(grad[0], 10.0 + 1.0);
    EXPECT_DOUBLE_EQ(grad[1], 20.0 + 0.25);
    EXPECT_DOUBLE_EQ(grad[2], 30.0 + 0.0625);
}

TEST_F(ProxRestoThreeVectorFixture, ObjectiveIsZeroAtTheSnapshotItself) {
    Eigen::VectorXd x(3);
    x << 0.5, 2.0, 4.0; // x == x_R -> delta = 0.
    EXPECT_DOUBLE_EQ(r.proximal_objective(x), 0.0);
}

// -----------------------------------------------------------------------------
// entry_permitted truth table.
// -----------------------------------------------------------------------------

TEST(ProxRestoEntryPermitted, NearFeasibleGuardBoundary) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1e-6;
    settings.max_feas_rest_ = 2; // budget open throughout this test.
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    const SolverContext ctx = ProxRestoContext(settings, solver, scratch, zero);

    ProximalSwitchRestoration r;
    const double threshold = kNearFeasibleGuardFactor * settings.econ_tol_; // 1e-7

    // Exactly at the boundary: refused (guard is "<=").
    EXPECT_FALSE(r.entry_permitted(threshold, ctx));
    // Just below the boundary: still near-feasible -> refused.
    EXPECT_FALSE(r.entry_permitted(threshold * 0.5, ctx));
    // Strictly above the boundary: guard does not fire -> permitted (budget open).
    EXPECT_TRUE(r.entry_permitted(threshold * 2.0, ctx));
    // Comfortably infeasible: permitted.
    EXPECT_TRUE(r.entry_permitted(1.0, ctx));
}

TEST(ProxRestoEntryPermitted, BudgetExhaustionAfterMaxEntries) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1e-6;
    settings.max_feas_rest_ = 2;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    const SolverContext ctx = ProxRestoContext(settings, solver, scratch, zero);

    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 1.0;
    const ProgressMeasures ref{1.0, 0.0, 0.0};

    // A comfortably-infeasible violation, well clear of the near-feasible guard.
    const double violation = 1.0;
    EXPECT_TRUE(r.entry_permitted(violation, ctx)); // 0 entries so far < 2.
    r.enter_restoration(ref, x0, 0.01);
    EXPECT_EQ(r.entries(), 1);
    EXPECT_TRUE(r.entry_permitted(violation, ctx)); // 1 < 2, still open.
    r.enter_restoration(ref, x0, 0.01);
    EXPECT_EQ(r.entries(), 2);
    EXPECT_FALSE(r.entry_permitted(violation, ctx)); // 2 >= max_feas_rest_ -> refused.
}

TEST(ProxRestoEntryPermitted, ZeroBudgetAlwaysRefuses) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1e-6;
    settings.max_feas_rest_ = 0;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    const SolverContext ctx = ProxRestoContext(settings, solver, scratch, zero);

    ProximalSwitchRestoration r;
    // Even a large, clearly-infeasible violation is refused: 0 entries so far
    // >= max_feas_rest_ == 0 before any entry has ever happened.
    EXPECT_FALSE(r.entry_permitted(1e6, ctx));
    EXPECT_EQ(r.entries(), 0);
}

// -----------------------------------------------------------------------------
// is_active lifecycle.
// -----------------------------------------------------------------------------

TEST(ProxRestoLifecycle, EnterActivatesExitDeactivates) {
    ProximalSwitchRestoration r;
    EXPECT_FALSE(r.is_active());

    Eigen::VectorXd x0(1);
    x0 << 2.0;
    const ProgressMeasures ref{1.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, 0.01);
    EXPECT_TRUE(r.is_active());

    r.exit_restoration();
    EXPECT_FALSE(r.is_active());
}

TEST(ProxRestoLifecycle, ReenteringAfterExitIsActiveAgain) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 2.0;
    const ProgressMeasures ref{1.0, 0.0, 0.0};
    r.enter_restoration(ref, x0, 0.01);
    r.exit_restoration();
    ASSERT_FALSE(r.is_active());

    r.enter_restoration(ref, x0, 0.02);
    EXPECT_TRUE(r.is_active());
    EXPECT_EQ(r.entries(), 2); // both episodes counted -- exit_restoration()
                               // alone does not clear the entry counter.
}

// -----------------------------------------------------------------------------
// reset() clears everything.
// -----------------------------------------------------------------------------

TEST(ProxRestoReset, ClearsAllState) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(2);
    x0 << 1.0, 2.0;
    const ProgressMeasures ref{0.7, 0.3, 0.1};
    r.enter_restoration(ref, x0, 0.01);
    r.note_iteration();
    r.note_iteration();
    ASSERT_TRUE(r.is_active());
    ASSERT_EQ(r.entries(), 1);
    ASSERT_EQ(r.iterations_in_mode(), 2);
    ASSERT_GT(r.scaling().size(), 0);
    ASSERT_GT(r.snapshot().size(), 0);
    ASSERT_NE(r.zeta(), 0.0);

    r.reset();
    EXPECT_FALSE(r.is_active());
    EXPECT_EQ(r.entries(), 0);
    EXPECT_EQ(r.iterations_in_mode(), 0);
    EXPECT_EQ(r.scaling().size(), 0);
    EXPECT_EQ(r.snapshot().size(), 0);
    EXPECT_EQ(r.proximal_diagonal().size(), 0);
    EXPECT_DOUBLE_EQ(r.zeta(), 0.0);
    EXPECT_DOUBLE_EQ(r.reference().infeasibility, 0.0);
    EXPECT_DOUBLE_EQ(r.reference().objective, 0.0);
    EXPECT_DOUBLE_EQ(r.reference().auxiliary, 0.0);
}

// -----------------------------------------------------------------------------
// reference() accessor.
// -----------------------------------------------------------------------------

TEST(ProxRestoReference, ReturnsTheEntryPointPassedIn) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 1.0;
    const ProgressMeasures ref{0.42, 1.23, 9.9};
    r.enter_restoration(ref, x0, 0.01);
    EXPECT_DOUBLE_EQ(r.reference().infeasibility, 0.42);
    EXPECT_DOUBLE_EQ(r.reference().objective, 1.23);
    EXPECT_DOUBLE_EQ(r.reference().auxiliary, 9.9);
}

// -----------------------------------------------------------------------------
// append_diagnostics.
// -----------------------------------------------------------------------------

TEST(ProxRestoDiagnostics, NeverEnteredReportsZeroZero) {
    ProximalSwitchRestoration r;
    PSIOPT::SolveResult result;
    r.append_diagnostics(result);
    // Constructed but never entered: 0/0 is the correct report (only the
    // interface's default no-op / "no strategy at all" path uses the -1
    // sentinel -- see restoration.h).
    EXPECT_EQ(result.last_feas_rest_entries_, 0);
    EXPECT_EQ(result.last_feas_rest_iters_, 0);
}

TEST(ProxRestoDiagnostics, ReportsEntriesAndIterationsInMode) {
    ProximalSwitchRestoration r;
    Eigen::VectorXd x0(1);
    x0 << 1.0;
    const ProgressMeasures ref{1.0, 0.0, 0.0};

    r.enter_restoration(ref, x0, 0.01);
    r.note_iteration();
    r.note_iteration();
    r.note_iteration();
    r.exit_restoration();
    r.enter_restoration(ref, x0, 0.02);
    r.note_iteration();

    PSIOPT::SolveResult result;
    r.append_diagnostics(result);
    EXPECT_EQ(result.last_feas_rest_entries_, 2);
    EXPECT_EQ(result.last_feas_rest_iters_, 4);
}

} // namespace
