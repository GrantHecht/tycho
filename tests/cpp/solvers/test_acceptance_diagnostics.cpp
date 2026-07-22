///////////////////////////////////////////////////////////////////////////////
// Unit tests for AcceptanceStrategy::append_diagnostics() — the solver-level
// observability hook added alongside the funnel/filter acceptance strategies.
//
// Covers:
//   - the default (base-class) body is a no-op: a strategy that does not
//     override it leaves the SolveResult untouched;
//   - a fake strategy's override IS invoked through the virtual dispatch, the
//     same call shape run_phase_sequence() uses (see psiopt.cpp);
//   - the two real overrides (FunnelAcceptance, FilterAcceptance) report their
//     documented fields (funnel_acceptance.h / filter_acceptance.h);
//   - PSIOPT::SolveResult::reset_accumulators() restores the three sentinel
//     values (-1.0 / -1 / -1) documented on the fields in psiopt.h.
//
// UNITY RULE: anonymous namespace does not protect names against the unity
// build — every helper/class here is prefixed Diag* to stay globally unique
// across tests/cpp/ (grep-confirmed no other "Diag"-prefixed symbol exists).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/acceptance_strategy.h"
#include "tycho/detail/solvers/globalization/filter_acceptance.h"
#include "tycho/detail/solvers/globalization/funnel_acceptance.h"
#include "tycho/detail/solvers/globalization/progress_measures.h"
#include "tycho/detail/solvers/psiopt.h"

#include <gtest/gtest.h>

namespace {

using tycho::solvers::AcceptanceStrategy;
using tycho::solvers::FilterAcceptance;
using tycho::solvers::FunnelAcceptance;
using tycho::solvers::kFunnelInfeasibilityFactor;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;

// File-unique helper.
ProgressMeasures DiagMakePm(double infeasibility, double objective, double auxiliary) {
    ProgressMeasures pm;
    pm.infeasibility = infeasibility;
    pm.objective = objective;
    pm.auxiliary = auxiliary;
    return pm;
}

// Bare AcceptanceStrategy: implements only the pure-virtual surface and does
// NOT override append_diagnostics(), so it exercises the base class's default
// no-op body. drives_classic_path() is arbitrary here (never read by these
// tests); is_iterate_acceptable()/is_infeasibility_sufficiently_reduced()
// return fixed values since neither is exercised by these tests either.
class DiagBareAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return false; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        return true;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return true;
    }
    void reset() override {}
};

// Fake AcceptanceStrategy whose append_diagnostics() override is invoked as a
// unit: increments a call counter and stamps a recognizable value into
// last_funnel_width_ so the test can tell the OVERRIDE ran (as opposed to the
// base's no-op).
class DiagFakeAcceptance : public AcceptanceStrategy {
  public:
    bool drives_classic_path() const override { return false; }
    bool is_iterate_acceptable(const ProgressMeasures &, const ProgressMeasures &,
                               const ProgressMeasures &, double, double) override {
        return true;
    }
    bool is_infeasibility_sufficiently_reduced(const ProgressMeasures &,
                                               const ProgressMeasures &) const override {
        return true;
    }
    void reset() override {}

    void append_diagnostics(PSIOPT::SolveResult &result) const override {
        ++calls_;
        result.last_funnel_width_ = 42.0;
    }

    mutable int calls_ = 0;
};

// The default body is a no-op: a SolveResult freshly reset to its sentinel
// values is untouched by a strategy that doesn't override append_diagnostics.
TEST(AcceptanceDiagnostics, DefaultIsNoop) {
    DiagBareAcceptance strategy;
    PSIOPT::SolveResult result;
    result.reset_accumulators();
    ASSERT_DOUBLE_EQ(result.last_funnel_width_, -1.0);
    ASSERT_EQ(result.last_filter_size_, -1);
    ASSERT_EQ(result.last_filter_resets_, -1);

    // Call it through a base-class reference, exactly like run_phase_sequence()
    // does through the acceptance_ unique_ptr<AcceptanceStrategy>.
    AcceptanceStrategy &base = strategy;
    base.append_diagnostics(result);

    EXPECT_DOUBLE_EQ(result.last_funnel_width_, -1.0);
    EXPECT_EQ(result.last_filter_size_, -1);
    EXPECT_EQ(result.last_filter_resets_, -1);
}

// A strategy's override IS invoked through virtual dispatch off a base-class
// reference (the exact call shape run_phase_sequence() uses:
// this->acceptance_->append_diagnostics(this->result_)).
TEST(AcceptanceDiagnostics, FakeStrategyOverrideIsInvoked) {
    DiagFakeAcceptance strategy;
    PSIOPT::SolveResult result;
    result.reset_accumulators();

    AcceptanceStrategy &base = strategy;
    base.append_diagnostics(result);

    EXPECT_EQ(strategy.calls_, 1);
    EXPECT_DOUBLE_EQ(result.last_funnel_width_, 42.0);

    base.append_diagnostics(result);
    EXPECT_EQ(strategy.calls_, 2);
}

// FunnelAcceptance::append_diagnostics() reports the current width_ verbatim.
// θ₀ = 4.0 ⇒ κ̄·θ₀ = 1.5·4 = 6.0 > τ̄ = 1.0 ⇒ τ = 6.0 (init rule, see
// funnel_acceptance.h). Priming trial well outside the funnel so the base's
// membership test rejects it before any width update runs.
TEST(AcceptanceDiagnostics, FunnelReportsWidth) {
    FunnelAcceptance funnel;
    const bool primed =
        funnel.is_iterate_acceptable(DiagMakePm(4.0, 0.0, 0.0), DiagMakePm(100.0, 0.0, 0.0),
                                     DiagMakePm(0.0, 0.0, 0.0), 1.0, 1.0);
    ASSERT_FALSE(primed);
    ASSERT_DOUBLE_EQ(funnel.funnel_width(), kFunnelInfeasibilityFactor * 4.0);

    PSIOPT::SolveResult result;
    result.reset_accumulators();
    static_cast<AcceptanceStrategy &>(funnel).append_diagnostics(result);

    EXPECT_DOUBLE_EQ(result.last_funnel_width_, funnel.funnel_width());
    EXPECT_DOUBLE_EQ(result.last_funnel_width_, kFunnelInfeasibilityFactor * 4.0);
    // Untouched by FunnelAcceptance's override (funnel doesn't report filter
    // fields).
    EXPECT_EQ(result.last_filter_size_, -1);
    EXPECT_EQ(result.last_filter_resets_, -1);
}

// FunnelAcceptance::append_diagnostics() reports the -1.0 sentinel when the
// acceptance test never ran (e.g. phase converged at initial iterate). The
// width_ stays at its +∞ uninitialized sentinel if is_iterate_acceptable was
// never called; append_diagnostics converts +∞ to -1.0.
TEST(AcceptanceDiagnostics, DiagFunnelUninitializedWidthSentinel) {
    FunnelAcceptance funnel;
    // Do NOT call is_iterate_acceptable — width_ remains at +∞.
    ASSERT_FALSE(std::isfinite(funnel.funnel_width()));

    PSIOPT::SolveResult result;
    result.reset_accumulators();
    static_cast<AcceptanceStrategy &>(funnel).append_diagnostics(result);

    EXPECT_DOUBLE_EQ(result.last_funnel_width_, -1.0);
    // Untouched by FunnelAcceptance's override.
    EXPECT_EQ(result.last_filter_size_, -1);
    EXPECT_EQ(result.last_filter_resets_, -1);
}

// FilterAcceptance::append_diagnostics() reports filter_size() and
// filter_resets(). One accepted H-type step (m_f = 0 routes every call to the
// H-type delegate) augments the filter to size 1 and leaves the reset counter
// at 0 (nowhere near the kFilterResetTrigger streak).
TEST(AcceptanceDiagnostics, FilterReportsSizeAndResets) {
    FilterAcceptance filter;
    const bool accepted = filter.is_iterate_acceptable(DiagMakePm(/*theta=*/4.0, /*phi=*/20.0, 0.0),
                                                       DiagMakePm(/*theta=*/1.0, /*phi=*/20.0, 0.0),
                                                       DiagMakePm(0.0, 0.0, 0.0), 1.0, 1.0);
    ASSERT_TRUE(accepted);
    ASSERT_EQ(filter.filter_size(), 1u);
    ASSERT_EQ(filter.filter_resets(), 0);

    PSIOPT::SolveResult result;
    result.reset_accumulators();
    static_cast<AcceptanceStrategy &>(filter).append_diagnostics(result);

    EXPECT_EQ(result.last_filter_size_, 1);
    EXPECT_EQ(result.last_filter_resets_, 0);
    // Untouched by FilterAcceptance's override (filter doesn't report the
    // funnel field).
    EXPECT_DOUBLE_EQ(result.last_funnel_width_, -1.0);
}

// reset_accumulators() restores all three sentinels, regardless of what a
// prior append_diagnostics() call left behind.
TEST(AcceptanceDiagnostics, ResetAccumulatorsRestoresSentinels) {
    PSIOPT::SolveResult result;
    result.last_funnel_width_ = 6.0;
    result.last_filter_size_ = 3;
    result.last_filter_resets_ = 2;

    result.reset_accumulators();

    EXPECT_DOUBLE_EQ(result.last_funnel_width_, -1.0);
    EXPECT_EQ(result.last_filter_size_, -1);
    EXPECT_EQ(result.last_filter_resets_, -1);
}

} // namespace
