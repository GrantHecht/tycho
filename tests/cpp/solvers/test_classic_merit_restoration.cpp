///////////////////////////////////////////////////////////////////////////////
// Unit tests for ClassicMeritAcceptance::is_infeasibility_sufficiently_reduced
// — the restoration-exit test for the classic merit line search.
//
// Classic merit has no Uno counterpart (Uno pairs restoration with its
// filter/funnel strategies, not a monolithic merit line search), so the shape
// is Ipopt IpRestoConvCheck's relative-reduction floor:
//   θ_trial ≤ max(kKappaResto·θ_ref, econ_tol_)
// with kKappaResto = 0.9 and econ_tol_ standing in (single-tolerance
// adaptation) for Ipopt's Min(tol, constr_viol_tol). ClassicMeritAcceptance
// reads econ_tol_ through its SolverContext, so these tests build a minimal
// all-zero-dimension context (only settings_.econ_tol_ is read by this method)
// via TychoTest::InertSolverContext.
//
// Every boundary is hand-computed in the comments.
///////////////////////////////////////////////////////////////////////////////

#include "progress_measures_test_utils.h"
#include "solver_test_utils.h"

#include "tycho/detail/solvers/globalization/merit_acceptance.h"

#include <gtest/gtest.h>

namespace {

using tycho::solvers::ClassicMeritAcceptance;
using tycho::solvers::kKappaResto;
using tycho::solvers::ProgressMeasures;
using TychoTest::InertSolverContext;
using TychoTest::pm;

// The relative floor dominates: econ_tol_ = 1e-6, θ_ref = 1.0 ⇒
// floor = max(0.9·1, 1e-6) = 0.9.
//   • θ_trial = 0.9 (boundary): 0.9 ≤ 0.9 ⇒ EXIT (≤ is inclusive).
//   • θ_trial = 0.9·1.0001 (just above): > 0.9 ⇒ NO.
//   • θ_trial = 0.5 (below): ≤ 0.9 ⇒ EXIT.
TEST(ClassicMeritRestoration, RelativeFloorBoundary) {
    InertSolverContext inert;
    inert.settings_.econ_tol_ = 1.0e-6;
    ClassicMeritAcceptance a(inert.ctx());

    const ProgressMeasures ref = pm(1.0);
    const double floor = kKappaResto * 1.0; // 0.9 (dominates 1e-6)

    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(floor)));
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(0.5)));
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(ref, pm(floor * 1.0001)));
}

// The econ-tolerance floor waives the relative test when θ_ref is tiny:
// econ_tol_ = 1e-6, θ_ref = 1e-9 ⇒ 0.9·1e-9 = 9e-10 < 1e-6 ⇒ floor = 1e-6.
//   • θ_trial = 5e-7 ≤ 1e-6 ⇒ EXIT, even though 5e-7 > 9e-10 (the relative test
//     alone would reject).
//   • θ_trial = 2e-6 > 1e-6 ⇒ NO (above the tolerance floor).
TEST(ClassicMeritRestoration, ToleranceFloorWaivesRelativeWhenRefTiny) {
    InertSolverContext inert;
    inert.settings_.econ_tol_ = 1.0e-6;
    ClassicMeritAcceptance a(inert.ctx());

    const ProgressMeasures ref = pm(1.0e-9);
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(5.0e-7)));
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(ref, pm(2.0e-6)));
}

// The floor tracks the injected econ_tol_ (not a hardcoded constant): with a
// larger tolerance the same tiny-θ_ref trial that failed above now passes.
// econ_tol_ = 1e-5, θ_ref = 1e-9 ⇒ floor = max(9e-10, 1e-5) = 1e-5; θ_trial =
// 2e-6 ≤ 1e-5 ⇒ EXIT.
TEST(ClassicMeritRestoration, FloorTracksSettingsEconTol) {
    InertSolverContext inert;
    inert.settings_.econ_tol_ = 1.0e-5;
    ClassicMeritAcceptance a(inert.ctx());

    const ProgressMeasures ref = pm(1.0e-9);
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, pm(2.0e-6)));
}

} // namespace
