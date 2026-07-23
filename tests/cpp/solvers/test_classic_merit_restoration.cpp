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
// all-zero-dimension context (only settings_.econ_tol_ is read by this method),
// the same pattern test_proximal_restoration.cpp uses for entry_permitted.
//
// Every boundary is hand-computed in the comments.
//
// UNITY RULE: the unity build defeats anonymous namespaces for ODR, so every
// file-local helper here is prefixed ClassicResto* to stay globally unique
// across tests/cpp/.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/globalization/merit_acceptance.h"
#include "tycho/detail/solvers/globalization/solver_context.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

namespace {

using tycho::solvers::ClassicMeritAcceptance;
using tycho::solvers::kKappaResto;
using tycho::solvers::ProgressMeasures;
using tycho::solvers::PSIOPT;
using tycho::solvers::SolverContext;

// Builds a minimal all-zero-dimension SolverContext (only settings_.econ_tol_ is
// read by is_infeasibility_sufficiently_reduced), mirroring
// test_proximal_restoration.cpp's ProxRestoContext.
SolverContext ClassicRestoContext(PSIOPT::Settings &settings, tycho::solvers::KktSolverType &solver,
                                  Eigen::VectorXd &scratch, int &zero) {
    return SolverContext{nullptr, solver,  settings, zero,    zero,    zero,
                         zero,    zero,    scratch,  scratch, scratch, scratch};
}

ProgressMeasures ClassicRestoMakePm(double infeasibility) {
    ProgressMeasures pm;
    pm.infeasibility = infeasibility;
    return pm;
}

// The relative floor dominates: econ_tol_ = 1e-6, θ_ref = 1.0 ⇒
// floor = max(0.9·1, 1e-6) = 0.9.
//   • θ_trial = 0.9 (boundary): 0.9 ≤ 0.9 ⇒ EXIT (≤ is inclusive).
//   • θ_trial = 0.9·1.0001 (just above): > 0.9 ⇒ NO.
//   • θ_trial = 0.5 (below): ≤ 0.9 ⇒ EXIT.
TEST(ClassicMeritRestoration, RelativeFloorBoundary) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1.0e-6;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    ClassicMeritAcceptance a(ClassicRestoContext(settings, solver, scratch, zero));

    const ProgressMeasures ref = ClassicRestoMakePm(1.0);
    const double floor = kKappaResto * 1.0; // 0.9 (dominates 1e-6)

    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, ClassicRestoMakePm(floor)));
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, ClassicRestoMakePm(0.5)));
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(ref, ClassicRestoMakePm(floor * 1.0001)));
}

// The econ-tolerance floor waives the relative test when θ_ref is tiny:
// econ_tol_ = 1e-6, θ_ref = 1e-9 ⇒ 0.9·1e-9 = 9e-10 < 1e-6 ⇒ floor = 1e-6.
//   • θ_trial = 5e-7 ≤ 1e-6 ⇒ EXIT, even though 5e-7 > 9e-10 (the relative test
//     alone would reject).
//   • θ_trial = 2e-6 > 1e-6 ⇒ NO (above the tolerance floor).
TEST(ClassicMeritRestoration, ToleranceFloorWaivesRelativeWhenRefTiny) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1.0e-6;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    ClassicMeritAcceptance a(ClassicRestoContext(settings, solver, scratch, zero));

    const ProgressMeasures ref = ClassicRestoMakePm(1.0e-9);
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, ClassicRestoMakePm(5.0e-7)));
    EXPECT_FALSE(a.is_infeasibility_sufficiently_reduced(ref, ClassicRestoMakePm(2.0e-6)));
}

// The floor tracks the injected econ_tol_ (not a hardcoded constant): with a
// larger tolerance the same tiny-θ_ref trial that failed above now passes.
// econ_tol_ = 1e-5, θ_ref = 1e-9 ⇒ floor = max(9e-10, 1e-5) = 1e-5; θ_trial =
// 2e-6 ≤ 1e-5 ⇒ EXIT.
TEST(ClassicMeritRestoration, FloorTracksSettingsEconTol) {
    PSIOPT::Settings settings;
    settings.econ_tol_ = 1.0e-5;
    tycho::solvers::KktSolverType solver;
    Eigen::VectorXd scratch;
    int zero = 0;
    ClassicMeritAcceptance a(ClassicRestoContext(settings, solver, scratch, zero));

    const ProgressMeasures ref = ClassicRestoMakePm(1.0e-9);
    EXPECT_TRUE(a.is_infeasibility_sufficiently_reduced(ref, ClassicRestoMakePm(2.0e-6)));
}

} // namespace
