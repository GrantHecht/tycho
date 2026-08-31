///////////////////////////////////////////////////////////////////////////////
// Persistence-based divergence classification (InteriorPointSolver converge_check).
//
// converge_check() no longer aborts the solve the first time a monitored
// residual crosses its divergence threshold. The per-iterate divergent
// predicate is unchanged (any of the four residuals non-finite or past its
// threshold), but the DIVERGING verdict now splits by kind:
//
//   * A non-finite residual (NaN/Inf) is an unrecoverable hard error and aborts
//     immediately, regardless of history length -- exactly as before.
//   * A finite residual merely past threshold is treated as a possibly-
//     recoverable transient: DIVERGING is declared only once the trailing
//     window of kDivergencePersistIters iterates is ALL divergent. Histories
//     shorter than the window cannot declare DIVERGING on a finite overshoot.
//
// Layer 1 drives converge_check() directly on synthetic iterate histories (via
// a friend harness) to pin the truth table. Layer 2 solves through the public
// API: the classic Maratos-effect example -- which used to abort at iteration
// two on a single ~5e15 equality-residual excursion -- now converges to the
// textbook optimum at defaults; a genuinely divergent problem (concave,
// unbounded-below objective) still aborts once the window fills.
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>

using namespace tycho;
using namespace TychoTest;

using tycho::solvers::InteriorPointSolver;
using tycho::solvers::IterateInfo;
using tycho::solvers::kDivergencePersistIters;
using tycho::solvers::OptimizationProblem;

// -----------------------------------------------------------------------------
// Friend harness: reaches InteriorPointSolver's private converge_check() so the trailing-
// window logic can be exercised on hand-built iterate histories without solving
// a real NLP. converge_check() reads only settings_ (divergence tolerances) and
// the iterate history, so a default-constructed optimizer at default tolerances
// is a faithful stand-in. Defined in the global namespace to match the
// `friend class ::DivergencePersistenceHarness` declaration in interior_point_solver.h.
// -----------------------------------------------------------------------------
class DivergencePersistenceHarness {
  public:
    tycho::ConvergenceFlags check(std::vector<IterateInfo> &iters) {
        return solver_.converge_check(iters);
    }
    InteriorPointSolver::Settings &settings() { return solver_.settings_; }

  private:
    InteriorPointSolver solver_;
};

namespace {

// Default divergence tolerance is 1e15 across all four families. A "divergent"
// iterate pushes one finite residual past that; a "recovered" iterate sits well
// below the divergence tolerance yet above the (1e-6) convergence tolerance, so
// it classifies NOTCONVERGED rather than CONVERGED; a "converged" iterate is a
// default IterateInfo (all residuals zero). A NaN iterate carries a non-finite
// residual.
constexpr double kDivPersBigResidual = 1.0e16; // > div tol (1e15): finite overshoot
constexpr double kDivPersMidResidual = 1.0;    // < div tol, > conv tol: not converged

IterateInfo divpers_divergent_iterate() {
    IterateInfo it;
    it.econ_inf_ = kDivPersBigResidual;
    return it;
}

IterateInfo divpers_recovered_iterate() {
    IterateInfo it;
    it.econ_inf_ = kDivPersMidResidual;
    return it;
}

IterateInfo divpers_converged_iterate() { return IterateInfo{}; }

IterateInfo divpers_nan_iterate() {
    IterateInfo it;
    it.kkt_inf_ = std::numeric_limits<double>::quiet_NaN();
    return it;
}

// =============================================================================
// Layer 1 -- truth table on synthetic histories
// =============================================================================

TEST(DivergencePersistence, ConstantIsThree) {
    // The window is a documented policy choice; lock its value.
    EXPECT_EQ(kDivergencePersistIters, 3);
}

TEST(DivergencePersistence, OneTrailingDivergentContinues) {
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_recovered_iterate(), divpers_recovered_iterate(),
                                   divpers_divergent_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::NOTCONVERGED);
}

TEST(DivergencePersistence, TwoTrailingDivergentContinues) {
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_recovered_iterate(), divpers_divergent_iterate(),
                                   divpers_divergent_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::NOTCONVERGED);
}

TEST(DivergencePersistence, ThreeTrailingDivergentDiverges) {
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_divergent_iterate(), divpers_divergent_iterate(),
                                   divpers_divergent_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::DIVERGING);
}

TEST(DivergencePersistence, WindowBrokenByRecoveryContinues) {
    // divergent, recovered, divergent, divergent: the trailing window of three
    // contains the recovered iterate, so the run of divergent iterates is broken
    // and DIVERGING is NOT declared.
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_divergent_iterate(), divpers_recovered_iterate(),
                                   divpers_divergent_iterate(), divpers_divergent_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::NOTCONVERGED);
}

TEST(DivergencePersistence, FourTrailingDivergentDiverges) {
    // Once a full window of divergent iterates accrues, DIVERGING fires even
    // though the history is longer than the window.
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_recovered_iterate(), divpers_divergent_iterate(),
                                   divpers_divergent_iterate(), divpers_divergent_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::DIVERGING);
}

TEST(DivergencePersistence, ShortHistoryAllDivergentContinues) {
    // Two divergent iterates -- fewer than the window -- cannot declare
    // DIVERGING on a finite overshoot.
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_divergent_iterate(), divpers_divergent_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::NOTCONVERGED);
}

TEST(DivergencePersistence, NonFiniteLatestDivergesImmediately) {
    // A NaN in the newest iterate aborts immediately, independent of how many
    // divergent iterates precede it (here: none -- the window is not full).
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_recovered_iterate(), divpers_recovered_iterate(),
                                   divpers_nan_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::DIVERGING);
}

TEST(DivergencePersistence, ShortHistoryNonFiniteDivergesImmediately) {
    // Non-finite exempts the window entirely: a single NaN iterate aborts.
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_nan_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::DIVERGING);
}

TEST(DivergencePersistence, InfiniteLatestDivergesImmediately) {
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_recovered_iterate()};
    iters.back().barr_inf_ = std::numeric_limits<double>::infinity();
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::DIVERGING);
}

// Default-path bit-identity: when no iterate ever trips a divergence threshold,
// converge_check() must classify exactly as it did before this change -- the
// finite-overshoot branch is skipped entirely.
TEST(DivergencePersistence, NonDivergingNonConvergedIsNotConverged) {
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_recovered_iterate(), divpers_recovered_iterate(),
                                   divpers_recovered_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::NOTCONVERGED);
}

TEST(DivergencePersistence, ConvergedHistoryConverges) {
    DivergencePersistenceHarness h;
    std::vector<IterateInfo> iters{divpers_converged_iterate()};
    EXPECT_EQ(h.check(iters), tycho::ConvergenceFlags::CONVERGED);
}

TEST(DivergencePersistence, ThresholdTripHonoredOnEachResidualFamily) {
    // The per-iterate predicate spans all four residual families; a full window
    // divergent on any one of them diverges.
    auto window_on = [](double IterateInfo::*field) {
        DivergencePersistenceHarness h;
        std::vector<IterateInfo> iters(kDivergencePersistIters);
        for (auto &it : iters)
            it.*field = kDivPersBigResidual;
        return h.check(iters);
    };
    EXPECT_EQ(window_on(&IterateInfo::kkt_inf_), tycho::ConvergenceFlags::DIVERGING);
    EXPECT_EQ(window_on(&IterateInfo::econ_inf_), tycho::ConvergenceFlags::DIVERGING);
    EXPECT_EQ(window_on(&IterateInfo::icon_inf_), tycho::ConvergenceFlags::DIVERGING);
    EXPECT_EQ(window_on(&IterateInfo::barr_inf_), tycho::ConvergenceFlags::DIVERGING);
}

// =============================================================================
// Layer 2 -- through the public API
// =============================================================================

// Maratos-effect example (Nocedal & Wright, Numerical Optimization 2e, Example
// 15.4): min 2(x1^2+x2^2-1)-x1 s.t. x1^2+x2^2-1=0, optimum x*=(1,0), obj -1.
// Started exactly on the constraint manifold at (0,1), InteriorPointSolver's first steps blow
// the equality residual up to ~5e15 for a single iteration. Under the former
// per-iterate abort this killed the solve at iteration two; under persistence
// the transient is tolerated and the solve reaches the optimum at defaults.
TEST(DivergencePersistence, MaratosCorpusConvergesAtDefaults) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars((Eigen::VectorXd(2) << 0.0, 1.0).finished());

    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(GenericFunction<-1, 1>(2.0 * (x0 * x0 + x1 * x1 - 1.0) - x0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 1.0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }

    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    auto flag = prob.solve(&ipm).flag_;

    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    const auto &r = ipm.result();
    EXPECT_NEAR(r.obj_val_, -1.0, 1e-4);
    EXPECT_LE(r.iter_num_, 60);
}

// Genuine divergence: a concave, unbounded-below objective (-x0^2) with a
// trivially satisfied equality (x1 = 0). The perturbed-Hessian step drives x0
// away from the origin, improving the true objective every iteration, so the
// line search keeps accepting and the KKT residual grows without bound. The
// persistence window fills within a few iterations and the solve still aborts.
TEST(DivergencePersistence, GenuineDivergenceStillAborts) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars((Eigen::VectorXd(2) << 2.0, 0.0).finished());

    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        prob.add_objective(GenericFunction<-1, 1>(-1.0 * (x0 * x0)),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x1 = args.coeff<1>();
        prob.add_equal_con(GenericFunction<-1, -1>(x1), (Eigen::VectorXi(2) << 0, 1).finished());
    }

    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    ipm.set_max_iters(60); // fail fast if divergence is not detected
    auto flag = prob.solve(&ipm).flag_;

    EXPECT_EQ(flag, tycho::ConvergenceFlags::DIVERGING);
}

// Exhaustion is Ipopt-faithful: with the perturbation ladder disabled outright
// (max_refac_ = 0), every wrong-inertia factorization exhausts immediately. The
// forced step rejection must consult the recovery chain and -- with none
// configured at defaults -- abort the phase as SINGULAR_KKT promptly, instead
// of crawling to max_iters on a singular system.
//
// Backend-dependent, like the sibling ClassicDegeneracyLatchTracksSingularity
// test in test_inertia_regularization.cpp (see its comment for the same
// caveat in more detail): the trigger here is a genuine zero eigenvalue
// (rank deficiency), and Accelerate reports inertia honestly (no Pardiso-style
// static pivot perturbation), so the premise -- an honest wrong-inertia/
// singular report reaching Singular() -- holds. On a pivot-perturbing backend
// (MKL Pardiso, Windows/Linux) the masked deficiency may report a spuriously
// correct inertia and never trip Singular()/kkt_exhausted, so this test is
// gated to the honest-inertia backend rather than asserted platform-wide.
#ifdef USE_ACCELERATE_SPARSE
TEST(DivergencePersistence, ExhaustedInertiaCorrectionAbortsAsSingularKkt) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars((Eigen::VectorXd(2) << 0.0, 1.0).finished());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(GenericFunction<-1, 1>(2.0 * (x0 * x0 + x1 * x1 - 1.0) - x0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 1.0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(0);
    ipm.settings().max_refac_ = 0;

    auto flag = prob.solve(&ipm).flag_;

    EXPECT_EQ(flag, tycho::ConvergenceFlags::SINGULAR_KKT);
    EXPECT_LE(ipm.result().iter_num_, 10);
}
#endif

// A merit-retry recovery link must not be able to resolve the exhaustion. Every
// link in the chain (SOC, extended backtracking, the watchdog relaxation, the
// soft feasibility pre-stage) can only re-test -- or relax acceptance of -- the
// very direction the exhausted factorization produced, so exhaustion bypasses
// the chain and routes straight to re-center/restoration/abort.
//
// The instance is the smallest one that makes the bypass observable: the
// indefinite quadratic 0.5*x0^2 - 0.5*x1^2, a saddle at the origin, started at
// (2, 1). With no constraints the KKT matrix IS the Hessian diag(1, -1), whose
// inertia carries one excess negative eigenvalue, so with the ladder disabled
// every factorization exhausts at once. The Newton direction (-2, -1)
// nonetheless DECREASES the objective (1.5 -> 0) and lands exactly on the
// saddle, where the KKT residual is zero -- so if extended backtracking were
// allowed to re-accept it, the solve would commit the wrong-inertia step and
// report CONVERGED at a saddle point.
//
// Unlike the rank-deficiency trigger of the sibling test above, wrong inertia
// by eigenvalue COUNT is backend-portable: a pivot-perturbing backend masks
// zero pivots, but still reports honest positive/negative counts for a
// well-conditioned indefinite matrix. Hence no backend gate here.
TEST(DivergencePersistence, ExhaustedInertiaCorrectionIsNotResolvedByExtendedBacktracking) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars((Eigen::VectorXd(2) << 2.0, 1.0).finished());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(GenericFunction<-1, 1>(0.5 * (x0 * x0) - 0.5 * (x1 * x1)),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    ipm.settings().max_refac_ = 0;         // ladder off: exhaust immediately
    ipm.settings().ls_extended_iters_ = 2; // extended backtracking armed

    auto flag = prob.solve(&ipm).flag_;

    EXPECT_EQ(flag, tycho::ConvergenceFlags::SINGULAR_KKT);
    EXPECT_LE(ipm.result().iter_num_, 10);
}

} // namespace
