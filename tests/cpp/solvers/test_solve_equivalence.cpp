///////////////////////////////////////////////////////////////////////////////
// Equivalence pins: the new solve() surface vs the five retired-to-be mode
// methods (solve/optimize/solve_optimize/solve_optimize_solve/optimize_solve).
// Both surfaces are live on BackendProblemBase today, so this file is the
// migration's own proof that the new surface is a pure re-surfacing rather
// than a behavior change.
//
// DIRECT ROWS (solve(ipm,{mode=Optimal}) vs optimize(); solve(ipm,{mode=
// Feasible}) vs solve()) pin BIT-IDENTITY: exact `==` on the primal vector,
// the equality/inequality multiplier vectors, and the objective. Both arms
// reduce to exactly one call into the same interior-point engine code path
// (run_nlp_solver's JetJobModes::Optimize/Solve branches call
// optimizer_->optimize(x)/solve(x) directly; the new pipeline's
// run_engine_stage does the same on the engine it is handed), so there is no
// arithmetic-ordering freedom between them -- a difference here is a defect
// in the pipeline, not an expected reassociation residue.
//
// COMPOSED ROWS (solve_optimize, solve_optimize_solve, optimize_solve) do
// NOT pin bit-identity -- see the comment block above that section below for
// why, and what is pinned instead.
//
// Every arm is configured for single-threaded, single-partition, fully
// deterministic evaluation (num_partitions=1, qp_threads=1) so that a
// difference in the comparison can only be a defect, never scheduling noise
// from a parallel reduction. Each arm gets a freshly built problem instance
// and, on the new-surface side, a freshly constructed engine.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/engines.h"
#include "tycho/detail/solvers_vf/optimization_problem.h"

// oc_test_utils.h (tests/cpp/optimal_control/) supplies the Brachistochrone/
// LinearODE trajectory-construction code this file duplicates into its own
// probe phase subclasses below -- reused rather than re-derived because it is
// the standing brach/linear fixture geometry every other solver test already
// shares.
#include "oc_test_utils.h"

#include <tycho/vector_functions.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>

namespace {

using tycho::solvers::BackendProblemBase;
using tycho::solvers::EngineRef;
using tycho::solvers::InteriorPointSolver;
using tycho::solvers::Mode;
using tycho::solvers::OptimizationProblem;
using tycho::solvers::SolveOptions;
using tycho::solvers::SolveResult;

///////////////////////////////////////////////////////////////////////////////
// Probe phase subclass: exposes the protected active_eq_lmults_/
// active_iq_lmults_ (ODEPhaseBase keeps them protected) so the pins below can
// read the FULL declared-space multiplier vectors directly, the same way
// SolvePipelinePhaseResultSnapshotPhase (test_solve_pipeline.cpp) exposes
// active_eq_lmults_ for its own snapshot check. OptimalControlProblemBase's
// own active_eq_lmults_/active_iq_lmults_ (the link-constraint multipliers)
// are public already, so no probe subclass is needed at the OCP level.
///////////////////////////////////////////////////////////////////////////////

struct SolveEquivalenceProbeBrachPhase : ODEPhase<TychoTest::BrachODE> {
    using ODEPhase<TychoTest::BrachODE>::ODEPhase;
    Eigen::VectorXd eq_lmults() const { return this->active_eq_lmults_; }
    Eigen::VectorXd iq_lmults() const { return this->active_iq_lmults_; }
};

///////////////////////////////////////////////////////////////////////////////
// Fixture builders.
///////////////////////////////////////////////////////////////////////////////

/// @brief The standing Brachistochrone convergence fixture
/// (solver_test_utils.h's make_brach_solver_phase), duplicated onto the probe
/// subclass above rather than reused -- make_brach_phase()/
/// make_brach_solver_phase() return the plain ODEPhase<BrachODE> type, not
/// this test-local subclass (the same reason test_solve_pipeline.cpp
/// duplicates make_linear_phase() for its own snapshot phase).
std::shared_ptr<SolveEquivalenceProbeBrachPhase>
solve_equivalence_build_brach_phase(int n_segs = 16) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;
    const int n_pts = n_segs * 3 + 1;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    TychoTest::BrachODE ode(g);
    auto phase = std::make_shared<SolveEquivalenceProbeBrachPhase>(ode, TranscriptionModes::LGL3,
                                                                   traj, n_segs);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    constexpr double u_lower = -0.1, u_upper = 2.0;
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 4, u_lower, u_upper);

    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    phase->optimizer_->set_print_level(3);
    return phase;
}

/// @brief One leg of a 2-phase Brachistochrone relay: the same geometry as
/// solve_equivalence_build_brach_phase() above, guessed over the segment
/// [xs,ys] -> [xe,ye] starting from guess time `ts`, fixing its own front
/// boundary only when `fix_front` is true. The second leg passes
/// `fix_front=false`: its front (x, y, v, t) is supplied entirely by the
/// OCP-level link constraint in solve_equivalence_build_two_phase_ocp()
/// below, not restated here -- fixing it independently AND linking it would
/// make the two declarations redundant.
std::shared_ptr<SolveEquivalenceProbeBrachPhase>
solve_equivalence_build_relay_leg(int n_segs, double xs, double ys, double vs, double ts, double xe,
                                  double ye, bool fix_front) {
    constexpr double g = 9.81;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;
    const int n_pts = n_segs * 3 + 1;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = xs + (xe - xs) * s;
        pt[1] = ys + (ye - ys) * s;
        pt[2] = vs + g * s * tf_guess * std::cos(theta_guess);
        pt[3] = ts + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    TychoTest::BrachODE ode(g);
    auto phase = std::make_shared<SolveEquivalenceProbeBrachPhase>(ode, TranscriptionModes::LGL3,
                                                                   traj, n_segs);

    if (fix_front) {
        Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
        Eigen::VectorXd front_val(4);
        front_val << xs, ys, vs, ts;
        phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);
    }

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xe, ye;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    constexpr double u_lower = -0.1, u_upper = 2.0;
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 4, u_lower, u_upper);

    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    phase->optimizer_->set_print_level(3);
    return phase;
}

/// @brief A genuine 2-phase OCP: a Brachistochrone relay split across two
/// legs (different mesh sizes, so a slicing/offset bug cannot hide behind two
/// identically-shaped phases) that hand off through one link-equality
/// constraint (x, y, v, t continuity at the waypoint) and share a real
/// objective -- each leg's own delta-time objective, whose sum is the total
/// transit time the single-phase Brachistochrone minimizes. This gives the
/// OCP-level assertions in the direct rows below (link multiplier vectors,
/// objective) real content: a nonempty link-equality vector and a nonzero
/// objective, rather than an empty vector or a zero compared to itself.
struct SolveEquivalenceOcpBundle {
    std::shared_ptr<OptimalControlProblemBase> ocp;
    std::shared_ptr<SolveEquivalenceProbeBrachPhase> phase0;
    std::shared_ptr<SolveEquivalenceProbeBrachPhase> phase1;
};

SolveEquivalenceOcpBundle solve_equivalence_build_two_phase_ocp() {
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xm = 5.0, ym = 7.5; // relay waypoint between the two legs
    constexpr double xf = 10.0, yf = 5.0;

    SolveEquivalenceOcpBundle bundle;
    bundle.ocp = std::make_shared<OptimalControlProblemBase>();
    bundle.phase0 =
        solve_equivalence_build_relay_leg(6, x0, y0, v0, t0, xm, ym, /*fix_front=*/true);
    bundle.phase1 =
        solve_equivalence_build_relay_leg(8, xm, ym, v0, t0 + 1.0, xf, yf, /*fix_front=*/false);
    bundle.ocp->add_phase(bundle.phase0);
    bundle.ocp->add_phase(bundle.phase1);

    // The one link constraint: continuity of (x, y, v, t) from leg 1's end to
    // leg 2's start. x and y are already fixed to the same waypoint by
    // phase0's own back boundary value above; v and t are genuinely free
    // until this link ties them, and phase1 declares no front boundary of its
    // own, so this is the sole source of its starting state, not a redundant
    // restatement of an independent fixing.
    Eigen::VectorXi link_vars(4);
    link_vars << 0, 1, 2, 3;
    bundle.ocp->add_direct_link_equal_con(0, PhaseRegionFlags::Back, link_vars, 1,
                                          PhaseRegionFlags::Front, link_vars, ScaleModes::AUTO);

    bundle.ocp->optimizer_->set_print_level(3);
    return bundle;
}

/// @brief A small VF OptimizationProblem carrying BOTH an equality and an
/// inequality constraint on independent variables (min x0^2 + x1^2 s.t.
/// x0 - 1 = 0, 2 - x1 <= 0; optimum x0=1, x1=2), so both multiplier vectors
/// this file compares are genuinely nonzero rather than vacuously empty.
std::unique_ptr<OptimizationProblem> solve_equivalence_build_vf_nlp() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    auto prob = std::make_unique<OptimizationProblem>();
    prob->set_vars(Eigen::VectorXd::Constant(2, 0.0));
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob->add_objective(GenericFunction<-1, 1>(x0 * x0 + x1 * x1),
                            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<1>();
        auto x0 = args.coeff<0>();
        prob->add_equal_con(GenericFunction<-1, -1>(x0 - 1.0),
                            (Eigen::VectorXi(1) << 0).finished());
    }
    {
        auto args = Arguments<1>();
        auto x1 = args.coeff<0>();
        prob->add_inequal_con(GenericFunction<-1, -1>(2.0 - x1),
                              (Eigen::VectorXi(1) << 1).finished());
    }
    prob->optimizer_->set_print_level(3);
    return prob;
}

///////////////////////////////////////////////////////////////////////////////
// Deterministic-settings helpers, applied to every arm below.
///////////////////////////////////////////////////////////////////////////////

void solve_equivalence_make_deterministic(BackendProblemBase &prob) {
    prob.set_num_partitions(1);
    prob.optimizer_->set_qp_threads(1);
    prob.optimizer_->set_print_level(3);
}

void solve_equivalence_make_engine_deterministic(InteriorPointSolver &engine) {
    engine.set_qp_threads(1);
    engine.set_print_level(3);
}

///////////////////////////////////////////////////////////////////////////////
// Comparison helpers. Bit-identity failures name the first differing index
// and its values, so a mismatch reports a divergence's location and
// magnitude rather than just "not equal".
///////////////////////////////////////////////////////////////////////////////

Eigen::VectorXd solve_equivalence_flatten(const std::vector<Eigen::VectorXd> &traj) {
    int total = 0;
    for (const auto &v : traj) {
        total += static_cast<int>(v.size());
    }
    Eigen::VectorXd out(total);
    int offset = 0;
    for (const auto &v : traj) {
        out.segment(offset, v.size()) = v;
        offset += static_cast<int>(v.size());
    }
    return out;
}

::testing::AssertionResult SolveEquivalenceVectorsBitIdentical(const char *a_name,
                                                               const char *b_name,
                                                               const Eigen::VectorXd &a,
                                                               const Eigen::VectorXd &b) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure()
               << a_name << ".size()=" << a.size() << " != " << b_name << ".size()=" << b.size();
    }
    for (int i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return ::testing::AssertionFailure()
                   << "first differing index " << i << ": " << a_name << "[" << i << "]=" << a[i]
                   << " vs " << b_name << "[" << i << "]=" << b[i] << " (delta=" << (a[i] - b[i])
                   << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

double solve_equivalence_rel_diff(double a, double b) {
    const double scale = std::max({std::abs(a), std::abs(b), 1.0});
    return std::abs(a - b) / scale;
}

/// @brief Whole-vector relative closeness: ||a-b|| / max(||a||, 1). A
/// per-component check (even a relative one) is the wrong instrument for the
/// composed rows -- a handful of trajectory entries (e.g. an early control
/// value with almost no effect on the objective) sit near zero and are
/// genuinely under-determined early in a run, so they can carry a much larger
/// RELATIVE difference than the well-conditioned bulk of the solution without
/// that difference meaning the two arms disagree on the solution. The norm
/// puts that in proportion to the whole vector, which is what
/// "outcome-equivalence" is actually claiming.
::testing::AssertionResult SolveEquivalenceVectorsRelClose(const char *a_name, const char *b_name,
                                                           const Eigen::VectorXd &a,
                                                           const Eigen::VectorXd &b,
                                                           double rel_tol) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure()
               << a_name << ".size()=" << a.size() << " != " << b_name << ".size()=" << b.size();
    }
    const double diff_norm = (a - b).norm();
    const double scale = std::max(a.norm(), 1.0);
    const double rel = diff_norm / scale;
    if (rel > rel_tol) {
        return ::testing::AssertionFailure()
               << "||" << a_name << " - " << b_name << "|| / max(||" << a_name << "||,1) = " << rel
               << " (tol=" << rel_tol << ", ||diff||=" << diff_norm << ", ||" << a_name
               << "||=" << a.norm() << ")";
    }
    return ::testing::AssertionSuccess();
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// DIRECT ROWS: solve(ipm, {mode=Optimal}) === optimize(); solve(ipm,
// {mode=Feasible}) === solve(). Bit-identical on all three fixtures.
///////////////////////////////////////////////////////////////////////////////

TEST(SolveEquivalence, DirectOptimal_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    const Eigen::VectorXd initial_primal = solve_equivalence_flatten(old_phase->return_traj());
    const tycho::ConvergenceFlags old_flag = old_phase->optimize();
    const double old_obj = old_phase->optimizer_->result().obj_val_;
    const Eigen::VectorXd old_bound_lmults = old_phase->optimizer_->result().bound_lmults_;

    auto new_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase);
    InteriorPointSolver ipm;
    solve_equivalence_make_engine_deterministic(ipm);
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.mode = Mode::Optimal;
    SolveResult result = new_phase->solve(ref, opts);

    // Neither arm is vacuously green: the fixture must actually converge, and
    // the solution must have genuinely moved off the initial guess.
    EXPECT_EQ(tycho::ConvergenceFlags::CONVERGED, old_flag);
    EXPECT_GT((solve_equivalence_flatten(old_phase->return_traj()) - initial_primal).norm(), 1e-3);

    EXPECT_EQ(old_flag, result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.primal", "new.primal", solve_equivalence_flatten(old_phase->return_traj()),
        solve_equivalence_flatten(new_phase->return_traj())));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.eq_lmults", "new.eq_lmults", old_phase->eq_lmults(), new_phase->eq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.iq_lmults", "new.iq_lmults", old_phase->iq_lmults(), new_phase->iq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.bound_lmults", "new.bound_lmults",
                                                    old_bound_lmults, ipm.result().bound_lmults_));
    EXPECT_EQ(old_obj, result.stages_.back().objective_);
}

TEST(SolveEquivalence, DirectFeasible_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    const Eigen::VectorXd initial_primal = solve_equivalence_flatten(old_phase->return_traj());
    const tycho::ConvergenceFlags old_flag = old_phase->solve();
    const double old_obj = old_phase->optimizer_->result().obj_val_;
    const Eigen::VectorXd old_bound_lmults = old_phase->optimizer_->result().bound_lmults_;

    auto new_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase);
    InteriorPointSolver ipm;
    solve_equivalence_make_engine_deterministic(ipm);
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.mode = Mode::Feasible;
    SolveResult result = new_phase->solve(ref, opts);

    // A feasibility-only solve can legitimately land on ACCEPTABLE rather
    // than CONVERGED (test_interior_point_solver_convergence.cpp's
    // BrachistochroneSolveOnly accepts the same range) -- but it must reach
    // one of those, and the solution must have moved off the initial guess.
    EXPECT_LE(old_flag, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GT((solve_equivalence_flatten(old_phase->return_traj()) - initial_primal).norm(), 1e-3);

    EXPECT_EQ(old_flag, result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.primal", "new.primal", solve_equivalence_flatten(old_phase->return_traj()),
        solve_equivalence_flatten(new_phase->return_traj())));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.eq_lmults", "new.eq_lmults", old_phase->eq_lmults(), new_phase->eq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.iq_lmults", "new.iq_lmults", old_phase->iq_lmults(), new_phase->iq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.bound_lmults", "new.bound_lmults",
                                                    old_bound_lmults, ipm.result().bound_lmults_));
    EXPECT_EQ(old_obj, result.stages_.back().objective_);
}

TEST(SolveEquivalence, DirectOptimal_TwoPhaseOcp) {
    auto old_bundle = solve_equivalence_build_two_phase_ocp();
    solve_equivalence_make_deterministic(*old_bundle.ocp);
    Eigen::VectorXd initial_primal(0);
    {
        const Eigen::VectorXd p0 = solve_equivalence_flatten(old_bundle.phase0->return_traj());
        const Eigen::VectorXd p1 = solve_equivalence_flatten(old_bundle.phase1->return_traj());
        initial_primal.resize(p0.size() + p1.size());
        initial_primal << p0, p1;
    }
    const tycho::ConvergenceFlags old_flag = old_bundle.ocp->optimize();
    const double old_obj = old_bundle.ocp->optimizer_->result().obj_val_;
    const Eigen::VectorXd old_bound_lmults = old_bundle.ocp->optimizer_->result().bound_lmults_;

    auto new_bundle = solve_equivalence_build_two_phase_ocp();
    solve_equivalence_make_deterministic(*new_bundle.ocp);
    InteriorPointSolver ipm;
    solve_equivalence_make_engine_deterministic(ipm);
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.mode = Mode::Optimal;
    SolveResult result = new_bundle.ocp->solve(ref, opts);

    // Neither arm is vacuously green: the fixture must actually converge, and
    // the solution must have genuinely moved off the initial guess.
    EXPECT_EQ(tycho::ConvergenceFlags::CONVERGED, old_flag);
    {
        const Eigen::VectorXd p0 = solve_equivalence_flatten(old_bundle.phase0->return_traj());
        const Eigen::VectorXd p1 = solve_equivalence_flatten(old_bundle.phase1->return_traj());
        Eigen::VectorXd final_primal(p0.size() + p1.size());
        final_primal << p0, p1;
        EXPECT_GT((final_primal - initial_primal).norm(), 1e-3);
    }

    EXPECT_EQ(old_flag, result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.phase0.primal", "new.phase0.primal",
        solve_equivalence_flatten(old_bundle.phase0->return_traj()),
        solve_equivalence_flatten(new_bundle.phase0->return_traj())));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.phase1.primal", "new.phase1.primal",
        solve_equivalence_flatten(old_bundle.phase1->return_traj()),
        solve_equivalence_flatten(new_bundle.phase1->return_traj())));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase0.eq_lmults", "new.phase0.eq_lmults",
                                                    old_bundle.phase0->eq_lmults(),
                                                    new_bundle.phase0->eq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase1.eq_lmults", "new.phase1.eq_lmults",
                                                    old_bundle.phase1->eq_lmults(),
                                                    new_bundle.phase1->eq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase0.iq_lmults", "new.phase0.iq_lmults",
                                                    old_bundle.phase0->iq_lmults(),
                                                    new_bundle.phase0->iq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase1.iq_lmults", "new.phase1.iq_lmults",
                                                    old_bundle.phase1->iq_lmults(),
                                                    new_bundle.phase1->iq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.link_eq_lmults", "new.link_eq_lmults",
                                                    old_bundle.ocp->active_eq_lmults_,
                                                    new_bundle.ocp->active_eq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.link_iq_lmults", "new.link_iq_lmults",
                                                    old_bundle.ocp->active_iq_lmults_,
                                                    new_bundle.ocp->active_iq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.bound_lmults", "new.bound_lmults",
                                                    old_bound_lmults, ipm.result().bound_lmults_));
    EXPECT_EQ(old_obj, result.stages_.back().objective_);
}

TEST(SolveEquivalence, DirectFeasible_TwoPhaseOcp) {
    auto old_bundle = solve_equivalence_build_two_phase_ocp();
    solve_equivalence_make_deterministic(*old_bundle.ocp);
    Eigen::VectorXd initial_primal(0);
    {
        const Eigen::VectorXd p0 = solve_equivalence_flatten(old_bundle.phase0->return_traj());
        const Eigen::VectorXd p1 = solve_equivalence_flatten(old_bundle.phase1->return_traj());
        initial_primal.resize(p0.size() + p1.size());
        initial_primal << p0, p1;
    }
    const tycho::ConvergenceFlags old_flag = old_bundle.ocp->solve();
    const double old_obj = old_bundle.ocp->optimizer_->result().obj_val_;
    const Eigen::VectorXd old_bound_lmults = old_bundle.ocp->optimizer_->result().bound_lmults_;

    auto new_bundle = solve_equivalence_build_two_phase_ocp();
    solve_equivalence_make_deterministic(*new_bundle.ocp);
    InteriorPointSolver ipm;
    solve_equivalence_make_engine_deterministic(ipm);
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.mode = Mode::Feasible;
    SolveResult result = new_bundle.ocp->solve(ref, opts);

    // A feasibility-only solve can legitimately land on ACCEPTABLE rather
    // than CONVERGED -- but it must reach one of those, and the solution must
    // have moved off the initial guess.
    EXPECT_LE(old_flag, tycho::ConvergenceFlags::ACCEPTABLE);
    {
        const Eigen::VectorXd p0 = solve_equivalence_flatten(old_bundle.phase0->return_traj());
        const Eigen::VectorXd p1 = solve_equivalence_flatten(old_bundle.phase1->return_traj());
        Eigen::VectorXd final_primal(p0.size() + p1.size());
        final_primal << p0, p1;
        EXPECT_GT((final_primal - initial_primal).norm(), 1e-3);
    }

    EXPECT_EQ(old_flag, result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.phase0.primal", "new.phase0.primal",
        solve_equivalence_flatten(old_bundle.phase0->return_traj()),
        solve_equivalence_flatten(new_bundle.phase0->return_traj())));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.phase1.primal", "new.phase1.primal",
        solve_equivalence_flatten(old_bundle.phase1->return_traj()),
        solve_equivalence_flatten(new_bundle.phase1->return_traj())));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase0.eq_lmults", "new.phase0.eq_lmults",
                                                    old_bundle.phase0->eq_lmults(),
                                                    new_bundle.phase0->eq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase1.eq_lmults", "new.phase1.eq_lmults",
                                                    old_bundle.phase1->eq_lmults(),
                                                    new_bundle.phase1->eq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase0.iq_lmults", "new.phase0.iq_lmults",
                                                    old_bundle.phase0->iq_lmults(),
                                                    new_bundle.phase0->iq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.phase1.iq_lmults", "new.phase1.iq_lmults",
                                                    old_bundle.phase1->iq_lmults(),
                                                    new_bundle.phase1->iq_lmults()));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.link_eq_lmults", "new.link_eq_lmults",
                                                    old_bundle.ocp->active_eq_lmults_,
                                                    new_bundle.ocp->active_eq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.link_iq_lmults", "new.link_iq_lmults",
                                                    old_bundle.ocp->active_iq_lmults_,
                                                    new_bundle.ocp->active_iq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.bound_lmults", "new.bound_lmults",
                                                    old_bound_lmults, ipm.result().bound_lmults_));
    EXPECT_EQ(old_obj, result.stages_.back().objective_);
}

TEST(SolveEquivalence, DirectOptimal_VfProblem) {
    auto old_prob = solve_equivalence_build_vf_nlp();
    solve_equivalence_make_deterministic(*old_prob);
    const tycho::ConvergenceFlags old_flag = old_prob->optimize();
    const double old_obj = old_prob->optimizer_->result().obj_val_;
    const Eigen::VectorXd old_bound_lmults = old_prob->optimizer_->result().bound_lmults_;

    auto new_prob = solve_equivalence_build_vf_nlp();
    solve_equivalence_make_deterministic(*new_prob);
    InteriorPointSolver ipm;
    solve_equivalence_make_engine_deterministic(ipm);
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.mode = Mode::Optimal;
    SolveResult result = new_prob->solve(ref, opts);

    // Self-validating against the known analytic optimum (x0=1, x1=2) rather
    // than only comparing the two arms to each other -- a change that broke
    // both arms identically would otherwise stay invisible here.
    EXPECT_EQ(tycho::ConvergenceFlags::CONVERGED, old_flag);
    ASSERT_EQ(old_prob->active_variables_.size(), 2);
    EXPECT_NEAR(old_prob->active_variables_[0], 1.0, 1e-6);
    EXPECT_NEAR(old_prob->active_variables_[1], 2.0, 1e-6);

    EXPECT_EQ(old_flag, result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.primal", "new.primal", old_prob->active_variables_, new_prob->active_variables_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.eq_lmults", "new.eq_lmults",
                                                    old_prob->active_eq_lmults_,
                                                    new_prob->active_eq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.iq_lmults", "new.iq_lmults",
                                                    old_prob->active_iq_lmults_,
                                                    new_prob->active_iq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.bound_lmults", "new.bound_lmults",
                                                    old_bound_lmults, ipm.result().bound_lmults_));
    EXPECT_EQ(old_obj, result.stages_.back().objective_);
}

TEST(SolveEquivalence, DirectFeasible_VfProblem) {
    auto old_prob = solve_equivalence_build_vf_nlp();
    solve_equivalence_make_deterministic(*old_prob);
    const Eigen::VectorXd initial_primal = old_prob->active_variables_;
    const tycho::ConvergenceFlags old_flag = old_prob->solve();
    const double old_obj = old_prob->optimizer_->result().obj_val_;
    const Eigen::VectorXd old_bound_lmults = old_prob->optimizer_->result().bound_lmults_;

    auto new_prob = solve_equivalence_build_vf_nlp();
    solve_equivalence_make_deterministic(*new_prob);
    InteriorPointSolver ipm;
    solve_equivalence_make_engine_deterministic(ipm);
    EngineRef ref = &ipm;
    SolveOptions opts;
    opts.mode = Mode::Feasible;
    SolveResult result = new_prob->solve(ref, opts);

    // Feasibility only (no objective to optimize), so the analytic optimum
    // does not apply here -- but the equality constraint's target (x0=1) and
    // the inequality's feasible half-space (x1>=2) both must hold, and the
    // starting guess (0, 0) violates both, so a genuine solve must move.
    EXPECT_LE(old_flag, tycho::ConvergenceFlags::ACCEPTABLE);
    ASSERT_EQ(old_prob->active_variables_.size(), 2);
    EXPECT_NEAR(old_prob->active_variables_[0], 1.0, 1e-6);
    EXPECT_GE(old_prob->active_variables_[1], 2.0 - 1e-6);
    EXPECT_GT((old_prob->active_variables_ - initial_primal).norm(), 1e-3);

    EXPECT_EQ(old_flag, result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical(
        "old.primal", "new.primal", old_prob->active_variables_, new_prob->active_variables_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.eq_lmults", "new.eq_lmults",
                                                    old_prob->active_eq_lmults_,
                                                    new_prob->active_eq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.iq_lmults", "new.iq_lmults",
                                                    old_prob->active_iq_lmults_,
                                                    new_prob->active_iq_lmults_));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("old.bound_lmults", "new.bound_lmults",
                                                    old_bound_lmults, ipm.result().bound_lmults_));
    EXPECT_EQ(old_obj, result.stages_.back().objective_);
}

///////////////////////////////////////////////////////////////////////////////
// COMPOSED ROWS: solve_optimize(), solve_optimize_solve(), optimize_solve().
//
// These do NOT pin bit-identity. The old fused calls run their internal
// phases (a feasibility pass, an optimality pass, and -- for the two chain
// methods -- a possible closing feasibility pass) as ONE continuous run on a
// single engine, carrying its FULL internal state (dual/slack iterate, the
// barrier parameter, filter/watchdog/proximal-regularization state, ...)
// across phases directly inside that call. The new staged pipeline runs the
// same phases as separate solve() calls chained through the engine-neutral
// WarmStartData currency: a presolve stage's export seeds the main stage
// that follows it (see src/solvers/solve_pipeline.cpp's WARM SEEDING note),
// and the explicit
// chain between the two top-level calls in a chain row seeds the second call
// from the first's export the same way. WarmStartData carries the primal,
// duals, and slacks -- the pieces the pipeline documents as portable between
// engines -- but not the engine-internal bookkeeping the old fused call kept
// hot across its internal phases (mu, filter/watchdog counters, proximal
// regularization). So the value chain closes most, but not quite all, of the
// gap: both arms reach the same answer, to a very tight but not bit-exact
// tolerance.
//
// What is pinned instead, on the standing Brachistochrone phase fixture:
//   - the two arms agree on the convergence flag exactly;
//   - the two arms agree on the primal solution (whole-vector relative norm)
//     and the objective (relative) to a tight tolerance -- see
//     kSolveEquivalenceOutcomeRelTol below for the measured basis of the
//     actual number used;
//   - the new pipeline is deterministic run to run: the same new-surface
//     call, made twice from two independently built (but identically
//     constructed) problem/engine pairs, is bit-identical to itself.
//
// kSolveEquivalenceOutcomeRelTol calibration: with the presolve-seeds-main
// chain in place, the rows that include a presolve stage
// (ComposedSolveOptimize, ComposedSolveOptimizeSolve_Converged) measured a
// peak objective relative difference of ~1.45e-9 and a peak whole-vector
// primal relative difference of ~4.04e-9 against the old fused call, on this
// fixture -- roughly 3-4x a bit-identity (1e-9) bound, down from ~1.3e-9 /
// ~1e-7 (a per-component, not whole-vector, measure) before the chain
// existed. The residual gap is the engine-internal bookkeeping
// WarmStartData does not carry (see above), not a defect in the chain
// itself. 1e-8 keeps roughly an order of magnitude of margin over the
// measured values while staying far tighter than the engine's own
// convergence tolerance.
//
// The two chain rows (solve_optimize_solve / optimize_solve) each carry two
// fixtures: one where the OPT stage converges (the trailing feasibility pass
// is skipped on both surfaces) and one where it is capped short of
// convergence (the trailing pass actually runs on both surfaces) -- both
// branches of the old conditional trailing SOLVE are exercised.
///////////////////////////////////////////////////////////////////////////////

namespace {

/// @brief One arm's post-solve snapshot: full declared-space primal, both
/// multiplier vectors, and the objective, all as value copies.
struct SolveEquivalencePhaseSnapshot {
    Eigen::VectorXd primal;
    Eigen::VectorXd eq_lmults;
    Eigen::VectorXd iq_lmults;
    double objective = 0.0;
};

SolveEquivalencePhaseSnapshot
solve_equivalence_snapshot(const SolveEquivalenceProbeBrachPhase &phase, double objective) {
    return SolveEquivalencePhaseSnapshot{solve_equivalence_flatten(phase.return_traj()),
                                         phase.eq_lmults(), phase.iq_lmults(), objective};
}

/// @brief Cap the engine's OPT-stage iteration budget tightly enough that a
/// Brachistochrone-sized problem cannot converge within it (mirrors
/// test_solve_pipeline.cpp's NonConvergenceIsAValueNotAThrow, one iteration).
constexpr int kSolveEquivalenceTightMaxIters = 1;

/// @brief Outcome-equivalence relative tolerance for the composed rows (see
/// the comment block above this section). Measured peak values on this
/// fixture, with the presolve-seeds-main warm chain in place: objective
/// relative difference ~1.45e-9, whole-vector primal relative difference
/// ~4.04e-9 (both from the presolve-including rows -- the engine-internal
/// bookkeeping WarmStartData does not carry across separate top-level engine
/// calls is the remaining source, see above). This keeps roughly an order of
/// magnitude of margin over both, while remaining far tighter than the
/// engine's own convergence tolerance.
constexpr double kSolveEquivalenceOutcomeRelTol = 1e-8;

/// @brief Result of running the new-surface chain: `first_opts` on `phase`,
/// then -- only if that stage's flag is not CONVERGED -- a second
/// Mode::Feasible call warm-started from the first call's export. `engine` is
/// reused for both calls, matching the way the old fused methods run every
/// internal phase on the same engine object.
struct SolveEquivalenceChainRun {
    SolveResult final_result;
    bool second_call_ran = false;
};

SolveEquivalenceChainRun solve_equivalence_run_chain(SolveEquivalenceProbeBrachPhase &phase,
                                                     InteriorPointSolver &engine,
                                                     const SolveOptions &first_opts) {
    EngineRef ref = &engine;
    SolveResult r1 = phase.solve(ref, first_opts);
    if (r1.flag_ != tycho::ConvergenceFlags::CONVERGED) {
        SolveOptions opts2;
        opts2.mode = Mode::Feasible;
        opts2.warm = &r1.warm_;
        SolveResult r2 = phase.solve(ref, opts2);
        return SolveEquivalenceChainRun{std::move(r2), true};
    }
    return SolveEquivalenceChainRun{std::move(r1), false};
}

} // namespace

TEST(SolveEquivalence, ComposedSolveOptimize_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    const tycho::ConvergenceFlags old_flag = old_phase->solve_optimize();
    const SolveEquivalencePhaseSnapshot old_snap =
        solve_equivalence_snapshot(*old_phase, old_phase->optimizer_->result().obj_val_);

    SolveOptions opts;
    opts.presolve = true;

    auto new_phase_b = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_b);
    InteriorPointSolver ipm_b;
    solve_equivalence_make_engine_deterministic(ipm_b);
    EngineRef ref_b = &ipm_b;
    SolveResult r_b = new_phase_b->solve(ref_b, opts);
    const SolveEquivalencePhaseSnapshot new_snap_b =
        solve_equivalence_snapshot(*new_phase_b, r_b.stages_.back().objective_);

    auto new_phase_c = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_c);
    InteriorPointSolver ipm_c;
    solve_equivalence_make_engine_deterministic(ipm_c);
    EngineRef ref_c = &ipm_c;
    SolveResult r_c = new_phase_c->solve(ref_c, opts);
    const SolveEquivalencePhaseSnapshot new_snap_c =
        solve_equivalence_snapshot(*new_phase_c, r_c.stages_.back().objective_);

    // Two-run determinism of the new surface.
    EXPECT_EQ(r_b.flag_, r_c.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.primal", "newC.primal", new_snap_b.primal,
                                                    new_snap_c.primal));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.eq_lmults", "newC.eq_lmults",
                                                    new_snap_b.eq_lmults, new_snap_c.eq_lmults));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.iq_lmults", "newC.iq_lmults",
                                                    new_snap_b.iq_lmults, new_snap_c.iq_lmults));
    EXPECT_EQ(new_snap_b.objective, new_snap_c.objective);

    // Outcome-equivalence: old fused call vs the new staged pipeline.
    EXPECT_EQ(old_flag, r_b.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsRelClose("old.primal", "new.primal", old_snap.primal,
                                                new_snap_b.primal, kSolveEquivalenceOutcomeRelTol));
    EXPECT_LE(solve_equivalence_rel_diff(old_snap.objective, new_snap_b.objective),
              kSolveEquivalenceOutcomeRelTol);
}

TEST(SolveEquivalence, ComposedSolveOptimizeSolve_Converged_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    const tycho::ConvergenceFlags old_flag = old_phase->solve_optimize_solve();
    const SolveEquivalencePhaseSnapshot old_snap =
        solve_equivalence_snapshot(*old_phase, old_phase->optimizer_->result().obj_val_);

    SolveOptions first_opts;
    first_opts.presolve = true;

    auto new_phase_b = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_b);
    InteriorPointSolver ipm_b;
    solve_equivalence_make_engine_deterministic(ipm_b);
    SolveEquivalenceChainRun chain_b = solve_equivalence_run_chain(*new_phase_b, ipm_b, first_opts);
    // The fixture is meant to converge on the OPT stage, so the trailing
    // feasibility pass must be skipped -- assert the branch actually taken so
    // a fixture regression fails loudly here rather than silently testing the
    // wrong half of the conditional.
    ASSERT_FALSE(chain_b.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_b =
        solve_equivalence_snapshot(*new_phase_b, chain_b.final_result.stages_.back().objective_);

    auto new_phase_c = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_c);
    InteriorPointSolver ipm_c;
    solve_equivalence_make_engine_deterministic(ipm_c);
    SolveEquivalenceChainRun chain_c = solve_equivalence_run_chain(*new_phase_c, ipm_c, first_opts);
    ASSERT_FALSE(chain_c.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_c =
        solve_equivalence_snapshot(*new_phase_c, chain_c.final_result.stages_.back().objective_);

    EXPECT_EQ(chain_b.final_result.flag_, chain_c.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.primal", "newC.primal", new_snap_b.primal,
                                                    new_snap_c.primal));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.eq_lmults", "newC.eq_lmults",
                                                    new_snap_b.eq_lmults, new_snap_c.eq_lmults));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.iq_lmults", "newC.iq_lmults",
                                                    new_snap_b.iq_lmults, new_snap_c.iq_lmults));
    EXPECT_EQ(new_snap_b.objective, new_snap_c.objective);

    EXPECT_EQ(old_flag, chain_b.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsRelClose("old.primal", "new.primal", old_snap.primal,
                                                new_snap_b.primal, kSolveEquivalenceOutcomeRelTol));
    EXPECT_LE(solve_equivalence_rel_diff(old_snap.objective, new_snap_b.objective),
              kSolveEquivalenceOutcomeRelTol);
}

TEST(SolveEquivalence, ComposedSolveOptimizeSolve_NotConverged_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    old_phase->optimizer_->settings().max_iters_ = kSolveEquivalenceTightMaxIters;
    const tycho::ConvergenceFlags old_flag = old_phase->solve_optimize_solve();
    const SolveEquivalencePhaseSnapshot old_snap =
        solve_equivalence_snapshot(*old_phase, old_phase->optimizer_->result().obj_val_);

    SolveOptions first_opts;
    first_opts.presolve = true;

    auto new_phase_b = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_b);
    InteriorPointSolver ipm_b;
    solve_equivalence_make_engine_deterministic(ipm_b);
    ipm_b.settings().max_iters_ = kSolveEquivalenceTightMaxIters;
    SolveEquivalenceChainRun chain_b = solve_equivalence_run_chain(*new_phase_b, ipm_b, first_opts);
    // This fixture is meant to leave the OPT stage short of convergence, so
    // the trailing feasibility pass must actually run.
    ASSERT_TRUE(chain_b.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_b =
        solve_equivalence_snapshot(*new_phase_b, chain_b.final_result.stages_.back().objective_);

    auto new_phase_c = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_c);
    InteriorPointSolver ipm_c;
    solve_equivalence_make_engine_deterministic(ipm_c);
    ipm_c.settings().max_iters_ = kSolveEquivalenceTightMaxIters;
    SolveEquivalenceChainRun chain_c = solve_equivalence_run_chain(*new_phase_c, ipm_c, first_opts);
    ASSERT_TRUE(chain_c.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_c =
        solve_equivalence_snapshot(*new_phase_c, chain_c.final_result.stages_.back().objective_);

    EXPECT_EQ(chain_b.final_result.flag_, chain_c.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.primal", "newC.primal", new_snap_b.primal,
                                                    new_snap_c.primal));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.eq_lmults", "newC.eq_lmults",
                                                    new_snap_b.eq_lmults, new_snap_c.eq_lmults));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.iq_lmults", "newC.iq_lmults",
                                                    new_snap_b.iq_lmults, new_snap_c.iq_lmults));
    EXPECT_EQ(new_snap_b.objective, new_snap_c.objective);

    EXPECT_EQ(old_flag, chain_b.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsRelClose("old.primal", "new.primal", old_snap.primal,
                                                new_snap_b.primal, kSolveEquivalenceOutcomeRelTol));
    EXPECT_LE(solve_equivalence_rel_diff(old_snap.objective, new_snap_b.objective),
              kSolveEquivalenceOutcomeRelTol);
}

TEST(SolveEquivalence, ComposedOptimizeSolve_Converged_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    const tycho::ConvergenceFlags old_flag = old_phase->optimize_solve();
    const SolveEquivalencePhaseSnapshot old_snap =
        solve_equivalence_snapshot(*old_phase, old_phase->optimizer_->result().obj_val_);

    SolveOptions first_opts; // mode=Optimal, presolve=false: just the OPT stage.

    auto new_phase_b = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_b);
    InteriorPointSolver ipm_b;
    solve_equivalence_make_engine_deterministic(ipm_b);
    SolveEquivalenceChainRun chain_b = solve_equivalence_run_chain(*new_phase_b, ipm_b, first_opts);
    ASSERT_FALSE(chain_b.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_b =
        solve_equivalence_snapshot(*new_phase_b, chain_b.final_result.stages_.back().objective_);

    auto new_phase_c = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_c);
    InteriorPointSolver ipm_c;
    solve_equivalence_make_engine_deterministic(ipm_c);
    SolveEquivalenceChainRun chain_c = solve_equivalence_run_chain(*new_phase_c, ipm_c, first_opts);
    ASSERT_FALSE(chain_c.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_c =
        solve_equivalence_snapshot(*new_phase_c, chain_c.final_result.stages_.back().objective_);

    EXPECT_EQ(chain_b.final_result.flag_, chain_c.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.primal", "newC.primal", new_snap_b.primal,
                                                    new_snap_c.primal));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.eq_lmults", "newC.eq_lmults",
                                                    new_snap_b.eq_lmults, new_snap_c.eq_lmults));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.iq_lmults", "newC.iq_lmults",
                                                    new_snap_b.iq_lmults, new_snap_c.iq_lmults));
    EXPECT_EQ(new_snap_b.objective, new_snap_c.objective);

    EXPECT_EQ(old_flag, chain_b.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsRelClose("old.primal", "new.primal", old_snap.primal,
                                                new_snap_b.primal, kSolveEquivalenceOutcomeRelTol));
    EXPECT_LE(solve_equivalence_rel_diff(old_snap.objective, new_snap_b.objective),
              kSolveEquivalenceOutcomeRelTol);
}

TEST(SolveEquivalence, ComposedOptimizeSolve_NotConverged_BrachPhase) {
    auto old_phase = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*old_phase);
    old_phase->optimizer_->settings().max_iters_ = kSolveEquivalenceTightMaxIters;
    const tycho::ConvergenceFlags old_flag = old_phase->optimize_solve();
    const SolveEquivalencePhaseSnapshot old_snap =
        solve_equivalence_snapshot(*old_phase, old_phase->optimizer_->result().obj_val_);

    SolveOptions first_opts; // mode=Optimal, presolve=false: just the OPT stage.

    auto new_phase_b = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_b);
    InteriorPointSolver ipm_b;
    solve_equivalence_make_engine_deterministic(ipm_b);
    ipm_b.settings().max_iters_ = kSolveEquivalenceTightMaxIters;
    SolveEquivalenceChainRun chain_b = solve_equivalence_run_chain(*new_phase_b, ipm_b, first_opts);
    ASSERT_TRUE(chain_b.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_b =
        solve_equivalence_snapshot(*new_phase_b, chain_b.final_result.stages_.back().objective_);

    auto new_phase_c = solve_equivalence_build_brach_phase();
    solve_equivalence_make_deterministic(*new_phase_c);
    InteriorPointSolver ipm_c;
    solve_equivalence_make_engine_deterministic(ipm_c);
    ipm_c.settings().max_iters_ = kSolveEquivalenceTightMaxIters;
    SolveEquivalenceChainRun chain_c = solve_equivalence_run_chain(*new_phase_c, ipm_c, first_opts);
    ASSERT_TRUE(chain_c.second_call_ran);
    const SolveEquivalencePhaseSnapshot new_snap_c =
        solve_equivalence_snapshot(*new_phase_c, chain_c.final_result.stages_.back().objective_);

    EXPECT_EQ(chain_b.final_result.flag_, chain_c.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.primal", "newC.primal", new_snap_b.primal,
                                                    new_snap_c.primal));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.eq_lmults", "newC.eq_lmults",
                                                    new_snap_b.eq_lmults, new_snap_c.eq_lmults));
    EXPECT_TRUE(SolveEquivalenceVectorsBitIdentical("newB.iq_lmults", "newC.iq_lmults",
                                                    new_snap_b.iq_lmults, new_snap_c.iq_lmults));
    EXPECT_EQ(new_snap_b.objective, new_snap_c.objective);

    EXPECT_EQ(old_flag, chain_b.final_result.flag_);
    EXPECT_TRUE(SolveEquivalenceVectorsRelClose("old.primal", "new.primal", old_snap.primal,
                                                new_snap_b.primal, kSolveEquivalenceOutcomeRelTol));
    EXPECT_LE(solve_equivalence_rel_diff(old_snap.objective, new_snap_b.objective),
              kSolveEquivalenceOutcomeRelTol);
}
