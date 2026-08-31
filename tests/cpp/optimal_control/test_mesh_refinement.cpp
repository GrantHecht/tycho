///////////////////////////////////////////////////////////////////////////////
// Mesh refinement tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, MeshRefinementConvergence) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-4); // relaxed tolerance
    phase->set_max_mesh_iters(5);
    phase->print_mesh_info_ = false;

    phase->solve(&ipm, {.presolve = true});
    EXPECT_TRUE(phase->mesh_converged_) << "Mesh should converge with relaxed tolerance";
}

TEST_F(OptimalControlTest, MeshRefinementIterates) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-7); // tight tolerance forces refinement
    phase->set_max_mesh_iters(3);
    phase->print_mesh_info_ = false;

    phase->solve(&ipm, {.presolve = true});
    EXPECT_GT(phase->mesh_iters_.size(), 0u) << "Should have at least one mesh iteration";
}

///////////////////////////////////////////////////////////////////////////////
// OC review §1.8 — construct-and-discard throw guard
//
// The `else` branch in the `get_space` lambdas of return_costate_traj() /
// return_traj_error() is unreachable through the shipping transcription enum
// (num_tran_card_states_ is always 2, 3, or 4), so there is no way to fire the
// new `throw` in a black-box test. This is a defensiveness smoke test
// confirming the supported modes still succeed without throwing, documenting
// that the throw only guards a future unsupported transcription mode.
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, CostateErrorEstimationSupportedModesDoNotThrow) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);

    phase->solve(&ipm, {.presolve = true});
    EXPECT_NO_THROW((void)phase->return_costate_traj());
    EXPECT_NO_THROW((void)phase->return_traj_error());
}

///////////////////////////////////////////////////////////////////////////////
// OC review §1.4 — calc_switches() normalized-tolerance test degenerates for
// non-O(1) controls
//
// `unddiff` (meant to test the normalized `und` matrix against
// rel_switch_tol_) was computed from the same raw expression as `udiff`, so
// `und` was dead. For a control whose magnitude isn't O(1), that makes the
// "relative" check degenerate into another absolute-scale check: every raw
// step whose magnitude exceeds rel_switch_tol_ (0.3) gets flagged as a
// switch, not just genuine jumps.
///////////////////////////////////////////////////////////////////////////////

TEST_F(OptimalControlTest, ControlSwitchDetectionUsesNormalizedDiff) {
    auto phase = make_bangbang_phase(/*u_scale=*/1000.0);

    Eigen::VectorXd sw = phase->calc_switches();

    ASSERT_EQ(sw.size(), 1) << "Only the true control jump should be flagged as a switch; a raw "
                               "(non-normalized) diff test spuriously flags every "
                               "large-magnitude step as a switch.";
    EXPECT_NEAR(sw[0], kBangBangSwitchTime, 1e-2);
}

///////////////////////////////////////////////////////////////////////////////
// OC review §1.7 + §3.4 — single-segment de Boor OOB + zero-error/density
// mesh div-zero cluster
//
// A single-defect-interval (num_blocks == 1) mesh has no de Boor stencil
// neighbor. Pre-fix, get_meshinfo_deboor()'s `i == 0` branch unconditionally
// read yvecs[i + 1]/hs[i + 1], an out-of-bounds access when num_blocks == 1.
// Fixed (OC §1.7) by short-circuiting to a zero local error estimate when
// there is no neighbor segment to difference against.
//
// The num_blocks == 1 state cannot be constructed directly (the validated
// set_traj overload throws for fewer than 2 segments,
// ode_phase_base.cpp:499-501); it is reached in production only through the
// adaptive-mesh loop: update_mesh() -> refine_traj_manual() has no
// segment-count guard, and its clamp chain bottoms out at min_segments_,
// which may be 1.
//
// That zero error/density is itself the exactly-resolved (converged) case:
// pre-fix, dividing by an all-zero distribution integral / zero max-error
// produced NaN in MeshIterateInfo's ctor (gmean_error_, distintegral_) and in
// calc_bins()'s per-bin slope. Fixed (OC §3.4) by treating a zero-error mesh
// as converged and producing finite, linearly-spaced bins instead.
///////////////////////////////////////////////////////////////////////////////

TEST(MeshRobustness, RefinementToSingleSegmentNoCrashConverges) {
    // §1.7 end-to-end: construct with 2 segments (the constructible minimum)
    // and drive the mesh down to 1 segment through the refinement loop.
    // force_one_mesh_iter_ makes iteration 0 call update_mesh() even though
    // the linear mesh's error is already ~0; with min_segments_ = 1,
    // mesh_red_factor_ = 0.25, and num_extra_segs_ = 0, update_mesh computes
    // n = clamp(ceil(2 * 0.25) + 0, ...) = 1 (each per-entry error term is
    // ~0, far below the 0.25 reduction floor; mesh_tol_ = 1e-3 keeps any
    // solver round-off orders of magnitude inside that margin). The second
    // check_mesh() then runs the de Boor estimator with num_blocks == 1 --
    // pre-fix, an out-of-bounds read of yvecs[1]/hs[1] (§1.7).
    auto phase = make_linear_phase(/*nsegs=*/2);
    tycho::solvers::InteriorPointSolver ipm;
    ipm.set_print_level(3);
    phase->print_mesh_info_ = false;
    phase->set_adaptive_mesh(true);
    phase->set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);
    phase->set_mesh_tol(1e-3);
    phase->set_min_segments(1);
    phase->set_mesh_red_factor(0.25);
    phase->num_extra_segs_ = 0;
    phase->force_one_mesh_iter_ = true;

    EXPECT_NO_THROW(
        phase->solve(&ipm, {.mode = tycho::solvers::Mode::Feasible})); // ASan-clean, no UB
    EXPECT_TRUE(phase->mesh_converged_); // zero-error mesh => converged, kept
    ASSERT_FALSE(phase->mesh_iters_.empty());
    // Proves the final check_mesh() really ran on a single-segment mesh,
    // i.e. the §1.7 num_blocks == 1 guard path was actually exercised.
    EXPECT_EQ(phase->mesh_iters_.back().numsegs_, 1);
    EXPECT_TRUE(phase->mesh_iters_.back().converged_);
}

TEST(MeshRobustness, GetMeshInfoZeroDensityNoCrashFiniteBins) {
    // Directly exercises get_mesh_info()'s distint-normalize guard (OC §3.4,
    // ode_phase_base.h): the exact-binary linear trajectory cancels the de
    // Boor derivative estimate bit-exactly (see make_linear_phase docs), so
    // every density entry -- and hence distint's last entry -- is exactly
    // 0.0. Pre-fix, normalizing by it produced NaN bins. get_mesh_info is
    // the Python-exposed diagnostic (sole caller: ode_phase_base_bind.cpp);
    // no solve needed, the ctor already populates active_traj_.
    auto phase = make_linear_phase(/*nsegs=*/2);

    auto [tsnd, bins, error] = phase->get_mesh_info(/*integ=*/false, /*n=*/4);

    EXPECT_TRUE(bins.allFinite());
    EXPECT_DOUBLE_EQ(bins[0], 0.0);
    EXPECT_DOUBLE_EQ(bins[bins.size() - 1], 1.0);
}

TEST(MeshRobustness, MeshIterateInfoZeroErrorNoNaNFiniteBins) {
    // Directly exercises MeshIterateInfo's ctor (gmean_error_, distintegral_
    // normalize) and calc_bins() (OC §3.4, mesh_iterate_info.h) with an
    // all-zero error/distribution -- the "fully resolved" mesh case that
    // pre-fix produced NaN via log(0) and 0/0.
    constexpr int n = 4;
    Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    Eigen::VectorXd error = Eigen::VectorXd::Zero(n + 1);
    Eigen::VectorXd distribution = Eigen::VectorXd::Zero(n + 1);

    MeshIterateInfo info(n, /*tol=*/1e-6, times, error, distribution);

    EXPECT_TRUE(std::isfinite(info.gmean_error_));
    EXPECT_DOUBLE_EQ(info.gmean_error_, 0.0);
    EXPECT_TRUE(info.distintegral_.allFinite());

    Eigen::VectorXd bins = info.calc_bins(n);
    ASSERT_EQ(bins.size(), n + 1);
    EXPECT_TRUE(bins.allFinite());
    EXPECT_DOUBLE_EQ(bins[0], 0.0);
    EXPECT_DOUBLE_EQ(bins[bins.size() - 1], 1.0);
}
