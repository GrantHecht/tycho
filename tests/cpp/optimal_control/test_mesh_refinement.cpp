///////////////////////////////////////////////////////////////////////////////
// Mesh refinement tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(OptimalControlTest, MeshRefinementConvergence) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer_->set_print_level(0);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-4); // relaxed tolerance
    phase->set_max_mesh_iters(5);
    phase->print_mesh_info_ = false;

    phase->solve_optimize();
    EXPECT_TRUE(phase->mesh_converged_) << "Mesh should converge with relaxed tolerance";
}

TEST_F(OptimalControlTest, MeshRefinementIterates) {
    auto phase = make_brach_phase(50, 8); // coarse: 8 segments
    phase->optimizer_->set_print_level(0);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1e-7); // tight tolerance forces refinement
    phase->set_max_mesh_iters(3);
    phase->print_mesh_info_ = false;

    phase->solve_optimize();
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
    phase->optimizer_->set_print_level(0);

    phase->solve_optimize();
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
// That zero error/density is itself the exactly-resolved (converged) case:
// pre-fix, dividing by an all-zero distribution integral / zero max-error
// produced NaN in MeshIterateInfo's ctor (gmean_error_, distintegral_) and in
// calc_bins()'s per-bin slope. Fixed (OC §3.4) by treating a zero-error mesh
// as converged and producing finite, linearly-spaced bins instead.
///////////////////////////////////////////////////////////////////////////////

TEST(MeshRobustness, SingleSegmentLinearDynamicsNoCrashFiniteBins) {
    // Linear (exactly representable) dynamics + single-segment adaptive mesh:
    // pre-fix hits yvecs[1]/hs[1] OOB (§1.7) and 0/0 density -> NaN bins (§3.4).
    auto phase = make_linear_phase(/*nsegs=*/1);
    phase->optimizer_->set_print_level(0);
    phase->set_min_segments(1);
    phase->set_adaptive_mesh(true);
    phase->set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);

    EXPECT_NO_THROW(phase->solve());     // ASan-clean, no UB
    EXPECT_TRUE(phase->mesh_converged_); // zero-error mesh => converged, kept
}

TEST(MeshRobustness, GetMeshInfoZeroDensityNoCrashFiniteBins) {
    // Directly exercises get_mesh_info()'s distint-normalize guard (OC §3.4,
    // ode_phase_base.h): a single-segment phase's de Boor error/density is
    // identically zero (no stencil neighbor, §1.7), so distint's last entry
    // is exactly 0 -- pre-fix, normalizing by it produced NaN bins.
    auto phase = make_linear_phase(/*nsegs=*/1);

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
