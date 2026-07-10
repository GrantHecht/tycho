///////////////////////////////////////////////////////////////////////////////
// StateFunction mixed (state-region + parameter-region) constructor tests
//
// OC review §1.19 (LIVE, Python-reachable via ode_phase_bind.h's
// StateConstraint/StateObjective mixed init<FuncType, PhaseRegionFlags,
// VectorXi, PhaseRegionFlags, VectorXi> overload): the `StaticParams` arm of
// state_function.h's mixed constructor overwrote the caller's state
// `region_flag_` (e.g. Front) with `PhaseRegionFlags::Params` while leaving
// `xtu_vars_` populated with state indices -- silently discarding the
// caller's requested state region. The sibling `ODEParams` arm never had
// this bug: it preserves `Reg`. Post-fix, both arms are symmetric.
//
// This also exercises the downstream consumer directly: ODEPhaseBase forwards
// a StateFunction's region_flag_/xtu_vars_/op_vars_/sp_vars_ verbatim into
// PhaseIndexer::add_equality/add_inequality/add_objective, which in turn call
// make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, ...) -- so the fix must
// also produce a correctly-sized/offset variable-index table downstream, not
// just a cosmetically-correct region_flag_.
///////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace tycho::oc;

namespace {

using StateConstraint = OptimalControlProblemBase::StateConstraint;

/// @brief A trivial scalar-output (squared-norm), @p in_rows-input, type-erased
/// VectorFunction, for use only as a StateFunction payload in these tests --
/// its math is never evaluated, only its `output_rows()` (StateFunction's
/// ctors size `output_scales_` from it).
tycho::vf::GenericFunction<-1, -1> make_dummy_scalar_vf(int in_rows) {
    auto args = Arguments<-1>(in_rows);
    return tycho::vf::GenericFunction<-1, -1>(args.squared_norm());
}

} // namespace

// ---------------------------------------------------------------------------
// §1.19 Step 1 (brief) -- the StaticParams arm must preserve the caller's
// state region instead of reclassifying to Params.
// ---------------------------------------------------------------------------
TEST(StateFunctionRegions, StaticParamArmPreservesStateRegion) {
    auto f = make_dummy_scalar_vf(4);
    Eigen::VectorXi xtuv(4);
    xtuv << 0, 1, 2, 3;
    Eigen::VectorXi pv(1);
    pv << 0;

    StateConstraint sc(f, PhaseRegionFlags::Front, xtuv, PhaseRegionFlags::StaticParams, pv);

    EXPECT_EQ(sc.region_flag_, PhaseRegionFlags::Front); // pre-fix: Params
    ASSERT_EQ(sc.sp_vars_.size(), 1);
    EXPECT_EQ(sc.sp_vars_[0], 0);
    ASSERT_EQ(sc.xtu_vars_.size(), 4);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(sc.xtu_vars_[i], i);
    EXPECT_EQ(sc.op_vars_.size(), 0);
}

// The fix must generalize beyond Front -- any state region survives.
TEST(StateFunctionRegions, StaticParamArmPreservesNonFrontStateRegion) {
    auto f = make_dummy_scalar_vf(2);
    Eigen::VectorXi xtuv(2);
    xtuv << 0, 1;
    Eigen::VectorXi pv(1);
    pv << 0;

    StateConstraint sc(f, PhaseRegionFlags::Path, xtuv, PhaseRegionFlags::StaticParams, pv);

    EXPECT_EQ(sc.region_flag_, PhaseRegionFlags::Path);
    ASSERT_EQ(sc.sp_vars_.size(), 1);
    EXPECT_EQ(sc.sp_vars_[0], 0);
}

// ---------------------------------------------------------------------------
// Symmetry with the sibling ODEParams arm (which never had the bug): the two
// arms must now behave identically modulo which parameter vector they fill.
// ---------------------------------------------------------------------------
TEST(StateFunctionRegions, ODEParamArmAlreadyPreservedStateRegion) {
    auto f = make_dummy_scalar_vf(4);
    Eigen::VectorXi xtuv(4);
    xtuv << 0, 1, 2, 3;
    Eigen::VectorXi pv(1);
    pv << 0;

    StateConstraint sc(f, PhaseRegionFlags::Front, xtuv, PhaseRegionFlags::ODEParams, pv);

    EXPECT_EQ(sc.region_flag_, PhaseRegionFlags::Front);
    ASSERT_EQ(sc.op_vars_.size(), 1);
    EXPECT_EQ(sc.op_vars_[0], 0);
    EXPECT_EQ(sc.sp_vars_.size(), 0);
}

TEST(StateFunctionRegions, StaticAndODEParamArmsAreSymmetric) {
    auto f = make_dummy_scalar_vf(4);
    Eigen::VectorXi xtuv(4);
    xtuv << 0, 1, 2, 3;
    Eigen::VectorXi pv(1);
    pv << 0;

    StateConstraint sp(f, PhaseRegionFlags::Back, xtuv, PhaseRegionFlags::StaticParams, pv);
    StateConstraint op(f, PhaseRegionFlags::Back, xtuv, PhaseRegionFlags::ODEParams, pv);

    EXPECT_EQ(sp.region_flag_, op.region_flag_);
    EXPECT_EQ(sp.region_flag_, PhaseRegionFlags::Back);
    EXPECT_EQ(sp.xtu_vars_.size(), op.xtu_vars_.size());
    // sp carries its parameter vector in sp_vars_ (op_vars_ empty); op is the
    // mirror image.
    EXPECT_EQ(sp.sp_vars_.size(), 1);
    EXPECT_EQ(sp.op_vars_.size(), 0);
    EXPECT_EQ(op.op_vars_.size(), 1);
    EXPECT_EQ(op.sp_vars_.size(), 0);
}

// ---------------------------------------------------------------------------
// check_param_region_invariant() must NOT throw for either arm: the resulting
// region_flag_ is a state region (Front/Back/Path/...), not a
// Params/ODEParams/StaticParams region, so the "no xtu indices on a param
// region" guard added in the Task-8 remediation (state_function.h) never
// engages here -- confirming the two fixes compose correctly.
// ---------------------------------------------------------------------------
TEST(StateFunctionRegions, StaticParamArmConstructionDoesNotThrowInvariant) {
    auto f = make_dummy_scalar_vf(4);
    Eigen::VectorXi xtuv(4);
    xtuv << 0, 1, 2, 3;
    Eigen::VectorXi pv(1);
    pv << 0;

    EXPECT_NO_THROW({
        StateConstraint sc(f, PhaseRegionFlags::Front, xtuv, PhaseRegionFlags::StaticParams, pv);
        sc.check_param_region_invariant();
    });
}

// ---------------------------------------------------------------------------
// Downstream indexer consumption (ODEPhaseBase::add_equal_con et al. forward
// region_flag_/xtu_vars_/op_vars_/sp_vars_ verbatim into
// PhaseIndexer::make_Vindex_Cindex -- see ode_phase_base.cpp:909). This is
// the C++-side equivalent of the Python-reachable path through
// ode_phase_bind.h:179-180 (StateConstraint's mixed init) -> add_equal_con.
//
// Discriminates pre- vs post-fix: pre-fix, region_flag_ == Params routes into
// make_Vindex_Cindex's Params/ODEParams/StaticParams branch, which ignores
// rxtuv entirely and sizes v_index_ to only (opsize + spsize) == 1 row --
// silently dropping all 4 requested state indices. Post-fix, region_flag_ ==
// Front routes into the Front branch, which lays out state indices (from
// ode_first_state_locs_) followed by op/sp indices, sizing v_index_ to
// xsize + opsize + spsize == 5 rows.
// ---------------------------------------------------------------------------
TEST(StateFunctionRegions, IndexerConsumesStaticParamArmAsStateRegion) {
    auto f = make_dummy_scalar_vf(4);
    Eigen::VectorXi xtuv(4);
    xtuv << 0, 1, 2, 3;
    Eigen::VectorXi pv(1);
    pv << 0;

    StateConstraint sc(f, PhaseRegionFlags::Front, xtuv, PhaseRegionFlags::StaticParams, pv);

    // Xv=4, Uv=0, OPv=0, SPv=2 (two static params so the location table is
    // non-trivial and distinguishable from the ODE-state locations).
    PhaseIndexer idx(/*Xv=*/4, /*Uv=*/0, /*OPv=*/0, /*SPv=*/2);
    idx.set_dimensions(/*DCS=*/3, /*Dnum=*/2, /*BlockCon=*/false);

    int next_c = 0;
    auto vc = idx.make_Vindex_Cindex(sc.region_flag_, sc.xtu_vars_, sc.op_vars_, sc.sp_vars_,
                                     f.output_rows(), next_c);
    const Eigen::MatrixXi &vindex = vc[0];

    // Post-fix: 4 state indices + 0 op + 1 sp == 5 rows, 1 column (Front is a
    // single-application region). Pre-fix this would instead have been
    // 0 op + 1 sp == 1 row (rxtuv silently dropped).
    ASSERT_EQ(vindex.cols(), 1);
    ASSERT_EQ(vindex.rows(), 5);

    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(vindex(i, 0), idx.ode_first_state_locs_[i])
            << "row " << i << " should carry the Front state-index binding (pre-fix: dropped)";
    EXPECT_EQ(vindex(4, 0), idx.static_param_locs_[0]);
}
