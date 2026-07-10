///////////////////////////////////////////////////////////////////////////////
// PhaseIndexer::make_Vindex_Cindex correctness + contract-guard tests
//
// OC review §1.2 + §3.7: exercises PhaseIndexer's index-building logic
// directly (it is internal transcription plumbing, not Python-reachable for
// the blocked-control FrontNodalBackPath region), covering:
//   - §1.2: the blocked FrontNodalBackPath branch's ODE/static-param writes
//     must land in column i-1 (matching every other write in the function),
//     not column i (which leaves column 0 unwritten and heap-overflows the
//     matrix on the final loop iteration).
//   - §3.7(a): blocked InnerPath/FrontNodalBackPath must not resize to a
//     negative column count when num_defects_ == 1.
///////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace tycho::oc;

// §1.2 — blocked FrontNodalBackPath must write ODE-param/static-param rows to
// column i-1, not column i.
TEST(PhaseIndexer, BlockedFrontNodalBackPathParamColumns) {
    // Xv states, Uv controls, OPv ode-params, SPv static-params.
    constexpr int Xv = 4, Uv = 1, OPv = 1, SPv = 0;
    PhaseIndexer idx(Xv, Uv, OPv, SPv);
    // defect_cardinal_states=3, num_defects=4, blocked_controls=true.
    idx.set_dimensions(/*DCS=*/3, /*Dnum=*/4, /*BlockCon=*/true);

    // A control-only relative-var binding (rxtuv[i] >= xt_vars() for all i)
    // makes IsOnlyControl true, which selects the blocked branch. idx.xt_vars()
    // is the first index past the state+time block, i.e. the first control var.
    Eigen::VectorXi rxtuv(1);
    rxtuv << idx.xt_vars();
    Eigen::VectorXi rodepv(1);
    rodepv << 0; // the phase's single ODE parameter
    Eigen::VectorXi rstatpv(0);

    int nextc = 0;
    auto vc = idx.make_Vindex_Cindex(PhaseRegionFlags::FrontNodalBackPath, rxtuv, rodepv, rstatpv,
                                     /*orows=*/1, nextc);
    const Eigen::MatrixXi &vindex = vc[0];

    // num_defects_ - 2 interior nodal applications (i = 1 .. num_defects_-2).
    ASSERT_EQ(vindex.cols(), idx.num_defects_ - 2);
    ASSERT_GT(vindex.cols(), 0);

    // Row layout per application: [first-state (rxtuv), path-state (rxtuv),
    // last-state (rxtuv), ode-param, static-param]. With rxtuv.size()==1,
    // opsize==1, spsize==0, the ODE-param row is the last row.
    int param_row = vindex.rows() - 1;
    EXPECT_GE(vindex(param_row, 0), 0)
        << "column 0 param row unwritten -- blocked FrontNodalBackPath used col i not i-1";

    // Every column's param row must be a valid (non-negative) NLP location;
    // the pre-fix bug also heap-overflowed column num_defects_-2 (one past the
    // last valid column) on the final loop iteration.
    for (int c = 0; c < vindex.cols(); ++c) {
        EXPECT_GE(vindex(param_row, c), 0) << "column " << c << " param row unwritten";
    }
}

// §3.7(a) — blocked InnerPath must not resize its index matrices to a
// negative column count when num_defects_ == 1 (num_defects_-2 == -1
// pre-fix); it must instead produce zero columns without crashing.
TEST(PhaseIndexer, SingleDefectBlockedInnerPathDoesNotResizeNegative) {
    PhaseIndexer idx(/*Xv=*/2, /*Uv=*/1, /*OPv=*/0, /*SPv=*/0);
    idx.set_dimensions(/*DCS=*/3, /*Dnum=*/1, /*BlockCon=*/true);

    Eigen::VectorXi rx(1);
    rx << idx.xt_vars();
    Eigen::VectorXi empty(0);

    int nextc = 0;
    EXPECT_NO_THROW({
        auto vc = idx.make_Vindex_Cindex(PhaseRegionFlags::InnerPath, rx, empty, empty,
                                         /*orows=*/1, nextc);
        EXPECT_EQ(vc[0].cols(), 0);
    });
}

// §3.7(a) — same guard for blocked FrontNodalBackPath.
TEST(PhaseIndexer, SingleDefectBlockedFrontNodalBackPathDoesNotResizeNegative) {
    PhaseIndexer idx(/*Xv=*/2, /*Uv=*/1, /*OPv=*/0, /*SPv=*/0);
    idx.set_dimensions(/*DCS=*/3, /*Dnum=*/1, /*BlockCon=*/true);

    Eigen::VectorXi rx(1);
    rx << idx.xt_vars();
    Eigen::VectorXi empty(0);

    int nextc = 0;
    EXPECT_NO_THROW({
        auto vc = idx.make_Vindex_Cindex(PhaseRegionFlags::FrontNodalBackPath, rx, empty, empty,
                                         /*orows=*/1, nextc);
        EXPECT_EQ(vc[0].cols(), 0);
    });
}
