///////////////////////////////////////////////////////////////////////////////
// Regression test for the canonical KKT lock-column protocol (InteriorPointSolver 2.3).
//
// Background: a cross-partition data race existed because kkt_fill_hess (the
// objective Hessian scatter in dense_function_base.h) locked on the OUTER
// variable while kkt_fill_all (the constraint scatter) and
// NonLinearProgram::get_mat_space's clash detection locked on the CANONICAL
// (min-endpoint) column. Two partitions writing the same symmetric physical
// KKT slot in mirrored local order could therefore take different mutexes
// and race on a non-atomic +=. The fix introduced a single shared keying
// function, tycho::solvers::kkt_canonical_lock_col(a, b) == min(a, b)
// (include/tycho/detail/solvers/indexing_data.h), used identically by
// kkt_fill_all, kkt_fill_hess, and get_mat_space's clash detection, so all
// claimants of a slot agree on the lock column by construction.
//
// This test guards two things:
//   1. TEST(KktCanonicalLockCol, ...): the helper's contract directly (min
//      semantics, symmetry f(a,b) == f(b,a)).
//   2. TEST(KktCanonicalLockMirrorPair, ...): a functional replica of the
//      real assembly -- a two-"partition" NLP where a constraint
//      (kkt_fill_all, via constraints_jacobian_adjointgradient_adjointhessian)
//      and an objective (kkt_fill_hess, via objective_gradient_hessian) both
//      write a shared, mirror-ordered Hessian pair -- driven through the
//      REAL template bodies (not a hand-rolled model), replicating
//      analyze_sparsity + get_mat_space's canonical clash detection, and
//      comparing every physical KKT slot against an independently computed
//      dense reference (to a tolerance far below any keying error but above
//      the few-ULP summation-order drift on the contended slot -- see the
//      comment at the comparison loop).
//
// IMPORTANT SCOPE NOTE: this is a deterministic, single-threaded run. The
// value comparison catches keying/accounting corruption -- e.g. someone
// reintroducing a second, diverging keying convention at a scatter site, or
// editing kkt_canonical_lock_col itself -- by making the assembled values
// drift from the independent reference. It CANNOT catch a timing race: with
// only one thread there is no concurrent access to observe. The original bug
// this guards against was a genuine data race under concurrent partitions;
// this test provides drift/regression protection for the keying contract
// that made the race possible, not race detection itself.
//
// Kept as a light TU: only tycho/vector_functions.h is included (which pulls
// in tycho/detail/solvers/indexing_data.h transitively via computable_base.h)
// -- not the tycho/tycho.h umbrella -- so this belongs in
// TYCHO_TEST_LIGHT_SOURCES rather than the heavy tycho_tests target.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/vector_functions.h>

#include <gtest/gtest.h>

#include <mutex>
#include <vector>

using tycho::solvers::kkt_canonical_lock_col;
using tycho::solvers::SolverIndexingData;

///////////////////////////////////////////////////////////////////////////////
// (a) kkt_canonical_lock_col's contract directly.
///////////////////////////////////////////////////////////////////////////////

TEST(KktCanonicalLockCol, MinSemantics) {
    EXPECT_EQ(kkt_canonical_lock_col(3, 7), 3);
    EXPECT_EQ(kkt_canonical_lock_col(7, 3), 3);
    EXPECT_EQ(kkt_canonical_lock_col(0, 0), 0);
    EXPECT_EQ(kkt_canonical_lock_col(5, 5), 5);
    EXPECT_EQ(kkt_canonical_lock_col(0, 100), 0);
}

TEST(KktCanonicalLockCol, SymmetricUnderArgumentSwap) {
    // Both scatter sites (kkt_fill_all with mirror-ordered locals, kkt_fill_hess with
    // ascending locals) must resolve the same physical (a, b) pair to the same column
    // regardless of which order they present the endpoints in.
    const int cases[][2] = {{0, 1}, {1, 0}, {10, 2}, {2, 10}, {43, 44}, {44, 43}, {8, 8}};
    for (auto &c : cases) {
        EXPECT_EQ(kkt_canonical_lock_col(c[0], c[1]), kkt_canonical_lock_col(c[1], c[0]))
            << "a=" << c[0] << " b=" << c[1];
    }
}

///////////////////////////////////////////////////////////////////////////////
// (b) Functional replica: mirror-ordered shared Hessian pair written by BOTH a
// constraint (kkt_fill_all) and an objective (kkt_fill_hess) across two
// "partitions", checked against an independent dense reference.
///////////////////////////////////////////////////////////////////////////////

TEST(KktCanonicalLockMirrorPair, ScatterMatchesDenseReference) {
    // f: R^2 -> R, nonlinear (norm), so both Jacobian and (non-degenerate) Hessian
    // elements are exercised.
    auto f = tycho::vf::Arguments<2>().norm();

    // Partition 0: constraint, local -> global order [1, 0] (MIRROR).
    Eigen::MatrixXi vindexA(2, 1);
    vindexA << 1, 0;
    Eigen::MatrixXi cindexA(1, 1);
    cindexA << 0;
    SolverIndexingData dataA(2, 1, vindexA, cindexA);

    // Partition 1: objective, local -> global order [0, 1] (ascending).
    Eigen::MatrixXi vindexB(2, 1);
    vindexB << 0, 1;
    SolverIndexingData dataB(2, vindexB);

    const int kkt_dim = 3; // 2 primal + 1 constraint row
    const int conoffset = 2;
    const int nelems = 5 + 3; // A: 3 hess + 2 jac ; B: 3 hess

    Eigen::VectorXi rows = Eigen::VectorXi::Constant(nelems, -1);
    Eigen::VectorXi cols = Eigen::VectorXi::Constant(nelems, -1);
    Eigen::VectorXi part_ids(nelems);
    int freeloc = 0;
    f.get_kkt_space(rows, cols, freeloc, conoffset, true, true, dataA);
    part_ids.head(freeloc).setConstant(0);
    int b_start = freeloc;
    f.get_kkt_space(rows, cols, freeloc, 0, false, true, dataB);
    part_ids.segment(b_start, freeloc - b_start).setConstant(1);
    ASSERT_EQ(freeloc, nelems);

    int gfree = 0;
    Eigen::VectorXi gxrows(4);
    dataA.get_gradient_space(gxrows, gfree);
    dataB.get_gradient_space(gxrows, gfree);
    int cfree = 0;
    Eigen::VectorXi fxrows(1);
    dataA.get_constraint_space(fxrows, cfree);

    // Keep the raw (pre-swap) coordinates for the reference accumulation below.
    std::vector<std::pair<int, int>> raw(nelems);
    for (int i = 0; i < nelems; i++)
        raw[i] = {rows[i], cols[i]};

    // ---- analyze_sparsity, replicated verbatim (NonLinearProgram::analyze_sparsity) ----
    Eigen::SparseMatrix<double, Eigen::RowMajor> KKTmat(kkt_dim, kkt_dim);
    {
        std::vector<Eigen::Triplet<double>> kktvec;
        for (int i = 0; i < nelems; i++) {
            int row = rows[i], col = cols[i];
            if (col <= row) {
                kktvec.emplace_back(col, row, 1.0);
            } else {
                rows[i] = col;
                cols[i] = row;
                kktvec.emplace_back(row, col, 1.0);
            }
        }
        KKTmat.setFromTriplets(kktvec.begin(), kktvec.end());
        KKTmat.makeCompressed();
    }
    Eigen::VectorXi locs = Eigen::VectorXi::Constant(nelems, -1);
    {
        Eigen::VectorXi innerNNZ(kkt_dim);
        for (int i = 0; i < kkt_dim; i++)
            innerNNZ[i] = KKTmat.row(i).nonZeros();
        for (int i = 0; i < nelems; i++) {
            int row = rows[i], col = cols[i];
            for (int k = 0; k < innerNNZ[col]; k++) {
                if (KKTmat.innerIndexPtr()[KKTmat.outerIndexPtr()[col] + k] == row) {
                    locs[i] = KKTmat.outerIndexPtr()[col] + k;
                    break;
                }
            }
        }
        for (int i = 0; i < nelems; i++)
            ASSERT_GE(locs[i], 0) << "kkt location resolved for elem " << i;
    }

    // ---- get_mat_space clash detection, canonical keying via the SHARED helper ----
    Eigen::MatrixXi KKTclash(2, kkt_dim);
    KKTclash.setZero();
    for (int i = 0; i < nelems; i++)
        KKTclash(part_ids[i], kkt_canonical_lock_col(rows[i], cols[i])) = 1;
    Eigen::VectorXi clashes(kkt_dim);
    int nclash = 0;
    for (int i = 0; i < kkt_dim; i++)
        clashes[i] = (KKTclash.col(i).sum() > 1) ? nclash++ : -1;
    std::vector<std::mutex> locks(nclash);
    EXPECT_NE(clashes[0], -1) << "canonical col 0 contested (mirror pair + shared diag)";
    EXPECT_NE(clashes[1], -1) << "canonical col 1 contested (shared diag (1,1))";
    EXPECT_EQ(clashes[2], -1) << "constraint row col 2 uncontested";

    // Structural agreement: both scatter sites and the clash detection key through
    // kkt_canonical_lock_col, so for every physical slot all claimants map to the same
    // column by construction -- invariant under the analyze_sparsity swap above.
    for (int i = 0; i < nelems; i++)
        EXPECT_EQ(kkt_canonical_lock_col(rows[i], cols[i]),
                  kkt_canonical_lock_col(raw[i].first, raw[i].second));

    // ---- run BOTH scatters through the REAL template bodies ----
    Eigen::VectorXd X(2);
    X << 0.8, -1.7;
    Eigen::VectorXd L(1);
    L << 2.5;
    const double ObjScale = 1.75;

    for (int k = 0; k < KKTmat.nonZeros(); k++)
        KKTmat.valuePtr()[k] = 0.0;

    Eigen::VectorXd FX = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd AGX = Eigen::VectorXd::Zero(4);
    f.constraints_jacobian_adjointgradient_adjointhessian(X, L, FX, AGX, KKTmat, locs, clashes,
                                                          locks, dataA);
    double Val = 0.0;
    Eigen::VectorXd GX = Eigen::VectorXd::Zero(4);
    f.objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat, locs, clashes, locks, dataB);

    // ---- independent dense reference: recompute local Jacobian/Hessians directly and
    // accumulate into a dense lower-triangle mirror using the RAW recorded coordinates.
    // This is deliberately NOT a call through kkt_fill_all/kkt_fill_hess -- it must be
    // independent of the code under test to be a meaningful check. ----
    Eigen::MatrixXd Mref = Eigen::MatrixXd::Zero(kkt_dim, kkt_dim);
    auto accum = [&](int r, int c, double v) { Mref(std::max(r, c), std::min(r, c)) += v; };
    {
        // Partition 0 (constraint, locals map to globals [1, 0]): x_local = (X[1], X[0]).
        Eigen::VectorXd x(2), fx(1), gx(2), lm(1);
        Eigen::MatrixXd jx(1, 2), hx(2, 2);
        x << X[1], X[0];
        lm << L[0];
        fx.setZero();
        jx.setZero();
        gx.setZero();
        hx.setZero();
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
        int e = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = i; j < 2; j++)
                accum(raw[e].first, raw[e].second, hx(j, i)), e++;
            accum(raw[e].first, raw[e].second, jx(0, i)), e++;
        }
    }
    {
        // Partition 1 (objective, locals map to globals [0, 1]).
        Eigen::VectorXd x(2), fx(1), gx(2), lm(1);
        Eigen::MatrixXd jx(1, 2), hx(2, 2);
        x << X[0], X[1];
        lm << ObjScale;
        fx.setZero();
        jx.setZero();
        gx.setZero();
        hx.setZero();
        f.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
        int e = 5;
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
                accum(raw[e].first, raw[e].second, hx(j, i)), e++;
    }

    // ---- compare: every physical slot's accumulated value ----
    // NOT bit-for-bit, and deliberately so. Six of the eight slots do agree
    // exactly, but the diagonal slot (1,1) -- the one both partitions write --
    // drifts by ~4 ULP (measured 2.2e-16 on a value of 0.410) because the
    // scatter path and this reference sum the two contributions in a different
    // order, and the objective side additionally applies ObjScale at a
    // different point in the expression under the project's -ffast-math build.
    // 1e-13 is three decades above that measured drift and many decades below
    // any keying/accounting error, which changes a slot's value by a whole
    // contribution (order 1 here), not by a rounding step.
    for (int i = 0; i < nelems; i++) {
        double got = KKTmat.valuePtr()[locs[i]];
        double want = Mref(rows[i], cols[i]); // rows/cols are canonical (lower) now
        EXPECT_NEAR(got, want, 1e-13) << "slot (" << rows[i] << "," << cols[i] << ")";
    }

    // The mirror pair (1, 0) is the slot both partitions actually contend for -- assert it
    // explicitly in addition to the loop above so a future refactor that reorders `nelems`
    // still exercises the exact regression scenario this test was written for.
    EXPECT_NEAR(KKTmat.valuePtr()[locs[1]], Mref(1, 0), 1e-13);
}
