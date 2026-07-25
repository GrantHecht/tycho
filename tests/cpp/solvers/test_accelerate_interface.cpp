///////////////////////////////////////////////////////////////////////////////
// Apple Accelerate sparse-solver interface tests.
//
// LEAF-HEADER TU: includes only accelerate_interface.h + Eigen + gtest, never
// tycho/tycho.h or test_utils.h, so this stays in TYCHO_TEST_LIGHT_SOURCES
// (~200 MB, seconds to compile) rather than the 4-7 GB heavy target. Do not
// add includes that pull in the tycho umbrella.
//
// macOS-only: the interface under test is Apple-only, so off-platform this
// compiles to an empty TU -- the inverse of solvers/test_jet_mkl_guard.cpp.
//
// Design: docs/dev/plans/2026-07-24-accelerate-interface-overhaul-design.md
///////////////////////////////////////////////////////////////////////////////

#ifdef USE_ACCELERATE_SPARSE

#include "tycho/detail/solvers/linear/accelerate_interface.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using LDLT = Eigen::AccelerateLDLTTPP<SpMat, Eigen::Upper>;
using LLT = Eigen::AccelerateLLT<SpMat, Eigen::Upper>;

// A = [[4,1,0],[1,3,1],[0,1,2]] -- SPD, with BOTH triangles populated so the
// triangle-canonicalization path (PR 88's P1 fix) is exercised.
SpMat spd_both_triangles() {
    SpMat A(3, 3);
    A.insert(0, 0) = 4.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = 1.0;
    A.insert(1, 1) = 3.0;
    A.insert(1, 2) = 1.0;
    A.insert(2, 1) = 1.0;
    A.insert(2, 2) = 2.0;
    A.makeCompressed();
    return A;
}

// Dense upper triangle, SPD. Shares its nonzero pattern with
// indefinite_full_upper() so one can be refactored into the other.
SpMat spd_full_upper() {
    SpMat A(3, 3);
    A.insert(0, 0) = 4.0;
    A.insert(0, 1) = 1.0;
    A.insert(0, 2) = 1.0;
    A.insert(1, 1) = 4.0;
    A.insert(1, 2) = 1.0;
    A.insert(2, 2) = 4.0;
    A.makeCompressed();
    return A;
}

// Same pattern as spd_full_upper(), but indefinite: Cholesky MUST fail on it.
// (LDLT^TPP will not -- it handles indefinite matrices by design, and probing
// showed it reports SparseStatusOK even for a zero matrix. Failure tests must
// use LLT.)
SpMat indefinite_full_upper() {
    SpMat A(3, 3);
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 8.0;
    A.insert(0, 2) = 8.0;
    A.insert(1, 1) = 1.0;
    A.insert(1, 2) = 8.0;
    A.insert(2, 2) = 1.0;
    A.makeCompressed();
    return A;
}

// Expand a triangle-stored sparse matrix to its full dense symmetric form,
// for residual checks.
Eigen::MatrixXd dense_symmetric(const SpMat &A) {
    Eigen::MatrixXd d = Eigen::MatrixXd(A);
    Eigen::MatrixXd full = d;
    full.triangularView<Eigen::Lower>() = d.transpose();
    return full;
}

TEST(AccelerateInterface, SolvesAnSpdSystem) {
    LDLT s;
    const SpMat A = spd_both_triangles();
    s.compute(A);
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);

    EXPECT_LT((Eigen::MatrixXd(A) * x - b).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(AccelerateInterface, DefaultConstructedDimsAreZero) {
    LDLT s;
    EXPECT_EQ(s.rows(), 0);
    EXPECT_EQ(s.cols(), 0);
}

#ifdef NDEBUG
// info() guards with eigen_assert(m_isInitialized), which only vanishes under
// NDEBUG -- so this test is Release-only by construction. Pre-fix, info_ is
// uninitialized and this reads indeterminate memory (it may pass by luck; it
// is a regression guard, not a demonstration).
TEST(AccelerateInterface, DefaultConstructedInfoIsInitialized) {
    LDLT s;
    EXPECT_EQ(s.info(), Eigen::Success);
}
#endif

TEST(AccelerateInterface, ReleaseRestoresDefaultConstructedState) {
    LDLT s;
    s.compute(spd_both_triangles());
    ASSERT_EQ(s.info(), Eigen::Success);
    ASSERT_EQ(s.rows(), 3);

    s.release();

    EXPECT_EQ(s.rows(), 0);
    EXPECT_EQ(s.cols(), 0);
    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);

    // ...and the solver is still reusable afterwards.
    s.compute(spd_both_triangles());
    EXPECT_EQ(s.info(), Eigen::Success);
}

TEST(AccelerateInterface, InertiaOnKnownIndefiniteMatrix) {
    SpMat A(3, 3);
    A.insert(0, 0) = 1.0;
    A.insert(1, 1) = 1.0;
    A.insert(2, 2) = -1.0;
    A.makeCompressed();

    LDLT s;
    s.compute(A);
    ASSERT_EQ(s.info(), Eigen::Success);
    EXPECT_EQ(s.peigs(), 2);
    EXPECT_EQ(s.neigs(), 1);
    EXPECT_EQ(s.zeigs(), 0);
}

// PSIOPT's inertia-correction loop reads neigs()/peigs() right after
// Compute()/Refactor() without consulting info(), so cached inertia must never
// outlive the factorization it describes.
TEST(AccelerateInterface, SetMatrixClearsStaleInertia) {
    SpMat indefinite(3, 3);
    indefinite.insert(0, 0) = 1.0;
    indefinite.insert(1, 1) = 1.0;
    indefinite.insert(2, 2) = -1.0;
    indefinite.makeCompressed();

    LDLT s;
    s.compute(indefinite);
    ASSERT_EQ(s.info(), Eigen::Success);
    ASSERT_GT(s.peigs() + s.neigs(), 0);

    s.set_matrix(spd_both_triangles());

    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);
}

TEST(AccelerateInterface, ReinitializeClearsStaleInertia) {
    SpMat indefinite(3, 3);
    indefinite.insert(0, 0) = 1.0;
    indefinite.insert(1, 1) = 1.0;
    indefinite.insert(2, 2) = -1.0;
    indefinite.makeCompressed();

    LDLT s;
    s.compute(indefinite);
    ASSERT_GT(s.peigs() + s.neigs(), 0);

    s.reinitialize_internal_matrix_representation();

    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);
}

// doAnalysis() destroys the numeric factorization, so any cached inertia
// describes something that no longer exists. analyze_pattern_internal()
// reaches that state without ever calling doFactorization().
TEST(AccelerateInterface, AnalyzePatternInternalClearsStaleInertia) {
    SpMat indefinite(3, 3);
    indefinite.insert(0, 0) = 1.0;
    indefinite.insert(1, 1) = 1.0;
    indefinite.insert(2, 2) = -1.0;
    indefinite.makeCompressed();

    LDLT s;
    s.compute(indefinite);
    ASSERT_EQ(s.info(), Eigen::Success);
    ASSERT_GT(s.peigs() + s.neigs(), 0);

    s.analyze_pattern_internal();

    EXPECT_EQ(s.peigs(), 0);
    EXPECT_EQ(s.neigs(), 0);
    EXPECT_EQ(s.zeigs(), 0);
}

// Drives doRefactorization() to failure the way PSIOPT does: mutate values in
// the internal matrix in place (pattern unchanged), then refactor.
void break_factorization_in_place(LLT &s) {
    s.get_matrix().coeffRef(0, 1) = 8.0;
    s.get_matrix().coeffRef(0, 2) = 8.0;
    s.refactorize_internal();
}

TEST(AccelerateInterface, FailedFactorizationIsReported) {
    LLT s;
    s.analyze_pattern(spd_full_upper());
    ASSERT_EQ(s.info(), Eigen::Success);

    s.factorize(indefinite_full_upper());

    EXPECT_NE(s.info(), Eigen::Success);
}

// Contract, matching PardisoImpl::_solve_impl: a solve on a bad factorization
// reports through info() and leaves the destination alone. Zeroing it instead
// is deferred to issue #105 (both backends together).
TEST(AccelerateInterface, SolveAfterFailedRefactorReportsAndLeavesDestination) {
    LLT s;
    s.compute(spd_full_upper());
    ASSERT_EQ(s.info(), Eigen::Success);

    break_factorization_in_place(s);
    ASSERT_NE(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x(3);
    x.setConstant(-12345.0);

    x = s.solve(b);

    EXPECT_NE(s.info(), Eigen::Success);
    EXPECT_DOUBLE_EQ(x[0], -12345.0);
    EXPECT_DOUBLE_EQ(x[1], -12345.0);
    EXPECT_DOUBLE_EQ(x[2], -12345.0);
}

// Pre-fix this is worse than stale: the main SparseSolve no-ops, then the
// refinement loop computes r = A*x - b (SparseMultiplyAdd takes the MATRIX, so
// it works fine), the inner SparseSolve also no-ops leaving r as the raw
// residual rather than a correction, and x -= r actively corrupts x -- once per
// refinement iteration.
TEST(AccelerateInterface, SolveAfterFailedRefactorDoesNotCorruptUnderRefinement) {
    LLT s;
    s.set_iterative_refinement(true);
    s.set_iterative_refinement_iterations(2);
    s.compute(spd_full_upper());
    ASSERT_EQ(s.info(), Eigen::Success);

    break_factorization_in_place(s);
    ASSERT_NE(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x(3);
    x.setConstant(-12345.0);

    x = s.solve(b);

    EXPECT_NE(s.info(), Eigen::Success);
    EXPECT_DOUBLE_EQ(x[0], -12345.0);
    EXPECT_DOUBLE_EQ(x[1], -12345.0);
    EXPECT_DOUBLE_EQ(x[2], -12345.0);
}

// The reason doRefactorization deliberately does NOT reset the numeric
// factorization on failure: Apple documents the failed object as refactorable,
// and PSIOPT's perturb-and-retry loop depends on that cheap recovery.
TEST(AccelerateInterface, RefactorRecoversAfterAFailedRefactor) {
    LLT s;
    const SpMat A = spd_full_upper();
    s.compute(A);
    ASSERT_EQ(s.info(), Eigen::Success);

    const double saved01 = s.get_matrix().coeff(0, 1);
    const double saved02 = s.get_matrix().coeff(0, 2);

    break_factorization_in_place(s);
    ASSERT_NE(s.info(), Eigen::Success);

    s.get_matrix().coeffRef(0, 1) = saved01;
    s.get_matrix().coeffRef(0, 2) = saved02;
    s.refactorize_internal();
    ASSERT_EQ(s.info(), Eigen::Success);

    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    const Eigen::VectorXd x = s.solve(b);
    EXPECT_LT((dense_symmetric(A) * x - b).cwiseAbs().maxCoeff(), 1e-10);
}

} // namespace

#endif // USE_ACCELERATE_SPARSE
