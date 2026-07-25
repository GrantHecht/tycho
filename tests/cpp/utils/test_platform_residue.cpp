///////////////////////////////////////////////////////////////////////////////
// Platform residue regression tests (CODEBASE_REVIEW.md §1.3, PR 7 task 8).
//
// Covers four independently-fixed correctness/robustness bugs:
//   P5 — tycho::utils::get_core_count() Windows failure-path fallback +
//        hardware_concurrency()==0 propagation guard (get_core_count.cpp).
//   P6 — tycho::solvers::SolverIndexingData ctor/setter dimension validation
//        and thread_split(Threads<=0) guard (indexing_data.h).
//   P7 — tycho::solvers::ensure_solver_initialized() one-shot return-value
//        contract: nonzero only for the call that actually ran the
//        initializer, 0.0 on every call thereafter (solver_init.cpp).
//   P8 — tycho::utils::factorial() range guard: throws std::invalid_argument
//        outside [0, 12] instead of infinite-recursing (negative input) or
//        silently overflowing int (13! and above) (math_functions.h).
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/solver_init.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"

#include <gtest/gtest.h>

using tycho::solvers::SolverIndexingData;
using MatrixXi = Eigen::MatrixXi;

// -----------------------------------------------------------------------------
// P5 — get_core_count()
// -----------------------------------------------------------------------------

TEST(PlatformResidueTest, GetCoreCountAtLeastOne) {
    // Header contract: "The physical core count, or 1 if detection fails."
    // Must never observably return 0, regardless of platform or
    // hardware_concurrency() behavior on this host.
    EXPECT_GE(tycho::utils::get_core_count(), 1);
}

// -----------------------------------------------------------------------------
// P8 — factorial()
// -----------------------------------------------------------------------------

TEST(PlatformResidueTest, FactorialInRangeValues) {
    EXPECT_EQ(tycho::utils::factorial(0), 1);
    EXPECT_EQ(tycho::utils::factorial(5), 120);
    EXPECT_EQ(tycho::utils::factorial(12), 479001600); // largest int-safe value
}

TEST(PlatformResidueTest, FactorialNegativeThrows) {
    EXPECT_THROW(tycho::utils::factorial(-1), std::invalid_argument);
}

TEST(PlatformResidueTest, FactorialOverflowThrows) {
    // 13! overflows a 32-bit int; guarded rather than left as UB.
    EXPECT_THROW(tycho::utils::factorial(13), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// P6 — SolverIndexingData
// -----------------------------------------------------------------------------

TEST(PlatformResidueTest, SolverIndexingDataGoodDimsConstructs) {
    // input_size_=3, output_size_=2, one function application (1 column).
    MatrixXi vindex(3, 1);
    vindex << 0, 1, 2;
    MatrixXi cindex(2, 1);
    cindex << 0, 1;

    SolverIndexingData data(3, 2, vindex, cindex);
    EXPECT_EQ(data.input_size_, 3);
    EXPECT_EQ(data.output_size_, 2);
    EXPECT_EQ(data.num_appl(), 1);
}

TEST(PlatformResidueTest, SolverIndexingDataMismatchedVIndexRowsThrows) {
    // vindex has 2 rows but input_size_ is declared as 3 -> invariant
    // violated (v_loc(i, V) for i in [0, input_size_) would read OOB /
    // garbage columns downstream). Must throw rather than silently store.
    MatrixXi bad_vindex(2, 1);
    bad_vindex << 0, 1;
    MatrixXi cindex(2, 1);
    cindex << 0, 1;
    EXPECT_THROW(SolverIndexingData(3, 2, bad_vindex, cindex), std::invalid_argument);
}

TEST(PlatformResidueTest, SolverIndexingDataMismatchedCIndexRowsThrows) {
    // cindex has 1 row but output_size_ is declared as 2.
    MatrixXi vindex(3, 1);
    vindex << 0, 1, 2;
    MatrixXi bad_cindex(1, 1);
    bad_cindex << 0;
    EXPECT_THROW(SolverIndexingData(3, 2, vindex, bad_cindex), std::invalid_argument);
}

TEST(PlatformResidueTest, SolverIndexingDataThreadSplitZeroThrows) {
    MatrixXi vindex(3, 1);
    vindex << 0, 1, 2;
    MatrixXi cindex(2, 1);
    cindex << 0, 1;
    SolverIndexingData data(3, 2, vindex, cindex);

    // Previously a SIGFPE (integer division by zero); now a catchable
    // exception. Reachable from Python via prob.num_partitions = 0.
    EXPECT_THROW(data.thread_split(0), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// P7 — ensure_solver_initialized()
//
// The once_flag guarding the real initializer is process-global (a function-
// local static), so this test cannot assume it is the first caller in the
// process — another TEST elsewhere in this binary (or gtest_discover_tests's
// enumeration run) may have already run the initializer. We therefore only
// assert the documented invariant that IS robust to call order:
//   - the return value is never negative, and
//   - a call that is NOT the process-wide first call returns exactly 0.0.
// We call the function twice back-to-back within this TEST; regardless of
// whether the first of these two calls is the process-wide first call, the
// SECOND of the two is guaranteed not to be (the once_flag was already
// consumed by the first call in this pair, if not earlier), so it must
// return 0.0 per the header contract ("0.0 if already initialized").
// -----------------------------------------------------------------------------

TEST(PlatformResidueTest, EnsureSolverInitializedOneShotContract) {
    double first_call = tycho::solvers::ensure_solver_initialized();
    EXPECT_GE(first_call, 0.0); // may be 0.0 if an earlier test already initialized

    double second_call = tycho::solvers::ensure_solver_initialized();
    EXPECT_EQ(second_call, 0.0); // never the process-wide first call -> exactly 0.0
}
