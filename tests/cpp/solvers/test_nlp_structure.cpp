///////////////////////////////////////////////////////////////////////////////
// NLP structure tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using TychoTest::make_brach_solver_phase;
using TychoTest::SolverTest;

// BackendProblemBase/OptimizationProblem live in tycho::solvers; this
// file previously relied on the TychoTest -> tycho::solvers using-directive
// leak (fixed in solver_test_utils.h) to see them unqualified.
using tycho::solvers::BackendProblemBase;
using tycho::solvers::OptimizationProblem;

TEST_F(SolverTest, NLPDimensionsConsistency) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp_;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->primal_vars_, 0);
    EXPECT_GT(nlp->equal_cons_, 0);
    // KKT dimension = primal vars + equality multipliers + 2*inequality (slack + dual)
    EXPECT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + nlp->equal_cons_ + 2 * nlp->inequal_cons_);
}

TEST_F(SolverTest, NLPSparsityNonEmpty) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp_;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->kkt_coeff_rows_.size(), 0);
    EXPECT_GT(nlp->kkt_coeff_cols_.size(), 0);
    EXPECT_EQ(nlp->num_kkt_elems_, nlp->num_user_kkt_elems_ + nlp->num_solver_kkt_elems_);
}

// Regression: the direct-NLP ctor once silently overrode base defaults
// with a single partition, diverging from Phase/OCP's own default. The
// paired QP-thread-default half of this regression retired along with the
// problem-owned optimizer: QP thread count is now a setting on
// whichever InteriorPointSolver engine the caller constructs and passes to
// solve(), not a value BackendProblemBase's constructor can default.
TEST(OptimizationProblemDefaults, MatchesBaseInitPartitions) {
    OptimizationProblem prob;
    EXPECT_EQ(prob.num_partitions_, BackendProblemBase::default_num_partitions());
}
