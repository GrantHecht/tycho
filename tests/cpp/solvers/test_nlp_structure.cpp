///////////////////////////////////////////////////////////////////////////////
// NLP structure tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

TEST_F(SolverTest, NLPDimensionsConsistency) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->PrimalVars, 0);
    EXPECT_GT(nlp->EqualCons, 0);
    // KKT dimension = primal vars + equality multipliers + 2*inequality (slack + dual)
    EXPECT_EQ(nlp->KKTdim, nlp->PrimalVars + nlp->EqualCons + 2 * nlp->InequalCons);
}

TEST_F(SolverTest, NLPSparsityNonEmpty) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->KKTcoeffRows.size(), 0);
    EXPECT_GT(nlp->KKTcoeffCols.size(), 0);
    EXPECT_EQ(nlp->numKKTElems, nlp->numUserKKTElems + nlp->numSolverKKTElems);
}
