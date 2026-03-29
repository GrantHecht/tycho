///////////////////////////////////////////////////////////////////////////////
// NLP structure tests
///////////////////////////////////////////////////////////////////////////////

#include "solver_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(SolverTest, NLPDimensionsConsistency) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->primal_vars_, 0);
    EXPECT_GT(nlp->equal_cons_, 0);
    // KKT dimension = primal vars + equality multipliers + 2*inequality (slack + dual)
    EXPECT_EQ(nlp->kkt_dim_, nlp->primal_vars_ + nlp->equal_cons_ + 2 * nlp->inequal_cons_);
}

TEST_F(SolverTest, NLPSparsityNonEmpty) {
    auto phase = make_brach_solver_phase(16);
    phase->transcribe(false, false);

    auto &nlp = phase->nlp;
    ASSERT_NE(nlp, nullptr);
    EXPECT_GT(nlp->kkt_coeff_rows_.size(), 0);
    EXPECT_GT(nlp->kkt_coeff_cols_.size(), 0);
    EXPECT_EQ(nlp->num_kkt_elems_, nlp->num_user_kkt_elems_ + nlp->num_solver_kkt_elems_);
}
