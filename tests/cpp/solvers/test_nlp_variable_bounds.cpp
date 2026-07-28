///////////////////////////////////////////////////////////////////////////////
// NLP variable-bound contract: NonLinearProgram::set_variable_bound,
// clear_variable_bounds, has_variable_bounds, and the x_lower_/x_upper_
// vectors make_nlp materializes from the staged declarations.
//
// Every test here hand-builds a NonLinearProgram directly (no PSIOPT, no
// Phase/transcription) with empty objective/constraint lists, so make_nlp
// only has to run its own bookkeeping over primal variables.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/solvers/non_linear_program.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>

using tycho::solvers::NonLinearProgram;

namespace {
constexpr double kNlpVarBoundsInf = std::numeric_limits<double>::infinity();
}

TEST(NlpVariableBounds, DefaultsAreUnboundedAndHasNoBounds) {
    NonLinearProgram nlp(1);
    nlp.make_nlp(4, 0, 0);

    ASSERT_EQ(nlp.x_lower_.size(), 4);
    ASSERT_EQ(nlp.x_upper_.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(nlp.x_lower_[i], -kNlpVarBoundsInf);
        EXPECT_EQ(nlp.x_upper_[i], kNlpVarBoundsInf);
    }
    EXPECT_FALSE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, SingleSetRoundTrips) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(2, -1.5, 3.5);
    nlp.make_nlp(5, 0, 0);

    EXPECT_DOUBLE_EQ(nlp.x_lower_[2], -1.5);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[2], 3.5);

    // Every other index is left unbounded.
    for (int i = 0; i < 5; ++i) {
        if (i == 2)
            continue;
        EXPECT_EQ(nlp.x_lower_[i], -kNlpVarBoundsInf);
        EXPECT_EQ(nlp.x_upper_[i], kNlpVarBoundsInf);
    }
    EXPECT_TRUE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, TightestWinsOverOverlappingDeclarations) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, 0.0, 10.0);
    nlp.set_variable_bound(1, 2.0, 6.0);
    nlp.make_nlp(3, 0, 0);

    EXPECT_DOUBLE_EQ(nlp.x_lower_[1], 2.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[1], 6.0);
}

TEST(NlpVariableBounds, ConflictingBoundsThrowWithIndexInMessage) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(3, 5.0, 10.0);
    nlp.set_variable_bound(3, 20.0, 30.0); // merged: lower=max(5,20)=20, upper=min(10,30)=10

    try {
        nlp.make_nlp(6, 0, 0);
        FAIL() << "expected std::invalid_argument for a lower > upper conflict";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("3"), std::string::npos) << msg;
        EXPECT_NE(msg.find("20"), std::string::npos) << msg;
        EXPECT_NE(msg.find("10"), std::string::npos) << msg;
    }
}

TEST(NlpVariableBounds, EqualBoundsAreAcceptedAsAFixedVariable) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(0, 4.0, 4.0);
    nlp.make_nlp(2, 0, 0);

    EXPECT_DOUBLE_EQ(nlp.x_lower_[0], 4.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[0], 4.0);
}

TEST(NlpVariableBounds, OutOfRangeIndexThrows) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(10, -1.0, 1.0); // primal_vars_ will only be 3

    EXPECT_THROW(nlp.make_nlp(3, 0, 0), std::invalid_argument);
}

TEST(NlpVariableBounds, ReMakeNlpPreservesStagedBounds) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, -2.0, 2.0);

    nlp.make_nlp(4, 0, 0);
    EXPECT_DOUBLE_EQ(nlp.x_lower_[1], -2.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[1], 2.0);

    // Re-transcription: make_nlp runs again without an intervening
    // clear_variable_bounds() call. The staged declaration must still apply.
    nlp.make_nlp(4, 0, 0);
    EXPECT_DOUBLE_EQ(nlp.x_lower_[1], -2.0);
    EXPECT_DOUBLE_EQ(nlp.x_upper_[1], 2.0);
    EXPECT_TRUE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, ClearVariableBoundsDropsStagedRecords) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(1, -2.0, 2.0);
    nlp.make_nlp(4, 0, 0);
    ASSERT_TRUE(nlp.has_variable_bounds());

    nlp.clear_variable_bounds();
    EXPECT_FALSE(nlp.has_variable_bounds());

    nlp.make_nlp(4, 0, 0);
    EXPECT_FALSE(nlp.has_variable_bounds());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(nlp.x_lower_[i], -kNlpVarBoundsInf);
        EXPECT_EQ(nlp.x_upper_[i], kNlpVarBoundsInf);
    }
}

TEST(NlpVariableBounds, BothBoundsInfiniteIsANoOp) {
    NonLinearProgram nlp(1);
    nlp.set_variable_bound(0, -kNlpVarBoundsInf, kNlpVarBoundsInf);
    EXPECT_TRUE(nlp.staged_variable_bounds_.empty());

    nlp.make_nlp(2, 0, 0);
    EXPECT_FALSE(nlp.has_variable_bounds());
}

TEST(NlpVariableBounds, NanBoundThrowsImmediately) {
    NonLinearProgram nlp(1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(nlp.set_variable_bound(0, nan, 1.0), std::invalid_argument);
    EXPECT_THROW(nlp.set_variable_bound(0, -1.0, nan), std::invalid_argument);
    EXPECT_TRUE(nlp.staged_variable_bounds_.empty());
}
