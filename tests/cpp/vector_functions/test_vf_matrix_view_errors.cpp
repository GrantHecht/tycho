///////////////////////////////////////////////////////////////////////////////
// MatrixFunctionView error-path tests
//
// Pins §1 of the PR #42 review fix: MatrixFunctionView must throw
// std::invalid_argument (not assert) on non-positive rows/cols or on a
// rows*cols mismatch with the wrapped function's output size. Asserts
// compile out under NDEBUG in Release builds.
///////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class MatrixFunctionViewErrorsTest : public VectorFunctionFixture {};

TEST_F(MatrixFunctionViewErrorsTest, NonPositiveRowsThrows) {
    // Arguments<4> returns a 4-element identity; row_matrix tries to lay it out.
    auto args = Arguments<4>();
    EXPECT_THROW((void)vf::row_matrix(args, 0, 4), std::invalid_argument);
    EXPECT_THROW((void)vf::row_matrix(args, -1, 4), std::invalid_argument);
}

TEST_F(MatrixFunctionViewErrorsTest, NonPositiveColsThrows) {
    auto args = Arguments<4>();
    EXPECT_THROW((void)vf::row_matrix(args, 4, 0), std::invalid_argument);
    EXPECT_THROW((void)vf::row_matrix(args, 4, -2), std::invalid_argument);
}

TEST_F(MatrixFunctionViewErrorsTest, SizeMismatchThrows) {
    // 4-element function reshaped as 3x2 (=6) must throw.
    auto args = Arguments<4>();
    EXPECT_THROW((void)vf::row_matrix(args, 3, 2), std::invalid_argument);
    EXPECT_THROW((void)vf::row_matrix(args, 2, 3), std::invalid_argument);
}

TEST_F(MatrixFunctionViewErrorsTest, ValidSizeAccepted) {
    auto args = Arguments<6>();
    EXPECT_NO_THROW((void)vf::row_matrix(args, 2, 3));
    EXPECT_NO_THROW((void)vf::row_matrix(args, 3, 2));
    EXPECT_NO_THROW((void)vf::row_matrix(args, 1, 6));
    EXPECT_NO_THROW((void)vf::row_matrix(args, 6, 1));
}
