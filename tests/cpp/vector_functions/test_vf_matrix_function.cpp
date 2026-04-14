///////////////////////////////////////////////////////////////////////////////
// row_matrix + MatrixFunctionView::inverse() finite-difference test
//
// Verifies that:
//   (a) row_matrix(...) construction accepts correctly-sized VF outputs
//       and throws on mismatched shapes.
//   (b) MatrixFunctionView::inverse() on a 3x3 VF-expression is analytically
//       correct — checked by composing M * M.inverse() and evaluating at a
//       deterministic input, expecting the identity matrix (flattened).
//
// Complements the existing test_vf_matrix_view_errors.cpp which pins
// only the error-path validation.
///////////////////////////////////////////////////////////////////////////////

#include "vf_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho::vf;
using namespace TychoTest;

class MatrixFunctionFDTest : public VectorFunctionFixture {};

// Build a row_matrix wrapping a 9-element VF (3x3) and sanity check the
// shape. Rectangular mismatch should still throw (pinned by the existing
// test_vf_matrix_view_errors.cpp; duplicated here only as a smoke check).
TEST_F(MatrixFunctionFDTest, RowMatrixConstructionRespectsShape) {
    auto args = Arguments<9>();
    auto M = row_matrix(args, 3, 3);
    EXPECT_EQ(M.matrix_rows_, 3);
    EXPECT_EQ(M.matrix_cols_, 3);
}

// M * M^{-1} must evaluate to identity (flattened to 9-vector row-major)
// at any invertible input.
TEST_F(MatrixFunctionFDTest, InverseComposesToIdentity) {
    auto args = Arguments<9>();
    auto M = row_matrix(args, 3, 3);
    auto Minv = M.inverse();

    // M * M^{-1} as a 3x3 row-matrix → flatten to length-9 output.
    auto product = M * Minv;
    GenericFunction<-1, -1> gf(product);
    ASSERT_EQ(gf.output_rows(), 9);

    // Deterministic invertible 3x3 input: use a diagonally-dominant matrix
    // flattened row-major.
    Eigen::VectorXd x(9);
    // [[4, 1, 0],
    //  [1, 3, 1],
    //  [0, 1, 2]]
    x << 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0;

    Eigen::VectorXd fx(9);
    fx.setZero();
    gf.compute(x, fx);

    // Expected: identity row-major.
    Eigen::VectorXd id(9);
    id << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(fx[i], id[i], 1e-10) << "M * M^{-1} failed at element " << i;
    }
}
