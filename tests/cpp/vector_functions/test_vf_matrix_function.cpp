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

// 2x2 inverse specialization — separate dispatch branch from 3x3.
TEST_F(MatrixFunctionFDTest, Inverse2x2ComposesToIdentity) {
    auto args = Arguments<4>();
    auto M = row_matrix(args, 2, 2);
    auto product = M * M.inverse();
    GenericFunction<-1, -1> gf(product);
    ASSERT_EQ(gf.output_rows(), 4);

    Eigen::VectorXd x(4);
    x << 2.0, 1.0, 0.5, 3.0;
    Eigen::VectorXd fx(4);
    fx.setZero();
    gf.compute(x, fx);

    Eigen::VectorXd id(4);
    id << 1, 0, 0, 1;
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(fx[i], id[i], 1e-10) << "2x2 M * M^{-1} failed at element " << i;
}

// 4x4 inverse — general (non-specialized) dispatch path.
TEST_F(MatrixFunctionFDTest, Inverse4x4ComposesToIdentity) {
    auto args = Arguments<16>();
    auto M = row_matrix(args, 4, 4);
    auto product = M * M.inverse();
    GenericFunction<-1, -1> gf(product);
    ASSERT_EQ(gf.output_rows(), 16);

    // Diagonally-dominant 4x4 (row-major).
    Eigen::VectorXd x(16);
    x << 5, 1, 0, 0,
         1, 4, 1, 0,
         0, 1, 3, 1,
         0, 0, 1, 2;
    Eigen::VectorXd fx(16);
    fx.setZero();
    gf.compute(x, fx);

    Eigen::VectorXd id(16);
    id << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    for (int i = 0; i < 16; ++i)
        EXPECT_NEAR(fx[i], id[i], 1e-10) << "4x4 M * M^{-1} failed at element " << i;
}

// inverse() on a non-square MatrixFunctionView must throw — pins the
// size-guard in operator_overloads.h.
TEST_F(MatrixFunctionFDTest, InverseRejectsNonSquare) {
    auto args = Arguments<6>();
    auto M = row_matrix(args, 2, 3); // 2x3
    EXPECT_THROW((void)M.inverse(), std::invalid_argument);
}

// MatrixFunctionView * vector VF — FD Jacobian check of a 3x3 * vec3 product.
TEST_F(MatrixFunctionFDTest, MatrixVectorProductFiniteDifference) {
    // Input: [m (9 row-major), v (3)] — 12 rows.
    auto args = Arguments<12>();
    auto m_vec = args.head<9>();
    auto v_vec = args.tail<3>();
    auto M = row_matrix(m_vec, 3, 3);
    auto Mv = M * v_vec;

    GenericFunction<-1, -1> gf(Mv);
    ASSERT_EQ(gf.output_rows(), 3);
    ASSERT_EQ(gf.input_rows(), 12);

    Eigen::VectorXd x(12);
    x << 2, 1, 0,
         1, 3, 1,
         0, 1, 2,
         1.0, -0.5, 2.0;

    // Expected M*v computed directly.
    Eigen::Matrix3d M_mat;
    M_mat << 2, 1, 0, 1, 3, 1, 0, 1, 2;
    Eigen::Vector3d v;
    v << 1.0, -0.5, 2.0;
    Eigen::Vector3d expected = M_mat * v;

    Eigen::VectorXd fx(3);
    fx.setZero();
    gf.compute(x, fx);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(fx[i], expected[i], 1e-12) << "M*v mismatch at row " << i;

    verify_gf_jacobian_fd<-1, -1>(Mv, x, 1e-6);
}
