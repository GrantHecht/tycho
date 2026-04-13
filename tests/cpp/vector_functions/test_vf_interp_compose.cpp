///////////////////////////////////////////////////////////////////////////////
// VF interp() / lgl_interp() composition tests
//
// Pins §1 of the PR #42 review fix: every interp overload must throw
// std::invalid_argument (not assert) when handed a null table pointer.
// Asserts compile out under NDEBUG in Release builds, so the previous
// code could segfault in production instead of surfacing a diagnostic.
///////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class InterpComposeNullTableTest : public VectorFunctionFixture {};

TEST_F(InterpComposeNullTableTest, Interp1D_VectorInput_Throws) {
    std::shared_ptr<oc::InterpTable1D> null_table;
    auto t = Arguments<1>();
    EXPECT_THROW((void)vf::interp(null_table, t), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, InterpScalar1D_Throws) {
    std::shared_ptr<oc::InterpTable1D> null_table;
    auto t = Arguments<1>();
    EXPECT_THROW((void)vf::interp_scalar(null_table, t), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, Interp2D_TwoScalars_Throws) {
    std::shared_ptr<oc::InterpTable2D> null_table;
    auto args = Arguments<2>();
    auto x = args.template segment<1>(0);
    auto y = args.template segment<1>(1);
    EXPECT_THROW((void)vf::interp(null_table, x, y), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, Interp2D_PackedVector_Throws) {
    std::shared_ptr<oc::InterpTable2D> null_table;
    auto xy = Arguments<2>();
    EXPECT_THROW((void)vf::interp(null_table, xy), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, Interp3D_ThreeScalars_Throws) {
    std::shared_ptr<oc::InterpTable3D> null_table;
    auto args = Arguments<3>();
    auto x = args.template segment<1>(0);
    auto y = args.template segment<1>(1);
    auto z = args.template segment<1>(2);
    EXPECT_THROW((void)vf::interp(null_table, x, y, z), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, Interp3D_PackedVector_Throws) {
    std::shared_ptr<oc::InterpTable3D> null_table;
    auto xyz = Arguments<3>();
    EXPECT_THROW((void)vf::interp(null_table, xyz), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, Interp4D_FourScalars_Throws) {
    std::shared_ptr<oc::InterpTable4D> null_table;
    auto args = Arguments<4>();
    auto x = args.template segment<1>(0);
    auto y = args.template segment<1>(1);
    auto z = args.template segment<1>(2);
    auto w = args.template segment<1>(3);
    EXPECT_THROW((void)vf::interp(null_table, x, y, z, w), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, Interp4D_PackedVector_Throws) {
    std::shared_ptr<oc::InterpTable4D> null_table;
    auto xyzw = Arguments<4>();
    EXPECT_THROW((void)vf::interp(null_table, xyzw), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, LglInterp_NoTimeExpr_Throws) {
    std::shared_ptr<oc::LGLInterpTable> null_table;
    Eigen::VectorXi vars(1);
    vars << 0;
    EXPECT_THROW((void)vf::lgl_interp(null_table, vars), std::invalid_argument);
}

TEST_F(InterpComposeNullTableTest, LglInterp_WithTimeExpr_Throws) {
    std::shared_ptr<oc::LGLInterpTable> null_table;
    Eigen::VectorXi vars(1);
    vars << 0;
    auto t = Arguments<1>();
    EXPECT_THROW((void)vf::lgl_interp(null_table, vars, t), std::invalid_argument);
}
