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

///////////////////////////////////////////////////////////////////////////////
// Happy-path round-trip tests — compose vf::interp with identity VF inputs
// and assert the composition matches a direct table query. Guards against
// stack() argument-reorder regressions in the 2D/3D/4D overloads.
///////////////////////////////////////////////////////////////////////////////

namespace {

Eigen::VectorXd linspace_compose(double a, double b, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v[i] = a + (b - a) * double(i) / double(n - 1);
    return v;
}

} // namespace

class InterpComposeHappyPathTest : public VectorFunctionFixture {};

TEST_F(InterpComposeHappyPathTest, Interp1D_MatchesDirect) {
    const int n = 41;
    auto ts = linspace_compose(0.0, 6.0, n);
    Eigen::VectorXd vs(n);
    for (int i = 0; i < n; ++i) vs[i] = std::sin(0.7 * ts[i]) + 0.3 * ts[i];

    auto table = std::make_shared<oc::InterpTable1D>(ts, vs, InterpType::Cubic);
    auto arg = Arguments<1>();
    auto composed = vf::interp(table, arg);

    Eigen::VectorXd x(1);
    x << 2.37;
    Eigen::VectorXd fx(1);
    fx.setZero();
    composed.compute(x, fx);
    Eigen::VectorXd expected = table->interp(x[0]);
    EXPECT_NEAR(fx[0], expected[0], 1e-12);
}

TEST_F(InterpComposeHappyPathTest, Interp2D_PackedVector_MatchesDirect) {
    auto xs = linspace_compose(0.0, 3.0, 13);
    auto ys = linspace_compose(-1.0, 2.0, 11);
    oc::InterpTable2D::MatType zs(ys.size(), xs.size());
    for (int j = 0; j < ys.size(); ++j)
        for (int i = 0; i < xs.size(); ++i)
            zs(j, i) = std::sin(0.8 * xs[i]) * std::cos(0.6 * ys[j]) + 0.1 * xs[i] * ys[j];

    auto table = std::make_shared<oc::InterpTable2D>(xs, ys, zs, InterpType::Cubic);
    auto args = Arguments<2>();
    auto composed = vf::interp(table, args);

    Eigen::VectorXd x(2);
    x << 1.42, 0.55;
    Eigen::VectorXd fx(1);
    fx.setZero();
    composed.compute(x, fx);
    EXPECT_NEAR(fx[0], table->interp(x[0], x[1]), 1e-12);
}

TEST_F(InterpComposeHappyPathTest, Interp2D_TwoScalars_MatchesDirect) {
    auto xs = linspace_compose(0.0, 3.0, 13);
    auto ys = linspace_compose(-1.0, 2.0, 11);
    oc::InterpTable2D::MatType zs(ys.size(), xs.size());
    for (int j = 0; j < ys.size(); ++j)
        for (int i = 0; i < xs.size(); ++i)
            zs(j, i) = 0.4 * xs[i] + 0.2 * ys[j] * ys[j];

    auto table = std::make_shared<oc::InterpTable2D>(xs, ys, zs, InterpType::Cubic);
    auto args = Arguments<2>();
    auto xseg = args.template segment<1>(0);
    auto yseg = args.template segment<1>(1);
    auto composed = vf::interp(table, xseg, yseg);

    Eigen::VectorXd x(2);
    x << 2.1, 0.9;
    Eigen::VectorXd fx(1);
    fx.setZero();
    composed.compute(x, fx);
    EXPECT_NEAR(fx[0], table->interp(x[0], x[1]), 1e-12);
}

TEST_F(InterpComposeHappyPathTest, Interp3D_PackedVector_MatchesDirect) {
    const int n = 7;
    auto xs = linspace_compose(0.0, 2.0, n);
    auto ys = linspace_compose(0.0, 2.0, n);
    auto zs = linspace_compose(0.0, 2.0, n);
    Eigen::Tensor<double, 3> fs(n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                fs(i, j, k) = std::sin(0.5 * xs[i]) + 0.3 * ys[j] * ys[j] - 0.2 * zs[k];

    auto table = std::make_shared<oc::InterpTable3D>(xs, ys, zs, fs, InterpType::Cubic, false);
    auto args = Arguments<3>();
    auto composed = vf::interp(table, args);

    Eigen::VectorXd x(3);
    x << 0.85, 1.2, 1.65;
    Eigen::VectorXd fx(1);
    fx.setZero();
    composed.compute(x, fx);
    EXPECT_NEAR(fx[0], table->interp(x[0], x[1], x[2]), 1e-12);
}

TEST_F(InterpComposeHappyPathTest, Interp4D_PackedVector_MatchesDirect) {
    const int n = 6;
    auto xs = linspace_compose(0.0, 2.0, n);
    auto ys = linspace_compose(0.0, 2.0, n);
    auto zs = linspace_compose(0.0, 2.0, n);
    auto ws = linspace_compose(0.0, 2.0, n);
    Eigen::Tensor<double, 4> fs(n, n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    fs(i, j, k, l) = std::sin(0.4 * xs[i]) * std::cos(0.3 * ys[j]) +
                                     0.1 * zs[k] + 0.05 * ws[l] * ws[l];

    auto table = std::make_shared<oc::InterpTable4D>(xs, ys, zs, ws, fs, InterpType::Cubic, false);
    auto args = Arguments<4>();
    auto composed = vf::interp(table, args);

    Eigen::VectorXd x(4);
    x << 0.75, 1.1, 1.4, 0.6;
    Eigen::VectorXd fx(1);
    fx.setZero();
    composed.compute(x, fx);
    EXPECT_NEAR(fx[0], table->interp(x[0], x[1], x[2], x[3]), 1e-12);
}
