// InterpTable 3D/4D smoke tests — construct, evaluate at nodes, and pin the
// default out-of-bounds throw contract for the high-dimensional tables.

#include "test_utils.h"
#include <fmt/color.h>
#include <gtest/gtest.h>
#include <tycho/tycho.h>

#include <cmath>

using namespace tycho;
using namespace TychoTest;

namespace {

Eigen::VectorXd linspace_nd(double a, double b, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v[i] = a + (b - a) * double(i) / double(n - 1);
    return v;
}

double ref_fn_3d(double x, double y, double z) {
    return std::sin(0.5 * x) + 0.3 * y * y - 0.2 * z;
}

double ref_fn_4d(double x, double y, double z, double w) {
    return std::sin(0.4 * x) * std::cos(0.3 * y) + 0.1 * z + 0.05 * w * w;
}

} // namespace

class InterpTable3DTest : public VectorFunctionFixture {};
class InterpTable4DTest : public VectorFunctionFixture {};

TEST_F(InterpTable3DTest, ConstructAndEvalSmoke) {
    const int n = 7;
    auto xs = linspace_nd(0.0, 2.0, n);
    auto ys = linspace_nd(0.0, 2.0, n);
    auto zs = linspace_nd(0.0, 2.0, n);

    Eigen::Tensor<double, 3> fs(n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) fs(i, j, k) = ref_fn_3d(xs[i], ys[j], zs[k]);

    oc::InterpTable3D table(xs, ys, zs, fs, InterpType::Cubic, false);

    // Interior point — cubic should track to ~1e-3 at this grid density.
    double v = table.interp(0.85, 1.2, 1.65);
    EXPECT_NEAR(v, ref_fn_3d(0.85, 1.2, 1.65), 5e-3);
}

TEST_F(InterpTable3DTest, OutOfBoundsThrowsByDefault) {
    const int n = 6;
    auto xs = linspace_nd(0.0, 1.0, n);
    auto ys = linspace_nd(0.0, 1.0, n);
    auto zs = linspace_nd(0.0, 1.0, n);
    Eigen::Tensor<double, 3> fs(n, n, n);
    fs.setZero();

    oc::InterpTable3D table(xs, ys, zs, fs, InterpType::Cubic, false);

    EXPECT_THROW((void)table.interp(-0.1, 0.5, 0.5), std::invalid_argument);
    EXPECT_THROW((void)table.interp(0.5, 1.2, 0.5), std::invalid_argument);
    EXPECT_THROW((void)table.interp(0.5, 0.5, 2.0), std::invalid_argument);
}

TEST_F(InterpTable4DTest, ConstructAndEvalSmoke) {
    const int n = 6;
    auto xs = linspace_nd(0.0, 2.0, n);
    auto ys = linspace_nd(0.0, 2.0, n);
    auto zs = linspace_nd(0.0, 2.0, n);
    auto ws = linspace_nd(0.0, 2.0, n);

    Eigen::Tensor<double, 4> fs(n, n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    fs(i, j, k, l) = ref_fn_4d(xs[i], ys[j], zs[k], ws[l]);

    oc::InterpTable4D table(xs, ys, zs, ws, fs, InterpType::Cubic, false);

    double v = table.interp(0.75, 1.1, 1.4, 0.6);
    EXPECT_NEAR(v, ref_fn_4d(0.75, 1.1, 1.4, 0.6), 1e-2);
}

TEST_F(InterpTable3DTest, CubicFirstDerivativeMatchesFiniteDifference) {
    const int n = 11;
    auto xs = linspace_nd(0.0, 2.0, n);
    auto ys = linspace_nd(0.0, 2.0, n);
    auto zs = linspace_nd(0.0, 2.0, n);
    Eigen::Tensor<double, 3> fs(n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) fs(i, j, k) = ref_fn_3d(xs[i], ys[j], zs[k]);

    oc::InterpTable3D table(xs, ys, zs, fs, InterpType::Cubic, false);
    const double h = 1e-5;
    const double x = 0.85, y = 1.2, z = 1.05;
    auto [v, grad] = table.interp_deriv1(x, y, z);
    double dfdx_fd = (table.interp(x + h, y, z) - table.interp(x - h, y, z)) / (2 * h);
    double dfdy_fd = (table.interp(x, y + h, z) - table.interp(x, y - h, z)) / (2 * h);
    double dfdz_fd = (table.interp(x, y, z + h) - table.interp(x, y, z - h)) / (2 * h);
    EXPECT_NEAR(v, table.interp(x, y, z), 1e-12);
    EXPECT_NEAR(grad[0], dfdx_fd, 1e-5);
    EXPECT_NEAR(grad[1], dfdy_fd, 1e-5);
    EXPECT_NEAR(grad[2], dfdz_fd, 1e-5);
}

TEST_F(InterpTable4DTest, CubicFirstDerivativeMatchesFiniteDifference) {
    const int n = 9;
    auto xs = linspace_nd(0.0, 2.0, n);
    auto ys = linspace_nd(0.0, 2.0, n);
    auto zs = linspace_nd(0.0, 2.0, n);
    auto ws = linspace_nd(0.0, 2.0, n);
    Eigen::Tensor<double, 4> fs(n, n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    fs(i, j, k, l) = ref_fn_4d(xs[i], ys[j], zs[k], ws[l]);

    oc::InterpTable4D table(xs, ys, zs, ws, fs, InterpType::Cubic, false);
    const double h = 1e-5;
    const double x = 0.75, y = 1.1, z = 1.05, w = 0.6;
    auto [v, grad] = table.interp_deriv1(x, y, z, w);
    double dfdx_fd = (table.interp(x + h, y, z, w) - table.interp(x - h, y, z, w)) / (2 * h);
    double dfdy_fd = (table.interp(x, y + h, z, w) - table.interp(x, y - h, z, w)) / (2 * h);
    double dfdz_fd = (table.interp(x, y, z + h, w) - table.interp(x, y, z - h, w)) / (2 * h);
    double dfdw_fd = (table.interp(x, y, z, w + h) - table.interp(x, y, z, w - h)) / (2 * h);
    EXPECT_NEAR(v, table.interp(x, y, z, w), 1e-12);
    EXPECT_NEAR(grad[0], dfdx_fd, 1e-4);
    EXPECT_NEAR(grad[1], dfdy_fd, 1e-4);
    EXPECT_NEAR(grad[2], dfdz_fd, 1e-4);
    EXPECT_NEAR(grad[3], dfdw_fd, 1e-4);
}

TEST_F(InterpTable3DTest, UnevenGridConstructorReproducesNodes) {
    // 1D/2D tests pin the even/uneven dispatch already; mirror the check
    // for 3D since the constructor branches are nearly-identical template
    // instantiations and have been a historical source of bugs.
    Eigen::VectorXd xs(7);
    xs << 0.0, 0.15, 0.5, 1.05, 1.4, 1.85, 2.4;
    Eigen::VectorXd ys(6);
    ys << 0.0, 0.3, 0.7, 1.2, 1.7, 2.2;
    Eigen::VectorXd zs(5);
    zs << 0.0, 0.4, 0.95, 1.55, 2.1;
    Eigen::Tensor<double, 3> fs(xs.size(), ys.size(), zs.size());
    for (int i = 0; i < xs.size(); ++i)
        for (int j = 0; j < ys.size(); ++j)
            for (int k = 0; k < zs.size(); ++k)
                fs(i, j, k) = ref_fn_3d(xs[i], ys[j], zs[k]);

    oc::InterpTable3D table(xs, ys, zs, fs, InterpType::Cubic, false);

    // Spot-check a handful of nodes — they must be reproduced exactly even on
    // an uneven grid.
    for (int i : {0, 3, 6})
        for (int j : {0, 2, 5})
            for (int k : {0, 4})
                EXPECT_NEAR(table.interp(xs[i], ys[j], zs[k]), fs(i, j, k), 1e-10)
                    << "node (" << i << "," << j << "," << k << ")";
}

TEST_F(InterpTable4DTest, UnevenGridConstructorReproducesNodes) {
    Eigen::VectorXd xs(6);
    xs << 0.0, 0.2, 0.55, 1.0, 1.45, 2.0;
    Eigen::VectorXd ys(5);
    ys << 0.0, 0.3, 0.85, 1.4, 2.0;
    Eigen::VectorXd zs(5);
    zs << 0.0, 0.35, 0.9, 1.5, 2.0;
    Eigen::VectorXd ws(5);
    ws << 0.0, 0.4, 0.95, 1.5, 2.0;
    Eigen::Tensor<double, 4> fs(xs.size(), ys.size(), zs.size(), ws.size());
    for (int i = 0; i < xs.size(); ++i)
        for (int j = 0; j < ys.size(); ++j)
            for (int k = 0; k < zs.size(); ++k)
                for (int l = 0; l < ws.size(); ++l)
                    fs(i, j, k, l) = ref_fn_4d(xs[i], ys[j], zs[k], ws[l]);

    oc::InterpTable4D table(xs, ys, zs, ws, fs, InterpType::Cubic, false);

    for (int i : {0, 5})
        for (int j : {0, 2, 4})
            for (int k : {0, 4})
                for (int l : {0, 2, 4})
                    EXPECT_NEAR(table.interp(xs[i], ys[j], zs[k], ws[l]), fs(i, j, k, l), 1e-10)
                        << "node (" << i << "," << j << "," << k << "," << l << ")";
}

TEST_F(InterpTable4DTest, OutOfBoundsThrowsByDefault) {
    const int n = 5;
    auto xs = linspace_nd(0.0, 1.0, n);
    auto ys = linspace_nd(0.0, 1.0, n);
    auto zs = linspace_nd(0.0, 1.0, n);
    auto ws = linspace_nd(0.0, 1.0, n);
    Eigen::Tensor<double, 4> fs(n, n, n, n);
    fs.setZero();

    oc::InterpTable4D table(xs, ys, zs, ws, fs, InterpType::Cubic, false);

    EXPECT_THROW((void)table.interp(-0.1, 0.5, 0.5, 0.5), std::invalid_argument);
    EXPECT_THROW((void)table.interp(0.5, 0.5, 0.5, 1.2), std::invalid_argument);
}
