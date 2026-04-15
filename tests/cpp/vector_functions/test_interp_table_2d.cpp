// InterpTable2D behavioral tests — pins cubic/linear dispatch and the
// default throw-on-out-of-bounds contract for the 2D interpolation surface.

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

#include <cmath>

using namespace tycho;
using namespace TychoTest;

namespace {

Eigen::VectorXd linspace_2d(double a, double b, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v[i] = a + (b - a) * double(i) / double(n - 1);
    return v;
}

// Smooth bivariate reference — cubic should track to <1e-4, linear to the
// per-cell error floor.
double ref_fn_2d(double x, double y) { return std::sin(0.8 * x) * std::cos(0.6 * y) + 0.1 * x * y; }

oc::InterpTable2D::MatType eval_grid_2d(const Eigen::VectorXd &xs, const Eigen::VectorXd &ys) {
    oc::InterpTable2D::MatType z(ys.size(), xs.size());
    for (int j = 0; j < ys.size(); ++j)
        for (int i = 0; i < xs.size(); ++i) z(j, i) = ref_fn_2d(xs[i], ys[j]);
    return z;
}

} // namespace

class InterpTable2DTest : public VectorFunctionFixture {};

TEST_F(InterpTable2DTest, CubicReproducesAnalyticSurface) {
    auto xs = linspace_2d(0.0, 3.0, 13);
    auto ys = linspace_2d(-1.0, 2.0, 11);
    auto zs = eval_grid_2d(xs, ys);

    oc::InterpTable2D table(xs, ys, zs, InterpType::Cubic);

    for (auto [x, y] : {std::pair{0.37, 0.25}, {1.42, -0.5}, {2.71, 1.33}, {0.9, 1.8}}) {
        double z = table.interp(x, y);
        EXPECT_NEAR(z, ref_fn_2d(x, y), 5e-4) << "cubic mismatch at (" << x << "," << y << ")";
    }
}

TEST_F(InterpTable2DTest, LinearMatchesAtNodesAndCellMidpoint) {
    auto xs = linspace_2d(0.0, 2.0, 9);
    auto ys = linspace_2d(0.0, 2.0, 9);
    auto zs = eval_grid_2d(xs, ys);

    oc::InterpTable2D table(xs, ys, zs, InterpType::Linear);

    // Nodes exact.
    for (int j = 0; j < ys.size(); ++j)
        for (int i = 0; i < xs.size(); ++i)
            EXPECT_NEAR(table.interp(xs[i], ys[j]), zs(j, i), 1e-12);

    // Cell midpoint — linear bilinear is mean of 4 corners.
    for (int j = 0; j < ys.size() - 1; ++j)
        for (int i = 0; i < xs.size() - 1; ++i) {
            double xm = 0.5 * (xs[i] + xs[i + 1]);
            double ym = 0.5 * (ys[j] + ys[j + 1]);
            double zm = 0.25 * (zs(j, i) + zs(j, i + 1) + zs(j + 1, i) + zs(j + 1, i + 1));
            EXPECT_NEAR(table.interp(xm, ym), zm, 1e-12);
        }
}

TEST_F(InterpTable2DTest, OutOfBoundsThrowsByDefault) {
    auto xs = linspace_2d(0.0, 1.0, 6);
    auto ys = linspace_2d(0.0, 1.0, 6);
    auto zs = eval_grid_2d(xs, ys);

    oc::InterpTable2D table(xs, ys, zs, InterpType::Cubic);

    EXPECT_THROW((void)table.interp(-0.1, 0.5), std::invalid_argument);
    EXPECT_THROW((void)table.interp(0.5, 1.2), std::invalid_argument);
    EXPECT_THROW((void)table.interp(1.5, 1.5), std::invalid_argument);
}

TEST_F(InterpTable2DTest, UnevenGridDetectedAndEvaluates) {
    Eigen::VectorXd xs(6);
    xs << 0.0, 0.2, 0.55, 1.1, 1.8, 2.9;
    Eigen::VectorXd ys(6);
    ys << -1.0, -0.3, 0.1, 0.6, 1.2, 2.0;
    auto zs = eval_grid_2d(xs, ys);

    oc::InterpTable2D table(xs, ys, zs, InterpType::Cubic);

    // Node values exact regardless of spacing — also verifies that the
    // uneven-grid construction did not silently fall back to the even path
    // (which would mis-place nodes and break this assertion).
    for (int j = 0; j < ys.size(); ++j)
        for (int i = 0; i < xs.size(); ++i)
            EXPECT_NEAR(table.interp(xs[i], ys[j]), zs(j, i), 1e-10);
}

TEST_F(InterpTable2DTest, ConstructorRejectsTooSmallAxes) {
    Eigen::VectorXd xs = linspace_2d(0.0, 1.0, 4); // < 5
    Eigen::VectorXd ys = linspace_2d(0.0, 1.0, 6);
    oc::InterpTable2D::MatType zs(6, 4);
    zs.setZero();
    EXPECT_THROW(oc::InterpTable2D(xs, ys, zs, InterpType::Cubic), std::invalid_argument);
}

TEST_F(InterpTable2DTest, CubicFirstDerivativeMatchesFiniteDifference) {
    auto xs = linspace_2d(0.0, 3.0, 21);
    auto ys = linspace_2d(-1.0, 2.0, 17);
    auto zs = eval_grid_2d(xs, ys);

    oc::InterpTable2D table(xs, ys, zs, InterpType::Cubic);
    const double h = 1e-5;
    for (auto [x, y] : {std::pair{1.42, 0.5}, {0.9, 1.1}, {2.3, -0.25}}) {
        auto [v, grad] = table.interp_deriv1(x, y);
        double dzdx_fd = (table.interp(x + h, y) - table.interp(x - h, y)) / (2 * h);
        double dzdy_fd = (table.interp(x, y + h) - table.interp(x, y - h)) / (2 * h);
        EXPECT_NEAR(v, table.interp(x, y), 1e-12);
        EXPECT_NEAR(grad[0], dzdx_fd, 1e-6) << "∂z/∂x FD mismatch at (" << x << "," << y << ")";
        EXPECT_NEAR(grad[1], dzdy_fd, 1e-6) << "∂z/∂y FD mismatch at (" << x << "," << y << ")";
    }
}

TEST_F(InterpTable2DTest, ConstructorRejectsMismatchedShape) {
    auto xs = linspace_2d(0.0, 1.0, 6);
    auto ys = linspace_2d(0.0, 1.0, 5);
    oc::InterpTable2D::MatType bad(6, 5); // rows/cols swapped
    bad.setZero();
    EXPECT_THROW(oc::InterpTable2D(xs, ys, bad, InterpType::Cubic), std::invalid_argument);
}
