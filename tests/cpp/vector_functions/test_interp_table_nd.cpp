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
