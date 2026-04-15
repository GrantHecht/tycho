// InterpTable1D behavioral tests — pins the enum-constructor, cubic/linear
// dispatch, even/uneven axis detection, and the default throw-on-out-of-bounds
// contract introduced by the PR #42 review fix.

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

#include <cmath>

using namespace tycho;
using namespace TychoTest;

namespace {

Eigen::VectorXd linspace_1d(double a, double b, int n) {
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v[i] = a + (b - a) * double(i) / double(n - 1);
    return v;
}

// Smooth reference that cubic should track to high precision and linear
// should track to the piecewise-linear error floor.
double ref_fn_1d(double t) { return std::sin(0.7 * t) + 0.3 * t; }

} // namespace

class InterpTable1DTest : public VectorFunctionFixture {};

TEST_F(InterpTable1DTest, CubicScalarReproducesAnalyticFunction) {
    const int n = 41;
    auto ts = linspace_1d(0.0, 6.0, n);
    Eigen::VectorXd vs(n);
    for (int i = 0; i < n; ++i) vs[i] = ref_fn_1d(ts[i]);

    oc::InterpTable1D table(ts, vs, InterpType::Cubic);

    for (double t : {0.0, 0.37, 1.85, 3.14159, 4.6, 5.999}) {
        Eigen::VectorXd v = table.interp(t);
        EXPECT_EQ(v.size(), 1);
        EXPECT_NEAR(v[0], ref_fn_1d(t), 1e-5) << "cubic mismatch at t=" << t;
    }
}

TEST_F(InterpTable1DTest, LinearScalarMatchesAtNodes) {
    const int n = 21;
    auto ts = linspace_1d(-1.0, 1.0, n);
    Eigen::VectorXd vs(n);
    for (int i = 0; i < n; ++i) vs[i] = ref_fn_1d(ts[i]);

    oc::InterpTable1D table(ts, vs, InterpType::Linear);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd v = table.interp(ts[i]);
        EXPECT_NEAR(v[0], vs[i], 1e-12) << "node " << i;
    }
    // Midpoint of a linear segment is the arithmetic mean — exact.
    for (int i = 0; i < n - 1; ++i) {
        double tm = 0.5 * (ts[i] + ts[i + 1]);
        double vm = 0.5 * (vs[i] + vs[i + 1]);
        Eigen::VectorXd v = table.interp(tm);
        EXPECT_NEAR(v[0], vm, 1e-12);
    }
}

TEST_F(InterpTable1DTest, UnevenGridDetectedAndEvaluates) {
    Eigen::VectorXd ts(7);
    ts << 0.0, 0.3, 0.7, 1.5, 2.1, 3.0, 4.4;
    Eigen::VectorXd vs(7);
    for (int i = 0; i < 7; ++i) vs[i] = ref_fn_1d(ts[i]);

    oc::InterpTable1D table(ts, vs, InterpType::Cubic);

    // Interior nodes reproduce exactly regardless of spacing — also verifies
    // that uneven-grid construction did not silently fall back to the even
    // path (which would mis-place the nodes and break this assertion).
    for (int i = 0; i < 7; ++i) {
        Eigen::VectorXd v = table.interp(ts[i]);
        EXPECT_NEAR(v[0], vs[i], 1e-12) << "node " << i;
    }
}

TEST_F(InterpTable1DTest, OutOfBoundsThrowsByDefault) {
    auto ts = linspace_1d(0.0, 1.0, 5);
    Eigen::VectorXd vs(5);
    for (int i = 0; i < 5; ++i) vs[i] = ref_fn_1d(ts[i]);

    oc::InterpTable1D table(ts, vs, InterpType::Cubic);

    EXPECT_THROW((void)table.interp(-0.1), std::invalid_argument);
    EXPECT_THROW((void)table.interp(1.5), std::invalid_argument);
}

TEST_F(InterpTable1DTest, MatrixConstructorAxisSelection) {
    // Two-component value function, stored as 2 x tsize.
    const int n = 17;
    auto ts = linspace_1d(0.0, 2.0, n);
    Eigen::MatrixXd vs(2, n);
    for (int i = 0; i < n; ++i) {
        vs(0, i) = ref_fn_1d(ts[i]);
        vs(1, i) = std::cos(ts[i]);
    }

    oc::InterpTable1D table(ts, vs, 1, InterpType::Cubic);

    double t = 1.234;
    Eigen::VectorXd v = table.interp(t);
    ASSERT_EQ(v.size(), 2);
    EXPECT_NEAR(v[0], ref_fn_1d(t), 1e-5);
    EXPECT_NEAR(v[1], std::cos(t), 1e-5);

    // Transposed axis: store tsize x 2 and set axis=0.
    Eigen::MatrixXd vs_t = vs.transpose();
    oc::InterpTable1D table_t(ts, vs_t, 0, InterpType::Cubic);
    Eigen::VectorXd v_t = table_t.interp(t);
    ASSERT_EQ(v_t.size(), 2);
    EXPECT_NEAR(v_t[0], ref_fn_1d(t), 1e-5);
    EXPECT_NEAR(v_t[1], std::cos(t), 1e-5);
}

TEST_F(InterpTable1DTest, CubicFirstDerivativeMatchesAnalytic) {
    const int n = 81;
    auto ts = linspace_1d(0.0, 6.0, n);
    Eigen::VectorXd vs(n);
    for (int i = 0; i < n; ++i) vs[i] = ref_fn_1d(ts[i]);

    oc::InterpTable1D table(ts, vs, InterpType::Cubic);
    for (double t : {0.5, 1.7, 2.9, 4.1, 5.2}) {
        auto [v, dv] = table.interp_deriv1(t);
        double dv_ref = 0.7 * std::cos(0.7 * t) + 0.3;
        EXPECT_NEAR(v[0], ref_fn_1d(t), 1e-5);
        EXPECT_NEAR(dv[0], dv_ref, 2e-3) << "derivative mismatch at t=" << t;
    }
}
