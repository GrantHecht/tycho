// ChebTable construction + evaluation tests.
// Task 1: node generation, DCT-I coefficients, round-trip.
// Task 2: derivative/multichannel correctness.

#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>
#include <cmath>

using namespace tycho;
using namespace TychoTest;

class ChebTableTest : public VectorFunctionFixture {};

// Chebyshev T_k via cos(k*acos(x)) for x in [-1,1].
static double Tk(int k, double x) { return std::cos(k * std::acos(x)); }

TEST_F(ChebTableTest, ChebPointsEndpointsAndCount) {
    auto pts = oc::ChebTable::cheb_points(4, -2.0, 6.0);  // n=4 -> 5 nodes
    ASSERT_EQ(pts.size(), 5);
    EXPECT_NEAR(pts[0], 6.0, 1e-12);   // xi_0 = +1 -> ub
    EXPECT_NEAR(pts[4], -2.0, 1e-12);  // xi_4 = -1 -> lb
}

TEST_F(ChebTableTest, ReproducesT3CoefficientsExactly) {
    const int n = 6;
    const double lb = -1.0, ub = 1.0;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(1, n + 1);                // 1 output channel, n+1 samples
    for (int j = 0; j <= n; ++j) vals(0, j) = Tk(3, pts[j]);  // f = T_3 on [-1,1]
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n);
    // Eval at a few points must match T_3 exactly (interp of a low-degree poly).
    for (double t : {-0.9, -0.3, 0.25, 0.8}) {
        EXPECT_NEAR(tab.eval(t)[0], Tk(3, t), 1e-11) << "t=" << t;
    }
}

TEST_F(ChebTableTest, RoundTripReproducesNodeValues) {
    const int n = 24;
    const double lb = 0.0, ub = 6.0;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    auto f = [](double t) { return std::sin(0.7 * t) + 0.3 * t; };
    oc::ChebTable::MatType vals(1, n + 1);
    for (int j = 0; j <= n; ++j) vals(0, j) = f(pts[j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n);
    for (int j = 0; j <= n; ++j)
        EXPECT_NEAR(tab.eval(pts[j])[0], vals(0, j), 1e-12) << "node " << j;
}

TEST_F(ChebTableTest, FirstAndSecondDerivativeMatchAnalytic) {
    const int n = 30;
    const double lb = 0.0, ub = 6.0;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    auto f   = [](double t) { return std::sin(0.7 * t) + 0.3 * t; };
    auto df  = [](double t) { return 0.7 * std::cos(0.7 * t) + 0.3; };
    auto d2f = [](double t) { return -0.49 * std::sin(0.7 * t); };
    oc::ChebTable::MatType vals(1, n + 1);
    for (int j = 0; j <= n; ++j) vals(0, j) = f(pts[j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n);

    for (double t : {0.5, 1.7, 2.9, 4.1, 5.2}) {
        auto [v, dv]       = tab.eval_deriv1(t);
        auto [v2, dv2, d2] = tab.eval_deriv2(t);
        EXPECT_NEAR(v[0],  f(t),   1e-9)  << "t=" << t;
        EXPECT_NEAR(dv[0], df(t),  1e-7)  << "t=" << t;
        EXPECT_NEAR(d2[0], d2f(t), 1e-5)  << "t=" << t;
        // central-difference cross-check on dv
        const double e = 1e-5;
        double fd = (f(t + e) - f(t - e)) / (2 * e);
        EXPECT_NEAR(dv[0], fd, 1e-6) << "FD t=" << t;
    }
}

TEST_F(ChebTableTest, MultiChannel) {
    const int n = 20;
    const double lb = -1.0, ub = 1.0;
    auto pts = oc::ChebTable::cheb_points(n, lb, ub);
    oc::ChebTable::MatType vals(2, n + 1);
    for (int j = 0; j <= n; ++j) { vals(0, j) = Tk(2, pts[j]); vals(1, j) = Tk(5, pts[j]); }
    auto tab = oc::ChebTable::from_values(vals, lb, ub, n);
    ASSERT_EQ(tab.output_dim(), 2);
    for (double t : {-0.4, 0.1, 0.7}) {
        auto v = tab.eval(t);
        EXPECT_NEAR(v[0], Tk(2, t), 1e-10);
        EXPECT_NEAR(v[1], Tk(5, t), 1e-10);
    }
}
