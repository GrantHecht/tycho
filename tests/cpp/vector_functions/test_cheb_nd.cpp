// N-D ChebTable value evaluation tests (Task 1).
// Tests for N-D cheb_points, from_values (separable DCT-I), and tensor-Clenshaw eval.
#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>
#include <cmath>

using namespace tycho;
using namespace TychoTest;

// Helper: Chebyshev polynomial T_k(x)
static double Tk(int k, double x) { return std::cos(k * std::acos(std::clamp(x, -1.0, 1.0))); }

class ChebNdTest : public VectorFunctionFixture {};

// Build a 2-D table of f(x,y) = T2(x)*T3(y) on [-1,1]^2 and check exact reproduction.
TEST_F(ChebNdTest, TensorProductPolyExact2D) {
    std::vector<int> orders{4, 5};
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);  // vector<VectorXd>, per axis
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);  // (tsize, olen=1), row-major (x outer, y inner)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            vals(i * ny + j, 0) = Tk(2, nodes[0][i]) * Tk(3, nodes[1][j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders);
    ASSERT_EQ(tab.input_dim(), 2);
    for (double x : {-0.6, 0.1, 0.7})
        for (double y : {-0.3, 0.4, 0.9}) {
            Eigen::VectorXd q(2);
            q << x, y;
            EXPECT_NEAR(tab.eval(q)[0], Tk(2, x) * Tk(3, y), 1e-10) << x << "," << y;
        }
}

// 3-D round-trip at grid nodes for a smooth function.
TEST_F(ChebNdTest, RoundTrip3D) {
    std::vector<int> orders{6, 5, 4};
    Eigen::VectorXd lb(3), ub(3);
    lb << 0, -1, 1;
    ub << 2, 1, 3;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    auto f = [](double x, double y, double z) { return std::sin(x) * std::cos(y) + 0.3 * z; };
    const int nx = orders[0] + 1, ny = orders[1] + 1, nz = orders[2] + 1;
    oc::ChebTable::MatType vals(nx * ny * nz, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                vals((i * ny + j) * nz + k, 0) = f(nodes[0][i], nodes[1][j], nodes[2][k]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                Eigen::VectorXd q(3);
                q << nodes[0][i], nodes[1][j], nodes[2][k];
                EXPECT_NEAR(tab.eval(q)[0], f(nodes[0][i], nodes[1][j], nodes[2][k]), 1e-11);
            }
}
