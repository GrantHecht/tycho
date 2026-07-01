// N-D ChebTable value evaluation tests (Task 1) and Jacobian tests (Task 2).
// Tests for N-D cheb_points, from_values (separable DCT-I), tensor-Clenshaw eval,
// multichannel value eval, and analytic Jacobian vs FD / closed-form.
#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>
#include <cmath>

using namespace tycho;
using namespace TychoTest;

// Helper: Chebyshev polynomial T_k(x)
static double Tk(int k, double x) { return std::cos(k * std::acos(std::clamp(x, -1.0, 1.0))); }

// Local finite-difference Jacobian helper (Task 2 plan Step 1)
static Eigen::MatrixXd fd_jacobian(const oc::ChebTable &t, const Eigen::VectorXd &x,
                                   double e = 1e-5) {
    Eigen::MatrixXd J(t.output_dim(), x.size());
    for (int d = 0; d < x.size(); ++d) {
        Eigen::VectorXd xp = x, xm = x;
        xp[d] += e;
        xm[d] -= e;
        J.col(d) = (t.eval(xp) - t.eval(xm)) / (2 * e);
    }
    return J;
}

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

// Task 1 review carryover: multichannel (olen=2) 2-D value test.
// Two independent functions stacked as 2 columns: f0(x,y)=T2(x)*T3(y), f1(x,y)=sin(x)*cos(y).
TEST_F(ChebNdTest, Multichannel2D) {
    std::vector<int> orders{6, 6};
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 2);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            int row = i * ny + j;
            // Two DISTINCT tensor-product polynomials (degrees <= 6, exactly
            // representable at order 6), so the two channels carry independent
            // coefficients: a channel-mixing bug fails hard at 1e-10. A
            // transcendental channel would only interpolate to ~1e-6 off-grid
            // and mask that with interpolation error.
            vals(row, 0) = Tk(2, nodes[0][i]) * Tk(3, nodes[1][j]);
            vals(row, 1) = Tk(4, nodes[0][i]) * Tk(1, nodes[1][j]);
        }
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders);
    ASSERT_EQ(tab.input_dim(), 2);
    ASSERT_EQ(tab.output_dim(), 2);
    for (double x : {-0.7, 0.0, 0.5})
        for (double y : {-0.4, 0.3, 0.8}) {
            Eigen::VectorXd q(2);
            q << x, y;
            auto v = tab.eval(q);
            EXPECT_NEAR(v[0], Tk(2, x) * Tk(3, y), 1e-10) << "ch0 at " << x << "," << y;
            EXPECT_NEAR(v[1], Tk(4, x) * Tk(1, y), 1e-10) << "ch1 at " << x << "," << y;
        }
}

// Task 2: Jacobian of smooth 2-D function vs central differences to 1e-6.
// f(x,y) = exp(sin(x)) * cos(y)
TEST_F(ChebNdTest, JacobianVsFD_Smooth2D) {
    std::vector<int> orders{10, 10};
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            vals(i * ny + j, 0) =
                std::exp(std::sin(nodes[0][i])) * std::cos(nodes[1][j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders);

    for (double x : {-0.5, 0.0, 0.6}) {
        for (double y : {-0.3, 0.4, 0.7}) {
            Eigen::VectorXd q(2);
            q << x, y;
            Eigen::MatrixXd Janalytic = tab.eval_jacobian(q);
            Eigen::MatrixXd Jfd = fd_jacobian(tab, q);
            EXPECT_NEAR(Janalytic(0, 0), Jfd(0, 0), 1e-6)
                << "df/dx FD mismatch at " << x << "," << y;
            EXPECT_NEAR(Janalytic(0, 1), Jfd(0, 1), 1e-6)
                << "df/dy FD mismatch at " << x << "," << y;
        }
    }
}

// Jacobian on a NON-UNIT domain so hinv[d]=2/(ub-lb) != 1 on each axis.
// Every other Jacobian test uses [-1,1] where hinv==1, so a missing or wrong
// chain-rule factor would go undetected. Span={4,3} -> hinv={0.5, 2/3}. The FD
// reference is the physical-coordinate derivative of the SAME interpolant, so it
// pins the scaling independent of interpolation accuracy.
TEST_F(ChebNdTest, JacobianNonUnitDomain) {
    std::vector<int> orders{8, 8};
    Eigen::VectorXd lb(2), ub(2);
    lb << 0, -1;
    ub << 4, 2;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            vals(i * ny + j, 0) = std::exp(std::sin(nodes[0][i])) * std::cos(nodes[1][j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders);

    for (double x : {0.5, 1.7, 3.2}) {
        for (double y : {-0.6, 0.4, 1.5}) {
            Eigen::VectorXd q(2);
            q << x, y;
            Eigen::MatrixXd Janalytic = tab.eval_jacobian(q);
            Eigen::MatrixXd Jfd = fd_jacobian(tab, q);
            EXPECT_NEAR(Janalytic(0, 0), Jfd(0, 0), 1e-6)
                << "df/dx (hinv[0]=0.5) mismatch at " << x << "," << y;
            EXPECT_NEAR(Janalytic(0, 1), Jfd(0, 1), 1e-6)
                << "df/dy (hinv[1]=2/3) mismatch at " << x << "," << y;
        }
    }
}

// Task 2: Jacobian of T2(x)*T3(y) vs closed-form gradient to 1e-9.
// T2(x) = 2x^2 - 1,  T2'(x) = 4x
// T3(y) = 4y^3 - 3y, T3'(y) = 12y^2 - 3
// df/dx = T2'(x)*T3(y) = 4x * T3(y)
// df/dy = T2(x)*T3'(y) = T2(x) * (12y^2 - 3)
TEST_F(ChebNdTest, JacobianVsClosedForm_TensorPoly) {
    std::vector<int> orders{4, 5};
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            vals(i * ny + j, 0) = Tk(2, nodes[0][i]) * Tk(3, nodes[1][j]);
    auto tab = oc::ChebTable::from_values(vals, lb, ub, orders);

    auto T2 = [](double x) { return 2.0 * x * x - 1.0; };
    auto T3 = [](double y) { return 4.0 * y * y * y - 3.0 * y; };
    auto dT2 = [](double x) { return 4.0 * x; };            // T2'(x) = 4x
    auto dT3 = [](double y) { return 12.0 * y * y - 3.0; };  // T3'(y) = 12y^2-3

    for (double x : {-0.6, 0.1, 0.7}) {
        for (double y : {-0.3, 0.4, 0.9}) {
            Eigen::VectorXd q(2);
            q << x, y;
            Eigen::MatrixXd J = tab.eval_jacobian(q);
            double expected_dx = dT2(x) * T3(y);
            double expected_dy = T2(x) * dT3(y);
            EXPECT_NEAR(J(0, 0), expected_dx, 1e-9)
                << "df/dx closed-form mismatch at " << x << "," << y;
            EXPECT_NEAR(J(0, 1), expected_dy, 1e-9)
                << "df/dy closed-form mismatch at " << x << "," << y;
        }
    }
}
