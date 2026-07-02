// N-D ChebFunction VectorFunction wrapper tests (Task 3).
// Covers ChebFunction<2>, ChebFunction<3>, ChebFunction<-1>:
//   - Value, Jacobian via verify_jacobian_analytical
//   - Hessian symmetry + adjoint gradient via verify_hessian_consistency
//   - Composition with Arguments<N>
#include "test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>
#include <cmath>

using namespace tycho;
using namespace TychoTest;

class ChebNdVfTest : public VectorFunctionFixture {};

// Build a smooth 2-D table: f(x,y) = exp(sin(x)) * cos(y) on [-1,1]^2.
static std::shared_ptr<oc::ChebTable> smooth_2d() {
    std::vector<int> orders{8, 8};
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
    return std::make_shared<oc::ChebTable>(
        oc::ChebTable::from_values(vals, lb, ub, orders));
}

// Build a smooth 3-D table: f(x,y,z) = sin(x)*cos(y) + 0.3*z on [0,2]x[-1,1]x[1,3].
static std::shared_ptr<oc::ChebTable> smooth_3d() {
    std::vector<int> orders{6, 6, 6};
    Eigen::VectorXd lb(3), ub(3);
    lb << 0, -1, 1;
    ub << 2, 1, 3;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1, nz = orders[2] + 1;
    oc::ChebTable::MatType vals(nx * ny * nz, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                vals((i * ny + j) * nz + k, 0) =
                    std::sin(nodes[0][i]) * std::cos(nodes[1][j]) + 0.3 * nodes[2][k];
    return std::make_shared<oc::ChebTable>(
        oc::ChebTable::from_values(vals, lb, ub, orders));
}

// ChebFunction<2>: Jacobian analytical check and Hessian consistency.
TEST_F(ChebNdVfTest, Function2D_JacobianAndHessian) {
    auto tab = smooth_2d();
    oc::ChebFunction<2> f(tab);
    Eigen::VectorXd x(2);
    x << 0.3, -0.4;
    Eigen::MatrixXd expJ = tab->eval_jacobian(x);
    verify_jacobian_analytical(f, x, expJ, 1e-9);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(f, x, lm, 1e-9);
}

// ChebFunction<3>: Jacobian analytical check and Hessian consistency.
TEST_F(ChebNdVfTest, Function3D_JacobianAndHessian) {
    auto tab = smooth_3d();
    oc::ChebFunction<3> f(tab);
    Eigen::VectorXd x(3);
    x << 0.7, 0.3, 2.1;
    Eigen::MatrixXd expJ = tab->eval_jacobian(x);
    verify_jacobian_analytical(f, x, expJ, 1e-9);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(f, x, lm, 1e-9);
}

// ChebFunction<-1> (dynamic): Jacobian and Hessian with dynamic IR.
TEST_F(ChebNdVfTest, FunctionDynamic_JacobianAndHessian) {
    auto tab = smooth_2d();
    oc::ChebFunction<-1> f(tab);
    Eigen::VectorXd x(2);
    x << -0.2, 0.5;
    Eigen::MatrixXd expJ = tab->eval_jacobian(x);
    verify_jacobian_analytical(f, x, expJ, 1e-9);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(f, x, lm, 1e-9);
}

// Composition: ChebFunction<-1>(2d_tab).eval(Arguments<2>()) composes into a
// GenericFunction that reproduces tab->eval at off-grid points.
TEST_F(ChebNdVfTest, DynamicInputWrapperComposes) {
    auto tab = smooth_2d();
    auto g = vf::GenericFunction<-1, -1>(oc::ChebFunction<-1>(tab).eval(Arguments<2>()));
    Eigen::VectorXd x(2);
    x << 0.25, 0.1;
    Eigen::VectorXd out(1);
    out.setZero();
    g.compute(x, out);
    EXPECT_NEAR(out[0], tab->eval(x)[0], 1e-12);
}

// Adjoint Hessian = Sigma_o adjvars[o] * H_o:
// For a 2-channel table and lm=[2,3], the adjoint Hessian must equal
// 2*H_0 + 3*H_1. Build a 2-output table and verify by comparing
// verify_hessian_consistency with a non-unit lm.
TEST_F(ChebNdVfTest, MultichannelAdjointHessian) {
    std::vector<int> orders{8, 8};
    Eigen::VectorXd lb(2), ub(2);
    lb << -1, -1;
    ub << 1, 1;
    auto nodes = oc::ChebTable::cheb_points(orders, lb, ub);
    const int nx = orders[0] + 1, ny = orders[1] + 1;
    oc::ChebTable::MatType vals(nx * ny, 2);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            double xi = nodes[0][i], yi = nodes[1][j];
            vals(i * ny + j, 0) = std::exp(std::sin(xi)) * std::cos(yi);
            vals(i * ny + j, 1) = std::sin(xi) * std::sin(yi);
        }
    auto tab = std::make_shared<oc::ChebTable>(
        oc::ChebTable::from_values(vals, lb, ub, orders));
    oc::ChebFunction<2> f(tab);
    Eigen::VectorXd x(2);
    x << 0.4, -0.3;
    // Non-unit adjvars: adjoint Hessian = 2*H_0 + 3*H_1
    Eigen::VectorXd lm(2);
    lm << 2.0, 3.0;
    verify_hessian_consistency(f, x, lm, 1e-9);
}
