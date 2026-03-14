///////////////////////////////////////////////////////////////////////////////
// VectorFunction DSL tests
//
// Tests the core VF expression system, autodiff jacobians, and type erasure
// via GenericFunction. Requires MemoryManager initialization before compute
// calls.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <gtest/gtest.h>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// Brachistochrone ODE — same as examples/cpp_examples/brachistochrone
///////////////////////////////////////////////////////////////////////////////

struct BrachODE_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>(); // [x, y, v, t, theta]
        auto v = args.coeff<2>();
        auto theta = args.coeff<4>();

        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);

        return StackedOutputs{xdot, ydot, vdot};
    }
};

BUILD_ODE_FROM_EXPRESSION(BrachODE, BrachODE_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Test fixture — initialise MemoryManager once for the suite
///////////////////////////////////////////////////////////////////////////////

class VectorFunctionTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() { MemoryManager::resize(64, 64); }
};

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(VectorFunctionTest, BrachistochroneDimensions) {
    BrachODE ode(9.81);
    EXPECT_EQ(ode.IRows(), 5);
    EXPECT_EQ(ode.ORows(), 3);
}

TEST_F(VectorFunctionTest, BrachistochroneComputeKnownValues) {
    constexpr double g = 9.81;
    BrachODE ode(g);

    // Input: [x=0, y=10, v=5, t=0, theta=pi/4]
    Eigen::VectorXd x(5);
    x << 0.0, 10.0, 5.0, 0.0, M_PI / 4.0;

    Eigen::VectorXd fx(3);
    fx.setZero();
    Eigen::MatrixXd jx(3, 5);
    jx.setZero();

    ode.compute_jacobian(x, fx, jx);

    // xdot = sin(pi/4) * 5 = 5/sqrt(2) ≈ 3.5355
    EXPECT_NEAR(fx[0], 5.0 * std::sin(M_PI / 4.0), 1e-12);
    // ydot = -cos(pi/4) * 5 = -5/sqrt(2) ≈ -3.5355
    EXPECT_NEAR(fx[1], -5.0 * std::cos(M_PI / 4.0), 1e-12);
    // vdot = g * cos(pi/4) ≈ 6.937
    EXPECT_NEAR(fx[2], g * std::cos(M_PI / 4.0), 1e-12);
}

TEST_F(VectorFunctionTest, JacobianVsFiniteDifference) {
    constexpr double g = 9.81;
    BrachODE ode(g);

    Eigen::VectorXd x(5);
    x << 1.0, 8.0, 3.0, 0.5, 0.7;

    // Autodiff jacobian
    Eigen::VectorXd fx(3);
    fx.setZero();
    Eigen::MatrixXd jx(3, 5);
    jx.setZero();
    ode.compute_jacobian(x, fx, jx);

    // Central finite-difference jacobian
    constexpr double h = 1e-7;
    Eigen::MatrixXd jx_fd(3, 5);
    for (int j = 0; j < 5; ++j) {
        Eigen::VectorXd xp = x, xm = x;
        xp[j] += h;
        xm[j] -= h;

        Eigen::VectorXd fxp(3), fxm(3);
        fxp.setZero();
        fxm.setZero();
        Eigen::MatrixXd dummy(3, 5);
        dummy.setZero();

        ode.compute_jacobian(xp, fxp, dummy);
        dummy.setZero();
        ode.compute_jacobian(xm, fxm, dummy);

        jx_fd.col(j) = (fxp - fxm) / (2.0 * h);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(jx(i, j), jx_fd(i, j), 1e-5)
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(VectorFunctionTest, GenericFunctionCopyEvaluate) {
    constexpr double g = 9.81;
    BrachODE ode(g);

    // Erase into GenericFunction<-1,-1>
    GenericFunction<-1, -1> gf(ode);
    EXPECT_EQ(gf.IRows(), 5);
    EXPECT_EQ(gf.ORows(), 3);

    // Copy the generic function
    GenericFunction<-1, -1> gf2(gf);

    // Evaluate both at the same input
    Eigen::VectorXd x(5);
    x << 2.0, 7.0, 4.0, 0.3, 1.2;

    Eigen::VectorXd fx1(3), fx2(3);
    fx1.setZero();
    fx2.setZero();
    Eigen::MatrixXd jx1(3, 5), jx2(3, 5);
    jx1.setZero();
    jx2.setZero();

    gf.compute_jacobian(x, fx1, jx1);
    gf2.compute_jacobian(x, fx2, jx2);

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(fx1[i], fx2[i]) << "Output mismatch at " << i;
        for (int j = 0; j < 5; ++j) {
            EXPECT_DOUBLE_EQ(jx1(i, j), jx2(i, j))
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}
