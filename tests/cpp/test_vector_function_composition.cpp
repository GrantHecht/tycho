///////////////////////////////////////////////////////////////////////////////
// VectorFunction composition / integration tests
//
// Exercises the VF DSL as users actually use it: building ODE expressions,
// composing functions, and verifying adjoint/hessian correctness.
///////////////////////////////////////////////////////////////////////////////

#include "Astro/CR3BPModel.h"
#include "Astro/KeplerModel.h"
#include "Tycho.h"
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class VFCompositionTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// ODE expression tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, BrachistochroneAdjointConsistency) {
    BrachODE ode(9.81);
    Eigen::VectorXd x = deterministic_random_vector(5, 200, 0.1, 5.0);
    Eigen::VectorXd lm = deterministic_random_vector(3, 201, -1.0, 1.0);
    verify_adjoint_consistency(ode, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, BrachistochroneGenericODEErasure) {
    BrachODE ode(9.81);
    GenericFunction<-1, -1> gf(ode);
    EXPECT_EQ(gf.IRows(), 5);
    EXPECT_EQ(gf.ORows(), 3);

    Eigen::VectorXd x(5);
    x << 0, 10, 5, 0, M_PI / 4.0;
    Eigen::VectorXd fx_orig(3), fx_gf(3);
    fx_orig.setZero();
    fx_gf.setZero();
    ode.compute(x, fx_orig);
    gf.compute(x, fx_gf);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(fx_orig[i], fx_gf[i], 1e-14);
    }
}

TEST_F(VFCompositionTest, KeplerODEDimensions) {
    Kepler kep(398600.4418);
    EXPECT_EQ(kep.IRows(), 7); // [x,y,z,vx,vy,vz,t]
    EXPECT_EQ(kep.ORows(), 6); // [dx,dy,dz,dvx,dvy,dvz]
}

TEST_F(VFCompositionTest, KeplerODEAdjointConsistency) {
    Kepler kep(398600.4418);
    Eigen::VectorXd x(7);
    x << 7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0;
    Eigen::VectorXd lm = deterministic_random_vector(6, 202, -1.0, 1.0);
    verify_adjoint_consistency(kep, x, lm, 1e-8);
}

TEST_F(VFCompositionTest, CR3BPODEAdjointConsistency) {
    double mu = 0.012150585; // Earth-Moon
    CR3BP cr3bp(mu);
    EXPECT_EQ(cr3bp.IRows(), 7);
    EXPECT_EQ(cr3bp.ORows(), 6);

    Eigen::VectorXd x(7);
    x << 0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0;
    Eigen::VectorXd lm = deterministic_random_vector(6, 203, -1.0, 1.0);
    verify_adjoint_consistency(cr3bp, x, lm, 1e-10);
}

///////////////////////////////////////////////////////////////////////////////
// Complex composition chains
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, FiveNestingLevels) {
    auto args = Arguments<4>();
    auto l1 = 2.0 * args;
    auto l2 = l1.Sin();
    auto l3 = l2.Square();
    auto l4 = l3.norm();
    auto l5 = 3.0 * l3;

    Eigen::VectorXd x = deterministic_random_vector(4, 210, 0.5, 3.0);
    Eigen::VectorXd lm4(1);
    lm4 << 1.0;
    verify_adjoint_consistency(l4, x, lm4, 1e-10);

    Eigen::VectorXd lm5 = deterministic_random_vector(4, 211, -1.0, 1.0);
    verify_adjoint_consistency(l5, x, lm5, 1e-10);
}

TEST_F(VFCompositionTest, MixedArithmeticComposition) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto cw = a.cwiseProduct(b); // element-wise
    auto n = cw.norm();          // scalar
    auto dp = a.dot(b);          // scalar

    Eigen::VectorXd x = deterministic_random_vector(6, 220, 0.5, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(n, x, lm, 1e-10);
    verify_adjoint_consistency(dp, x, lm, 1e-10);
}

TEST_F(VFCompositionTest, NormOfComposition) {
    auto args = Arguments<3>();
    auto sq = args.Square();
    auto n = sq.norm();
    // ||x^2|| at x=[3,4,0]: ||[9,16,0]|| = sqrt(81+256) = sqrt(337)
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 0.0;
    Eigen::VectorXd fx(1);
    fx.setZero();
    n.compute(x, fx);
    EXPECT_NEAR(fx[0], std::sqrt(337.0), 1e-10);
}

TEST_F(VFCompositionTest, DotOfCompositions) {
    auto args = Arguments<6>();
    auto a = args.template head<3>().Sin();
    auto b = args.template tail<3>().Cos();
    auto dp = a.dot(b);
    Eigen::VectorXd x = deterministic_random_vector(6, 221, 0.1, 3.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_adjoint_consistency(dp, x, lm, 1e-10);
}

///////////////////////////////////////////////////////////////////////////////
// Hessian / adjoint integration tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFCompositionTest, NormHessianSymmetry) {
    auto args = Arguments<3>();
    auto n = args.norm();
    Eigen::VectorXd x = deterministic_random_vector(3, 230, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(n, x, lm, 1e-11);
}

TEST_F(VFCompositionTest, SquaredNormHessian) {
    auto args = Arguments<3>();
    auto sn = args.squared_norm();
    Eigen::VectorXd x = deterministic_random_vector(3, 231, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    // Hessian of ||x||^2 = 2*I
    const int ir = 3;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, ir);
    Eigen::VectorXd gx(ir);
    Eigen::MatrixXd hx(ir, ir);
    fx.setZero();
    jx.setZero();
    gx.setZero();
    hx.setZero();
    sn.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

    // Hessian should be 2*I
    for (int i = 0; i < ir; ++i) {
        for (int j = 0; j < ir; ++j) {
            double expected = (i == j) ? 2.0 : 0.0;
            EXPECT_NEAR(hx(i, j), expected, 1e-11)
                << "Hessian mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(VFCompositionTest, DotProductHessian) {
    auto args = Arguments<6>();
    auto a = args.template head<3>();
    auto b = args.template tail<3>();
    auto dp = a.dot(b);
    Eigen::VectorXd x = deterministic_random_vector(6, 232, 1.0, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;

    // Hessian of a^T b: H = [0, I; I, 0] (6x6 block matrix)
    const int ir = 6;
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, ir);
    Eigen::VectorXd gx(ir);
    Eigen::MatrixXd hx(ir, ir);
    fx.setZero();
    jx.setZero();
    gx.setZero();
    hx.setZero();
    dp.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

    // Check structure: top-left 3x3 = 0, top-right 3x3 = I, etc.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(hx(i, j), 0.0, 1e-12);         // top-left
            EXPECT_NEAR(hx(i + 3, j + 3), 0.0, 1e-12); // bottom-right
            double expected_cross = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(hx(i, j + 3), expected_cross, 1e-12); // top-right
            EXPECT_NEAR(hx(i + 3, j), expected_cross, 1e-12); // bottom-left
        }
    }
}

TEST_F(VFCompositionTest, BrachistochroneHessianSymmetry) {
    BrachODE ode(9.81);
    // Need a scalar function for hessian — use squared norm of ODE output
    auto sq = ode.squared_norm();
    Eigen::VectorXd x = deterministic_random_vector(5, 233, 0.1, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(sq, x, lm, 1e-10);
}

TEST_F(VFCompositionTest, NestedScalarHessianSymmetry) {
    auto args = Arguments<4>();
    auto sq = args.Square();
    auto n = sq.norm(); // scalar
    Eigen::VectorXd x = deterministic_random_vector(4, 234, 0.5, 5.0);
    Eigen::VectorXd lm(1);
    lm << 1.0;
    verify_hessian_consistency(n, x, lm, 1e-10);
}
