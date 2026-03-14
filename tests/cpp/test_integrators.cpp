///////////////////////////////////////////////////////////////////////////////
// Integrator tests
//
// Tests Runge-Kutta integration using:
//   - Simple Harmonic Oscillator (x'' = -x, exact: x=cos(t), v=-sin(t))
//   - Kepler ODE (orbital mechanics)
//
// Validation: comparison against known analytical solutions and
// convergence order via h-refinement. NOT conservation checks.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include "Astro/KeplerModel.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace Tycho;
using namespace TychoTest;

///////////////////////////////////////////////////////////////////////////////
// Simple Harmonic Oscillator ODE: x'' = -x
// State: [x, v, t], augmented: [x, v, t], XVars=2, UVars=0, PVars=0
// ODE: dx/dt = v, dv/dt = -x
// Exact solution: x(t) = cos(t), v(t) = -sin(t) with x(0)=1, v(0)=0
///////////////////////////////////////////////////////////////////////////////

struct SHO_Impl : ODESize<2, 0, 0> {
    static auto Definition(double /*unused*/) {
        auto args = Arguments<3>(); // [x, v, t]
        auto x = args.coeff<0>();
        auto v = args.coeff<1>();
        auto xdot = v;
        auto vdot = (-1.0) * x;
        return StackedOutputs{xdot, vdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(SHO, SHO_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class IntegratorTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Basic integration
///////////////////////////////////////////////////////////////////////////////

TEST_F(IntegratorTest, SHOKnownSolutionAtPi) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0; // x=1, v=0, t=0

    auto xf = integ.integrate(x0, M_PI);
    // At t=pi: x=cos(pi)=-1, v=-sin(pi)=0
    EXPECT_NEAR(xf[0], -1.0, 1e-10);
    EXPECT_NEAR(xf[1], 0.0, 1e-10);
}

TEST_F(IntegratorTest, SHOKnownSolutionAt2Pi) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf = integ.integrate(x0, 2.0 * M_PI);
    // At t=2pi: x=cos(2pi)=1, v=-sin(2pi)=0
    EXPECT_NEAR(xf[0], 1.0, 1e-9);
    EXPECT_NEAR(xf[1], 0.0, 1e-9);
}

TEST_F(IntegratorTest, ForwardBackwardReversibility) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    // Integrate forward to t=1
    auto xf = integ.integrate(x0, 1.0);
    // Integrate backward from t=1 to t=0
    auto xr = integ.integrate(xf, 0.0);

    EXPECT_NEAR(xr[0], x0[0], 1e-9);
    EXPECT_NEAR(xr[1], x0[1], 1e-9);
}

TEST_F(IntegratorTest, BrachistochroneSmokeTest) {
    struct Brach_Impl : ODESize<3, 1, 0> {
        static auto Definition(double g) {
            auto args = Arguments<5>();
            auto v = args.coeff<2>();
            auto theta = args.coeff<4>();
            auto xdot = sin(theta) * v;
            auto ydot = cos(theta) * v * (-1.0);
            auto vdot = g * cos(theta);
            return StackedOutputs{xdot, ydot, vdot};
        }
    };
    BUILD_ODE_FROM_EXPRESSION(BrachInteg, Brach_Impl, double);

    BrachInteg ode(9.81);
    // Need to set control for integration
    Eigen::VectorXd u(1);
    u << M_PI / 4.0; // constant control
    Integrator<BrachInteg> integ(ode, 0.01, u);
    integ.setAbsTol(1e-12);

    Eigen::VectorXd x0(5);
    x0 << 0, 10, 0.1, 0, M_PI / 4.0;

    auto xf = integ.integrate(x0, 1.0);
    EXPECT_EQ(xf.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(xf[i])) << "Component " << i << " not finite";
    }
}

///////////////////////////////////////////////////////////////////////////////
// Convergence order tests
///////////////////////////////////////////////////////////////////////////////

static double sho_error(const std::string &method, double h) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, method, h);
    integ.Adaptive = false; // Fixed step

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 1.0;
    auto xf = integ.integrate(x0, tf);

    double x_exact = std::cos(tf);
    double v_exact = -std::sin(tf);
    return std::sqrt((xf[0] - x_exact) * (xf[0] - x_exact) +
                     (xf[1] - v_exact) * (xf[1] - v_exact));
}

TEST_F(IntegratorTest, DOPRI54ConvergenceOrder) {
    double h1 = 0.1, h2 = 0.05;
    double e1 = sho_error("DOPRI54", h1);
    double e2 = sho_error("DOPRI54", h2);

    // Expected order ~5 (uses DOPRI5 stepper): slope = log(e1/e2) / log(h1/h2)
    double slope = std::log(e1 / e2) / std::log(h1 / h2);
    EXPECT_GT(slope, 4.0) << "DOPRI54 convergence order too low: " << slope;
    EXPECT_LT(slope, 7.0) << "DOPRI54 convergence order unexpectedly high: " << slope;
}

TEST_F(IntegratorTest, DOPRI87ConvergenceOrder) {
    // Use larger steps to stay in the truncation-error regime (not roundoff).
    // The integrator internally recomputes step count and actual step size,
    // so the nominal h ratio is approximate; accept order >= 6.
    double h1 = 0.5, h2 = 0.25;
    double e1 = sho_error("DOPRI87", h1);
    double e2 = sho_error("DOPRI87", h2);

    double slope = std::log(e1 / e2) / std::log(h1 / h2);
    EXPECT_GT(slope, 6.0) << "DOPRI87 convergence order too low: " << slope;
    EXPECT_LT(slope, 11.0) << "DOPRI87 convergence order unexpectedly high: " << slope;
}

///////////////////////////////////////////////////////////////////////////////
// STM tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(IntegratorTest, SHOSTMVsAnalytical) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 1.0;

    auto [xf, stm] = integ.integrate_stm(x0, tf);

    // Analytical STM for SHO: rotation matrix
    // dx_f/dx_0 = cos(t),  dx_f/dv_0 = sin(t)
    // dv_f/dx_0 = -sin(t), dv_f/dv_0 = cos(t)
    double ct = std::cos(tf);
    double st = std::sin(tf);

    // STM is partial derivatives of [x,v] at tf w.r.t. [x,v,t] at t0
    // The first 2x2 block is the state transition
    EXPECT_NEAR(stm(0, 0), ct, 1e-8);  // dx/dx0
    EXPECT_NEAR(stm(0, 1), st, 1e-8);  // dx/dv0
    EXPECT_NEAR(stm(1, 0), -st, 1e-8); // dv/dx0
    EXPECT_NEAR(stm(1, 1), ct, 1e-8);  // dv/dv0
}

TEST_F(IntegratorTest, STMIdentityAtZeroTime) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto [xf, stm] = integ.integrate_stm(x0, 0.0);

    // STM at dt=0 should be close to identity for state block
    EXPECT_NEAR(stm(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(stm(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(stm(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(stm(1, 0), 0.0, 1e-10);
}

TEST_F(IntegratorTest, STMComposition) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double t1 = 0.5, t2 = 0.5;

    // Phi(0 -> 1.0) via single integration
    auto [xf_full, stm_full] = integ.integrate_stm(x0, t1 + t2);

    // Phi(0 -> 0.5) then Phi(0.5 -> 1.0)
    auto [xm, stm1] = integ.integrate_stm(x0, t1);
    auto [xf2, stm2] = integ.integrate_stm(xm, t1 + t2); // to tf=1.0

    // Phi(0->1) should approximately equal Phi(0.5->1) * Phi(0->0.5)
    // But only the state block (first 2x2)
    Eigen::MatrixXd composed = stm2.block(0, 0, 2, 2) * stm1.block(0, 0, 2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(stm_full(i, j), composed(i, j), 1e-7)
                << "STM composition mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(IntegratorTest, KeplerSTMDeterminant) {
    Kepler kep(398600.4418);
    Integrator<Kepler> integ(kep, "DOPRI87", 10.0);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    // Circular orbit initial state
    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0, 0, 0, v0, 0, 0;

    // Quarter period
    double T = 2.0 * M_PI * std::sqrt(r0 * r0 * r0 / 398600.4418);
    auto [xf, stm] = integ.integrate_stm(x0, T / 4.0);

    // State block of STM (6x6) should have determinant = 1 (symplecticity)
    Eigen::MatrixXd state_stm = stm.block(0, 0, 6, 6);
    double det = state_stm.determinant();
    EXPECT_NEAR(det, 1.0, 1e-5);
}

///////////////////////////////////////////////////////////////////////////////
// Adaptive step tests
///////////////////////////////////////////////////////////////////////////////

TEST_F(IntegratorTest, AdaptiveToleranceAdherence) {
    SHO ode(0.0);
    double tol = 1e-10;
    Integrator<SHO> integ(ode, "DOPRI87", 0.1);
    integ.setAbsTol(tol);
    integ.setRelTol(tol);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    auto xf = integ.integrate(x0, 5.0);

    double x_exact = std::cos(5.0);
    double v_exact = -std::sin(5.0);
    double err = std::sqrt((xf[0] - x_exact) * (xf[0] - x_exact) +
                           (xf[1] - v_exact) * (xf[1] - v_exact));
    // Error should be within a reasonable multiple of tolerance
    EXPECT_LT(err, tol * 1000);
}

TEST_F(IntegratorTest, TightVsLooseToleranceComparison) {
    SHO ode(0.0);

    Integrator<SHO> integ_tight(ode, "DOPRI87", 0.1);
    integ_tight.setAbsTol(1e-13);
    integ_tight.setRelTol(1e-13);

    Integrator<SHO> integ_loose(ode, "DOPRI87", 0.1);
    integ_loose.setAbsTol(1e-6);
    integ_loose.setRelTol(1e-6);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf_tight = integ_tight.integrate(x0, 3.0);
    auto xf_loose = integ_loose.integrate(x0, 3.0);

    double x_exact = std::cos(3.0);
    double err_tight = std::abs(xf_tight[0] - x_exact);
    double err_loose = std::abs(xf_loose[0] - x_exact);

    EXPECT_LT(err_tight, err_loose);
}

TEST_F(IntegratorTest, MethodComparison) {
    SHO ode(0.0);

    Integrator<SHO> integ54(ode, "DOPRI54", 0.1);
    integ54.setAbsTol(1e-10);
    integ54.setRelTol(1e-10);

    Integrator<SHO> integ87(ode, "DOPRI87", 0.1);
    integ87.setAbsTol(1e-10);
    integ87.setRelTol(1e-10);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf54 = integ54.integrate(x0, 2.0);
    auto xf87 = integ87.integrate(x0, 2.0);

    // Both should agree within tolerance
    EXPECT_NEAR(xf54[0], xf87[0], 1e-8);
    EXPECT_NEAR(xf54[1], xf87[1], 1e-8);
}

///////////////////////////////////////////////////////////////////////////////
// Dense output
///////////////////////////////////////////////////////////////////////////////

TEST_F(IntegratorTest, DenseOutputBoundaryConsistency) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 1.0;

    auto dense = integ.integrate_dense(x0, tf);
    ASSERT_GE(dense.size(), 2u);

    // First state should match initial
    EXPECT_NEAR(dense.front()[0], x0[0], 1e-12);
    EXPECT_NEAR(dense.front()[1], x0[1], 1e-12);

    // Last state should match single integration
    auto xf = integ.integrate(x0, tf);
    EXPECT_NEAR(dense.back()[0], xf[0], 1e-10);
    EXPECT_NEAR(dense.back()[1], xf[1], 1e-10);
}

TEST_F(IntegratorTest, DenseOutputVsAnalytical) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.setAbsTol(1e-13);
    integ.setRelTol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto dense = integ.integrate_dense(x0, M_PI);

    // Check several interior points against analytical solution
    for (const auto &state : dense) {
        double t = state[2]; // time component
        double x_exact = std::cos(t);
        double v_exact = -std::sin(t);
        EXPECT_NEAR(state[0], x_exact, 1e-8)
            << "x mismatch at t=" << t;
        EXPECT_NEAR(state[1], v_exact, 1e-8)
            << "v mismatch at t=" << t;
    }
}
