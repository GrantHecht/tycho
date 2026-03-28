///////////////////////////////////////////////////////////////////////////////
// STM (State Transition Matrix) tests
//
// Validates STM computation against analytical results for the SHO,
// checks identity at zero time, composition property, and symplecticity
// of the Kepler STM determinant.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include "tycho/detail/utils/thread_pool.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, SHOSTMVsAnalytical) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, "DOPRI87", 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

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
    integ.set_abs_tol(1e-13);

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
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

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
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    // Circular orbit initial state
    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0, 0, 0, v0, 0, 0;

    // Quarter period
    double T = 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / 398600.4418);
    auto [xf, stm] = integ.integrate_stm(x0, T / 4.0);

    // State block of STM (6x6) should have determinant = 1 (symplecticity)
    Eigen::MatrixXd state_stm = stm.block(0, 0, 6, 6);
    double det = state_stm.determinant();
    EXPECT_NEAR(det, 1.0, 1e-5);
}

TEST_F(IntegratorTest, STMParallelSingleTrajectoryMatchesSerial) {
    Kepler kep(398600.4418);
    Integrator<Kepler> integ(kep, "DOPRI87", 10.0);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    // Circular orbit initial state
    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0, 0, 0, v0, 0, 0;

    // Quarter period
    double T = 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / 398600.4418);
    double tf = T / 4.0;

    auto [xf_serial, stm_serial] = integ.integrate_stm(x0, tf);
    auto [xf_parallel, stm_parallel] = integ.integrate_stm_parallel(x0, tf, 4);

    // Final states should match
    for (int i = 0; i < xf_serial.size(); ++i) {
        EXPECT_NEAR(xf_parallel[i], xf_serial[i], 1e-8)
            << "State mismatch at index " << i;
    }

    // STMs should match
    for (int r = 0; r < stm_serial.rows(); ++r) {
        for (int c = 0; c < stm_serial.cols(); ++c) {
            EXPECT_NEAR(stm_parallel(r, c), stm_serial(r, c), 1e-6)
                << "STM mismatch at (" << r << "," << c << ")";
        }
    }
}

TEST_F(IntegratorTest, STMParallelSingleTrajectorySerialFallback) {
    ScopedThreadCount guard(1); // Force serial fallback (restores on scope exit)

    Kepler kep(398600.4418);
    Integrator<Kepler> integ(kep, "DOPRI87", 10.0);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    double r0 = 7000.0;
    double v0 = std::sqrt(398600.4418 / r0);
    Eigen::VectorXd x0(7);
    x0 << r0, 0, 0, 0, v0, 0, 0;

    double T = 2.0 * std::numbers::pi * std::sqrt(r0 * r0 * r0 / 398600.4418);
    double tf = T / 4.0;

    auto [xf_serial, stm_serial] = integ.integrate_stm(x0, tf);
    auto [xf_fallback, stm_fallback] = integ.integrate_stm_parallel(x0, tf, 4);

    for (int i = 0; i < xf_serial.size(); ++i) {
        EXPECT_NEAR(xf_fallback[i], xf_serial[i], 1e-8)
            << "State mismatch at index " << i;
    }

    for (int r = 0; r < stm_serial.rows(); ++r) {
        for (int c = 0; c < stm_serial.cols(); ++c) {
            EXPECT_NEAR(stm_fallback(r, c), stm_serial(r, c), 1e-6)
                << "STM mismatch at (" << r << "," << c << ")";
        }
    }
}
