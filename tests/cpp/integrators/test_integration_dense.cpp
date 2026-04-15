///////////////////////////////////////////////////////////////////////////////
// Dense output tests
//
// Validates dense (continuous) output from the integrator: boundary
// consistency with initial/final states, and agreement with analytical
// solution at interior points.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, DenseOutputBoundaryConsistency) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

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
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.01);
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto dense = integ.integrate_dense(x0, std::numbers::pi);

    // Check several interior points against analytical solution
    for (const auto &state : dense) {
        double t = state[2]; // time component
        double x_exact = std::cos(t);
        double v_exact = -std::sin(t);
        EXPECT_NEAR(state[0], x_exact, 1e-8) << "x mismatch at t=" << t;
        EXPECT_NEAR(state[1], v_exact, 1e-8) << "v mismatch at t=" << t;
    }
}
