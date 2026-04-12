///////////////////////////////////////////////////////////////////////////////
// Adaptive step tests
//
// Validates adaptive step-size control: tolerance adherence, tight vs loose
// tolerance comparison, and cross-method agreement.
///////////////////////////////////////////////////////////////////////////////

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

TEST_F(IntegratorTest, AdaptiveToleranceAdherence) {
    SHO ode(0.0);
    double tol = 1e-10;
    Integrator<SHO> integ(ode, IVPAlg::DOPRI87, 0.1);
    integ.set_abs_tol(tol);
    integ.set_rel_tol(tol);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    auto xf = integ.integrate(x0, 5.0);

    double x_exact = std::cos(5.0);
    double v_exact = -std::sin(5.0);
    double err =
        std::sqrt((xf[0] - x_exact) * (xf[0] - x_exact) + (xf[1] - v_exact) * (xf[1] - v_exact));
    // Error should be within a reasonable multiple of tolerance
    EXPECT_LT(err, tol * 1000);
}

TEST_F(IntegratorTest, TightVsLooseToleranceComparison) {
    SHO ode(0.0);

    Integrator<SHO> integ_tight(ode, IVPAlg::DOPRI87, 0.1);
    integ_tight.set_abs_tol(1e-13);
    integ_tight.set_rel_tol(1e-13);

    Integrator<SHO> integ_loose(ode, IVPAlg::DOPRI87, 0.1);
    integ_loose.set_abs_tol(1e-6);
    integ_loose.set_rel_tol(1e-6);

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

    Integrator<SHO> integ54(ode, IVPAlg::DOPRI54, 0.1);
    integ54.set_abs_tol(1e-10);
    integ54.set_rel_tol(1e-10);

    Integrator<SHO> integ87(ode, IVPAlg::DOPRI87, 0.1);
    integ87.set_abs_tol(1e-10);
    integ87.set_rel_tol(1e-10);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf54 = integ54.integrate(x0, 2.0);
    auto xf87 = integ87.integrate(x0, 2.0);

    // Both should agree within tolerance
    EXPECT_NEAR(xf54[0], xf87[0], 1e-8);
    EXPECT_NEAR(xf54[1], xf87[1], 1e-8);
}
