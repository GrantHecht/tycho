///////////////////////////////////////////////////////////////////////////////
// Runtime ODE integrator builder tests
//
// Exercises ODE::integrator() — the fluent IntegratorBuilder introduced in
// the PR #42 review response. Covers:
//   1. enum-to-stepper dispatch for every user-selectable IVPAlg value
//      (DOPRI54, DOPRI87) on a Simple Harmonic Oscillator
//   2. default method (DOPRI87) when .method() is not called
//   3. .step() omission throws
//   4. calling .control(...) twice throws
//
// Complements the templated Integrator<DODE> tests under this directory,
// which test the compile-time stepper code. This file tests the runtime
// wiring between the ODE builder's enum and the underlying integrator.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <numbers>
#include <tycho/detail/optimal_control/builder/ode_builder.h>
#include <tycho/detail/optimal_control/builder/runtime_ode.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class RuntimeIntegratorBuilderTest : public OptimalControlTest {};

namespace {

// Simple Harmonic Oscillator as a runtime ODE:
//   xdot = v
//   vdot = -x
ODE make_sho_ode() {
    return ODEBuilder(2, 0)
        .define([](auto &args) {
            auto x = args.x_var(0);
            auto v = args.x_var(1);
            return stack(v, -1.0 * x);
        })
        .build();
}

} // namespace

// DOPRI87: default method, step-only configuration.
TEST_F(RuntimeIntegratorBuilderTest, DefaultMethodSHOAtPi) {
    auto ode = make_sho_ode();
    auto integ = ode.integrator().step(0.01).build();
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf = integ.integrate(x0, std::numbers::pi);
    EXPECT_NEAR(xf[0], -1.0, 1e-10);
    EXPECT_NEAR(xf[1], 0.0, 1e-10);
}

// DOPRI54: explicit method selection.
TEST_F(RuntimeIntegratorBuilderTest, DOPRI54SHOAtPi) {
    auto ode = make_sho_ode();
    auto integ = ode.integrator().method(IVPAlg::DOPRI54).step(0.01).build();
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf = integ.integrate(x0, std::numbers::pi);
    EXPECT_NEAR(xf[0], -1.0, 1e-9);
    EXPECT_NEAR(xf[1], 0.0, 1e-9);
}

// DOPRI87: explicit method selection (matches default, exercises the enum arm).
TEST_F(RuntimeIntegratorBuilderTest, DOPRI87SHOAtPi) {
    auto ode = make_sho_ode();
    auto integ = ode.integrator().method(IVPAlg::DOPRI87).step(0.01).build();
    integ.set_abs_tol(1e-13);
    integ.set_rel_tol(1e-13);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    auto xf = integ.integrate(x0, std::numbers::pi);
    EXPECT_NEAR(xf[0], -1.0, 1e-10);
    EXPECT_NEAR(xf[1], 0.0, 1e-10);
}

// Calling build() without step() is a caller contract error.
TEST_F(RuntimeIntegratorBuilderTest, BuildWithoutStepThrows) {
    auto ode = make_sho_ode();
    EXPECT_THROW((void)ode.integrator().build(), std::logic_error);
}

// Calling .control(...) twice is a caller contract error.
TEST_F(RuntimeIntegratorBuilderTest, DoubleControlThrows) {
    auto ode = make_sho_ode();
    Eigen::VectorXd u_const(1);
    u_const << 0.0;
    EXPECT_THROW(
        {
            auto builder = ode.integrator().step(0.01).control(u_const);
            builder.control(u_const);
        },
        std::logic_error);
}
