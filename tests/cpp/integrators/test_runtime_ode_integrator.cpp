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

// Driven SHO with a scalar control: xdot = v, vdot = -x + u.
// Used by the Const-control and NamedLaw-registry tests which need u_vars > 0.
ODE make_driven_sho_ode_unnamed() {
    return ODEBuilder(2, 1)
        .define([](auto &args) {
            auto x = args.x_var(0);
            auto v = args.x_var(1);
            auto u = args.u_var(0);
            return stack(v, u - x);
        })
        .build();
}

ODE make_driven_sho_ode_named() {
    return ODEBuilder(2, 1)
        .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
        .define([](auto &args) {
            auto x = args.x_var(0);
            auto v = args.x_var(1);
            auto u = args.u_var(0);
            return stack(v, u - x);
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

// Const control: u = 0 on the driven SHO reduces to the unforced SHO, so
// x(π) = -1, v(π) = 0 from (1, 0, 0). Pins the ControlKind::Const build arm.
TEST_F(RuntimeIntegratorBuilderTest, ConstControlHappyPath) {
    auto ode = make_driven_sho_ode_unnamed();
    Eigen::VectorXd u_const(1);
    u_const << 0.0;

    auto integ = ode.integrator().step(0.01).control(u_const).build();
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    // Driven SHO has input layout [x, v, t, u]; integrator expects state
    // size 4 (3 state+time + 1 control) on the state axis.
    Eigen::Vector4d x0;
    x0 << 1.0, 0.0, 0.0, 0.0;

    auto xf = integ.integrate(x0, std::numbers::pi);
    EXPECT_NEAR(xf[0], -1.0, 1e-9);
    EXPECT_NEAR(xf[1], 0.0, 1e-9);
}

// NamedLaw on an ODE with no var_names registry must throw — pins the
// require_registry() guard on the IntegratorBuilder NamedLaw path.
TEST_F(RuntimeIntegratorBuilderTest, NamedLawOnRegistrylessOdeThrows) {
    auto ode = make_driven_sho_ode_unnamed(); // no .var_names(...)

    // Simple constant-zero scalar control law.
    auto u_args = Arguments<1>();
    auto u_zero = GenericFunction<-1, -1>(u_args * 0.0);

    EXPECT_THROW(
        {
            (void)ode.integrator().step(0.01).control(u_zero, {"t"}).build();
        },
        std::invalid_argument);
}

// NamedLaw on a named ODE resolves successfully. Pins the happy path on the
// registry lookup inside IntegratorBuilder::build for ControlKind::NamedLaw.
TEST_F(RuntimeIntegratorBuilderTest, NamedLawOnNamedOdeResolvesAndIntegrates) {
    auto ode = make_driven_sho_ode_named();

    // u(t) = 0 — reduces to unforced SHO.
    auto t_args = Arguments<1>();
    auto u_zero = GenericFunction<-1, -1>(t_args * 0.0);

    auto integ = ode.integrator().step(0.01).control(u_zero, {"t"}).build();
    integ.set_abs_tol(1e-12);
    integ.set_rel_tol(1e-12);

    Eigen::Vector4d x0;
    x0 << 1.0, 0.0, 0.0, 0.0;
    auto xf = integ.integrate(x0, std::numbers::pi);
    EXPECT_NEAR(xf[0], -1.0, 1e-9);
    EXPECT_NEAR(xf[1], 0.0, 1e-9);
}
