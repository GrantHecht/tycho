///////////////////////////////////////////////////////////////////////////////
// Shared utilities for integrator tests
//
// Provides the Simple Harmonic Oscillator (SHO) ODE, an error helper
// for convergence testing, and the IntegratorTest fixture base class.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>
#include "test_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <string>

namespace TychoTest {

using namespace tycho;

///////////////////////////////////////////////////////////////////////////////
// Simple Harmonic Oscillator ODE: x'' = -x
// State: [x, v, t], augmented: [x, v, t], x_vars=2, u_vars=0, p_vars=0
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
// SHO error helper for convergence order tests
///////////////////////////////////////////////////////////////////////////////

inline double sho_error(const std::string &method, double h) {
    SHO ode(0.0);
    Integrator<SHO> integ(ode, method, h);
    integ.adaptive_ = false; // Fixed step

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 1.0;
    auto xf = integ.integrate(x0, tf);

    double x_exact = std::cos(tf);
    double v_exact = -std::sin(tf);
    return std::sqrt((xf[0] - x_exact) * (xf[0] - x_exact) + (xf[1] - v_exact) * (xf[1] - v_exact));
}

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class IntegratorTest : public VectorFunctionFixture {};

} // namespace TychoTest
