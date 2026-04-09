///////////////////////////////////////////////////////////////////////////////
// Shared ODE definitions for benchmarks.
// Separated from bench_phases.h so TUs that only need raw VF evaluation
// (vector_functions, type_erasure, integrators) avoid pulling in the full
// optimal_control umbrella.
//
// Note: ode.h transitively includes ode_phase.h → psiopt.h, so ODE-using
// TUs still get the OC/solver template chain. The split nonetheless helps
// bench_utils and bench_kepler which don't include this header at all.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/vector_functions.h>
#include "tycho/detail/optimal_control/phase/ode.h"
#include <cmath>

using namespace tycho::vf;
using namespace tycho::oc;

///////////////////////////////////////////////////////////////////////////////
// ODE definitions — each benchmark executable is a single TU, so struct
// definitions in a header are fine (no ODR concern across link units).
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

struct Zermelo_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args = Arguments<4>(); // [x, y, t, theta]
        auto theta = args.coeff<3>();
        auto xdot = vMax * sin(theta) + 0.5;
        auto ydot = vMax * cos(theta) + (-0.3);
        return StackedOutputs{xdot, ydot};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloODE, Zermelo_Impl, double);

struct PolarLT_Impl : ODESize<4, 2, 0> {
    static auto Definition(double amax) {
        auto args = Arguments<7>(); // [r, theta, vr, vt, t, u, alpha]
        auto r = args.coeff<0>();
        auto vr = args.coeff<2>();
        auto vt = args.coeff<3>();
        auto u = args.coeff<5>();
        auto alpha = args.coeff<6>();
        auto rdot = vr;
        auto tdot = vt / r;
        auto vrdot = (vt * vt) / r + ((-1.0) / (r * r)) + amax * u * sin(alpha);
        auto vtdot = (vt * vr) / r * (-1.0) + amax * u * cos(alpha);
        return StackedOutputs{rdot, tdot, vrdot, vtdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(PolarLTODE, PolarLT_Impl, double);
