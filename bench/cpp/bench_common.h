///////////////////////////////////////////////////////////////////////////////
// Shared benchmark infrastructure — ODE definitions, memory init, constants
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "Tycho.h"
#include <numbers>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// Memory initialization (called once per executable via inline variable)
///////////////////////////////////////////////////////////////////////////////

namespace TychoBench {
inline void ensure_memory_initialized() {
    static bool done = [] {
        BumpAllocator::resize(256, 256);
        return true;
    }();
    (void)done;
}
namespace detail {
inline const bool g_init = (ensure_memory_initialized(), true);
} // namespace detail
} // namespace TychoBench

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

static constexpr double MU_EARTH = 398600.4418;

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

///////////////////////////////////////////////////////////////////////////////
// Shared Brachistochrone phase builder (used by OC and solver benchmarks)
///////////////////////////////////////////////////////////////////////////////

inline std::shared_ptr<ODEPhase<BrachODE>> make_brach_phase(int n_segs, int print_level = -1) {
    constexpr double g = 9.81;
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;
    constexpr double tf_guess = 1.0, theta_guess = 1.0;

    int n_pts = n_segs * 3 + 1; // LGL3 collocation
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    BrachODE ode(g);
    auto phase = std::make_shared<ODEPhase<BrachODE>>(ode, TranscriptionModes::LGL3, traj, n_segs);

    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);

    if (print_level >= 0) {
        phase->optimizer->PrintLevel = print_level;
    }

    return phase;
}
