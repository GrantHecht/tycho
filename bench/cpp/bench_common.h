///////////////////////////////////////////////////////////////////////////////
// Shared benchmark infrastructure — ODE definitions, memory init, constants
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <tycho/tycho.h>
#include <cmath>
#include <numbers>

using namespace tycho;
using namespace tycho::utils;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
using namespace tycho::solvers;

///////////////////////////////////////////////////////////////////////////////
// Runtime initialization (called once per executable via inline variable)
///////////////////////////////////////////////////////////////////////////////

namespace TychoBench {
inline void ensure_runtime_initialized() {
    static bool done = [] {
        BumpAllocator::resize(256, 256);
        double init_ms = tycho::solvers::ensure_solver_initialized();
#ifdef USE_ACCELERATE_SPARSE
        fmt::print("Accelerate init: {:.3f} ms (VECLIB_MAXIMUM_THREADS={})\n", init_ms,
                   TYCHO_DEFAULT_QP_THREADS);
#else
        if (init_ms > 0.0)
            fmt::print("MKL init: {:.3f} ms\n", init_ms);
#endif
        return true;
    }();
    (void)done;
}
namespace detail {
inline const bool g_init = (ensure_runtime_initialized(), true);
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
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    phase->add_lu_var_bound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);
    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    if (print_level >= 0) {
        phase->optimizer_->print_level_ = print_level;
    }

    return phase;
}

///////////////////////////////////////////////////////////////////////////////
// Shared PolarLT phase builder (used by solver benchmarks at larger scale)
///////////////////////////////////////////////////////////////////////////////

inline std::shared_ptr<ODEPhase<PolarLTODE>> make_polar_lt_phase(int n_segs, int print_level = -1) {
    constexpr double amax = 0.5;
    constexpr double r0 = 1.0, rf = 1.5;
    constexpr double theta0 = 0.0;
    constexpr double vr0 = 0.0, vrf = 0.0;
    const double vt0 = 1.0 / std::sqrt(r0);
    const double vtf = 1.0 / std::sqrt(rf);
    constexpr double t0 = 0.0, tf_guess = 3.0;
    constexpr double u_guess = 1.0;
    const double alpha_guess = std::numbers::pi / 2.0;

    int n_pts = n_segs * 3 + 1; // LGL3 collocation
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(7); // [r, theta, vr, vt, t, u, alpha]
        pt[0] = r0 + (rf - r0) * s;
        pt[1] = theta0 + 2.0 * std::numbers::pi * s;
        pt[2] = vr0 + (vrf - vr0) * s;
        pt[3] = vt0 + (vtf - vt0) * s;
        pt[4] = t0 + tf_guess * s;
        pt[5] = u_guess;
        pt[6] = alpha_guess;
        traj.push_back(pt);
    }

    PolarLTODE ode(amax);
    auto phase =
        std::make_shared<ODEPhase<PolarLTODE>>(ode, TranscriptionModes::LGL3, traj, n_segs);

    // Front boundary: r, theta, vr, vt, t
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(5, 0, 4);
    Eigen::VectorXd front_val(5);
    front_val << r0, theta0, vr0, vt0, t0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    // Back boundary: r, vr, vt (free theta, free time)
    Eigen::VectorXi back_idx(3);
    back_idx << 0, 2, 3;
    Eigen::VectorXd back_val(3);
    back_val << rf, vrf, vtf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    // Control bounds
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 5, 0.0, 1.0, 1.0);
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 6, -std::numbers::pi, std::numbers::pi, 1.0);

    // Minimize transfer time
    phase->add_delta_time_objective(1.0, ScaleModes::AUTO);

    if (print_level >= 0) {
        phase->optimizer_->print_level_ = print_level;
    }

    return phase;
}

// NOTE: A Betts low-thrust benchmark (MEE dynamics, 7 states, 3 controls,
// 1 param, mesh refinement) is planned but requires its own dedicated TU
// to avoid 11+ GB template instantiation cost per benchmark source file.
// The MEE expression tree with Arguments<12> and 7 complex outputs is too
// heavy for a shared header. See bench/cpp/solvers/ for future implementation.

#if 0 // Disabled: causes 11+ GB memory per TU when in shared header
struct MEELT_Impl : ODESize<7, 3, 1> {
    // mu parameter is the nondimensional gravitational parameter (always 1.0
    // for the Betts problem). Other constants are pre-computed nondimensional
    // values from the Betts (2009) example 6 scaling.
    static auto Definition(double mu) {
        // Pre-computed nondimensional constants:
        //   Lstar = Re = 20925662.73 ft
        //   Tstar = Lstar/sqrt(mu_dim/Lstar) ≈ 806.81 s
        //   Mstar = 1/g0 ≈ 0.031081 slugs
        //   Fstar = Mstar*Lstar/Tstar^2
        const double Thrust_nd = 4.452766e-3; // Thrust/Fstar
        const double gs_nd = 1.0;             // g0*Tstar^2/Lstar = 1 (by construction)
        const double Isp_nd = 11677.4;        // Isp*g0*Tstar/Lstar
        // [p, f, g, h, k, L, w, t, ur, ut, un, tau]
        auto args = Arguments<12>();
        auto p = args.coeff<0>();
        auto f = args.coeff<1>();
        auto g = args.coeff<2>();
        auto h = args.coeff<3>();
        auto k = args.coeff<4>();
        auto L = args.coeff<5>();
        auto w = args.coeff<6>();
        auto ur = args.coeff<8>();
        auto ut = args.coeff<9>();
        auto un = args.coeff<10>();
        auto tau = args.coeff<11>();

        auto sinL = sin(L);
        auto cosL = cos(L);
        auto sqp = sqrt(p) / std::sqrt(mu);
        auto ww = 1.0 + f * cosL + g * sinL;
        auto s2 = 1.0 + h * h + k * k;

        // Thrust acceleration in RTN
        auto acc = gs_nd * Thrust_nd * (1.0 + 0.01 * tau) / w;
        auto ar = acc * ur;
        auto at = acc * ut;
        auto an = acc * un;

        // hsinL_kcosL = h*sin(L) - k*cos(L) — appears in multiple rates
        auto hsinL_kcosL = h * sinL + k * cosL * (-1.0);

        // MEE rates (Betts eq. 6.42-6.47)
        auto pdot = 2.0 * (p / ww) * at * sqp;
        auto fdot = (ar * sinL + ((ww + 1.0) * cosL + f) * (at / ww) +
                     hsinL_kcosL * (g * an / ww) * (-1.0)) *
                    sqp;
        auto gdot = (ar * cosL * (-1.0) + ((ww + 1.0) * sinL + g) * (at / ww) +
                     hsinL_kcosL * (f * an / ww)) *
                    sqp;
        auto hdot = s2 * cosL * an / (2.0 * ww) * sqp;
        auto kdot = s2 * sinL * an / (2.0 * ww) * sqp;
        auto Ldot =
            (mu * (ww / p) * (ww / p) + (1.0 / ww) * hsinL_kcosL * an) * sqp;
        const double wdot_coeff = -Thrust_nd / Isp_nd;
        auto wdot = wdot_coeff * (1.0 + 0.01 * tau);

        return StackedOutputs{pdot, fdot, gdot, hdot, kdot, Ldot, wdot};
    }
};
BUILD_ODE_FROM_EXPRESSION(MEELTODE, MEELT_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// Betts low-thrust phase builder (used by large-scale solver benchmarks)
// Creates a simplified Betts problem (no J2/J3/J4) for benchmarking.
///////////////////////////////////////////////////////////////////////////////

namespace BettsConst {
constexpr double mu = 1.407645794e16;
constexpr double Re = 20925662.73;
constexpr double g0 = 32.174;
constexpr double Thrust = 4.446618e-3;
constexpr double Isp = 450.0;
constexpr double gs = g0;
constexpr double pt0 = 21837080.052835;
constexpr double ptf = 40007346.015232;
constexpr double ht0 = -0.25396764647494;
constexpr double Lstar = Re;
inline const double Tstar = Lstar / std::sqrt(mu / Lstar);
inline const double Mstar = 1.0 / g0;
inline const double Fstar = Mstar * Lstar / (Tstar * Tstar);
// Nondimensional
constexpr double mu_nd = 1.0;
inline const double Thrust_nd = Thrust / Fstar;
inline const double gs_nd = gs * Tstar * Tstar / Lstar;
inline const double Isp_nd = Isp * g0 * Tstar / Lstar;
inline const double pt0_nd = pt0 / Lstar;
inline const double ptf_nd = ptf / Lstar;
inline const double tfig_nd = 90000.0 / Tstar;
} // namespace BettsConst

inline std::shared_ptr<ODEPhase<MEELTODE>>
make_betts_lt_phase(int n_segs, TranscriptionModes tmode = TranscriptionModes::LGL3,
                    bool adaptive_mesh = false, int print_level = -1) {
    using namespace BettsConst;

    MEELTODE ode(mu_nd);

    // Initial state: [p, f, g, h, k, L, w, t, ur, ut, un, tau]
    Eigen::VectorXd X0(12);
    X0 << pt0_nd, 0, 0, ht0, 0, std::numbers::pi, 1.0, 0.0, 0, 1, 0, -25.0;

    // Final state guess (target orbit, reduced weight)
    Eigen::VectorXd Xf(12);
    Xf << ptf_nd, 0, 0, ht0, 0, std::numbers::pi + 40.0, 0.5, tfig_nd, 0, 1, 0, -25.0;

    // Linear interpolation initial guess
    int pts_per_seg = (tmode == TranscriptionModes::LGL3) ? 3 : 5;
    int n_pts = n_segs * pts_per_seg + 1;
    std::vector<Eigen::VectorXd> IG;
    IG.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        double s = static_cast<double>(i) / (n_pts - 1);
        IG.push_back(X0 + s * (Xf - X0));
    }

    auto phase =
        std::shared_ptr<ODEPhase<MEELTODE>>(new ODEPhase<MEELTODE>(ode, tmode, IG, n_segs));
    phase->set_auto_scaling(true);

    // Front boundary: MEEs + weight + time
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(8, 0, 7);
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, X0.head(8), ScaleModes::AUTO);

    // Back boundary: target semi-latus rectum
    Eigen::VectorXi back_p_idx(1);
    back_p_idx << 0;
    Eigen::VectorXd back_p_val(1);
    back_p_val << ptf_nd;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_p_idx, back_p_val, ScaleModes::AUTO);

    // Lower bound on final weight
    phase->add_lower_var_bound(PhaseRegionFlags::Back, 6, 0.05);

    // Throttle parameter bounds
    phase->add_lu_var_bound(PhaseRegionFlags::ODEParams, 0, -50.0, 0.0);

    // Control unit vector constraint: ||u|| = 1
    phase->set_control_mode(ControlModes::NoSpline);

    // Maximize final weight (minimize fuel)
    phase->add_value_objective(PhaseRegionFlags::Back, 6, -1.0);

    // Mesh refinement settings
    if (adaptive_mesh) {
        phase->set_adaptive_mesh(true);
        phase->set_mesh_error_estimator(MeshErrorEstimators::INTEGRATOR);
        phase->set_mesh_tol(1.0e-7);
    }

    phase->set_num_partitions(8, 8);

    if (print_level >= 0) {
        phase->optimizer_->print_level_ = print_level;
    }

    return phase;
}
#endif
