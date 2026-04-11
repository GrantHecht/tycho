///////////////////////////////////////////////////////////////////////////////
// Multi-Phase Cannonball — C++ example (Builder API)
//
// Ported from examples/python_examples/MultiPhaseCannon.py
// Source: Dymos optimal control library (OpenMDAO)
//
// Find the cannonball radius that maximises range, subject to a fixed
// launch energy budget.  Two phases: ascent (gamma -> 0) and descent
// (h -> 0).  ODE parameter: ball radius.
//
// State   : [v, gamma, h, r]  (speed, FPA, altitude, range)
// Control : none
// ODE param: [rad]            (cannonball radius)
//
// API features exercised:
//   - ODEBuilder(4, 0, 1) with ODE parameter
//   - ocp.add_direct_link_equal_con() for ODE param linking
//   - phase.add_inequal_con() with mixed XtUP + ODE param variable sources
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

///////////////////////////////////////////////////////////////////////////////
// Physical constants (non-dimensionalised)
///////////////////////////////////////////////////////////////////////////////

static constexpr double g0 = 9.81;
static constexpr double Lstar = 1000.0;    // m
static constexpr double Tstar = 60.0;      // sec
static constexpr double Mstar = 10.0;      // kg
static const double Astar = Lstar / (Tstar * Tstar);
static const double Vstar = Lstar / Tstar;
static const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
static const double Estar = Mstar * (Vstar * Vstar);

static constexpr double CD = 0.5;
static const double RhoAir = 1.225 / Rhostar;
static const double RhoIron = 7870.0 / Rhostar;
static const double h_scale = 8.44e3 / Lstar;
static const double E0 = 400000.0 / Estar;
static const double g = g0 / Astar;

///////////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////////

static double MFunc_val(double rad) {
    return (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
}

static double SFunc_val(double rad) {
    return M_PI * (rad * rad);
}

///////////////////////////////////////////////////////////////////////////////
// ODE factory — cannon dynamics with drag, ODE parameter = radius
///////////////////////////////////////////////////////////////////////////////

ODE make_cannon_ode() {
    // 4 states, 0 controls, 1 ODE parameter
    return ODEBuilder(4, 0, 1)
        .define([](auto &args) {
            auto v     = args.x_var(0);
            auto gamma = args.x_var(1);
            auto h     = args.x_var(2);
            auto r     = args.x_var(3);
            auto rad   = args.p_var(0);

            // Mass and cross-section from radius
            auto M = (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
            auto S = M_PI * (rad * rad);

            // Exponential atmosphere
            auto rho = RhoAir * exp(h * (-1.0 / h_scale));

            // Drag
            auto D = (0.5 * CD) * rho * (v * v) * S;

            auto vdot     = D * (-1.0) / M - g * sin(gamma);
            auto gammadot = g * cos(gamma) * (-1.0) / v;
            auto hdot     = v * sin(gamma);
            auto rdot     = v * cos(gamma);

            return stack(vdot, gammadot, hdot, rdot);
        })
        .var_names({{"v", 0}, {"gamma", 1}, {"h", 2}, {"r", 3}, {"t", 4}, {"rad", 5}})
        .build();
}

///////////////////////////////////////////////////////////////////////////////
// Energy constraint function: E(v, rad) - E0 <= 0
//
// Takes 2 inputs: [v, rad] -> 0.5 * M(rad) * v^2 - E0
// Scaled by 0.01 as in the Python example.
///////////////////////////////////////////////////////////////////////////////

auto make_energy_constraint() {
    auto args = Arguments<2>();
    auto v   = args.coeff<0>();
    auto rad = args.coeff<1>();

    auto M = (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
    auto E = 0.5 * M * (v * v);
    return GenericFunction<-1, -1>((E - E0) * 0.01);
}

///////////////////////////////////////////////////////////////////////////////
// Initial guess generation via ODE::integrator() with event detection
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    const double rad0 = 0.1 / Lstar;
    const double h0 = 100.0 / Lstar;
    const double r0 = 0.0;
    const double m0 = MFunc_val(rad0);
    const double gamma0 = 45.0 * M_PI / 180.0;
    const double v0 = std::sqrt(2.0 * E0 / m0) * 0.99;

    // ── Build ODE ──────────────────────────────────────────────────────
    auto ode = make_cannon_ode();

    std::cout << "MultiPhaseCannon (builder): generating initial guesses ...\n" << std::flush;

    // Create integrator
    auto integ = ode.integrator(0.01);
    integ.set_abs_tol(1.0e-14);

    // Initial state: [v, gamma, h, r, t, rad]
    Eigen::VectorXd IG = Eigen::VectorXd::Zero(6);
    IG[0] = v0;
    IG[1] = gamma0;
    IG[2] = h0;
    IG[3] = r0;
    IG[5] = rad0;

    // Ascent event: v*sin(gamma) crossing zero (hdot = 0 at apogee)
    // EventPack = (event_func, direction, action): direction=0 (any), action=1 (stop)
    using EventPack = decltype(integ)::EventPack;

    auto ascent_args = ODEArguments(4, 0, 1);
    auto ascent_event_expr = ascent_args.coeff(0) * sin(ascent_args.coeff(1));
    auto ascent_event = GenericFunction<-1, 1>(ascent_event_expr);

    auto [ascent_traj, ascent_events] =
        integ.integrate_dense(IG, 60.0 / Tstar,
                              std::vector<EventPack>{EventPack{ascent_event, 0, 1}}, false);

    // Descent event: h crossing zero (ground impact)
    auto descent_args = ODEArguments(4, 0, 1);
    auto descent_event_expr = descent_args.coeff(2); // h
    auto descent_event = GenericFunction<-1, 1>(descent_event_expr);

    auto [descent_traj, descent_events] =
        integ.integrate_dense(ascent_traj.back(), ascent_traj.back()[4] + 1000.0 / Tstar,
                              std::vector<EventPack>{EventPack{descent_event, 0, 1}}, false);

    std::cout << "  Ascent IG: " << ascent_traj.size() << " pts\n";
    std::cout << "  Descent IG: " << descent_traj.size() << " pts\n";
    const auto tmode = TranscriptionModes::LGL5;
    const int nsegs = 128;

    // ── Ascent phase ──────────────────────────────────────────────────
    auto aphase = ode.phase(tmode, ascent_traj, nsegs);

    // ODE param lower bound: rad >= 0
    // ODEParams region expects raw ODE param index (0), not phase vector index
    aphase.add_lower_var_bound(PhaseRegionFlags::ODEParams, 0, 0.0, 1.0);

    // Launch angle lower bound: gamma >= 0
    aphase.add_lower_var_bound(PhaseRegionFlags::Front, "gamma", 0.0, 1.0);

    // Front BC: h, r, t
    Eigen::VectorXd front_vals(3);
    front_vals << h0, r0, 0.0;
    aphase.add_boundary_value(PhaseRegionFlags::Front, {"h", "r", "t"}, front_vals);

    // Energy inequality at launch: 0.5 * M(rad) * v^2 - E0 <= 0
    // This references both state var 0 (v) and ODE param var 0 (rad).
    {
        auto efunc = make_energy_constraint();
        Eigen::VectorXi xvars(1); xvars << 0;   // v
        Eigen::VectorXi pvars(1); pvars << 0;   // rad (P-relative index)
        Eigen::VectorXi empty;
        aphase.add_inequal_con(
            PhaseRegionFlags::Front, efunc, xvars, pvars, empty, ScaleModes::AUTO);
    }

    // Back BC: gamma = 0 (top of arc)
    aphase.add_boundary_value(PhaseRegionFlags::Back, "gamma", 0.0);

    // ── Descent phase ─────────────────────────────────────────────────
    auto dphase = ode.phase(tmode, descent_traj, nsegs);

    // Back BC: h = 0 (ground)
    dphase.add_boundary_value(PhaseRegionFlags::Back, "h", 0.0);

    // Objective: maximise range (minimise -r at end)
    dphase.add_value_objective(PhaseRegionFlags::Back, "r", -1.0);

    // ── OCP ───────────────────────────────────────────────────────────
    OptimalControlProblem ocp;
    ocp.add_phase(aphase);
    ocp.add_phase(dphase);

    // Continuity in time-dependent vars: v, gamma, h, r, t
    ocp.add_forward_link_equal_con(aphase, dphase, {"v", "gamma", "h", "r", "t"});

    // Link ODE params (radius) between phases
    {
        Eigen::VectorXi pvar(1); pvar << 0;
        ocp.add_direct_link_equal_con(
            0, PhaseRegionFlags::ODEParams, pvar,
            1, PhaseRegionFlags::ODEParams, pvar);
    }

    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_print_level(1);

    std::cout << "Solving multi-phase cannon ...\n" << std::flush;
    const auto status = ocp.optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MultiPhaseCannon (builder): FAILED (status "
                  << static_cast<int>(status) << ")\n";
        return 1;
    }

    // ── Report ────────────────────────────────────────────────────────
    auto ascentTraj = aphase.return_traj();
    auto descentTraj = dphase.return_traj();

    double launch_angle = ascentTraj.front()[1] * 180.0 / M_PI;
    double opt_range = descentTraj.back()[3] * Lstar;
    double opt_rad = descentTraj.back()[5] * Lstar;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "MultiPhaseCannon (builder):\n";
    std::cout << "  Launch angle: " << launch_angle << " deg\n";
    std::cout << "  Optimized range: " << opt_range << " m\n";
    std::cout << "  Optimized radius: " << opt_rad << " m\n";

    return 0;
}
