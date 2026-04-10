///////////////////////////////////////////////////////////////////////////////
// Goddard Rocket — C++ example (Builder API)
//
// Ported from examples/python_examples/GoddardRocket.py
// Source: Betts, "Practical Methods for OC", Cambridge, 2009, Section 4.14
//
// Classic Goddard rocket problem with singular arc.
//
// State  : [h, v, m]   (altitude, velocity, mass)
// Control: [u]          (throttle 0..1)
//
// Multi-phase formulation:
//   Phase 1: full thrust (u = 1)
//   Phase 2: singular arc (path constraint determines u)
//   Phase 3: coast (u = 0)
//
// Objective: maximise final altitude (minimise -h at tf)
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

static constexpr double g0 = 32.2;
static constexpr double W = 203000.0;

static constexpr double Lstar = 10000.0;    // feet
static constexpr double Tstar = 60.0;       // sec
static constexpr double Mstar = 1.0;        // slugs

static const double Vstar = Lstar / Tstar;
static const double Fstar = Mstar * Lstar / (Tstar * Tstar);
static const double Astar = Lstar / (Tstar * Tstar);
static const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
static const double sigmastar = Mstar / Lstar;

static const double rho0 = 0.002378 / Rhostar;
static const double h_ref = 23800.0 / Lstar;
static const double g = g0 / Astar;
static const double Tmag = 200.0 / Fstar;
static const double c = 1580.94 / Vstar;
static const double sigma = 5.4915e-5 / sigmastar;

static constexpr double m0 = 3.0;
static constexpr double mf = 1.0;

///////////////////////////////////////////////////////////////////////////////
// ODE factory
///////////////////////////////////////////////////////////////////////////////

ODE make_goddard_ode() {
    return ODEBuilder(3, 1)
        .define([](auto &args) {
            auto h = args.x_var(0);
            auto v = args.x_var(1);
            auto m = args.x_var(2);
            auto u = args.u_var(0);

            auto hdot = v;
            auto vdot = (u * Tmag - sigma * (v * v) * exp(h * (-1.0 / h_ref))) / m - g;
            auto mdot = u * (-Tmag / c);

            return stack(hdot, vdot, mdot);
        })
        .var_names({{"h", 0}, {"v", 1}, {"m", 2}, {"t", 3}, {"u", 4}})
        .build();
}

///////////////////////////////////////////////////////////////////////////////
// Singular arc path constraint
//
// From Betts eq. 4.188: the constraint that defines the singular arc control
// t1 = (u*Tmag - sigma*v^2*exp(-h/h_ref)) - g*m
// t2 = (m*g / (1 + 4*(c/v) + 2*(c/v)^2)) * (c^2*(1+v/c)/(h_ref*g) - 1 - 2*c/v)
// constraint: t1 - t2 = 0
///////////////////////////////////////////////////////////////////////////////

auto make_path_constraint() {
    // Takes 4 args: [h, v, m, u]
    auto args = Arguments<4>();
    auto h = args.coeff<0>();
    auto v = args.coeff<1>();
    auto m = args.coeff<2>();
    auto u = args.coeff<3>();

    auto t1 = (u * Tmag - sigma * (v * v) * exp(h * (-1.0 / h_ref))) - g * m;

    auto cv = c / v;
    auto t2 = (m * g / (1.0 + 4.0 * cv + 2.0 * cv * cv)) *
              (c * c * (1.0 + v / c) / (h_ref * g) - 1.0 - 2.0 * cv);

    return GenericFunction<-1, -1>(t1 - t2);
}

///////////////////////////////////////////////////////////////////////////////
// Simple initial guess via Euler integration
///////////////////////////////////////////////////////////////////////////////

std::vector<Eigen::VectorXd> make_initial_guess() {
    // Integrate forward with u=1 until mass = mf, then u=0 until v < 0
    const double dt_phys = 0.01;  // physical dt in non-dim time
    const int max_steps = 100000;

    std::vector<Eigen::VectorXd> traj;

    double h = 0.0, v = 0.0, m_val = m0, t = 0.0;

    for (int step = 0; step < max_steps; ++step) {
        double u_val = (m_val > mf) ? 1.0 : 0.0;

        Eigen::VectorXd pt(5);
        pt << h, v, m_val, t, u_val;
        traj.push_back(pt);

        // Euler step
        double hdot = v;
        double rho = std::exp(-h / h_ref);
        double vdot = (u_val * Tmag - sigma * v * v * rho) / m_val - g;
        double mdot = -u_val * Tmag / c;

        h += hdot * dt_phys;
        v += vdot * dt_phys;
        m_val += mdot * dt_phys;
        t += dt_phys;

        // Stop when velocity goes negative (rocket falling)
        if (v < 0.0)
            break;
    }

    return traj;
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    std::cout << "GoddardRocket (builder): generating initial guess ...\n" << std::flush;

    auto TrajIG = make_initial_guess();
    if (TrajIG.size() < 10) {
        std::cerr << "GoddardRocket: initial guess too short\n";
        return 1;
    }

    std::cout << "  Initial guess: " << TrajIG.size() << " points, tf = "
              << TrajIG.back()[3] * Tstar << " s\n";

    // ── Build multi-phase formulation ──────────────────────────────────
    auto ode = make_goddard_ode();
    const int n = static_cast<int>(TrajIG.size()) / 3;

    std::vector<Eigen::VectorXd> TrajIG1(TrajIG.begin(), TrajIG.begin() + n);
    std::vector<Eigen::VectorXd> TrajIG2(TrajIG.begin() + n, TrajIG.begin() + 2 * n);
    std::vector<Eigen::VectorXd> TrajIG3(TrajIG.begin() + 2 * n, TrajIG.end() - 1);

    const auto tmode = TranscriptionModes::LGL3;

    // ── Phase 1: full thrust (u = 1) ──────────────────────────────────
    auto phase1 = ode.phase(tmode, TrajIG1, 32);

    // Front BC: h, v, m, t
    Eigen::VectorXd front_vals(4);
    front_vals << TrajIG[0][0], TrajIG[0][1], TrajIG[0][2], TrajIG[0][3];
    phase1.add_boundary_value(PhaseRegionFlags::Front, {"h", "v", "m", "t"}, front_vals);

    // Fix control on path: u = 1
    phase1.add_boundary_value(PhaseRegionFlags::Path, "u", 1.0);

    // ── Phase 2: singular arc ─────────────────────────────────────────
    auto phase2 = ode.phase(tmode, TrajIG2, 32);
    phase2.set_control_mode(ControlModes::NoSpline);

    // Throttle bounds
    phase2.add_lu_var_bound(PhaseRegionFlags::Path, "u", 0.0, 1.0, 1.0);

    // Singular arc constraint
    auto path_con = make_path_constraint();
    phase2.add_equal_con(PhaseRegionFlags::Path, path_con, {"h", "v", "m", "u"});

    // ── Phase 3: coast (u = 0) ────────────────────────────────────────
    auto phase3 = ode.phase(tmode, TrajIG3, 32);

    // Fix control on path: u = 0
    phase3.add_boundary_value(PhaseRegionFlags::Path, "u", 0.0);

    // Terminal BC: v = 0, m = mf
    Eigen::VectorXd back_vals(2);
    back_vals << 0.0, mf;
    phase3.add_boundary_value(PhaseRegionFlags::Back, {"v", "m"}, back_vals);

    // Objective: maximise altitude at end (minimise -h)
    phase3.add_value_objective(PhaseRegionFlags::Back, "h", -1.0);

    // ── OCP ───────────────────────────────────────────────────────────
    OptimalControlProblem ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);
    ocp.add_phase(phase3);

    // Continuity across all phases: h, v, m, t
    ocp.add_forward_link_equal_con(phase1, phase3, {"h", "v", "m", "t"});

    // Ensure non-negative phase durations
    phase1.add_lower_delta_time_bound(0.0);
    phase2.add_lower_delta_time_bound(0.0);
    phase3.add_lower_delta_time_bound(0.0);

    // ── Solve ─────────────────────────────────────────────────────────
    std::cout << "Solving multi-phase Goddard rocket ...\n" << std::flush;

    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_print_level(1);

    const auto status = ocp.optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "GoddardRocket (builder): FAILED (status " << static_cast<int>(status)
                  << ")\n";
        return 1;
    }

    // ── Report ────────────────────────────────────────────────────────
    auto traj1 = phase1.return_traj();
    auto traj2 = phase2.return_traj();
    auto traj3 = phase3.return_traj();

    double h_final = traj3.back()[0] * Lstar;
    double t_final = traj3.back()[3] * Tstar;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GoddardRocket (builder): max altitude = " << h_final << " ft"
              << ", tf = " << t_final << " s\n";
    std::cout << "  Phase 1 (thrust): " << traj1.size() << " pts, dt = "
              << (traj1.back()[3] - traj1.front()[3]) * Tstar << " s\n";
    std::cout << "  Phase 2 (singular): " << traj2.size() << " pts, dt = "
              << (traj2.back()[3] - traj2.front()[3]) * Tstar << " s\n";
    std::cout << "  Phase 3 (coast): " << traj3.size() << " pts, dt = "
              << (traj3.back()[3] - traj3.front()[3]) * Tstar << " s\n";

    return 0;
}
