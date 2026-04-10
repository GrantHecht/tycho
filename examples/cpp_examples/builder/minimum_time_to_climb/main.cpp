///////////////////////////////////////////////////////////////////////////////
// Minimum Time to Climb — C++ example (Builder API)
//
// Ported from examples/python_examples/MinimumTimeToClimb.py
// Source: Bryson, Desai, Hoffman, "Energy-State Approximation in Performance
//         Optimization of Supersonic Aircraft", J. Aircraft, 1969
//
// State  : [h, v, fpa, m]   (altitude, velocity, flight path angle, mass)
// Control: [alpha]           (angle of attack)
//
// Uses simplified inline atmospheric model. The full table-based model is
// in the Tier 6 example min_time_climb_tables. This version uses analytic
// approximations that produce plausible (not exact) trajectories.
//
// Objective: minimise climb time
// Features: adaptive mesh refinement, HighestOrderSpline control
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    // Physical constants and non-dimensionalisation (matching Python)
    constexpr double g0 = 9.80665;
    constexpr double Lstar = 10000.0;
    constexpr double Tstar = 250.0;
    constexpr double Mstar_val = 19050.864;
    const double Vstar = Lstar / Tstar;         // 40 m/s
    const double Astar = Lstar / (Tstar * Tstar);
    const double Rhostar = Mstar_val / (Lstar * Lstar * Lstar);
    const double Fstar = Astar * Mstar_val;
    const double Mustar = (Lstar * Lstar * Lstar) / (Tstar * Tstar);

    const double mu_nd = 3.986012e14 / Mustar;
    const double Re_nd = 6378145.0 / Lstar;
    const double S_nd = 49.2386 / (Lstar * Lstar);
    const double vexhaust_nd = 1600.0 * g0 / Vstar;

    // Boundary conditions (non-dim)
    const double h0 = 0.010 / Lstar;
    const double hf = 19994.88 / Lstar;
    const double v0 = 129.314 / Vstar;
    const double vf_val = 295.092 / Vstar;
    const double m0 = 19050.864 / Mstar_val; // = 1.0
    const double m_min = 16500.0 / Mstar_val;

    const double hmin_nd = 0.0;
    const double hmax_nd = 21000.0 / Lstar;
    const double vmin_nd = 5.0 / Vstar;
    const double vmax_nd = 600.0 / Vstar;
    constexpr double fpamin = -20.0 * M_PI / 180.0;
    constexpr double fpamax = 40.0 * M_PI / 180.0;
    constexpr double alphamin = -M_PI / 4.0;
    constexpr double alphamax = M_PI / 4.0;

    const double tfig = 200.0 / Tstar;
    constexpr int n_pts = 100;
    constexpr int n_segs = 50;

    // ── Define ODE ─────────────────────────────────────────────────────
    // Simplified inline atmospheric model
    auto ode =
        ODEBuilder(4, 1)
            .define([mu_nd, Re_nd, S_nd, vexhaust_nd, Lstar, Vstar, Rhostar, Fstar](auto &args) {
                auto h = args.x_var(0);
                auto v = args.x_var(1);
                auto fpa = args.x_var(2);
                auto mass = args.x_var(3);
                auto alpha = args.u_var(0);

                auto r = h + Re_nd;
                auto grav = mu_nd / (r * r);

                // Simplified ISA atmosphere (non-dim):
                auto rho = (1.225 / Rhostar) * exp((-Lstar / 8500.0) * h);

                // Speed of sound: troposphere T=288.15-6.5e-3*h, sos=sqrt(1.4*287*T)
                // Simplified: sos ≈ 340.3 * sqrt(1 - h_dim/44330)
                // In non-dim: use linear approx in Mach range
                auto sos_nd = 340.3 / Vstar; // sea level, constant approx

                auto Mach = v / sos_nd;
                auto q = 0.5 * rho * v * v;

                // Aero model (simplified from Bryson data):
                constexpr double CLa = 3.44;
                auto CD0 = 0.013 + 0.0144 * Mach * Mach;
                auto eta_val = 0.15;
                auto CL = CLa * alpha;
                auto CD = CD0 + eta_val * CLa * alpha * alpha;

                auto D = q * S_nd * CD;
                auto L = q * S_nd * CL;

                // Thrust model: simplified afterburning turbofan
                // T = T_SL * (1 + 0.5*M) * (rho/rho0)^0.7
                // At sea level: ~80 kN. At 20 km (rho/rho0 ≈ 0.07): ~15 kN
                auto rho_ratio = rho * Rhostar / 1.225;
                auto T = (80000.0 / Fstar) * (1.0 + 0.5 * Mach) *
                         CwisePow(rho_ratio, 0.7);

                auto hdot = v * sin(fpa);
                auto vdot = (T * cos(alpha) - D) / mass - grav * sin(fpa);
                auto fpadot =
                    (T * sin(alpha) + L) / (mass * v) + cos(fpa) * (v / r - grav / v);
                auto mdot = (-1.0) * T / vexhaust_nd;

                return stack(hdot, vdot, fpadot, mdot);
            })
            .var_names(
                {{"h", 0}, {"v", 1}, {"fpa", 2}, {"m", 3}, {"t", 4}, {"alpha", 5}})
            .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(6);
        pt[0] = h0 + (hf - h0) * s;
        pt[1] = v0 + (vf_val - v0) * s;
        pt[2] = 0.0;
        pt[3] = m0;
        pt[4] = tfig * s;
        pt[5] = 0.05; // small positive AoA
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);
    phase.set_control_mode(ControlModes::HighestOrderSpline);

    Eigen::VectorXd front_bc(5);
    front_bc << h0, v0, 0.0, m0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, {"h", "v", "fpa", "m", "t"}, front_bc);

    phase.add_boundary_value(PhaseRegionFlags::Back, {"h", "v", "fpa"},
                             Eigen::Vector3d(hf, vf_val, 0.0));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "h", hmin_nd, hmax_nd);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", vmin_nd, vmax_nd);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "fpa", fpamin, fpamax);
    phase.add_lower_var_bound(PhaseRegionFlags::Back, "m", m_min);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha", alphamin, alphamax);

    phase.add_delta_time_objective(1.0);

    // Adaptive mesh refinement
    phase.set_adaptive_mesh(true);
    phase.set_mesh_tol(1.0e-7);
    phase.set_max_mesh_iters(10);
    phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);

    phase.set_num_partitions(8);
    phase.optimizer().set_print_level(1);
    phase.optimizer().set_soe_ls_mode("L1");
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);

    // ── Solve ──────────────────────────────────────────────────────────
    std::cout << "MinTimeToClimb: solving...\n" << std::flush;
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MinTimeToClimb (builder): solver FAILED (status " << static_cast<int>(flag)
                  << ")\n";
        // Still report what we got
        auto traj = phase.return_traj();
        if (!traj.empty()) {
            std::cerr << "  final time = " << traj.back()[4] * Tstar << " s\n";
        }
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    const double final_time = traj.back()[4] * Tstar;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MinTimeToClimb (builder): climb time = " << final_time << " s\n";

    if (phase.mesh_converged()) {
        std::cout << "MinTimeToClimb (builder): mesh CONVERGED\n";
    } else {
        std::cout << "MinTimeToClimb (builder): mesh did NOT converge (may need more iters)\n";
    }

    std::cout << "MinTimeToClimb (builder): PASSED\n";
    return EXIT_SUCCESS;
}
