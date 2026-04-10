///////////////////////////////////////////////////////////////////////////////
// Space Shuttle Reentry — C++ example (Builder API)
//
// Ported from examples/python_examples/Reentry.py
// Source: Betts, "Practical Methods for OC", Cambridge, 2009
//
// State  : [h, theta, v, gamma, psi]   (alt, downrange, speed, FPA, heading)
// Control: [alpha, beta]                (AoA, bank angle)
//
// Two-step solve: unconstrained, then add heating rate constraint
// Objective: maximise final cross-range (theta at tf)
//
// API gaps: refine_traj_manual() not on Phase wrapper, uses base().
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
    // ── Non-dimensionalisation ─────────────────────────────────────────
    constexpr double g0 = 32.2;
    constexpr double W = 203000.0;
    constexpr double Lstar = 100000.0;
    constexpr double Tstar = 60.0;
    const double Mstar = W / g0;
    const double Vstar = Lstar / Tstar;
    const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
    const double Mustar = (Lstar * Lstar * Lstar) / (Tstar * Tstar);

    const double tmax = 2500.0 / Tstar;
    const double Re = 20902900.0 / Lstar;
    const double S_ref = 2690.0 / (Lstar * Lstar);
    const double m = (W / g0) / Mstar;
    const double mu = 0.140765e17 / Mustar;
    const double rho0 = 0.002378 / Rhostar;
    const double h_ref = 23800.0 / Lstar;

    constexpr double a0 = -0.20704, a1 = 0.029244;
    constexpr double b0 = 0.07854, b1 = -0.61592e-2, b2 = 0.621408e-3;
    constexpr double c0 = 1.0672181, c1 = -0.19213774e-1;
    constexpr double c2 = 0.21286289e-3, c3 = -0.10117e-5;
    constexpr double Qlimit = 70.0;

    // Boundary conditions
    const double ht0 = 260000.0 / Lstar;
    const double htf = 80000.0 / Lstar;
    const double vt0 = 25600.0 / Vstar;
    const double vtf = 2500.0 / Vstar;
    const double gammat0 = -1.0 * M_PI / 180.0;
    const double gammatf = -5.0 * M_PI / 180.0;
    const double psit0 = 90.0 * M_PI / 180.0;

    const double tf_guess = 1000.0 / Tstar;
    constexpr int n_pts = 200;
    constexpr int n_segs = 40;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(5, 2)
                   .define([Re, S_ref, m, mu, rho0, h_ref, a0, a1, b0, b1, b2](auto &args) {
                       auto h = args.x_var(0);
                       auto theta = args.x_var(1);
                       auto v = args.x_var(2);
                       auto gamma = args.x_var(3);
                       auto psi = args.x_var(4);
                       auto alpha = args.u_var(0);
                       auto beta_u = args.u_var(1);

                       auto alphadeg = (180.0 / M_PI) * alpha;
                       auto CL = a0 + a1 * alphadeg;
                       auto CD = b0 + b1 * alphadeg + b2 * alphadeg * alphadeg;
                       auto rho = rho0 * exp((-1.0 / h_ref) * h);
                       auto r = h + Re;
                       auto Lift = 0.5 * CL * S_ref * rho * v * v;
                       auto Drag = 0.5 * CD * S_ref * rho * v * v;
                       auto grav = mu / (r * r);

                       auto sgam = sin(gamma);
                       auto cgam = cos(gamma);
                       auto sbet = sin(beta_u);
                       auto cbet = cos(beta_u);
                       auto spsi = sin(psi);
                       auto cpsi = cos(psi);

                       auto hdot = v * sgam;
                       auto thetadot = (v / r) * cgam * cpsi;
                       auto vdot = (-1.0) * Drag / m - grav * sgam;
                       auto gammadot = (Lift / (m * v)) * cbet + cgam * (v / r - grav / v);
                       auto psidot =
                           Lift * sbet / (m * v * cgam) + (v / r) * cgam * spsi * tan(theta);

                       return stack(hdot, thetadot, vdot, gammadot, psidot);
                   })
                   .var_names({{"h", 0},
                               {"theta", 1},
                               {"v", 2},
                               {"gamma", 3},
                               {"psi", 4},
                               {"t", 5},
                               {"alpha", 6},
                               {"beta", 7}})
                   .build();

    // ── Heating rate function ──────────────────────────────────────────
    // Q = qa * qr where:
    //   qr = 17700 * sqrt(rhodim) * (0.0001*vdim)^3.07
    //   qa = c0 + c1*alphadeg + c2*alphadeg^2 + c3*alphadeg^3
    // Operates on [h, v, alpha]
    auto qfunc = [rho0, h_ref, Rhostar, Vstar, c0, c1, c2, c3]() {
        auto args = Arguments<3>();
        auto qh = args.coeff<0>();
        auto qv = args.coeff<1>();
        auto qalpha = args.coeff<2>();
        auto qalphadeg = (180.0 / M_PI) * qalpha;
        auto rhodim = rho0 * exp((-1.0 / h_ref) * qh) * Rhostar;
        auto vdim = qv * Vstar;

        // qr = 17700 * sqrt(rhodim) * (0.0001*vdim)^3.07
        auto base_val = 0.0001 * vdim;
        auto qr = 17700.0 * sqrt(rhodim) * CwisePow(base_val, 3.07);

        // qa = c0 + c1*alphadeg + c2*alphadeg^2 + c3*alphadeg^3
        auto qa = c0 + c1 * qalphadeg + c2 * qalphadeg * qalphadeg +
                  c3 * qalphadeg * qalphadeg * qalphadeg;

        return qa * qr;
    };

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(8);
        pt[0] = ht0 * (1.0 - s) + htf * s;
        pt[1] = 0.0;
        pt[2] = vt0 * (1.0 - s) + vtf * s;
        pt[3] = gammat0 * (1.0 - s) + gammatf * s;
        pt[4] = psit0;
        pt[5] = tf_guess * s;
        pt[6] = 0.0;
        pt[7] = 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);

    Eigen::VectorXd front_bc(6);
    front_bc << ht0, 0.0, vt0, gammat0, psit0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, {"h", "theta", "v", "gamma", "psi", "t"},
                             front_bc);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -89.0 * M_PI / 180.0,
                           89.0 * M_PI / 180.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "gamma", -89.0 * M_PI / 180.0,
                           89.0 * M_PI / 180.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha", -90.0 * M_PI / 180.0,
                           90.0 * M_PI / 180.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "beta", -90.0 * M_PI / 180.0,
                           1.0 * M_PI / 180.0, 1.0);
    phase.add_upper_delta_time_bound(tmax, 1.0);

    phase.add_boundary_value(PhaseRegionFlags::Back, {"h", "v", "gamma"},
                             Eigen::Vector3d(htf, vtf, gammatf));

    // Objective: maximise theta(tf) => minimise -theta
    phase.add_delta_var_objective("theta", -1.0);

    phase.set_num_partitions(8);
    phase.optimizer().set_soe_ls_mode("L1");
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_print_level(1);

    // ── Solve (unconstrained) ──────────────────────────────────────────
    std::cout << "Reentry: solve (unconstrained)...\n" << std::flush;
    phase.solve_optimize();

    // Refine to more segments — refine_traj_manual not on Phase wrapper
    std::cout << "Reentry: refining to 300 segments...\n" << std::flush;
    phase.base().refine_traj_manual(300);
    phase.optimize();

    auto traj1 = phase.return_traj();

    // ── Add heating constraint and re-solve ────────────────────────────
    std::cout << "Reentry: re-solving with heating rate constraint...\n" << std::flush;
    phase.add_upper_func_bound(PhaseRegionFlags::Path, GenericFunction<-1, 1>(qfunc()),
                               {"h", "v", "alpha"}, Qlimit, 1.0 / Qlimit);
    phase.optimize();

    auto traj2 = phase.return_traj();
    const double crossrange1 = traj1.back()[1] * 180.0 / M_PI;
    const double crossrange2 = traj2.back()[1] * 180.0 / M_PI;
    const double final_time1 = traj1.back()[5] * Tstar;
    const double final_time2 = traj2.back()[5] * Tstar;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reentry (builder): cross-range (no Q limit) = " << crossrange1
              << " deg, tf = " << final_time1 << " s\n";
    std::cout << "Reentry (builder): cross-range (Q limited)  = " << crossrange2
              << " deg, tf = " << final_time2 << " s\n";

    std::cout << "Reentry (builder): PASSED\n";
    return EXIT_SUCCESS;
}
