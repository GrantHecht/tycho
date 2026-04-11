///////////////////////////////////////////////////////////////////////////////
// Minimum Time to Climb (Table-based) — C++ example (Builder API)
//
// Ported from examples/python_examples/MinimumTimeToClimb.py
// Source: Bryson, Desai, Hoffman, "Energy-State Approximation in Performance
//         Optimization of Supersonic Aircraft", J. Aircraft, 1969
//
// State  : [h, v, fpa, m]   (altitude, velocity, flight path angle, mass)
// Control: [alpha]           (angle of attack)
//
// Demonstrates InterpTable1D and InterpTable2D usage in the C++ builder API.
// All aero/atmosphere/thrust data uses table interpolation matching the
// Python version exactly:
//   - InterpTable1D: rho(alt), speed_of_sound(alt), CLalpha(Mach),
//                    CD0(Mach), eta(Mach)
//   - InterpTable2D: Thrust(Mach, alt)
//
// Objective: minimise climb time
// Features: adaptive mesh refinement, HighestOrderSpline control
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
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

    // ── Aerodynamic data (function of Mach number) ────────────────────
    Eigen::VectorXd AeroMach(11);
    AeroMach << 0, 0.4, 0.6, 0.75, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8;

    Eigen::VectorXd Clalpha_data(11);
    Clalpha_data << 3.44, 3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44;

    Eigen::VectorXd CD0_data(11);
    CD0_data << 0.013, 0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035;

    Eigen::VectorXd eta_data(11);
    eta_data << 0.54, 0.54, 0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93;

    // ── Atmosphere data (1976 US standard atmosphere) ─────────────────
    // 45 data points: [altitude(m), density(kg/m^3), speed_of_sound(m/s)]
    // clang-format off
    Eigen::VectorXd alts(45);
    alts << -2000, 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000,
            18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000,
            38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000,
            58000, 60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000,
            78000, 80000, 82000, 84000, 86000;

    Eigen::VectorXd rhos(45);
    rhos << 1.478e00, 1.225e00, 1.007e00, 8.193e-01, 6.601e-01, 5.258e-01,
            4.135e-01, 3.119e-01, 2.279e-01, 1.665e-01, 1.216e-01, 8.891e-02,
            6.451e-02, 4.694e-02, 3.426e-02, 2.508e-02, 1.841e-02, 1.355e-02,
            9.887e-03, 7.257e-03, 5.366e-03, 3.995e-03, 2.995e-03, 2.259e-03,
            1.714e-03, 1.317e-03, 1.027e-03, 8.055e-04, 6.389e-04, 5.044e-04,
            3.962e-04, 3.096e-04, 2.407e-04, 1.860e-04, 1.429e-04, 1.091e-04,
            8.281e-05, 6.236e-05, 4.637e-05, 3.430e-05, 2.523e-05, 1.845e-05,
            1.341e-05, 9.690e-06, 6.955e-06;

    Eigen::VectorXd soss(45);
    soss << 3.479e02, 3.403e02, 3.325e02, 3.246e02, 3.165e02, 3.081e02,
            2.995e02, 2.951e02, 2.951e02, 2.951e02, 2.951e02, 2.951e02,
            2.964e02, 2.977e02, 2.991e02, 3.004e02, 3.017e02, 3.030e02,
            3.065e02, 3.101e02, 3.137e02, 3.172e02, 3.207e02, 3.241e02,
            3.275e02, 3.298e02, 3.298e02, 3.288e02, 3.254e02, 3.220e02,
            3.186e02, 3.151e02, 3.115e02, 3.080e02, 3.044e02, 3.007e02,
            2.971e02, 2.934e02, 2.907e02, 2.880e02, 2.853e02, 2.825e02,
            2.797e02, 2.769e02, 2.741e02;
    // clang-format on

    // ── Thrust data (Mach x altitude) ────────────────────────────────
    Eigen::VectorXd ThrustMach(10);
    ThrustMach << 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8;

    Eigen::VectorXd ThrustAlt(11);
    ThrustAlt << 304.8 * -0.5, 304.8 * 0, 304.8 * 5, 304.8 * 10, 304.8 * 15,
                 304.8 * 20, 304.8 * 25, 304.8 * 30, 304.8 * 40, 304.8 * 50,
                 304.8 * 70;

    // Raw data is 10 rows (Mach) x 11 cols (Alt), stored here transposed
    // to 11 rows (Alt) x 10 cols (Mach) matching InterpTable2D: rows=ys, cols=xs
    using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat ThrustData(11, 10);
    // clang-format off
    // Each row = one altitude, each column = one Mach number, values in lbf * 4448.2 -> Newtons
    ThrustData <<
        24.2, 28.0, 28.3, 30.8, 34.5, 37.9, 36.1, 36.1, 36.1, 36.1,
        24.2, 28.0, 28.3, 30.8, 34.5, 37.9, 36.1, 36.1, 36.1, 36.1,
        24.0, 24.6, 25.2, 27.2, 30.3, 34.3, 38.0, 36.6, 35.2, 33.8,
        20.3, 21.1, 21.9, 23.8, 26.6, 30.4, 34.9, 38.5, 42.1, 45.7,
        17.3, 18.1, 18.7, 20.5, 23.2, 26.8, 31.3, 36.1, 38.7, 41.3,
        14.5, 15.2, 15.9, 17.3, 19.8, 23.3, 27.3, 31.6, 35.7, 39.8,
        12.2, 12.8, 13.4, 14.7, 16.8, 19.8, 23.6, 28.1, 32.0, 34.6,
        10.2, 10.7, 11.2, 12.3, 14.1, 16.8, 20.1, 24.2, 28.1, 31.1,
         5.7,  6.5,  7.3,  8.1,  9.4, 11.2, 13.4, 16.2, 19.3, 21.7,
         3.4,  3.9,  4.4,  4.9,  5.6,  6.8,  8.3, 10.0, 11.9, 13.3,
         0.1,  0.2,  0.4,  0.8,  1.1,  1.4,  1.7,  2.2,  2.9,  3.1;
    // clang-format on
    ThrustData *= 4448.2; // Convert from klbf to Newtons

    // ── Build interpolation tables ───────────────────────────────────
    auto rhoTab = std::make_shared<InterpTable1D>(alts, rhos, 0, InterpType::Cubic);
    auto sosTab = std::make_shared<InterpTable1D>(alts, soss, 0, InterpType::Cubic);
    auto ClalphaTab = std::make_shared<InterpTable1D>(AeroMach, Clalpha_data, 0, InterpType::Cubic);
    auto etaTab = std::make_shared<InterpTable1D>(AeroMach, eta_data, 0, InterpType::Cubic);
    auto CD0Tab = std::make_shared<InterpTable1D>(AeroMach, CD0_data, 0, InterpType::Cubic);
    auto ThrustTab =
        std::make_shared<InterpTable2D>(ThrustMach, ThrustAlt, ThrustData, InterpType::Cubic);

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode =
        ODEBuilder(4, 1)
            .define([=](auto &args) {
                auto h = args.x_var(0);
                auto v = args.x_var(1);
                auto fpa = args.x_var(2);
                auto mass = args.x_var(3);
                auto alpha = args.u_var(0);

                auto rho = interp_scalar(rhoTab, h * Lstar) / Rhostar;
                auto sos = interp_scalar(sosTab, h * Lstar) / Vstar;
                auto Mach = v / sos;
                auto CD0 = interp_scalar(CD0Tab, Mach);
                auto Clalpha = interp_scalar(ClalphaTab, Mach);
                auto eta_val = interp_scalar(etaTab, Mach);
                auto Thrust = interp(ThrustTab, Mach, h * Lstar) / Fstar;
                auto CD = CD0 + eta_val * Clalpha * (alpha * alpha);
                auto CL = Clalpha * alpha;
                auto q = 0.5 * rho * v * v;
                auto D = q * S_nd * CD;
                auto L = q * S_nd * CL;
                auto r = h + Re_nd;
                auto hdot = v * sin(fpa);
                auto vdot = (Thrust * cos(alpha) - D) / mass - mu_nd * sin(fpa) / (r * r);
                auto fpadot = (Thrust * sin(alpha) + L) / (mass * v) +
                              cos(fpa) * (v / r - mu_nd / (v * (r * r)));
                auto mdot = (-1.0) * Thrust / vexhaust_nd;

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
    std::cout << "MinTimeClimbTables: solving...\n" << std::flush;
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MinTimeClimbTables (builder): solver FAILED (status "
                  << static_cast<int>(flag) << ")\n";
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
    std::cout << "MinTimeClimbTables (builder): climb time = " << final_time << " s\n";

    if (phase.mesh_converged()) {
        std::cout << "MinTimeClimbTables (builder): mesh CONVERGED\n";
    } else {
        std::cout << "MinTimeClimbTables (builder): mesh did NOT converge (may need more iters)\n";
    }

    std::cout << "MinTimeClimbTables (builder): PASSED\n";
    return EXIT_SUCCESS;
}
