///////////////////////////////////////////////////////////////////////////////
// Mountain Car — C++ example (Builder API)
//
// Ported from examples/python_examples/MountainCar.py
// Source: https://openmdao.github.io/dymos/examples/mountain_car/mountain_car.html
//
// State  : [x, v]   (position, velocity)
// Control: [u]       (thrust, -1..1)
//
// ODE:
//   xdot = v
//   vdot = 0.001*u - 0.0025*cos(3*x)
//
// Objective: minimise time (scaled 0.01)
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
    constexpr double x0 = -0.5;
    constexpr double v0 = 0.0;
    constexpr double xf = 0.52;
    constexpr double tf_guess = 500.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 128;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(2, 1)
                   .define([](auto &args) {
                       auto x = args.x_var(0);
                       auto v = args.x_var(1);
                       auto u = args.u_var(0);
                       auto xdot = v;
                       auto vdot = 0.001 * u - 0.0025 * cos(3.0 * x);
                       return stack(xdot, vdot);
                   })
                   .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf_guess * s;
        Eigen::VectorXd pt(4);
        pt << x0 + (xf - x0) * s, s, t, std::sin(s);
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "v", "t"},
                            Eigen::Vector3d(x0, v0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, "x", xf);

    // Back: final velocity must be non-negative
    phase.add_lower_var_bound(PhaseRegionFlags::Back, "v", 0.0, 1.0);

    // Path bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -1.2, 0.55, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", -0.07, 0.07, 100.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -1.0, 1.0, 1.0);

    // Objective: minimize time (scaled)
    phase.add_delta_time_objective(0.01);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MountainCar (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    const double final_time = traj.back()[2];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MountainCar (builder): final time = " << final_time << "\n";
    std::cout << "MountainCar (builder): PASSED\n";
    return EXIT_SUCCESS;
}
