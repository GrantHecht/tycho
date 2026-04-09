///////////////////////////////////////////////////////////////////////////////
// Bike Obstacle Avoidance — C++ example (Builder API)
//
// Ported from examples/python_examples/BikeObstacle.py
// Source: https://arxiv.org/pdf/2003.00142.pdf
//
// State  : [x, y, psi, v]   (position, heading, speed)
// Control: [acc, alpha]      (acceleration, steering angle)
//
// ODE:
//   beta = atan((la/(la+lb))*tan(alpha))
//   xdot = v*cos(psi + beta)
//   ydot = v*sin(psi + beta)
//   psidot = v*sin(beta)/lb
//   vdot = acc
//
// Obstacle: circular exclusion zone at (0, 50), r=5+2.5
// Objective: minimise time
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
    // Physical parameters
    constexpr double la = 1.58;   // front axle distance (m)
    constexpr double lb = 1.72;   // rear axle distance (m)
    constexpr double obsrad = 5.0;
    constexpr double margin = 2.5;
    constexpr double xobs = 0.0;
    constexpr double yobs = 50.0;

    // Boundary conditions
    constexpr double x0 = 0.0, y0 = 0.0;
    constexpr double psi0 = M_PI / 2.0;
    constexpr double v0 = 15.0;
    constexpr double xf = 0.0, yf = 100.0;

    // Bounds
    constexpr double accbound = 2.0;
    constexpr double vlbound = 5.0, vubound = 29.0;
    constexpr double alpha_max = M_PI / 6.0;  // 30 deg

    constexpr double tf_guess = yf / v0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 128;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(4, 2)
                   .define([la, lb](auto &args) {
                       auto psi = args.x_var(2);
                       auto v = args.x_var(3);
                       auto acc = args.u_var(0);
                       auto alpha = args.u_var(1);

                       auto beta = atan((la / (la + lb)) * tan(alpha));
                       auto xdot = v * cos(psi + beta);
                       auto ydot = v * sin(psi + beta);
                       auto psidot = v * sin(beta) / lb;
                       auto vdot = acc;

                       return stack(xdot, ydot, psidot, vdot);
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"psi", 2}, {"v", 3},
                               {"t", 4}, {"acc", 5}, {"alpha", 6}})
                   .build();

    // ── Obstacle constraint function ───────────────────────────────────
    // Returns 1 - ((x-xobs)/R)^2 - ((y-yobs)/R)^2 <= 0 for feasible
    auto obs_args = Arguments<2>();
    auto ox = obs_args.coeff<0>();
    auto oy = obs_args.coeff<1>();
    const double denom = obsrad + margin;
    auto ellips = ((ox - xobs) / denom) * ((ox - xobs) / denom) +
                  ((oy - yobs) / denom) * ((oy - yobs) / denom);
    auto obs_con = 1.0 - ellips;  // <= 0 for feasible

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf_guess * s;
        Eigen::VectorXd pt(7);
        pt[0] = obsrad + margin + 1.0;  // bias to side of obstacle
        pt[1] = yf * s;
        pt[2] = psi0;
        pt[3] = v0;
        pt[4] = t;
        pt[5] = 0.0;
        pt[6] = 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "y", "psi", "v", "t"},
                            Eigen::Matrix<double, 5, 1>(x0, y0, psi0, v0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"},
                            Eigen::Vector2d(xf, yf));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "v", vlbound, vubound);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "acc", -accbound, accbound);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha", -alpha_max, alpha_max);

    phase.add_inequal_con(PhaseRegionFlags::Path,
                         GenericFunction<-1, -1>(obs_con), {"x", "y"});

    phase.add_delta_time_objective(1.0);

    phase.optimizer().set_tols(1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "BikeObstacle (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    const double final_time = traj.back()[4];

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "BikeObstacle (builder): final time = " << final_time << " s\n";
    std::cout << "BikeObstacle (builder): PASSED\n";
    return EXIT_SUCCESS;
}
