///////////////////////////////////////////////////////////////////////////////
// Cart-Pole Swing-Up — C++ example (Builder API)
//
// Ported from examples/python_examples/CartPole.py
// Source: Kelly, M., "An introduction to trajectory optimization", SIAM Review, 2017
//
// State  : [x, theta, xdot, thetadot]   (cart pos, pole angle, velocities)
// Control: [F]                            (applied force)
//
// ODE: Uses mass matrix inversion M^{-1} * Q
//   M = [[m1+m2, m2*l*cos(theta)], [cos(theta), l]]
//   Q = [-g*sin(theta), F + m2*l*sin(theta)*thetadot^2]
//   [xddot, thetaddot] = M^{-1} * Q
//
// NOTE: Python uses vf.RowMatrix(...).inverse() * Q convenience API.
//       In C++, RowMatrix/inverse are not free functions; we compute the
//       2x2 inverse analytically via det = a*d - b*c.
//       Documented as API gap: no RowMatrix/inverse in C++ expression DSL.
//
// Objective: minimise integral(F^2)
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
    constexpr double l = 0.5;
    constexpr double m1 = 1.0;
    constexpr double m2 = 0.3;
    constexpr double g = 9.81;
    constexpr double Fmax = 20.0;
    constexpr double xmax = 2.0;
    constexpr double tf = 2.0;
    constexpr double xf = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 64;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(4, 1)
                   .define([l, m1, m2, g](auto &args) {
                       auto x = args.x_var(0);
                       auto theta = args.x_var(1);
                       auto xdot = args.x_var(2);
                       auto thetadot = args.x_var(3);
                       auto F = args.u_var(0);

                       // Mass matrix M (row-major):
                       //   [[m1+m2,            m2*l*cos(theta)],
                       //    [cos(theta),        l              ]]
                       // M = [[m1+m2, m2*l*cos(theta)], [cos(theta), l]]
                       double a = m1 + m2;
                       auto b_val = m2 * l * cos(theta);
                       auto c_val = cos(theta);
                       double d_val = l;

                       // Force vector Q:
                       auto Q0 = (-g) * sin(theta);
                       auto Q1 = F + m2 * l * sin(theta) * thetadot * thetadot;

                       // Analytic 2x2 inverse: M^{-1} = (1/det) * [[d, -b], [-c, a]]
                       auto det = a * d_val - b_val * c_val;
                       auto xddot = (d_val * Q0 - b_val * Q1) / det;
                       auto thetaddot = ((-1.0) * c_val * Q0 + a * Q1) / det;

                       return stack(xdot, thetadot, xddot, thetaddot);
                   })
                   .var_names({{"x", 0}, {"theta", 1}, {"xdot", 2}, {"thetadot", 3},
                               {"t", 4}, {"F", 5}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf * s;
        Eigen::VectorXd pt(6);
        pt << xf * s, M_PI * s, 0.0, 0.0, t, 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    // Fix initial state and time
    Eigen::VectorXd front_bc(5);
    front_bc << 0.0, 0.0, 0.0, 0.0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "theta", "xdot", "thetadot", "t"}, front_bc);

    // Fix final state and time
    Eigen::VectorXd back_bc(5);
    back_bc << xf, M_PI, 0.0, 0.0, tf;
    phase.add_boundary_value(PhaseRegionFlags::Back,
                            {"x", "theta", "xdot", "thetadot", "t"}, back_bc);

    // Bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "F", -Fmax, Fmax);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -xmax, xmax);

    // Integral objective: min integral(F^2)
    {
        auto obj_args = Arguments<1>();
        auto obj_expr = obj_args.coeff<0>() * obj_args.coeff<0>();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"F"});
    }

    phase.set_num_partitions(8);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "CartPole (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CartPole (builder): final theta = " << traj.back()[1]
              << " (expected ~" << M_PI << ")\n";
    std::cout << "CartPole (builder): PASSED\n";
    return EXIT_SUCCESS;
}
