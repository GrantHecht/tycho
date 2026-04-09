///////////////////////////////////////////////////////////////////////////////
// Free Flying Robot — C++ example (Builder API)
//
// Ported from examples/python_examples/FreeFlyingRobotExample.py
// Source: https://arxiv.org/pdf/1905.11898.pdf
//
// State  : [x, y, xdot, ydot, theta, omega]   (6 states)
// Control: [u0, u1, u2, u3]                    (4 thrusters)
//
// ODE:
//   [x,y]dot = [xdot, ydot]
//   [xdot,ydot]dot = [cos(theta), sin(theta)] * (u0-u1+u2-u3)
//   thetadot = omega
//   omegadot = (u0-u1)*alpha + (u3-u2)*beta
//
// Objective: minimise integral(u0 + u1 + u2 + u3) (fuel)
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
    constexpr double alpha = 0.2;
    constexpr double beta = 0.2;
    constexpr double tf = 12.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 128;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(6, 4)
                   .define([alpha, beta](auto &args) {
                       auto xdot_st = args.x_var(2);
                       auto ydot_st = args.x_var(3);
                       auto theta = args.x_var(4);
                       auto omega = args.x_var(5);

                       auto u0 = args.u_var(0);
                       auto u1 = args.u_var(1);
                       auto u2 = args.u_var(2);
                       auto u3 = args.u_var(3);

                       auto vscale = u0 - u1 + u2 - u3;
                       auto xdotdot = cos(theta) * vscale;
                       auto ydotdot = sin(theta) * vscale;
                       auto theta_dot = omega;
                       auto omega_dot = (u0 - u1) * alpha + (u3 - u2) * beta;

                       return stack(xdot_st, ydot_st, xdotdot, ydotdot,
                                   theta_dot, omega_dot);
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"xdot", 2}, {"ydot", 3},
                               {"theta", 4}, {"omega", 5}, {"t", 6},
                               {"u0", 7}, {"u1", 8}, {"u2", 9}, {"u3", 10}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    Eigen::VectorXd X0(7), XF(7);
    X0 << -10.0, -10.0, 0.0, 0.0, M_PI / 2.0, 0.0, 0.0;
    XF << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tf;

    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(11);
        pt.head<7>() = X0 + s * (XF - X0);
        pt.tail<4>().setConstant(0.5);
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "y", "xdot", "ydot", "theta", "omega", "t"},
                            X0);
    phase.add_boundary_value(PhaseRegionFlags::Back,
                            {"x", "y", "xdot", "ydot", "theta", "omega", "t"},
                            XF);

    // Control bounds: 0 <= ui <= 1
    for (const auto &u_name : {"u0", "u1", "u2", "u3"}) {
        phase.add_lu_var_bound(PhaseRegionFlags::Path, u_name, 0.0, 1.0, 1.0);
    }

    // Integral objective: min integral(u0 + u1 + u2 + u3)
    {
        auto obj_args = Arguments<4>();
        auto obj_expr = obj_args.sum();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr),
                                    {"u0", "u1", "u2", "u3"});
    }

    phase.optimizer().set_print_level(0);
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_tols(1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "FreeFlyingRobot (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FreeFlyingRobot (builder): converged, "
              << traj.size() << " nodes\n";
    std::cout << "FreeFlyingRobot (builder): PASSED\n";
    return EXIT_SUCCESS;
}
