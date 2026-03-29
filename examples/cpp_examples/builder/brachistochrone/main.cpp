///////////////////////////////////////////////////////////////////////////////
// Brachistochrone — C++ example (Builder API)
//
// Same problem as the static DSL version, but constructed at runtime via
// ODEBuilder.  Named variables replace raw index arrays throughout.
//
// State  : [x, y, v]   (position x, position y, speed v)
// Control: [theta]      (wire angle, measured from vertical)
//
// ODE dynamics:
//   xdot =  sin(theta) * v
//   ydot = -cos(theta) * v
//   vdot =  g * cos(theta)
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;
using namespace tycho::solvers;
using namespace tycho::astro;
using namespace tycho::utils;

int main() {
    constexpr double g = 9.81;

    // Boundary conditions
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;

    // Initial-guess parameters
    constexpr double tf_guess = 1.0;
    constexpr double theta_guess = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_defects = 32;

    // ── Define ODE via Builder API ──────────────────────────────────────
    auto ode = ODEBuilder(3, 1)
                   .define([g](auto &args) {
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
                   .build();

    // ── Initial trajectory guess ────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    // ── Construct phase and set up problem ──────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL3, traj, n_defects);

    // Front boundary: fix x, y, v, t
    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(x0, y0, v0, 0.0));

    // Back boundary: fix x, y
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(xf, yf));

    // Control bounds: theta in [-0.1, 2.0]
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    // Objective: minimise flight time
    phase.add_delta_time_objective(1.0);

    // ── Solve ───────────────────────────────────────────────────────────
    const auto status = phase.solve_optimize();

    if (status <= PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        const auto result = phase.return_traj();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nBrachistochrone (builder): optimal solution found\n";
        std::cout << "  Optimal time : " << result.back()[3] << " s\n";
        std::cout << "  Final x      : " << result.back()[0] << "\n";
        std::cout << "  Final y      : " << result.back()[1] << "\n";
        std::cout << "  Nodes        : " << result.size() << "\n";
    } else {
        std::cerr << "Brachistochrone (builder): FAILED (status " << static_cast<int>(status)
                  << ")\n";
        return 1;
    }

    return 0;
}
