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

    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;

    constexpr double tf_guess = 1.0;
    constexpr double theta_guess = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_defects = 32;

    auto ode = ODEBuilder(3, 1)
                   .define([g](auto &args) {
                       auto v = args.x_var(2);
                       auto theta = args.u_var(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), g * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"t", 3}, {"theta", 4}})
                   .build();

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

    auto phase = ode.phase(TranscriptionModes::LGL3, traj, n_defects);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y", "v", "t"},
                           Eigen::Vector4d(x0, y0, v0, 0.0));

    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, Eigen::Vector2d(xf, yf));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -0.1, 2.0);

    phase.add_delta_time_objective(1.0);

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
