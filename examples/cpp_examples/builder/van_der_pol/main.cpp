// Source: https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html

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
    constexpr double tf = 10.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 256;

    auto ode = ODEBuilder(2, 1)
                   .define([](auto &args) {
                       auto x0 = args.x_var(0);
                       auto x1 = args.x_var(1);
                       auto u = args.u_var(0);
                       auto x0dot = (1.0 - x1 * x1) * x0 - x1 + u;
                       auto x1dot = x0;
                       return stack(x0dot, x1dot);
                   })
                   .var_names({{"x0", 0}, {"x1", 1}, {"t", 2}, {"u", 3}})
                   .build();

    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double t = tf * static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(4);
        pt << 0.0, 0.0, t, 0.0;
        traj_ig.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL3, traj_ig, n_segs);
    phase.set_control_mode(ControlModes::BlockConstant);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x0", "x1", "t"},
                            Eigen::Vector3d(0.0, 1.0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x0", "x1", "t"},
                            Eigen::Vector3d(0.0, 0.0, tf));

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -0.75, 1.0);

    {
        auto obj_args = Arguments<3>();
        auto obj_expr = obj_args.squared_norm();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr),
                                    {"x0", "x1", "u"});
    }

    phase.optimizer().set_print_level(0);
    phase.set_num_partitions(8, 8);
    // Use individual tol setters so bar_tol keeps its default.
    phase.optimizer().set_kkt_tol(1.0e-8);
    phase.optimizer().set_econ_tol(1.0e-8);
    phase.optimizer().set_icon_tol(1.0e-8);

    const auto flag = phase.optimize();

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "VanDerPol (builder): optimize returned flag "
              << static_cast<int>(flag) << ", " << traj.size() << " nodes\n";
    std::cout << "VanDerPol (builder): PASSED\n";
    return EXIT_SUCCESS;
}
