// Source: Kelly, M., "An introduction to trajectory optimization", SIAM Review, 2017

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

    auto ode = ODEBuilder(4, 1)
                   .define([l, m1, m2, g](auto &args) {
                       auto x = args.x_var(0);
                       auto theta = args.x_var(1);
                       auto xdot = args.x_var(2);
                       auto thetadot = args.x_var(3);
                       auto F = args.u_var(0);

                       auto Q = stack((-g) * sin(theta),
                                      F + m2 * l * sin(theta) * thetadot * thetadot);

                       // Scalar constants must be promoted to VF expressions for stack()
                       auto l_vf = theta * 0.0 + l;
                       auto a_vf = theta * 0.0 + (m1 + m2);
                       auto Mvec = stack(cos(theta), l_vf, a_vf, m2 * l * cos(theta));
                       auto M = row_matrix(Mvec, 2, 2);
                       auto accel = M.inverse() * Q;

                       return stack(xdot, thetadot, accel);
                   })
                   .var_names({{"x", 0}, {"theta", 1}, {"xdot", 2}, {"thetadot", 3},
                               {"t", 4}, {"F", 5}})
                   .build();

    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        const double t = tf * s;
        Eigen::VectorXd pt(6);
        pt << xf * s, M_PI * s, 0.0, 0.0, t, 0.0;
        traj_ig.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    Eigen::VectorXd front_bc(5);
    front_bc << 0.0, 0.0, 0.0, 0.0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front,
                            {"x", "theta", "xdot", "thetadot", "t"}, front_bc);

    Eigen::VectorXd back_bc(5);
    back_bc << xf, M_PI, 0.0, 0.0, tf;
    phase.add_boundary_value(PhaseRegionFlags::Back,
                            {"x", "theta", "xdot", "thetadot", "t"}, back_bc);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "F", -Fmax, Fmax);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -xmax, xmax);

    {
        auto obj_args = Arguments<1>();
        auto obj_expr = obj_args.coeff<0>() * obj_args.coeff<0>();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"F"});
    }

    phase.set_num_partitions(8);
    phase.optimizer().set_print_level(1);

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
