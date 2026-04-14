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
    const double xt0 = 1.5;
    const double xtf = 1.0;
    const double tf = 10000.0;
    constexpr int nSeg = 50;

    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) {
                       auto x = args.x_var(0);
                       auto u = args.u_var(0);
                       return u - x;
                   })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    std::vector<Eigen::VectorXd> trajIG;
    trajIG.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        const double s = static_cast<double>(i) / 999.0;
        Eigen::VectorXd pt(3);
        pt << xt0 * (1.0 - s) + xtf * s, tf * s, 0.0;
        trajIG.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL7, trajIG, nSeg);

    phase.set_control_mode(ControlModes::NoSpline);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"}, Eigen::Vector2d(xt0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "t"}, Eigen::Vector2d(xtf, tf));

    {
        auto obj_args = Arguments<2>();
        auto obj_expr = obj_args.squared_norm() / 2.0;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"x", "u"});
    }

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -50.0, 50.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -50.0, 50.0);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_soe_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_print_level(2);
    phase.set_num_partitions(1);

    // MINDEG ordering — needed for reliable convergence at tf=10000
    phase.optimizer().set_qp_ordering_mode("MINDEG");

    phase.set_adaptive_mesh(true);
    phase.set_mesh_tol(1.0e-7);
    phase.optimizer().set_econ_tol(1.0e-7);
    phase.set_max_mesh_iters(10);
    phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);
    phase.set_mesh_error_criteria(MeshErrorAggregation::MAX);
    phase.set_mesh_inc_factor(5.0);
    phase.set_mesh_red_factor(0.5);
    phase.set_mesh_err_factor(10.0);

    std::cout << "Solving HyperSensitive problem (tf=" << std::fixed << std::setprecision(0) << tf
              << ") ...\n"
              << std::flush;

    const auto flag = phase.optimize_solve();

    if (phase.mesh_converged()) {
        std::cout << "HyperSens (builder): mesh CONVERGED\n";
    } else {
        std::cerr << "HyperSens (builder): mesh did NOT converge\n";
    }

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "HyperSens (builder): solver FAILED (status " << static_cast<int>(flag)
                  << ")\n";
        return 1;
    }

    auto traj = phase.return_traj();
    if (!traj.empty()) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  x(0)  = " << traj.front()[0] << "\n";
        std::cout << "  x(tf) = " << traj.back()[0] << "\n";
        std::cout << "  segments = " << traj.size() - 1 << "\n";
    }

    if (phase.mesh_converged()) {
        std::cout << "HyperSens (builder): PASS\n";
    } else {
        std::cout << "HyperSens (builder): FAIL (mesh did not converge)\n";
    }
    return phase.mesh_converged() ? 0 : 1;
}
