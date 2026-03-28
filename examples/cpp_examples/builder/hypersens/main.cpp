///////////////////////////////////////////////////////////////////////////////
// HyperSensitive Problem — C++ example (Builder API)
//
// Same problem as the static DSL version.  Tests integral objectives with
// named variables, adaptive mesh refinement, and MINDEG ordering.
//
// State  : [x]   (1 scalar state)
// Control: [u]   (1 scalar control)
//
// ODE: xdot = -x + u
//
// Objective: minimise integral of (x^2 + u^2) / 2
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;

int main() {
    const double xt0 = 1.5;
    const double xtf = 1.0;
    const double tf = 10000.0;
    constexpr int nSeg = 50;

    // ── Define ODE via Builder API ──────────────────────────────────────
    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) {
                       auto x = args.XVar(0);
                       auto u = args.UVar(0);
                       return u - x;
                   })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    // ── Initial trajectory guess ────────────────────────────────────────
    std::vector<Eigen::VectorXd> trajIG;
    trajIG.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        const double s = static_cast<double>(i) / 999.0;
        Eigen::VectorXd pt(3);
        pt << xt0 * (1.0 - s) + xtf * s, tf * s, 0.0;
        trajIG.push_back(pt);
    }

    // ── Construct phase ─────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL7, trajIG, nSeg);

    phase.set_control_mode(ControlModes::NoSpline);

    // Boundary conditions
    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"}, Eigen::Vector2d(xt0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "t"}, Eigen::Vector2d(xtf, tf));

    // Integral objective: minimise integral of (x^2 + u^2) / 2
    {
        auto obj_args = Arguments<2>();
        auto obj_expr = obj_args.squared_norm() / 2.0;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"x", "u"});
    }

    // Variable bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "x", -50.0, 50.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", -50.0, 50.0);

    // ── Solver settings ─────────────────────────────────────────────────
    phase.optimizer().set_OptLSMode("L1");
    phase.optimizer().set_SoeLSMode("L1");
    phase.optimizer().set_MaxLSIters(2);
    phase.optimizer().PrintLevel = 2;
    phase.set_num_partitions(1);

    // MINDEG ordering — needed for reliable convergence at tf=10000
    phase.optimizer().set_QPOrderingMode("MINDEG");

    // Adaptive mesh refinement
    phase.set_adaptive_mesh(true);
    phase.set_mesh_tol(1.0e-7);
    phase.optimizer().set_EContol(1.0e-7);
    phase.set_max_mesh_iters(10);
    phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);
    phase.set_mesh_error_criteria(MeshErrorAggregation::MAX);
    phase.set_mesh_inc_factor(5.0);
    phase.set_mesh_red_factor(0.5);
    phase.set_mesh_err_factor(10.0);

    // ── Solve ───────────────────────────────────────────────────────────
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

    // Print trajectory summary
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
