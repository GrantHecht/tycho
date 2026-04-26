///////////////////////////////////////////////////////////////////////////////
// HyperSensitive Problem — C++ example (static DSL, TU-split)
//
// Classic hypersensitive mesh refinement benchmark from Rao et al.
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.2114
//
// State  : [x]   (1 scalar state)
// Control: [u]   (1 scalar control)
//
// ODE:
//   xdot = -x + u
//
// Objective: minimize integral of (x^2 + u^2) / 2
//
// With tf = 10000 the problem is extremely sensitive — the default METIS
// pivot ordering produces a structurally singular factorization despite the
// problem being well-posed.  MINDEG ordering is stable for this case.
//
// Concrete HyperSens VectorFunction lives in hypersens_ode.cpp; this TU
// only sees the erased factory (see doc/user_guide_example_tu_split.md).
//
// Corresponds to the Python example in examples/MeshRefinement/HyperSensLong.py.
///////////////////////////////////////////////////////////////////////////////

#include "hypersens_ode.h"

#include <tycho/tycho.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::solvers;

int main() {
    const double xt0 = 1.5;
    const double xtf = 1.0;
    const double tf = 10000.0;
    constexpr int nSeg = 50;

    // Linear interpolation initial guess: x lerps from xt0 to xtf
    std::vector<Eigen::VectorXd> trajIG;
    trajIG.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        const double s = static_cast<double>(i) / 999.0;
        const double t = tf * s;
        Eigen::VectorXd pt(3);
        pt << xt0 * (1.0 - s) + xtf * s, t, 0.0;
        trajIG.push_back(pt);
    }

    auto erased = tycho_examples::make_hypersens_ode();
    ODE ode(std::move(erased), 1, 1, 0);
    auto phase = ode.phase(TranscriptionModes::LGL7, trajIG, nSeg);

    // Control mode
    phase.set_control_mode(ControlModes::NoSpline);

    // Boundary conditions
    Eigen::VectorXi front_idx(2);
    front_idx << 0, 1;
    Eigen::VectorXd front_val(2);
    front_val << xt0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xtf, tf;
    phase.add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    // Integral objective: minimize integral of (x^2 + u^2) / 2
    {
        auto obj_args = Arguments<2>();
        auto obj_expr = obj_args.squared_norm() / 2.0;
        Eigen::VectorXi obj_vars(2);
        obj_vars << 0, 2;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), obj_vars, ScaleModes::AUTO);
    }

    // Variable bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 0, -50.0, 50.0, 1.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 2, -50.0, 50.0, 1.0);

    // Solver settings
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_soe_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_print_level(2);
    phase.set_num_partitions(1);

    // MINDEG ordering — required for stability at tf=10000
    phase.optimizer().set_qp_ordering_mode("MINDEG");

    // Adaptive mesh refinement
    phase.set_adaptive_mesh(true);
    phase.set_mesh_tol(1.0e-7);
    phase.optimizer().set_econ_tol(1.0e-7);
    phase.set_max_mesh_iters(10);
    phase.set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);
    phase.set_mesh_error_criteria(MeshErrorAggregation::MAX);
    phase.set_mesh_inc_factor(5.0);
    phase.set_mesh_red_factor(0.5);
    phase.set_mesh_err_factor(10.0);

    // Solve
    std::cout << "Solving HyperSensitive problem (tf=" << std::fixed << std::setprecision(0) << tf
              << ") ...\n"
              << std::flush;

    const auto flag = phase.optimize_solve();

    if (phase.mesh_converged()) {
        std::cout << "HyperSens: mesh CONVERGED\n";
    } else {
        std::cerr << "HyperSens: mesh did NOT converge\n";
    }

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "HyperSens: solver FAILED (status " << static_cast<int>(flag) << ")\n";
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
        std::cout << "HyperSens: PASS\n";
    } else {
        std::cout << "HyperSens: FAIL (mesh did not converge)\n";
    }
    return phase.mesh_converged() ? 0 : 1;
}
