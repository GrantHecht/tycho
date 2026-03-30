///////////////////////////////////////////////////////////////////////////////
// HyperSensitive Problem — C++ example (static DSL)
//
// Classic hypersensitive mesh refinement benchmark from Rao et al.
// https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.2114
//
// State  : [x]   (1 scalar state)
// Control: [u]   (1 scalar control)
//
// Phase vector layout: [x, t, u]
//                       0  1  2
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
// Corresponds to the Python example in examples/MeshRefinement/HyperSensLong.py.
//
// === PAIN POINTS (for Phase 7 static DSL improvements) ===
// 1. Integral objectives require constructing a GenericFunction wrapper and
//    manually specifying which phase variables map to the function's inputs.
//    Python: phase.add_integral_objective(Args(2).squared_norm()/2, [0, 2])
//    C++:    GenericFunction<-1,1> + VectorXi index array
// 2. Boundary conditions require manual VectorXi/VectorXd construction for
//    index and value arrays — verbose compared to Python's list syntax.
// 3. Solver/mesh settings are string-based in Python but require enum imports
//    in C++.  Minor, but adds friction.
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

///////////////////////////////////////////////////////////////////////////////
// ODE definition
///////////////////////////////////////////////////////////////////////////////

// Linear version: xdot = -x + u
// The cubed version (xdot = -x^3 + u) would require a separate ODE type.
struct HyperSens_Impl : ODESize<1, 1, 0> {
    static auto Definition(double /*unused*/) {
        auto args = Arguments<3>(); // [x, t, u]
        auto x = args.coeff<0>();
        auto u = args.coeff<2>();
        return u - x;
    }
};
BUILD_ODE_FROM_EXPRESSION(HyperSens, HyperSens_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

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

    HyperSens ode(0.0); // dummy parameter (macro requires >=1 type arg)

    auto phase = std::make_shared<ODEPhase<HyperSens>>(ode, TranscriptionModes::LGL7);
    phase->set_traj(trajIG, nSeg);

    // Control mode
    phase->set_control_mode(ControlModes::NoSpline);

    // Boundary conditions
    Eigen::VectorXi front_idx(2);
    front_idx << 0, 1;
    Eigen::VectorXd front_val(2);
    front_val << xt0, 0.0;
    phase->add_boundary_value(PhaseRegionFlags::Front, front_idx, front_val,
                            ScaleModes::AUTO);

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xtf, tf;
    phase->add_boundary_value(PhaseRegionFlags::Back, back_idx, back_val,
                            ScaleModes::AUTO);

    // Integral objective: minimize integral of (x^2 + u^2) / 2
    {
        auto obj_args = Arguments<2>();
        auto obj_expr = obj_args.squared_norm() / 2.0;
        Eigen::VectorXi obj_vars(2);
        obj_vars << 0, 2;
        phase->add_integral_objective(GenericFunction<-1, 1>(obj_expr), obj_vars,
                                    ScaleModes::AUTO);
    }

    // Variable bounds
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 0, -50.0, 50.0, 1.0);
    phase->add_lu_var_bound(PhaseRegionFlags::Path, 2, -50.0, 50.0, 1.0);

    // Solver settings
    phase->optimizer->set_opt_ls_mode("L1");
    phase->optimizer->set_soe_ls_mode("L1");
    phase->optimizer->set_max_ls_iters(2);
    phase->optimizer->print_level_ = 2;
    phase->set_num_partitions(1);

    // MINDEG ordering — required for stability at tf=10000
    phase->optimizer->set_qp_ordering_mode("MINDEG");

    // Adaptive mesh refinement
    phase->set_adaptive_mesh(true);
    phase->set_mesh_tol(1.0e-7);
    phase->optimizer->set_econ_tol(1.0e-7);
    phase->set_max_mesh_iters(10);
    phase->set_mesh_error_estimator(MeshErrorEstimators::DEBOOR);
    phase->set_mesh_error_criteria(MeshErrorAggregation::MAX);
    phase->set_mesh_inc_factor(5.0);
    phase->set_mesh_red_factor(0.5);
    phase->set_mesh_err_factor(10.0);

    // Solve
    std::cout << "Solving HyperSensitive problem (tf=" << std::fixed
              << std::setprecision(0) << tf << ") ...\n"
              << std::flush;

    const auto flag = phase->optimize_solve();

    if (phase->mesh_converged_) {
        std::cout << "HyperSens: mesh CONVERGED\n";
    } else {
        std::cerr << "HyperSens: mesh did NOT converge\n";
    }

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "HyperSens: solver FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return 1;
    }

    // Print trajectory summary
    auto traj = phase->return_traj();
    if (!traj.empty()) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  x(0)  = " << traj.front()[0] << "\n";
        std::cout << "  x(tf) = " << traj.back()[0] << "\n";
        std::cout << "  segments = " << traj.size() - 1 << "\n";
    }

    if (phase->mesh_converged_) {
        std::cout << "HyperSens: PASS\n";
    } else {
        std::cout << "HyperSens: FAIL (mesh did not converge)\n";
    }
    return phase->mesh_converged_ ? 0 : 1;
}
