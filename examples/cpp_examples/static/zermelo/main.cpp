///////////////////////////////////////////////////////////////////////////////
// Zermelo Navigation — C++ example (TU-split)
//
// Minimum-time navigation through wind fields.  A boat travels from point A
// to point B at maximum speed vMax while the wind pushes it.  The control
// variable theta selects the heading.
//
// State  : [x, y]    (2D position)
// Control: [theta]   (heading angle, measured from +x)
//
// Four wind-model ODEs (no-wind, uniform, constant-direction, variable) live
// in zermelo_ode.cpp behind GenericFunction<-1,-1> factories. The `navigate`
// solver is now non-template — it accepts an erased ODE by value through the
// builder-API `tycho::ODE` wrapper, so all four wind models share one
// compilation of the phase/solver machinery.
//
// Corresponds to the Python example in examples/Zermelo.py.
///////////////////////////////////////////////////////////////////////////////

#include "zermelo_ode.h"

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::oc;
using namespace tycho::solvers;

namespace {

// Non-template navigator — takes an already-erased ODE factory output and
// drives PSIOPT. All four wind models share this single instantiation.
std::vector<Eigen::VectorXd>
navigate(tycho::vf::GenericFunction<-1, -1> erased_ode, const Eigen::VectorXd &A,
         const Eigen::VectorXd &B, double vMax) {
    constexpr int nSeg = 250;
    constexpr double tol = 1e-12;

    // Linear initial guess
    const double dist = (B - A).norm();
    const double t0 = dist / vMax;
    const Eigen::VectorXd d = (B - A) / dist;
    const double ang = std::atan2(d[1], d[0]);

    std::vector<Eigen::VectorXd> trajG;
    trajG.reserve(nSeg);
    for (int i = 0; i < nSeg; ++i) {
        const double s = static_cast<double>(i) / (nSeg - 1);
        Eigen::VectorXd pt(4);
        pt << A[0] + d[0] * s, A[1] + d[1] * s, t0 * s, ang;
        trajG.push_back(pt);
    }

    ODE ode(std::move(erased_ode), 2, 1, 0);
    auto phase = ode.phase(TranscriptionModes::LGL3, trajG, nSeg);
    phase.set_num_partitions(10);

    // Boundary conditions
    Eigen::VectorXi xy_idx(2);
    xy_idx << 0, 1;
    Eigen::VectorXi t_idx(1);
    t_idx << 2;
    Eigen::VectorXd t0_val(1);
    t0_val << 0.0;

    phase.add_boundary_value(PhaseRegionFlags::Front, xy_idx, A, ScaleModes::AUTO);
    phase.add_boundary_value(PhaseRegionFlags::Front, t_idx, t0_val, ScaleModes::AUTO);
    phase.add_boundary_value(PhaseRegionFlags::Back, xy_idx, B, ScaleModes::AUTO);

    // Control bounds
    phase.add_lu_var_bound(PhaseRegionFlags::Path, 3, -M_PI, M_PI, 1.0);

    // Minimise travel time
    phase.add_delta_time_objective(1.0, ScaleModes::AUTO);

    // Solver settings (match Python: only tolerances)
    phase.optimizer().set_econ_tol(tol);
    phase.optimizer().set_kkt_tol(tol);

    const auto status = phase.solve_optimize();
    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "  FAILED: navigation did not converge (status "
                  << static_cast<int>(status) << ")\n";
        return {};
    }
    return phase.return_traj();
}

} // namespace

int main() {
    int failures = 0;

    // Quick Jacobian sanity check on the variable-wind model — uses the
    // erased factory output, which forwards compute/jacobian through
    // DenseFunctionBase.
    {
        auto ode_test = tycho_examples::make_zermelo_var_wind_ode(1.25);
        Eigen::VectorXd tp(4);
        tp << 0.2, -0.5, 0.5, 1.0;
        Eigen::VectorXd fx(2);
        ode_test.compute(tp, fx);
        Eigen::MatrixXd jac(2, 4);
        ode_test.jacobian(tp, jac);
        // Python: [[ 0.842 -0.079 0 -1.052] [0.667 1.297 0 0.675]]
        bool ok = std::abs(jac(0, 1) - (-0.0787)) < 0.01;
        std::cout << "ODE Jacobian dy column: " << jac(0, 1) << " " << jac(1, 1)
                  << (ok ? "  OK" : "  BUG") << "\n\n";
        if (!ok) ++failures;
    }

    Eigen::VectorXd A(2), B(2);
    A << 0.0, -1.0;
    B << 1.0, 1.0;

    // No wind
    std::cout << "Solving: no wind ... " << std::flush;
    auto traj1 = navigate(tycho_examples::make_zermelo_no_wind_ode(1.0), A, B, 1.0);
    if (traj1.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf1 = traj1.back()[2];
        std::cout << std::fixed << std::setprecision(4) << "tf = " << tf1 << " s\n";
    }

    // Uniform wind
    std::cout << "Solving: uniform wind ... " << std::flush;
    constexpr double vM = 1.25;
    auto traj2 = navigate(
        tycho_examples::make_zermelo_uniform_wind_ode(vM, 0.5, 135.0 * M_PI / 180.0), A, B, vM);
    if (traj2.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf2 = traj2.back()[2];
        std::cout << "tf = " << tf2 << " s\n";
    }

    // Constant-direction wind
    std::cout << "Solving: constant-direction wind ... " << std::flush;
    auto traj3 = navigate(
        tycho_examples::make_zermelo_const_dir_wind_ode(vM, 45.0 * M_PI / 180.0), A, B, vM);
    if (traj3.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf3 = traj3.back()[2];
        std::cout << "tf = " << tf3 << " s\n";
    }

    // Variable-direction wind
    std::cout << "Solving: variable-direction wind ... " << std::flush;
    auto traj4 = navigate(tycho_examples::make_zermelo_var_wind_ode(vM), A, B, vM);
    if (traj4.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        double tf4 = traj4.back()[2];
        std::cout << "tf = " << tf4 << " s\n";
    }

    // Summary
    std::cout << "\nZermelo: " << (4 - failures) << "/4 wind models converged\n";
    return failures > 0 ? 1 : 0;
}
