///////////////////////////////////////////////////////////////////////////////
// Brachistochrone — C++ Benchmark
//
// Minimal benchmark based on main.cpp for timing the solve.
// All I/O removed except timing output.
///////////////////////////////////////////////////////////////////////////////

#include "Tycho.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// ODE definition via the VectorFunction DSL
///////////////////////////////////////////////////////////////////////////////

struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args  = Arguments<5>();  // [x, y, v, t, theta]
        auto v     = args.coeff<2>();
        auto theta = args.coeff<4>();

        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);

        return StackedOutputs{xdot, ydot, vdot};
    }
};

BUILD_ODE_FROM_EXPRESSION(Brachistochrone, Brachistochrone_Impl, double);

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    constexpr double g   = 9.81;

    // Boundary conditions
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;

    // Initial-guess parameters
    constexpr double tf_guess    = 1.0;
    constexpr double theta_guess = 1.0;
    constexpr int    n_pts       = 100;
    constexpr int    n_defects   = 32;

    // Build a linearly-interpolated initial trajectory
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5);
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    // Construct ODE and phase (LGL3 collocation)
    Brachistochrone ode(g);
    auto phase = std::make_shared<ODEPhase<Brachistochrone>>(ode, "LGL3", traj, n_defects);

    // ---- Constraints -------------------------------------------------------
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, std::string("auto"));

    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, std::string("auto"));

    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);

    // ---- Objective ---------------------------------------------------------
    phase->addDeltaTimeObjective(1.0, std::string("auto"));

    // Suppress optimizer output
    phase->optimizer->PrintLevel = 3;

    // ---- Solve (timed) -----------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
    const auto status = phase->solve_optimize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << duration << std::endl;

    if (status != PSIOPT::ConvergenceFlags::CONVERGED) {
        std::cerr << "FAIL" << std::endl;
        return 1;
    }

    return 0;
}
