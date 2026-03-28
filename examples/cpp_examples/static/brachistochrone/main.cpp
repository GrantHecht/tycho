///////////////////////////////////////////////////////////////////////////////
// Brachistochrone — C++ example
//
// Classic minimum-time brachistochrone problem: find the wire shape (angle
// theta(t)) that lets a bead slide from (x0,y0) to (xf,yf) under gravity in
// the shortest possible time.
//
// State  : [x, y, v]   (position x, position y, speed v)
// Control: [theta]     (wire angle, measured from vertical)
//
// Augmented phase vector layout: [x, y, v, t, theta]
//                                  0  1  2  3     4
//
// ODE dynamics:
//   xdot = sin(theta) * v
//   ydot = -cos(theta) * v
//   vdot = g * cos(theta)
//
// Corresponds to the Python example in examples/Brachistochrone.py.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;

///////////////////////////////////////////////////////////////////////////////
// ODE definition via the VectorFunction DSL
///////////////////////////////////////////////////////////////////////////////

struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = Arguments<5>(); // [x, y, v, t, theta]
        auto v = args.coeff<2>();
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
    constexpr double g = 9.81; // gravitational acceleration (m/s^2)

    // Boundary conditions
    constexpr double x0 = 0.0, y0 = 10.0, v0 = 0.0, t0 = 0.0;
    constexpr double xf = 10.0, yf = 5.0;

    // Initial-guess parameters
    constexpr double tf_guess = 1.0;    // guess for flight time (s)
    constexpr double theta_guess = 1.0; // constant control guess (rad)
    constexpr int n_pts = 100;          // number of initial-guess points
    constexpr int n_defects = 32;       // number of collocation segments

    // Build a linearly-interpolated initial trajectory
    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(5); // [x, y, v, t, theta]
        pt[0] = x0 + (xf - x0) * s;
        pt[1] = y0 + (yf - y0) * s;
        pt[2] = g * s * tf_guess * std::cos(theta_guess);
        pt[3] = t0 + tf_guess * s;
        pt[4] = theta_guess;
        traj.push_back(pt);
    }

    // Construct ODE and phase (LGL3 collocation)
    Brachistochrone ode(g);
    auto phase =
        std::make_shared<ODEPhase<Brachistochrone>>(ode, TranscriptionModes::LGL3, traj, n_defects);

    // ---- Constraints -------------------------------------------------------

    // Front boundary: x(0)=x0, y(0)=y0, v(0)=v0, t(0)=t0
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(4, 0, 3);
    Eigen::VectorXd front_val(4);
    front_val << x0, y0, v0, t0;
    phase->addBoundaryValue(PhaseRegionFlags::Front, front_idx, front_val, ScaleModes::AUTO);

    // Back boundary: x(tf)=xf, y(tf)=yf
    Eigen::VectorXi back_idx(2);
    back_idx << 0, 1;
    Eigen::VectorXd back_val(2);
    back_val << xf, yf;
    phase->addBoundaryValue(PhaseRegionFlags::Back, back_idx, back_val, ScaleModes::AUTO);

    // Control bounds: theta in [-0.1, 2.0]
    phase->addLUVarBound(PhaseRegionFlags::Path, 4, -0.1, 2.0, 1.0);

    // ---- Objective ---------------------------------------------------------
    phase->addDeltaTimeObjective(1.0, ScaleModes::AUTO);

    // ---- Solve -------------------------------------------------------------
    const auto status = phase->solve_optimize();

    if (status <= PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        const auto result = phase->returnTraj();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nBrachistochrone: optimal solution found\n";
        std::cout << "  Optimal time : " << result.back()[3] << " s\n";
        std::cout << "  Final x      : " << result.back()[0] << "\n";
        std::cout << "  Final y      : " << result.back()[1] << "\n";
        std::cout << "  Nodes        : " << result.size() << "\n";
    } else {
        std::cerr << "Brachistochrone: FAILED (status " << static_cast<int>(status) << ")\n";
        return 1;
    }

    return 0;
}
