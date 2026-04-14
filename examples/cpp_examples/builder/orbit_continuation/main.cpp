///////////////////////////////////////////////////////////////////////////////
// Orbit Continuation — C++ example (Builder API)
//
// Ported from examples/python_examples/OrbitContinuation.py
//
// Computes periodic orbits in the Earth-Moon CR3BP using numerical
// continuation. Starts with an L1 Lyapunov orbit initial guess and
// converges a single periodic orbit, then performs basic continuation
// along the x-coordinate.
//
// State  : [x, y, z, vx, vy, vz]  (6 states, no controls)
//
// Phase vector: [x, y, z, vx, vy, vz, t]
//                0  1  2   3   4   5   6
//
// This example demonstrates:
//   - CR3BP dynamics via builder API
//   - Periodic orbit boundary conditions
//   - Numerical continuation scheme
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

// Earth-Moon system
static constexpr double MuEarth = 3.986004418e14; // m^3/s^2
static constexpr double MuMoon = 4.9048695e12;    // m^3/s^2
static constexpr double LD = 384400000.0;          // m (Earth-Moon distance)

// Mass ratio
static const double mu = MuMoon / (MuEarth + MuMoon);

///////////////////////////////////////////////////////////////////////////////
// Build CR3BP ODE via builder API
///////////////////////////////////////////////////////////////////////////////

ODE make_cr3bp_ode(double mu_val) {
    auto args = ODEArguments(6, 0, 0);
    // State: [x, y, z, vx, vy, vz], time at index 6
    auto X = args.head<3>();
    auto V = args.segment<3>(3);

    Eigen::Vector3d p1loc;
    p1loc << -mu_val, 0.0, 0.0;
    Eigen::Vector3d p2loc;
    p2loc << 1.0 - mu_val, 0.0, 0.0;

    auto p1_const = Constant<7, 3>(7, p1loc);
    auto p2_const = Constant<7, 3>(7, p2loc);

    auto dvec = X - p1_const;
    auto rvec = X - p2_const;

    auto x = X.coeff<0>();
    auto y = X.coeff<1>();
    auto xdot = V.coeff<0>();
    auto ydot = V.coeff<1>();

    auto rot_x = 2.0 * ydot + x;
    auto rot_y = (-2.0) * xdot + y;

    auto rot_terms = StackedOutputs{rot_x, rot_y};

    auto acc = rot_terms.padded_lower<1>() - (1.0 - mu_val) * dvec.normalized_power<3>() -
               mu_val * rvec.normalized_power<3>();

    auto ode_expr = StackedOutputs{V, acc};

    return ODE(ode_expr, 6, 0)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"x", 0}, {"y", 1}, {"z", 2}, {"vx", 3}, {"vy", 4}, {"vz", 5}, {"t", 6}});
}

///////////////////////////////////////////////////////////////////////////////
// Solve for a periodic orbit (half-period symmetric)
///////////////////////////////////////////////////////////////////////////////

// Integrator step size for orbit propagation
static constexpr double integ_dt = 3.1415 / 10000.0;

std::vector<Eigen::VectorXd> solve_periodic(const ODE &ode, const Eigen::VectorXd &ig, double tf,
                                             const std::vector<int> &fix_init = {0, 1, 2}) {
    // Integrate initial guess using ODE::integrator()
    auto integ = ode.integrator().step(integ_dt).build();
    auto trajGuess = integ.integrate_dense(ig, tf, 1000);

    constexpr int nSeg = 150;
    auto phase = ode.phase(TranscriptionModes::LGL3, trajGuess, nSeg);
    phase.set_num_partitions(8);

    // Fix specified initial conditions
    for (int idx : fix_init) {
        phase.add_boundary_value(PhaseRegionFlags::Front, idx, ig[idx]);
    }
    // Initial y=0, vx=0, t=0
    Eigen::VectorXi front_fix(3);
    front_fix << 3, 6, -1; // dummy, will do individually
    phase.add_boundary_value(PhaseRegionFlags::Front, 1, 0.0); // y = 0 (if not already fixed)
    phase.add_boundary_value(PhaseRegionFlags::Front, 3, 0.0); // vx = 0
    phase.add_boundary_value(PhaseRegionFlags::Front, 6, 0.0); // t = 0

    // Back BC: y=0, vx=0
    phase.add_boundary_value(PhaseRegionFlags::Back, 1, 0.0);
    phase.add_boundary_value(PhaseRegionFlags::Back, 3, 0.0);
    phase.add_boundary_value(PhaseRegionFlags::Back, 5, 0.0);

    phase.optimizer().set_econ_tol(1e-12);
    auto flag = phase.solve();
    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        return {}; // caller checks for empty
    }

    return phase.return_traj();
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    auto ode = make_cr3bp_ode(mu);

    // L1 Lyapunov initial guess
    Eigen::VectorXd ig(7);
    ig.setZero();
    ig[0] = 0.8234; // x
    ig[4] = 0.1263; // vy
    double tf = 1.3;

    std::cout << "=== Orbit Continuation: L1 Lyapunov orbits in Earth-Moon CR3BP ===\n\n";

    // Solve first orbit
    std::cout << "Solving initial Lyapunov orbit...\n";
    auto traj = solve_periodic(ode, ig, tf);

    if (traj.empty()) {
        std::cerr << "Failed to converge initial orbit\n";
        return 1;
    }

    double period = traj.back()[6];
    std::cout << "  Initial orbit: x0 = " << std::fixed << std::setprecision(6) << traj[0][0]
              << ", period = " << std::setprecision(4) << period << "\n";

    // Continuation: step x0 inward toward Moon
    const double dx = -0.001;
    const double x_limit = 0.80;
    const int max_steps = 20;
    int n_orbits = 1;

    std::vector<std::vector<Eigen::VectorXd>> orbit_family;
    orbit_family.push_back(traj);

    for (int step = 0; step < max_steps; ++step) {
        auto prev = orbit_family.back();
        Eigen::VectorXd next_ig = prev[0];
        double next_tf = prev.back()[6];
        next_ig[0] += dx;

        auto sol = solve_periodic(ode, next_ig, next_tf);
        if (sol.empty()) {
            std::cout << "  Continuation stopped: solver failed at step " << step << "\n";
            break;
        }

        orbit_family.push_back(sol);
        n_orbits++;

        std::cout << "  Step " << step + 1 << ": x0 = " << std::fixed << std::setprecision(6)
                  << sol[0][0] << ", period = " << std::setprecision(4) << sol.back()[6] << "\n";

        if (sol[0][0] < x_limit) {
            std::cout << "  Reached continuation limit\n";
            break;
        }
    }

    // ── Verification ───────────────────────────────────────────────────
    std::cout << "\n=== Results ===\n";
    std::cout << "  Total orbits computed: " << n_orbits << "\n";

    // Check periodicity of last orbit
    auto &last = orbit_family.back();
    double y_err = std::abs(last.back()[1]);
    double vx_err = std::abs(last.back()[3]);
    std::cout << "  Last orbit periodicity: |y(tf)| = " << std::scientific << std::setprecision(2)
              << y_err << ", |vx(tf)| = " << vx_err << "\n";

    bool ok = (n_orbits >= 3) && (y_err < 1e-8) && (vx_err < 1e-8);
    if (ok) {
        std::cout << "\nOrbitContinuation: PASSED\n";
        return 0;
    } else {
        std::cerr << "\nOrbitContinuation: FAILED (periodicity or orbit count insufficient)\n";
        return 1;
    }
}
