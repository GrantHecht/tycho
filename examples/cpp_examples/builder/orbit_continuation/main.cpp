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
    // Integrate initial guess using ODE::integrator() — matches Python
    // odeItg = ode.integrator(dt); trajGuess = odeItg.integrate_dense(ig, tf, 1000).
    auto integ = ode.integrator().step(integ_dt).build();
    auto trajGuess = integ.integrate_dense(ig, tf, 1000);

    constexpr int nSeg = 150;
    auto phase = ode.phase(TranscriptionModes::LGL3, trajGuess, nSeg);
    phase.set_num_partitions(8);

    // Fix specified initial-guess components (matches Python's fixInit loop).
    for (int idx : fix_init) {
        phase.add_boundary_value(PhaseRegionFlags::Front, idx, ig[idx]);
    }
    // Python also fixes vx=0 and t=0 at the front unconditionally (indices
    // [3, 6]). Half-period symmetry is enforced via the Back BC below.
    phase.add_boundary_value(PhaseRegionFlags::Front, 3, 0.0);
    phase.add_boundary_value(PhaseRegionFlags::Front, 6, 0.0);

    // Back BC: y = 0, vx = 0, vz = 0 — half-period crossing of the x-z plane.
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
// Basic continuation: step ig[cIdx] by dx until sign(x[cIdx] - lim) flips.
// Mirrors Python's `contin()` — runs until the continuation variable crosses
// `lim` (sign change of (x[cIdx] - lim) from the previous iterate).
///////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<Eigen::VectorXd>>
contin(const ODE &ode, const Eigen::VectorXd &ig, double tf, int cIdx, double dx, double lim,
       const std::vector<int> &fix_init = {0, 1, 2}) {
    std::vector<std::vector<Eigen::VectorXd>> traj_list;
    auto first = solve_periodic(ode, ig, tf, fix_init);
    if (first.empty())
        return traj_list;
    traj_list.push_back(first);

    int sign = (traj_list.back().front()[cIdx] - lim) >= 0.0 ? 1 : -1;
    int sign_last = sign;
    while (sign == sign_last) {
        Eigen::VectorXd g = traj_list.back().front();
        double t = traj_list.back().back()[6];
        g[cIdx] += dx;

        auto sol = solve_periodic(ode, g, t, fix_init);
        if (sol.empty())
            break;

        traj_list.push_back(sol);
        sign_last = sign;
        sign = (traj_list.back().front()[cIdx] - lim) >= 0.0 ? 1 : -1;
    }
    return traj_list;
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    auto ode = make_cr3bp_ode(mu);

    std::cout << "=== Orbit Continuation: CR3BP L1 families (Earth-Moon) ===\n\n";

    // ── L1 Lyapunov family ─────────────────────────────────────────────
    Eigen::VectorXd ig_lyap(7);
    ig_lyap.setZero();
    ig_lyap[0] = 0.8234; // x
    ig_lyap[4] = 0.1263; // vy
    double tf_lyap = 1.3;

    std::cout << "L1 Lyapunov continuation...\n";
    auto lyap_family = contin(ode, ig_lyap, tf_lyap, /*cIdx=*/0, /*dx=*/-0.001, /*lim=*/0.77);
    if (lyap_family.empty()) {
        std::cerr << "  Failed to converge initial Lyapunov orbit\n";
        return EXIT_FAILURE;
    }
    std::cout << "  Computed " << lyap_family.size() << " Lyapunov orbits\n";

    // ── Northern L1 Halo family ────────────────────────────────────────
    Eigen::VectorXd ig_halo(7);
    ig_halo.setZero();
    ig_halo[0] = 0.8234;
    ig_halo[4] = 0.1263;
    double tf_halo = 1.3715;

    std::cout << "L1 Halo continuation...\n";
    auto halo_family =
        contin(ode, ig_halo, tf_halo, /*cIdx=*/2, /*dx=*/0.001, /*lim=*/0.214, {1, 2, 5});
    if (halo_family.empty()) {
        std::cerr << "  Failed to converge initial Halo orbit\n";
        return EXIT_FAILURE;
    }
    std::cout << "  Computed " << halo_family.size() << " Halo orbits\n";

    // ── Verification ───────────────────────────────────────────────────
    std::cout << "\n=== Results ===\n";
    std::cout << "  Lyapunov orbits: " << lyap_family.size() << "\n";
    std::cout << "  Halo orbits:     " << halo_family.size() << "\n";

    // Check periodicity (y, vx) at half-period crossing for both families.
    auto check_periodicity = [](const std::vector<Eigen::VectorXd> &orbit, const char *name) {
        double y_err = std::abs(orbit.back()[1]);
        double vx_err = std::abs(orbit.back()[3]);
        std::cout << "  " << name << " last orbit: |y(tf)| = " << std::scientific
                  << std::setprecision(2) << y_err << ", |vx(tf)| = " << vx_err << "\n";
        return y_err < 1e-8 && vx_err < 1e-8;
    };
    bool ok = check_periodicity(lyap_family.back(), "Lyapunov") &&
              check_periodicity(halo_family.back(), "Halo") && lyap_family.size() >= 3 &&
              halo_family.size() >= 3;
    if (ok) {
        std::cout << "\nOrbitContinuation: PASSED\n";
        return EXIT_SUCCESS;
    }
    std::cerr << "\nOrbitContinuation: FAILED (periodicity or orbit count insufficient)\n";
    return EXIT_FAILURE;
}
