///////////////////////////////////////////////////////////////////////////////
// Bryson-Denham — C++ example (Builder API)
//
// Ported from examples/python_examples/BrysonDenham.py
//
// State  : [x, v]   (position, velocity)
// Control: [u]       (acceleration)
//
// ODE: xdot = v, vdot = u
// Constraint: x <= 1/9
// Objective: minimise integral(u^2 / 2)
///////////////////////////////////////////////////////////////////////////////

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
    constexpr int n_pts = 100;
    constexpr int n_segs = 32;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(2, 1)
                   .define([](auto &args) {
                       auto v = args.x_var(1);
                       auto u = args.u_var(0);
                       return stack(v, u);
                   })
                   .var_names({{"x", 0}, {"v", 1}, {"t", 2}, {"u", 3}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(4);
        pt << 0.0, 1.0 - 2.0 * s, s, 0.0;  // v: 1 -> -1
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "v", "t"},
                            Eigen::Vector3d(0.0, 1.0, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "v", "t"},
                            Eigen::Vector3d(0.0, -1.0, 1.0));

    // State upper bound: x <= 1/9
    phase.add_upper_var_bound(PhaseRegionFlags::Path, "x", 1.0 / 9.0);

    // Integral objective: min integral(u^2 / 2)
    {
        auto obj_args = Arguments<1>();
        auto obj_expr = (obj_args.coeff<0>() * obj_args.coeff<0>()) / 2.0;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"u"});
    }

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_kkt_tol(1.0e-10);
    phase.optimizer().set_print_level(0);
    phase.set_num_partitions(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "BrysonDenham (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();

    // Verify state upper bound is respected
    double max_x = -1e30;
    for (const auto &pt : traj) {
        max_x = std::max(max_x, pt[0]);
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "BrysonDenham (builder): max x = " << max_x << "\n";

    if (max_x > 1.0 / 9.0 + 1e-4) {
        std::cerr << "BrysonDenham (builder): FAILED — state bound violated\n";
        return EXIT_FAILURE;
    }

    std::cout << "BrysonDenham (builder): PASSED\n";
    return EXIT_SUCCESS;
}
