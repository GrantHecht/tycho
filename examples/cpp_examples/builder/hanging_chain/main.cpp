///////////////////////////////////////////////////////////////////////////////
// Hanging Chain — C++ example (Builder API)
//
// Ported from examples/python_examples/HangingChain.py
//
// State  : [x]   (chain height)
// Control: [u]   (slope dx/ds)
//
// ODE: xdot = u  (parameterised by arc length s in [0,1])
//
// Integral constraint: integral sqrt(1 + u^2) ds = L  (chain length)
// Objective: minimise integral x * sqrt(1 + u^2) ds  (gravitational energy)
//
// NOTE: Python example uses Jet.map() for parallel sweep over L values.
//       This C++ version solves a single chain configuration (L=4).
//       Jet parallel solve is a documented API gap.
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
    constexpr double a = 1.0;   // left endpoint height
    constexpr double b = 3.0;   // right endpoint height
    constexpr double L = 4.0;   // chain length
    constexpr int n_segs = 500;
    constexpr int n_pts = n_segs;

    // ── Define ODE: xdot = u ───────────────────────────────────────────
    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) { return args.u_var(0); })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    // ── Energy integrand: x * sqrt(1 + u^2) ───────────────────────────
    auto energy_args = Arguments<2>();
    auto ex = energy_args.coeff<0>();
    auto eu = energy_args.coeff<1>();
    auto energy_expr = ex * sqrt(1.0 + eu * eu);

    // ── Length integrand: sqrt(1 + u^2) ────────────────────────────────
    auto len_args = Arguments<1>();
    auto lu = len_args.coeff<0>();
    auto length_expr = sqrt(1.0 + lu * lu);

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    // Match Python: tm=0.25 when b>a
    constexpr double tm = 0.25;
    for (int i = 0; i < n_pts; ++i) {
        const double s = static_cast<double>(i) / (n_pts - 1);
        double x_val = 2.0 * std::abs(b - a) * s * (s - 2.0 * tm) + a;
        double u_val = 2.0 * std::abs(b - a) * (2.0 * s - 2.0 * tm);
        Eigen::VectorXd pt(3);
        pt << x_val, s, u_val;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    Eigen::VectorXd sp(1);
    sp[0] = L;
    phase.set_static_params(sp);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"}, Eigen::Vector2d(a, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "t"}, Eigen::Vector2d(b, 1.0));

    // Lock static param to L
    phase.add_boundary_value(PhaseRegionFlags::StaticParams, Eigen::VectorXi::Constant(1, 0),
                             Eigen::Matrix<double, 1, 1>(L));

    // Upper bound on x
    phase.add_upper_var_bound(PhaseRegionFlags::Path, "x", std::max(a, b) + 0.001);

    // Integral objective: minimise gravitational energy
    phase.add_integral_objective(GenericFunction<-1, 1>(energy_expr), {"x", "u"});

    // Integral parameter constraint: length = L
    phase.add_integral_param_function(GenericFunction<-1, 1>(length_expr), {"u"}, 0);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_print_level(1);

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.solve_optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "HangingChain (builder): FAILED (status " << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    auto traj = phase.return_traj();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "HangingChain (builder): x(0) = " << traj.front()[0]
              << ", x(1) = " << traj.back()[0] << "\n";

    // Verify endpoints match boundary conditions
    if (std::abs(traj.front()[0] - a) > 1e-4 || std::abs(traj.back()[0] - b) > 1e-4) {
        std::cerr << "HangingChain (builder): FAILED — endpoints incorrect\n";
        return EXIT_FAILURE;
    }

    std::cout << "HangingChain (builder): PASSED\n";
    return EXIT_SUCCESS;
}
