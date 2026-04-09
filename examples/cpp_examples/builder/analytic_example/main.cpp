///////////////////////////////////////////////////////////////////////////////
// Analytic Example — C++ example (Builder API)
//
// Ported from examples/python_examples/AnalyticExample.py
// Source: https://www.hindawi.com/journals/aaa/2014/851720/
//
// State  : [x]   (1 state)
// Control: [u]   (1 control)
//
// ODE: xdot = 0.5*x + u
// Objective: minimise integral(u^2 + x*u + 1.25*x^2)
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
    constexpr double x0_val = 1.0;
    constexpr double t0 = 0.0;
    constexpr double tf = 1.0;
    constexpr int n_pts = 100;
    constexpr int n_segs = 20;

    // ── Define ODE ─────────────────────────────────────────────────────
    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) {
                       auto x = args.x_var(0);
                       auto u = args.u_var(0);
                       return 0.5 * x + u;
                   })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    // ── Initial guess ──────────────────────────────────────────────────
    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        const double t = t0 + (tf - t0) * static_cast<double>(i) / (n_pts - 1);
        Eigen::VectorXd pt(3);
        pt << x0_val, t, 0.0;
        traj_ig.push_back(pt);
    }

    // ── Phase setup ────────────────────────────────────────────────────
    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"},
                            Eigen::Vector2d(x0_val, t0));
    phase.add_boundary_value(PhaseRegionFlags::Back, "t", tf);

    // Integral objective: min integral(u^2 + x*u + 1.25*x^2)
    {
        auto obj_args = Arguments<2>();
        auto x = obj_args.coeff<0>();
        auto u = obj_args.coeff<1>();
        auto obj_expr = u * u + x * u + 1.25 * x * x;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"x", "u"});
    }

    // ── Solve ──────────────────────────────────────────────────────────
    const auto flag = phase.optimize();

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "AnalyticExample (builder): FAILED (status "
                  << static_cast<int>(flag) << ")\n";
        return EXIT_FAILURE;
    }

    // Verify against analytic solution at t=0:
    // u*(t) = -(tanh(1-t) + 0.5) * cosh(1-t) / cosh(1)
    // u*(0) = -(tanh(1) + 0.5) * cosh(1) / cosh(1) = -(tanh(1) + 0.5)
    auto traj = phase.return_traj();
    const double u_colloc = traj.front()[2];
    const double u_analytic = -(std::tanh(1.0) + 0.5);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "AnalyticExample (builder): u(0) collocation = " << u_colloc << "\n";
    std::cout << "AnalyticExample (builder): u(0) analytic    = " << u_analytic << "\n";

    if (std::abs(u_colloc - u_analytic) > 1e-4) {
        std::cerr << "AnalyticExample (builder): FAILED — control mismatch\n";
        return EXIT_FAILURE;
    }

    std::cout << "AnalyticExample (builder): PASSED\n";
    return EXIT_SUCCESS;
}
