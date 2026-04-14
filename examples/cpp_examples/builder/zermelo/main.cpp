#include <tycho/tycho.h>
#include <cmath>
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

std::vector<Eigen::VectorXd> navigate(ODE &ode, const Eigen::VectorXd &A,
                                      const Eigen::VectorXd &B, double vMax) {
    constexpr int nSeg = 250;
    constexpr double tol = 1e-12;

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

    auto phase = ode.phase(TranscriptionModes::LGL3, trajG, nSeg);
    phase.set_num_partitions(10);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "y"}, A);
    phase.add_boundary_value(PhaseRegionFlags::Front, "t", 0.0);
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "y"}, B);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "theta", -M_PI, M_PI);

    phase.add_delta_time_objective(1.0);

    phase.optimizer().set_econ_tol(tol);
    phase.optimizer().set_kkt_tol(tol);

    const auto status = phase.solve_optimize();
    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "  FAILED: navigation did not converge (status " << static_cast<int>(status)
                  << ")\n";
        return {};
    }
    return phase.return_traj();
}

ODE make_no_wind(double vMax) {
    return ODEBuilder(2, 1)
        .define([vMax](auto &args) {
            auto theta = args.u_var(0);
            return stack(vMax * cos(theta), vMax * sin(theta));
        })
        .var_names({{"x", 0}, {"y", 1}, {"t", 2}, {"theta", 3}})
        .build();
}

ODE make_uniform_wind(double vMax, double wVel, double wAng) {
    double wx = wVel * std::cos(wAng);
    double wy = wVel * std::sin(wAng);
    return ODEBuilder(2, 1)
        .define([vMax, wx, wy](auto &args) {
            auto theta = args.u_var(0);
            return stack(vMax * cos(theta) + wx, vMax * sin(theta) + wy);
        })
        .var_names({{"x", 0}, {"y", 1}, {"t", 2}, {"theta", 3}})
        .build();
}

ODE make_const_dir_wind(double vMax, double wAng) {
    // Wind speed depends on position: vel = cos(||pos||)
    // Direction is constant.
    double cwAng = std::cos(wAng);
    double swAng = std::sin(wAng);

    // Position-dependent wind requires the full-vector norm of pos,
    // so we use ODEArguments + from() to build the expression directly.
    auto XtU = ODEArguments(2, 1, 0);
    auto pos = XtU.segment(0, 2);
    auto theta = XtU.segment(XtU.xt_vars(), 1).coeff(0);
    auto vel = cos(pos.norm());

    auto expr = stack(vMax * cos(theta) + vel * cwAng, vMax * sin(theta) + vel * swAng);

    return ODEBuilder(2, 1)
        .from(expr)
        .var_names({{"x", 0}, {"y", 1}, {"t", 2}, {"theta", 3}})
        .build();
}

ODE make_var_wind(double vMax) {
    // Wind speed and direction both depend on position.
    // vel = sin(||pos||), ang = 2*(x+y)
    auto XtU = ODEArguments(2, 1, 0);
    auto pos = XtU.segment(0, 2);
    auto x = XtU.segment(0, 2).coeff(0);
    auto y = XtU.segment(0, 2).coeff(1);
    auto theta = XtU.segment(XtU.xt_vars(), 1).coeff(0);
    auto vel = sin(pos.norm());
    auto ang = 2.0 * (x + y);

    auto expr = stack(vMax * cos(theta) + vel * cos(ang), vMax * sin(theta) + vel * sin(ang));

    return ODEBuilder(2, 1)
        .from(expr)
        .var_names({{"x", 0}, {"y", 1}, {"t", 2}, {"theta", 3}})
        .build();
}

int main() {
    int failures = 0;

    {
        auto ode = make_var_wind(1.25);
        Eigen::VectorXd tp(4);
        tp << 0.2, -0.5, 0.5, 1.0;
        auto &func = ode.function();
        Eigen::VectorXd fx(2);
        func.compute(tp, fx);
        Eigen::MatrixXd jac(2, 4);
        func.jacobian(tp, jac);
        bool ok = std::abs(jac(0, 1) - (-0.0787)) < 0.01;
        std::cout << "ODE Jacobian dy column: " << jac(0, 1) << " " << jac(1, 1)
                  << (ok ? "  OK" : "  BUG") << "\n\n";
        if (!ok) ++failures;
    }

    Eigen::VectorXd A(2), B(2);
    A << 0.0, -1.0;
    B << 1.0, 1.0;

    std::cout << "Solving: no wind ... " << std::flush;
    auto ode_nw = make_no_wind(1.0);
    auto traj1 = navigate(ode_nw, A, B, 1.0);
    if (traj1.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        std::cout << std::fixed << std::setprecision(4) << "tf = " << traj1.back()[2] << " s\n";
    }

    std::cout << "Solving: uniform wind ... " << std::flush;
    constexpr double vM = 1.25;
    auto ode_uw = make_uniform_wind(vM, 0.5, 135.0 * M_PI / 180.0);
    auto traj2 = navigate(ode_uw, A, B, vM);
    if (traj2.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        std::cout << "tf = " << traj2.back()[2] << " s\n";
    }

    std::cout << "Solving: constant-direction wind ... " << std::flush;
    auto ode_cd = make_const_dir_wind(vM, 45.0 * M_PI / 180.0);
    auto traj3 = navigate(ode_cd, A, B, vM);
    if (traj3.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        std::cout << "tf = " << traj3.back()[2] << " s\n";
    }

    std::cout << "Solving: variable-direction wind ... " << std::flush;
    auto ode_vw = make_var_wind(vM);
    auto traj4 = navigate(ode_vw, A, B, vM);
    if (traj4.empty()) {
        ++failures;
        std::cout << "FAILED\n";
    } else {
        std::cout << "tf = " << traj4.back()[2] << " s\n";
    }

    std::cout << "\nZermelo (builder): " << (4 - failures) << "/4 wind models converged\n";
    return failures > 0 ? 1 : 0;
}
