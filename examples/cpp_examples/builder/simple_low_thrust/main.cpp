#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double mu = 1.0;
    constexpr double acc = 0.02;

    const double r0 = 1.0;
    const double v0 = std::sqrt(mu / r0);

    const double rf = 2.0;
    const double vf_val = std::sqrt(mu / rf);

    Eigen::VectorXd X0(7);
    X0.setZero();
    X0[0] = r0;
    X0[4] = v0;

    Eigen::VectorXd Xf(6);
    Xf.setZero();
    Xf[0] = rf;
    Xf[4] = vf_val;

    auto args = ODEArguments(6, 3, 0);
    auto R = args.head<3>();
    auto V = args.segment<3>(3);
    auto u = args.tail<3>();

    auto grav = (-mu) * R.normalized_power<3>();
    auto thrust = acc * u;
    auto ode_expr = StackedOutputs{V, grav + thrust};

    auto ode = ODE(ode_expr, 6, 3)
                   .var_group("R", 0, 3)
                   .var_group("V", 3, 3)
                   .var_names({{"t", 6}})
                   .var_group("u", 7, 3);

    // Forward-integrate with prograde thrust so the IG satisfies the
    // dynamics exactly (defects ≈ 0). A hand-rolled analytic spiral leaves
    // O(1) defects and places the solver on a near-saddle that is extremely
    // sensitive to floating-point ordering.
    Eigen::VectorXd XIG(10);
    XIG.setZero();
    XIG.head<7>() = X0;

    auto u_law_args = Arguments<3>();
    auto u_law = GenericFunction<-1, -1>(u_law_args.normalized() * 0.8);

    auto ig_integ = ode.integrator().step(0.01).control(u_law, {"V"}).build();
    auto trajIG = ig_integ.integrate_dense(XIG, 6.4 * M_PI, 100);

    constexpr int nSeg = 256;
    auto phase = ode.phase(TranscriptionModes::LGL3, trajIG, nSeg);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"R", "V", "t"}, X0);

    phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.001, 1.0, 1.0);

    phase.add_boundary_value(PhaseRegionFlags::Back, {"R", "V"}, Xf);

    phase.optimizer().set_print_level(1);
    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_delta_h(1.0e-6);

    std::cout << "=== Phase 1: Minimum time transfer ===\n";
    phase.add_delta_time_objective(1.0);
    auto flag_time = phase.solve_optimize();
    if (flag_time > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "SimpleLowThrust: time-optimal solve_optimize failed\n";
        return EXIT_FAILURE;
    }

    auto time_optimal = phase.return_traj();
    double tf_time = time_optimal.back()[6];
    std::cout << "  Time-optimal tf = " << std::fixed << std::setprecision(4) << tf_time
              << " (ND)\n";

    phase.remove_state_objective(-1);

    std::cout << "\n=== Phase 2: Minimum power transfer ===\n";
    {
        auto u_args = Arguments<3>();
        auto obj_expr = u_args.squared_norm() * 0.5;
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"u"});
    }

    phase.optimizer().set_max_ls_iters(0);
    auto flag_power = phase.optimize();
    if (flag_power > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "SimpleLowThrust: power-optimal optimize failed\n";
        return EXIT_FAILURE;
    }

    auto power_optimal = phase.return_traj();
    double tf_power = power_optimal.back()[6];
    std::cout << "  Power-optimal tf = " << std::fixed << std::setprecision(4) << tf_power
              << " (ND)\n";

    phase.remove_integral_objective(-1);

    std::cout << "\n=== Phase 3: Minimum fuel transfer ===\n";
    {
        auto u_args = Arguments<3>();
        auto obj_expr = u_args.norm();
        phase.add_integral_objective(GenericFunction<-1, 1>(obj_expr), {"u"});
    }
    phase.optimizer().set_max_ls_iters(2);
    auto flag_mass = phase.optimize();
    if (flag_mass > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "SimpleLowThrust: mass-optimal optimize failed\n";
        return EXIT_FAILURE;
    }

    auto mass_optimal = phase.return_traj();
    double tf_mass = mass_optimal.back()[6];
    std::cout << "  Mass-optimal tf = " << std::fixed << std::setprecision(4) << tf_mass
              << " (ND)\n";

    bool ok = true;
    auto check_traj = [&](const std::string &name, const std::vector<Eigen::VectorXd> &traj) {
        double rfinal = traj.back().head<3>().norm();
        double vfinal = traj.back().segment<3>(3).norm();
        double r_err = std::abs(rfinal - rf);
        double v_err = std::abs(vfinal - vf_val);
        std::cout << "  " << name << ": rf=" << std::fixed << std::setprecision(6) << rfinal
                  << " (err=" << std::scientific << std::setprecision(2) << r_err << ")"
                  << ", vf=" << std::fixed << std::setprecision(6) << vfinal
                  << " (err=" << std::scientific << std::setprecision(2) << v_err << ")\n";
        if (r_err > 1e-4 || v_err > 1e-4) {
            std::cerr << "  WARNING: " << name << " boundary error elevated\n";
            ok = false;
        }
    };

    std::cout << "\n=== Verification ===\n";
    check_traj("TimeOptimal", time_optimal);
    check_traj("PowerOptimal", power_optimal);
    check_traj("MassOptimal", mass_optimal);

    if (ok) {
        std::cout << "\nSimpleLowThrust: PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSimpleLowThrust: FAILED (elevated boundary errors)\n";
        return 1;
    }
}
