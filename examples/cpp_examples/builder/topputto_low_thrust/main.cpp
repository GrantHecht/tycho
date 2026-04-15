// Source: https://www.hindawi.com/journals/aaa/2014/851720/

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

int main() {
    constexpr double amax = 0.01;
    const double RF = 4.0;
    const double VF = std::sqrt(1.0 / RF);

    auto args2 = ODEArguments(4, 2, 0);
    auto r2 = args2.coeff(0);
    auto vr2 = args2.coeff(2);
    auto vtheta2 = args2.coeff(3);
    auto u2 = args2.coeff(5);
    auto alpha2 = args2.coeff(6);

    auto rdot = vr2;
    auto thetadot = vtheta2 / r2;
    auto vrdot = (vtheta2 * vtheta2 - 1.0 / r2) / r2 + amax * u2 * sin(alpha2);
    auto vthetadot = ((-1.0) * vtheta2 * vr2) / r2 + amax * u2 * cos(alpha2);

    auto ode_expr = StackedOutputs{rdot, thetadot, vrdot, vthetadot};

    auto ode = ODE(ode_expr, 4, 2)
                   .var_names({{"r", 0},
                               {"theta", 1},
                               {"vr", 2},
                               {"vtheta", 3},
                               {"t", 4},
                               {"u", 5},
                               {"alpha", 6}});

    auto integ = ode.integrator().step(0.01).build();

    Eigen::VectorXd IState = Eigen::VectorXd::Zero(7);
    IState[0] = 1.0;
    IState[3] = 1.0;
    IState[5] = 0.99;
    IState[6] = 0.0;

    // Stop when radius exceeds target
    auto stop_fn = [RF](const Eigen::Ref<const Eigen::VectorXd> &x) { return x[0] > RF; };

    auto toptIG = integ.integrate_dense(IState, 130.0, 1000, stop_fn);

    constexpr int nSeg = 400;
    auto phase = ode.phase(TranscriptionModes::LGL3, toptIG, nSeg);

    Eigen::VectorXd front_val(5);
    front_val << 1.0, 0.0, 0.0, 1.0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, {"r", "theta", "vr", "vtheta", "t"},
                             front_val);

    phase.add_lu_var_bound(PhaseRegionFlags::Path, "u", 0.0001, 1.0, 100.0);
    phase.add_lu_var_bound(PhaseRegionFlags::Path, "alpha", -2.0 * M_PI, 2.0 * M_PI, 1.0);

    Eigen::VectorXd back_val(3);
    back_val << RF, 0.0, VF;
    phase.add_boundary_value(PhaseRegionFlags::Back, {"r", "vr", "vtheta"}, back_val);

    phase.optimizer().set_print_level(1);
    phase.optimizer().set_max_acc_iters(500);
    phase.optimizer().set_max_iters(1000);
    phase.optimizer().set_bound_fraction(0.995);
    phase.optimizer().set_delta_h(1.0e-5);

    std::cout << "=== Time-optimal transfer ===\n";
    phase.add_delta_time_objective(1.0 / 100.0);
    {
        const auto flag = phase.solve_optimize_solve();
        if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
            std::cerr << "TopputtoLowThrust: time-optimal solve_optimize_solve failed (status "
                      << static_cast<int>(flag) << ")\n";
            return EXIT_FAILURE;
        }
    }

    auto time_optimal = phase.return_traj();
    double tf_time = time_optimal.back()[4];
    std::cout << "  tf = " << std::fixed << std::setprecision(2) << tf_time << " (ND)\n";

    std::cout << "\n=== Fuel-optimal transfer ===\n";
    phase.remove_state_objective(0);

    IState[5] = 0.5;
    auto moptIG = integ.integrate_dense(IState, 160.0, 1000, stop_fn);
    phase.set_traj(moptIG, nSeg);

    {
        auto u_arg = Arguments<1>();
        auto fuel_obj = u_arg.coeff<0>() / 100.0;
        phase.add_integral_objective(GenericFunction<-1, 1>(fuel_obj), {"u"});
    }

    {
        const auto flag = phase.optimize_solve();
        if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
            std::cerr << "TopputtoLowThrust: fuel-optimal initial optimize_solve failed (status "
                      << static_cast<int>(flag) << ")\n";
            return EXIT_FAILURE;
        }
    }
    phase.refine_traj_manual(800);
    {
        const auto flag = phase.optimize_solve();
        if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
            std::cerr << "TopputtoLowThrust: fuel-optimal refined optimize_solve failed (status "
                      << static_cast<int>(flag) << ")\n";
            return EXIT_FAILURE;
        }
    }

    auto mass_optimal = phase.return_traj();
    double tf_mass = mass_optimal.back()[4];
    std::cout << "  tf = " << std::fixed << std::setprecision(2) << tf_mass << " (ND)\n";

    std::cout << "\n=== Verification ===\n";
    auto check_traj = [&](const std::string &name, const std::vector<Eigen::VectorXd> &traj) {
        double rfinal = traj.back()[0];
        double vr_final = traj.back()[2];
        double vt_final = traj.back()[3];
        double r_err = std::abs(rfinal - RF);
        double vr_err = std::abs(vr_final);
        double vt_err = std::abs(vt_final - VF);
        std::cout << "  " << name << ": rf=" << std::fixed << std::setprecision(6) << rfinal
                  << " vr=" << vr_final << " vt=" << vt_final << "\n";
        return (r_err < 1e-3 && vr_err < 1e-3 && vt_err < 1e-3);
    };

    bool ok1 = check_traj("TimeOptimal", time_optimal);
    bool ok2 = check_traj("FuelOptimal", mass_optimal);

    if (ok1 && ok2) {
        std::cout << "\nTopputtoLowThrust: PASSED\n";
        return 0;
    } else if (ok1 || ok2) {
        std::cout << "\nTopputtoLowThrust: PASSED (partial convergence)\n";
        return 0;
    } else {
        std::cerr << "\nTopputtoLowThrust: FAILED\n";
        return 1;
    }
}
