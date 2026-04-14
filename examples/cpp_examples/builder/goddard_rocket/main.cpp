// Source: Betts, "Practical Methods for OC", Cambridge, 2009, Section 4.14

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

static constexpr double g0 = 32.2;
static constexpr double W = 203000.0;

static constexpr double Lstar = 10000.0;    // feet
static constexpr double Tstar = 60.0;       // sec
static constexpr double Mstar = 1.0;        // slugs

static const double Vstar = Lstar / Tstar;
static const double Fstar = Mstar * Lstar / (Tstar * Tstar);
static const double Astar = Lstar / (Tstar * Tstar);
static const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
static const double sigmastar = Mstar / Lstar;

static const double rho0 = 0.002378 / Rhostar;
static const double h_ref = 23800.0 / Lstar;
static const double g = g0 / Astar;
static const double Tmag = 200.0 / Fstar;
static const double c = 1580.94 / Vstar;
static const double sigma = 5.4915e-5 / sigmastar;

static constexpr double m0 = 3.0;
static constexpr double mf = 1.0;

ODE make_goddard_ode() {
    return ODEBuilder(3, 1)
        .define([](auto &args) {
            auto h = args.x_var(0);
            auto v = args.x_var(1);
            auto m = args.x_var(2);
            auto u = args.u_var(0);

            auto hdot = v;
            auto vdot = (u * Tmag - sigma * (v * v) * exp(h * (-1.0 / h_ref))) / m - g;
            auto mdot = u * (-Tmag / c);

            return stack(hdot, vdot, mdot);
        })
        .var_names({{"h", 0}, {"v", 1}, {"m", 2}, {"t", 3}, {"u", 4}})
        .build();
}

// From Betts eq. 4.188: constraint that defines the singular arc control.
auto make_path_constraint() {
    auto args = Arguments<4>();
    auto h = args.coeff<0>();
    auto v = args.coeff<1>();
    auto m = args.coeff<2>();
    auto u = args.coeff<3>();

    auto t1 = (u * Tmag - sigma * (v * v) * exp(h * (-1.0 / h_ref))) - g * m;

    auto cv = c / v;
    auto t2 = (m * g / (1.0 + 4.0 * cv + 2.0 * cv * cv)) *
              (c * c * (1.0 + v / c) / (h_ref * g) - 1.0 - 2.0 * cv);

    return GenericFunction<-1, -1>(t1 - t2);
}

std::vector<Eigen::VectorXd> make_initial_guess(const ODE &ode) {
    // u = 1 while m > mf, else 0
    auto ulaw_args = Arguments<1>();
    Eigen::Matrix<double, 1, 1> one_val, zero_val;
    one_val << 1.0;
    zero_val << 0.0;
    auto ulaw_expr = IfElseFunction(ulaw_args.coeff<0>() > mf,
                                    Constant<1, 1>(1, one_val),
                                    Constant<1, 1>(1, zero_val));
    auto ulaw = GenericFunction<-1, -1>(ulaw_expr);

    auto integ = ode.integrator().step(0.01).control(ulaw, {"m"}).build();

    Eigen::VectorXd X0 = Eigen::VectorXd::Zero(5);
    X0[2] = m0;
    X0[4] = 1.0;

    // Stop when velocity goes negative (rocket falling)
    auto stop_fn = [](const Eigen::Ref<const Eigen::VectorXd> &x) { return x[1] < 0.0; };

    return integ.integrate_dense(X0, 60.0 / Tstar, 1000, stop_fn);
}

int main() {
    auto ode = make_goddard_ode();

    std::cout << "GoddardRocket (builder): generating initial guess ...\n" << std::flush;

    auto TrajIG = make_initial_guess(ode);
    if (TrajIG.size() < 10) {
        std::cerr << "GoddardRocket: initial guess too short\n";
        return 1;
    }

    std::cout << "  Initial guess: " << TrajIG.size() << " points, tf = "
              << TrajIG.back()[3] * Tstar << " s\n";
    const int n = static_cast<int>(TrajIG.size()) / 3;

    std::vector<Eigen::VectorXd> TrajIG1(TrajIG.begin(), TrajIG.begin() + n);
    std::vector<Eigen::VectorXd> TrajIG2(TrajIG.begin() + n, TrajIG.begin() + 2 * n);
    std::vector<Eigen::VectorXd> TrajIG3(TrajIG.begin() + 2 * n, TrajIG.end() - 1);

    const auto tmode = TranscriptionModes::LGL3;

    // Phase 1: full thrust (u = 1)
    auto phase1 = ode.phase(tmode, TrajIG1, 32);

    Eigen::VectorXd front_vals(4);
    front_vals << TrajIG[0][0], TrajIG[0][1], TrajIG[0][2], TrajIG[0][3];
    phase1.add_boundary_value(PhaseRegionFlags::Front, {"h", "v", "m", "t"}, front_vals);

    phase1.add_boundary_value(PhaseRegionFlags::Path, "u", 1.0);

    // Phase 2: singular arc
    auto phase2 = ode.phase(tmode, TrajIG2, 32);
    phase2.set_control_mode(ControlModes::NoSpline);

    phase2.add_lu_var_bound(PhaseRegionFlags::Path, "u", 0.0, 1.0, 1.0);

    auto path_con = make_path_constraint();
    phase2.add_equal_con(PhaseRegionFlags::Path, path_con, {"h", "v", "m", "u"});

    // Phase 3: coast (u = 0)
    auto phase3 = ode.phase(tmode, TrajIG3, 32);

    phase3.add_boundary_value(PhaseRegionFlags::Path, "u", 0.0);

    Eigen::VectorXd back_vals(2);
    back_vals << 0.0, mf;
    phase3.add_boundary_value(PhaseRegionFlags::Back, {"v", "m"}, back_vals);

    phase3.add_value_objective(PhaseRegionFlags::Back, "h", -1.0);

    OptimalControlProblem ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);
    ocp.add_phase(phase3);

    ocp.add_forward_link_equal_con(phase1, phase3, {"h", "v", "m", "t"});

    phase1.add_lower_delta_time_bound(0.0);
    phase2.add_lower_delta_time_bound(0.0);
    phase3.add_lower_delta_time_bound(0.0);

    std::cout << "Solving multi-phase Goddard rocket ...\n" << std::flush;

    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_print_level(1);

    const auto status = ocp.optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "GoddardRocket (builder): FAILED (status " << static_cast<int>(status)
                  << ")\n";
        return 1;
    }

    auto traj1 = phase1.return_traj();
    auto traj2 = phase2.return_traj();
    auto traj3 = phase3.return_traj();

    double h_final = traj3.back()[0] * Lstar;
    double t_final = traj3.back()[3] * Tstar;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GoddardRocket (builder): max altitude = " << h_final << " ft"
              << ", tf = " << t_final << " s\n";
    std::cout << "  Phase 1 (thrust): " << traj1.size() << " pts, dt = "
              << (traj1.back()[3] - traj1.front()[3]) * Tstar << " s\n";
    std::cout << "  Phase 2 (singular): " << traj2.size() << " pts, dt = "
              << (traj2.back()[3] - traj2.front()[3]) * Tstar << " s\n";
    std::cout << "  Phase 3 (coast): " << traj3.size() << " pts, dt = "
              << (traj3.back()[3] - traj3.front()[3]) * Tstar << " s\n";

    return 0;
}
