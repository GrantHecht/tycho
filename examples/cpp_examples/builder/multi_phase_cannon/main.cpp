// Source: Dymos optimal control library (OpenMDAO)

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

static constexpr double g0 = 9.81;
static constexpr double Lstar = 1000.0;    // m
static constexpr double Tstar = 60.0;      // sec
static constexpr double Mstar = 10.0;      // kg
static const double Astar = Lstar / (Tstar * Tstar);
static const double Vstar = Lstar / Tstar;
static const double Rhostar = Mstar / (Lstar * Lstar * Lstar);
static const double Estar = Mstar * (Vstar * Vstar);

static constexpr double CD = 0.5;
static const double RhoAir = 1.225 / Rhostar;
static const double RhoIron = 7870.0 / Rhostar;
static const double h_scale = 8.44e3 / Lstar;
static const double E0 = 400000.0 / Estar;
static const double g = g0 / Astar;

static double MFunc_val(double rad) {
    return (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
}

static double SFunc_val(double rad) {
    return M_PI * (rad * rad);
}

ODE make_cannon_ode() {
    return ODEBuilder(4, 0, 1)
        .define([](auto &args) {
            auto v     = args.x_var(0);
            auto gamma = args.x_var(1);
            auto h     = args.x_var(2);
            auto r     = args.x_var(3);
            auto rad   = args.p_var(0);

            auto M = (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
            auto S = M_PI * (rad * rad);

            auto rho = RhoAir * exp(h * (-1.0 / h_scale));

            auto D = (0.5 * CD) * rho * (v * v) * S;

            auto vdot     = D * (-1.0) / M - g * sin(gamma);
            auto gammadot = g * cos(gamma) * (-1.0) / v;
            auto hdot     = v * sin(gamma);
            auto rdot     = v * cos(gamma);

            return stack(vdot, gammadot, hdot, rdot);
        })
        .var_names({{"v", 0}, {"gamma", 1}, {"h", 2}, {"r", 3}, {"t", 4}, {"rad", 5}})
        .build();
}

// Energy constraint: 0.5 * M(rad) * v^2 - E0 <= 0, scaled by 0.01.
auto make_energy_constraint() {
    auto args = Arguments<2>();
    auto v   = args.coeff<0>();
    auto rad = args.coeff<1>();

    auto M = (4.0 / 3.0) * (M_PI * RhoIron) * (rad * rad * rad);
    auto E = 0.5 * M * (v * v);
    return GenericFunction<-1, -1>((E - E0) * 0.01);
}

int main() {
    const double rad0 = 0.1 / Lstar;
    const double h0 = 100.0 / Lstar;
    const double r0 = 0.0;
    const double m0 = MFunc_val(rad0);
    const double gamma0 = 45.0 * M_PI / 180.0;
    const double v0 = std::sqrt(2.0 * E0 / m0) * 0.99;

    auto ode = make_cannon_ode();

    std::cout << "MultiPhaseCannon (builder): generating initial guesses ...\n" << std::flush;

    auto integ = ode.integrator().step(0.01).build();
    integ.set_abs_tol(1.0e-14);

    Eigen::VectorXd IG = Eigen::VectorXd::Zero(6);
    IG[0] = v0;
    IG[1] = gamma0;
    IG[2] = h0;
    IG[3] = r0;
    IG[5] = rad0;

    // Ascent event: v*sin(gamma) crossing zero (hdot = 0 at apogee)
    using EventPack = decltype(integ)::EventPack;

    auto ascent_args = ODEArguments(4, 0, 1);
    auto ascent_event_expr = ascent_args.coeff(0) * sin(ascent_args.coeff(1));
    auto ascent_event = GenericFunction<-1, 1>(ascent_event_expr);

    auto [ascent_traj, ascent_events] =
        integ.integrate_dense(IG, 60.0 / Tstar,
                              std::vector<EventPack>{EventPack{ascent_event, 0, 1}}, false);

    // Descent event: h crossing zero (ground impact)
    auto descent_args = ODEArguments(4, 0, 1);
    auto descent_event_expr = descent_args.coeff(2);
    auto descent_event = GenericFunction<-1, 1>(descent_event_expr);

    auto [descent_traj, descent_events] =
        integ.integrate_dense(ascent_traj.back(), ascent_traj.back()[4] + 1000.0 / Tstar,
                              std::vector<EventPack>{EventPack{descent_event, 0, 1}}, false);

    std::cout << "  Ascent IG: " << ascent_traj.size() << " pts\n";
    std::cout << "  Descent IG: " << descent_traj.size() << " pts\n";
    const auto tmode = TranscriptionModes::LGL5;
    const int nsegs = 128;

    auto aphase = ode.phase(tmode, ascent_traj, nsegs);

    // ODEParams region expects raw ODE param index, not phase vector index.
    aphase.add_lower_var_bound(PhaseRegionFlags::ODEParams, 0, 0.0, 1.0);

    aphase.add_lower_var_bound(PhaseRegionFlags::Front, "gamma", 0.0, 1.0);

    Eigen::VectorXd front_vals(3);
    front_vals << h0, r0, 0.0;
    aphase.add_boundary_value(PhaseRegionFlags::Front, {"h", "r", "t"}, front_vals);

    // Energy inequality references state var v (xvars=0) and ODE param rad (pvars=0).
    {
        auto efunc = make_energy_constraint();
        Eigen::VectorXi xvars(1); xvars << 0;
        Eigen::VectorXi pvars(1); pvars << 0;
        Eigen::VectorXi empty;
        aphase.add_inequal_con(
            PhaseRegionFlags::Front, efunc, xvars, pvars, empty, ScaleModes::AUTO);
    }

    aphase.add_boundary_value(PhaseRegionFlags::Back, "gamma", 0.0);

    auto dphase = ode.phase(tmode, descent_traj, nsegs);

    dphase.add_boundary_value(PhaseRegionFlags::Back, "h", 0.0);

    dphase.add_value_objective(PhaseRegionFlags::Back, "r", -1.0);

    OptimalControlProblem ocp;
    ocp.add_phase(aphase);
    ocp.add_phase(dphase);

    ocp.add_forward_link_equal_con(aphase, dphase, {"v", "gamma", "h", "r", "t"});

    {
        Eigen::VectorXi pvar(1); pvar << 0;
        ocp.add_direct_link_equal_con(
            0, PhaseRegionFlags::ODEParams, pvar,
            1, PhaseRegionFlags::ODEParams, pvar);
    }

    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_print_level(1);

    std::cout << "Solving multi-phase cannon ...\n" << std::flush;
    const auto status = ocp.optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "MultiPhaseCannon (builder): FAILED (status "
                  << static_cast<int>(status) << ")\n";
        return 1;
    }

    auto ascentTraj = aphase.return_traj();
    auto descentTraj = dphase.return_traj();

    double launch_angle = ascentTraj.front()[1] * 180.0 / M_PI;
    double opt_range = descentTraj.back()[3] * Lstar;
    double opt_rad = descentTraj.back()[5] * Lstar;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "MultiPhaseCannon (builder):\n";
    std::cout << "  Launch angle: " << launch_angle << " deg\n";
    std::cout << "  Optimized range: " << opt_range << " m\n";
    std::cout << "  Optimized radius: " << opt_rad << " m\n";

    return 0;
}
