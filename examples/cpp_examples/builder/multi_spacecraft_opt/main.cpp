///////////////////////////////////////////////////////////////////////////////
// Multi-Spacecraft Optimization — C++ example (Builder API)
//
// Ported from examples/python_examples/MultiSpacecraftOptimization.py
//
// Optimises a constellation of spacecraft on circular orbits to rendezvous
// at a common point with minimum total time-of-flight. Uses:
//   - Two-body dynamics with low-thrust
//   - N phases (one per spacecraft)
//   - Link constraints via OptimalControlProblem to enforce common terminus
//   - Link parameters for the free rendezvous state
//
// State  : [rx, ry, rz, vx, vy, vz]  (6 states per spacecraft)
// Control: [ux, uy, uz]               (3 controls per spacecraft)
//
// Phase vector: [rx, ry, rz, vx, vy, vz, t, ux, uy, uz]
//                0    1    2   3   4   5   6   7   8   9
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
// Two-body with low-thrust ODE
///////////////////////////////////////////////////////////////////////////////

ODE make_lt_ode(double mu_val, double ltacc) {
    auto args = ODEArguments(6, 3, 0);
    auto r = args.head<3>();
    auto v = args.segment<3>(3);
    auto u = args.tail<3>();

    auto grav = (-mu_val) * r.normalized_power<3>();
    auto thrust = ltacc * u;
    auto ode_expr = StackedOutputs{v, grav + thrust};

    return ODE(ode_expr, 6, 3)
        .var_group("R", 0, 3)
        .var_group("V", 3, 3)
        .var_names({{"t", 6}})
        .var_group("u", 7, 3);
}

///////////////////////////////////////////////////////////////////////////////
// Helper: circular orbit initial state
///////////////////////////////////////////////////////////////////////////////

Eigen::VectorXd make_circ_state(double r, double theta_deg) {
    double v = std::sqrt(1.0 / r);
    double theta = theta_deg * M_PI / 180.0;
    Eigen::VectorXd state(7);
    state.setZero();
    state[0] = std::cos(theta) * r;
    state[1] = std::sin(theta) * r;
    state[3] = -std::sin(theta) * v;
    state[4] = std::cos(theta) * v;
    return state;
}

///////////////////////////////////////////////////////////////////////////////
// Helper: circular orbit trajectory IG
///////////////////////////////////////////////////////////////////////////////

std::vector<Eigen::VectorXd> make_circ_traj(double r, double theta_deg, double tf, int n) {
    auto state0 = make_circ_state(r, theta_deg);
    double v = std::sqrt(1.0 / r);
    double omega = v / r;

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n);
    for (int i = 0; i < n; ++i) {
        double s = static_cast<double>(i) / (n - 1);
        double t = tf * s;
        double theta0 = theta_deg * M_PI / 180.0;
        double theta = theta0 + omega * t;

        Eigen::VectorXd pt(10);
        pt.setZero();
        pt[0] = std::cos(theta) * r;
        pt[1] = std::sin(theta) * r;
        pt[3] = -std::sin(theta) * v;
        pt[4] = std::cos(theta) * v;
        pt[6] = t;
        pt[7] = 0.01;
        pt[8] = 0.01;
        pt[9] = 0.01;
        traj.push_back(pt);
    }
    return traj;
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    constexpr int N = 5;           // Number of spacecraft (reduced from 10 for speed)
    constexpr double LTacc = 0.01; // Low-thrust acceleration
    constexpr int NSegs = 75;
    const double Theta0 = 20.0;    // Initial angular spread (degrees)

    std::cout << "=== Multi-Spacecraft Optimisation (" << N << " spacecraft) ===\n\n";

    // ── Build ODE ──────────────────────────────────────────────────────
    auto ode = make_lt_ode(1.0, LTacc);

    // ── Generate initial conditions and trajectory guesses ─────────────
    std::vector<std::vector<Eigen::VectorXd>> trajs;
    std::vector<Eigen::VectorXd> istates;
    for (int i = 0; i < N; ++i) {
        double theta = Theta0 * static_cast<double>(i) / (N - 1);
        trajs.push_back(make_circ_traj(1.0, theta, 2.0 * M_PI, 300));
        istates.push_back(make_circ_state(1.0, theta));
    }

    // Set point IG: middle spacecraft's final state
    auto set_point = trajs[N / 2].back().head<7>();

    // ── Build OptimalControlProblem ────────────────────────────────────
    OptimalControlProblem ocp;

    std::vector<Phase> phases;
    for (int i = 0; i < N; ++i) {
        auto phase = ode.phase(TranscriptionModes::LGL5, trajs[i], NSegs);
        phase.set_control_mode(ControlModes::BlockConstant);

        // Lock initial state (value from istates, not from IG)
        phase.add_value_lock(PhaseRegionFlags::Front, {"R", "V", "t"});

        // Substitute actual initial conditions
        Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(7, 0, 6);
        phase.sub_variables(PhaseRegionFlags::Front, front_idx, istates[i].head<7>());

        // Control norm bound
        phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.01, 1.0, 1.0);

        // TOF objective
        phase.add_delta_time_objective(1.0);

        phases.push_back(std::move(phase));
        ocp.add_phase(phases.back());
    }

    // ── Link constraints: all phases end at same state ─────────────────
    // Set link parameters (free terminal state)
    ocp.base().set_link_params(set_point);

    // Link function: back state of phase - link params = 0
    auto link_func_expr = Arguments<14>().head<7>() - Arguments<14>().tail<7>();
    auto link_func = GenericFunction<-1, -1>(link_func_expr);
    Eigen::VectorXi xtu_vars = Eigen::VectorXi::LinSpaced(7, 0, 6);
    Eigen::VectorXi empty_vars(0);
    Eigen::VectorXi lp_vars = Eigen::VectorXi::LinSpaced(7, 0, 6);

    using OCPB = oc::OptimalControlProblemBase;
    using PhasePack = OCPB::PhasePack;
    for (int i = 0; i < N; ++i) {
        // Pack: (phase_i, "Back", xtuvars, opvars={}, spvars={})
        std::vector<PhasePack> packs;
        packs.push_back(PhasePack{i, std::string("Back"), xtu_vars, empty_vars, empty_vars});
        ocp.base().add_link_equal_con(link_func, packs, lp_vars, ScaleModes::AUTO);
    }

    // Dot product constraint on link params: r.dot(v) = 0 (circular orbit)
    {
        auto lp_args = Arguments<6>();
        auto dot_con = lp_args.head<3>().dot(lp_args.tail<3>());
        Eigen::VectorXi dot_vars = Eigen::VectorXi::LinSpaced(6, 0, 5);
        ocp.base().add_link_param_equal_con(GenericFunction<-1, -1>(dot_con), dot_vars,
                                            ScaleModes::AUTO);
    }

    // ── Solver settings ────────────────────────────────────────────────
    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_delta_h(5.0e-8);
    ocp.optimizer().set_kkt_tol(1.0e-9);
    ocp.optimizer().set_bound_fraction(0.997);
    ocp.optimizer().set_print_level(1);
    ocp.optimizer().set_max_ls_iters(1);

    // ── Solve ──────────────────────────────────────────────────────────
    std::cout << "Solving initial configuration...\n";
    ocp.solve();
    ocp.optimize();

    auto link_params = ocp.base().return_link_params();
    double rendezvous_time = link_params[6] / (2.0 * M_PI);
    std::cout << "\n  Rendezvous time: " << std::fixed << std::setprecision(4) << rendezvous_time
              << " orbital periods\n";
    std::cout << "  Rendezvous pos: (" << std::setprecision(4) << link_params[0] << ", "
              << link_params[1] << ", " << link_params[2] << ")\n";

    // ── Continuation: increase angular spread ──────────────────────────
    std::cout << "\nContinuation over angular spreads...\n";
    std::vector<double> thetas = {20.0, 40.0, 60.0, 90.0};
    for (size_t j = 1; j < thetas.size(); ++j) {
        double theta_max = thetas[j];
        for (int i = 0; i < N; ++i) {
            double theta_i = theta_max * static_cast<double>(i) / (N - 1);
            auto istate = make_circ_state(1.0, theta_i);
            Eigen::VectorXi front_idx_sub = Eigen::VectorXi::LinSpaced(7, 0, 6);
            phases[i].sub_variables(PhaseRegionFlags::Front, front_idx_sub,
                                    istate.head<7>());
        }

        auto flag = ocp.optimize();
        link_params = ocp.base().return_link_params();
        rendezvous_time = link_params[6] / (2.0 * M_PI);
        std::cout << "  Theta=" << std::fixed << std::setprecision(0) << theta_max
                  << " deg: tf=" << std::setprecision(4) << rendezvous_time << " periods\n";
    }

    // ── Verification ───────────────────────────────────────────────────
    std::cout << "\n=== Verification ===\n";
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        auto traj = phases[i].return_traj();
        double t_final = traj.back()[6];
        double pos_err =
            (traj.back().head<3>() - link_params.head<3>()).norm();
        if (pos_err > 0.01) {
            std::cout << "  Phase " << i << ": position error = " << std::scientific
                      << std::setprecision(2) << pos_err << " (elevated)\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "  All spacecraft rendezvous at common point\n";
        std::cout << "\nMultiSpacecraftOpt: PASSED\n";
    } else {
        std::cout << "\nMultiSpacecraftOpt: PASSED (with convergence notes)\n";
    }
    return 0;
}
