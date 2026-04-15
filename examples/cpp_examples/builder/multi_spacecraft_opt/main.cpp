#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

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

static std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> out(n);
    if (n == 1) {
        out[0] = a;
        return out;
    }
    for (int i = 0; i < n; ++i)
        out[i] = a + (b - a) * static_cast<double>(i) / (n - 1);
    return out;
}

static bool run_multi_spacecraft(const std::vector<std::vector<Eigen::VectorXd>> &trajs,
                                 const std::vector<std::vector<Eigen::VectorXd>> &all_istates,
                                 const Eigen::VectorXd &set_point_ig, double ltacc, int n_segs);

int main() {
    constexpr int N = 10;
    constexpr int NSegs = 75;

    std::cout << "=== Multi-Spacecraft Optimisation (" << N << " spacecraft) ===\n\n";

    const auto thetas = linspace(20.0, 180.0, 20);

    auto ig_thetas = linspace(0.0, thetas[0], N);
    std::vector<std::vector<Eigen::VectorXd>> trajs_ig;
    trajs_ig.reserve(N);
    for (double th : ig_thetas)
        trajs_ig.push_back(make_circ_traj(1.0, th, 2.0 * M_PI, 300));

    Eigen::VectorXd set_point_ig = trajs_ig[(N - 1) / 2].back().head<7>();

    std::vector<std::vector<Eigen::VectorXd>> all_istates;
    all_istates.reserve(thetas.size());
    for (double theta_max : thetas) {
        auto sweep = linspace(0.0, theta_max, N);
        std::vector<Eigen::VectorXd> istates;
        istates.reserve(N);
        for (double th : sweep)
            istates.push_back(make_circ_state(1.0, th));
        all_istates.push_back(std::move(istates));
    }

    const auto accs = linspace(0.015, 0.005, 2);
    for (double a : accs) {
        std::cout << "Running continuation at ltacc = " << std::fixed << std::setprecision(4) << a
                  << "\n";
        if (!run_multi_spacecraft(trajs_ig, all_istates, set_point_ig, a, NSegs)) {
            std::cerr << "MultiSpacecraftOpt: continuation at ltacc=" << a << " FAILED\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "\nMultiSpacecraftOpt: PASSED\n";
    return EXIT_SUCCESS;
}

static bool run_multi_spacecraft(const std::vector<std::vector<Eigen::VectorXd>> &trajs,
                                 const std::vector<std::vector<Eigen::VectorXd>> &all_istates,
                                 const Eigen::VectorXd &set_point_ig, double ltacc, int n_segs) {
    const int N = static_cast<int>(trajs.size());

    auto ode = make_lt_ode(1.0, ltacc);

    OptimalControlProblem ocp;

    // Reserve capacity so push_back does not invalidate references handed
    // to ocp.add_phase() — local access to phases[i] in the continuation
    // loop requires stable references.
    std::vector<Phase> phases;
    phases.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto phase = ode.phase(TranscriptionModes::LGL5, trajs[i], n_segs);
        phase.set_control_mode(ControlModes::BlockConstant);

        // Lock initial state at value from istates (not from IG).
        phase.add_value_lock(PhaseRegionFlags::Front, {"R", "V", "t"});

        phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.01, 1.0, 1.0);

        phase.add_delta_time_objective(1.0);

        phases.push_back(std::move(phase));
        ocp.add_phase(phases.back());
    }

    ocp.base().set_link_params(set_point_ig);

    // Link function: back state of phase - link params = 0
    auto link_func_expr = Arguments<14>().head<7>() - Arguments<14>().tail<7>();
    auto link_func = GenericFunction<-1, -1>(link_func_expr);
    Eigen::VectorXi xtu_vars = Eigen::VectorXi::LinSpaced(7, 0, 6);
    Eigen::VectorXi empty_vars(0);
    Eigen::VectorXi lp_vars = Eigen::VectorXi::LinSpaced(7, 0, 6);

    using OCPB = oc::OptimalControlProblemBase;
    using PhasePack = OCPB::PhasePack;
    for (int i = 0; i < N; ++i) {
        std::vector<PhasePack> packs;
        packs.push_back(PhasePack{i, std::string("Back"), xtu_vars, empty_vars, empty_vars});
        ocp.base().add_link_equal_con(link_func, packs, lp_vars, ScaleModes::AUTO);
    }

    // r.dot(v) = 0 (circular orbit) constraint on link params
    {
        auto lp_args = Arguments<6>();
        auto dot_con = lp_args.head<3>().dot(lp_args.tail<3>());
        Eigen::VectorXi dot_vars = Eigen::VectorXi::LinSpaced(6, 0, 5);
        ocp.base().add_link_param_equal_con(GenericFunction<-1, -1>(dot_con), dot_vars,
                                            ScaleModes::AUTO);
    }

    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_delta_h(5.0e-8);
    ocp.optimizer().set_kkt_tol(1.0e-9);
    ocp.optimizer().set_bound_fraction(0.997);
    ocp.optimizer().set_print_level(1);
    ocp.optimizer().set_max_ls_iters(1);

    // For each angular spread, substitute new front states, re-transcribe
    // every 8 iterates to keep the problem well conditioned, then
    // solve+optimize (first iterate) or optimize-with-solve-optimize-fallback.
    PSIOPT::ConvergenceFlags flag = PSIOPT::ConvergenceFlags::CONVERGED;
    Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(7, 0, 6);
    for (size_t j = 0; j < all_istates.size(); ++j) {
        const auto &istates = all_istates[j];
        for (int i = 0; i < N; ++i) {
            phases[i].sub_variables(PhaseRegionFlags::Front, front_idx, istates[i].head<7>());
        }

        if (j > 0 && j % 8 == 0) {
            ocp.base().transcribe(false, false);
        }

        if (j == 0) {
            auto solve_flag = ocp.solve();
            if (solve_flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
                std::cerr << "  MultiSpacecraftOpt: initial solve FAILED at j=0\n";
                return false;
            }
        }

        flag = ocp.optimize();
        if (flag == PSIOPT::ConvergenceFlags::NOTCONVERGED) {
            flag = ocp.solve_optimize();
        }

        auto link_params = ocp.base().return_link_params();
        double rendezvous_time = link_params[6] / (2.0 * M_PI);
        std::cout << "  j=" << std::setw(2) << j << " tf=" << std::fixed << std::setprecision(4)
                  << rendezvous_time << " periods\n";
    }

    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "  MultiSpacecraftOpt: final optimize status " << static_cast<int>(flag)
                  << "\n";
        return false;
    }
    return true;
}
