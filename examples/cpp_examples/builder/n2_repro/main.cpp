///////////////////////////////////////////////////////////////////////////////
// n2_repro — minimal 2-phase multi-spacecraft reproducer used to isolate
// the value_lock regression (LockArgs::compute_impl incorrectly writing
// `fx = x`, which turned the lock into a self-referential constraint and
// caused multi_spacecraft_opt to diverge).
//
// This is effectively an N=2 slice of the multi_spacecraft_opt example,
// kept as a smallest-possible regression test. It uses add_value_lock +
// sub_variables on the Front region, which is the specific code path that
// exercised the bug.
//
// Forces qp_threads=1 so Pardiso is deterministic when the binary is run
// under external tools (valgrind, ASan, debugger).
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include <cmath>
#include <cstdio>
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

Eigen::VectorXd circ_state(double r, double theta_deg) {
    double v = std::sqrt(1.0 / r);
    double t = theta_deg * M_PI / 180.0;
    Eigen::VectorXd X(7);
    X.setZero();
    X[0] = std::cos(t) * r;
    X[1] = std::sin(t) * r;
    X[3] = -std::sin(t) * v;
    X[4] = std::cos(t) * v;
    return X;
}

// Build the IG by integrating unforced two-body dynamics (matches the
// Python MultiSpacecraftOptimization.py reproducer exactly).
std::vector<Eigen::VectorXd> circ_traj(double r, double theta_deg, double tf, int n) {
    auto args = ODEArguments(6, 0, 0);
    auto rvec = args.head<3>();
    auto vvec = args.segment<3>(3);
    auto grav = (-1.0) * rvec.normalized_power<3>();
    auto ode_expr = StackedOutputs{vvec, grav};
    ODE ig_ode(ode_expr, 6, 0);

    auto integ = ig_ode.integrator(0.01);
    auto X0 = circ_state(r, theta_deg);
    auto dense = integ.integrate_dense(X0, tf, n);

    std::vector<Eigen::VectorXd> traj;
    traj.reserve(n);
    for (auto &pt7 : dense) {
        Eigen::VectorXd pt(10);
        pt.head<7>() = pt7;
        pt[7] = 0.01;
        pt[8] = 0.01;
        pt[9] = 0.01;
        traj.push_back(pt);
    }
    return traj;
}

int main() {
    constexpr int N = 2;
    constexpr double LTacc = 0.01;
    constexpr int NSegs = 75;
    constexpr double Theta0 = 20.0;

    auto ode = make_lt_ode(1.0, LTacc);

    std::vector<std::vector<Eigen::VectorXd>> trajs;
    std::vector<Eigen::VectorXd> istates;
    for (int i = 0; i < N; ++i) {
        double theta = Theta0 * static_cast<double>(i) / (N - 1);
        trajs.push_back(circ_traj(1.0, theta, 2.0 * M_PI, 300));
        istates.push_back(circ_state(1.0, theta));
    }
    auto set_point = trajs[0].back().head<7>();

    OptimalControlProblem ocp;
    std::vector<Phase> phases;
    phases.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto phase = ode.phase(TranscriptionModes::LGL5, trajs[i], NSegs);
        phase.set_control_mode(ControlModes::BlockConstant);
        phase.add_value_lock(PhaseRegionFlags::Front, {"R", "V", "t"});
        Eigen::VectorXi front_idx = Eigen::VectorXi::LinSpaced(7, 0, 6);
        phase.sub_variables(PhaseRegionFlags::Front, front_idx, istates[i].head<7>());
        phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.01, 1.0, 1.0);
        phase.add_delta_time_objective(1.0);
        phases.push_back(std::move(phase));
        ocp.add_phase(phases.back());
    }

    ocp.base().set_link_params(set_point);

    // Link function: phase back state == link params
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

    // r·v = 0 link-param constraint
    {
        auto lp_args = Arguments<6>();
        auto dot_con = lp_args.head<3>().dot(lp_args.tail<3>());
        Eigen::VectorXi dot_vars = Eigen::VectorXi::LinSpaced(6, 0, 5);
        ocp.base().add_link_param_equal_con(GenericFunction<-1, -1>(dot_con), dot_vars,
                                            ScaleModes::AUTO);
    }

    // Force single-threaded Pardiso for deterministic behavior under
    // sanitizers / valgrind / gdb. Otherwise MKL Pardiso's multi-threaded
    // factorization produces run-to-run variance that obscures real bugs.
    ocp.optimizer().set_qp_threads(1);

    auto flag = ocp.solve();
    auto lp = ocp.base().return_link_params();
    double posn = std::sqrt(lp[0] * lp[0] + lp[1] * lp[1] + lp[2] * lp[2]);
    std::printf("RESULT n=2 flag=%d tf_nd=%.6f pos_norm=%.6f\n", static_cast<int>(flag),
                lp[6], posn);

    // Regression check: convergence should match asset_asrl's answer on
    // the same problem (tf ~ 0.95-1.1 orbital periods, rendezvous on unit
    // circle).
    double tf_periods = lp[6] / (2.0 * M_PI);
    if (flag > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::fprintf(stderr, "n2_repro: FAILED (did not converge, flag=%d)\n",
                     static_cast<int>(flag));
        return 1;
    }
    if (tf_periods < 0.8 || tf_periods > 1.3 || std::abs(posn - 1.0) > 0.05) {
        std::fprintf(stderr,
                     "n2_repro: FAILED (unphysical solution: tf=%.4f periods, "
                     "pos_norm=%.4f)\n",
                     tf_periods, posn);
        return 1;
    }
    std::printf("n2_repro: PASSED\n");
    return 0;
}
