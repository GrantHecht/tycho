#include <tycho/tycho.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

/// Build a Phase for a single chain-length value L.
auto make_chain_phase(double a, double b, int n_segs, double L) {

    auto ode = ODEBuilder(1, 1)
                   .define([](auto &args) { return args.u_var(0); })
                   .var_names({{"x", 0}, {"t", 1}, {"u", 2}})
                   .build();

    auto energy_args = Arguments<2>();
    auto ex = energy_args.coeff<0>();
    auto eu = energy_args.coeff<1>();
    auto energy_expr = ex * sqrt(1.0 + eu * eu);

    auto len_args = Arguments<1>();
    auto lu = len_args.coeff<0>();
    auto length_expr = sqrt(1.0 + lu * lu);

    std::vector<Eigen::VectorXd> traj_ig;
    traj_ig.reserve(n_segs);
    const double tm = (b > a) ? 0.25 : 0.75;
    for (int i = 0; i < n_segs; ++i) {
        const double s = static_cast<double>(i) / (n_segs - 1);
        double x_val = 2.0 * std::abs(b - a) * s * (s - 2.0 * tm) + a;
        double u_val = 2.0 * std::abs(b - a) * (2.0 * s - 2.0 * tm);
        Eigen::VectorXd pt(3);
        pt << x_val, s, u_val;
        traj_ig.push_back(pt);
    }

    auto phase = ode.phase(TranscriptionModes::LGL5, traj_ig, n_segs);

    Eigen::VectorXd sp(1);
    sp[0] = L;
    phase.set_static_params(sp);

    phase.add_boundary_value(PhaseRegionFlags::Front, {"x", "t"}, Eigen::Vector2d(a, 0.0));
    phase.add_boundary_value(PhaseRegionFlags::Back, {"x", "t"}, Eigen::Vector2d(b, 1.0));

    phase.add_boundary_value(PhaseRegionFlags::StaticParams, Eigen::VectorXi::Constant(1, 0),
                             Eigen::Matrix<double, 1, 1>(L));

    phase.add_upper_var_bound(PhaseRegionFlags::Path, "x", std::max(a, b) + 0.001);

    phase.add_integral_objective(GenericFunction<-1, 1>(energy_expr), {"x", "u"});

    phase.add_integral_param_function(GenericFunction<-1, 1>(length_expr), {"u"}, 0);

    phase.optimizer().set_opt_ls_mode("L1");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_print_level(0);

    phase.set_jet_job_mode(solvers::OptimizationProblemBase::JetJobModes::SolveOptimize);

    return phase;
}

int main() {
    constexpr double a = 1.0; // left endpoint height
    constexpr double b = 3.0; // right endpoint height
    constexpr int n_segs = 500;
    constexpr int n_jobs = 100;

    std::vector<std::shared_ptr<ODEPhaseBase>> jobs;
    jobs.reserve(n_jobs);

    for (int i = 0; i < n_jobs; ++i) {
        double L = 2.25 + (8.0 - 2.25) * static_cast<double>(i) / (n_jobs - 1);
        auto phase = make_chain_phase(a, b, n_segs, L);
        jobs.push_back(phase.base_ptr());
    }

    auto results = solvers::Jet::map(jobs, true);

    int converged = 0;
    for (int i = 0; i < n_jobs; ++i) {
        auto phase_ptr = std::dynamic_pointer_cast<ODEPhaseBase>(results[i]);
        if (!phase_ptr)
            continue;
        auto traj = phase_ptr->return_traj();
        if (std::abs(traj.front()[0] - a) < 1e-3 && std::abs(traj.back()[0] - b) < 1e-3) {
            ++converged;
        }
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "HangingChain (builder): " << converged << "/" << n_jobs << " converged\n";

    if (converged == n_jobs) {
        std::cout << "HangingChain (builder): PASSED\n";
        return EXIT_SUCCESS;
    } else {
        std::cerr << "HangingChain (builder): FAILED — only " << converged << "/" << n_jobs
                  << " converged\n";
        return EXIT_FAILURE;
    }
}
