// Source: Junkins et al., AIAA J. Guidance (2019).

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

static constexpr double Isp_dim = 3000.0;   // s
static constexpr double Tmag_dim = 0.32;    // N
static constexpr double mass_dim = 4000.0;  // kg
static constexpr double g0 = 9.80665;       // m/s^2

// Non-dimensionalisation using AU and sun parameters
static constexpr double AU = 1.49597870691e11;   // m
static constexpr double MuSun = 1.32712440018e20; // m^3/s^2
static constexpr double day = 86400.0;            // s

static const double tf_dim = 3534.0 * day;

static const double Lstar = AU;
static const double Tstar = std::sqrt(Lstar * Lstar * Lstar / MuSun);
static const double Vstar = Lstar / Tstar;
static const double Mstar = mass_dim;
static const double Astar = Lstar / (Tstar * Tstar);
static const double Fstar = Mstar * Astar;

static const double mu = MuSun / (Lstar * Lstar * Lstar / (Tstar * Tstar));
static const double Thrust = Tmag_dim / Fstar;
static const double Isp = Isp_dim / Tstar;
static const double gs = g0 / Astar;
static const double tf = tf_dim / Tstar;

// CSI thruster non-dim mass flow magnitude: |dm/dt| = T/(Isp·g₀).
// Non-dim ratio sqrt's out to Thrust_nd / (Isp_nd · gs_nd).
static const double mdot_nd = Thrust / (Isp * gs);

int main() {
    auto args = ODEArguments(7, 3, 0);
    auto MEEs = args.head<6>();        // [p, f, g, h, k, L]
    auto m = args.coeff(6);            // mass (non-dim, m/Mstar)
    auto u_vec = args.segment<3>(8);   // RTN direction; ||u|| = 1 via path bound

    // Mass-state formulation: a_nd = T_nd / m_nd, no gs factor (the
    // weight-state formulation would carry one — keeping the two consistent
    // with mdot below is what makes the min-fuel objective well-posed).
    auto acc = (Thrust / m) * u_vec;

    // MEE rates via analytic astro::MEEDynamics (9 inputs → 6 rates).
    auto mee_dyn = astro::MEEDynamics(mu);
    auto Xdot = GenericFunction<-1, -1>(mee_dyn.eval(stack(MEEs, acc)));

    // CSI mass flow: dm/dt = -T * ||u|| / (Isp·g₀), non-dim -mdot_nd * ||u||.
    // Must scale with ||u|| — otherwise the min-fuel objective is degenerate
    // (mass depletion independent of control → solver runs full thrust always).
    auto mdot = (-mdot_nd) * u_vec.norm();

    auto ode_expr = StackedOutputs{Xdot, mdot};

    auto ode = ODE(ode_expr, 7, 3)
                   .var_names({{"p", 0},
                               {"f", 1},
                               {"g", 2},
                               {"h", 3},
                               {"k", 4},
                               {"L", 5},
                               {"m", 6},
                               {"t", 7}})
                   .var_group("u", 8, 3);

    // From Junkins paper
    Eigen::VectorXd X0_mee(6), XF_mee(6);
    X0_mee << 0.99969, -0.00376, 0.01628, -7.702e-6, 6.188e-7, 14.161;
    XF_mee << 1.5536, 0.15303, -0.51994, 0.01618, 0.11814, 46.3302;

    const int nPts = 500;
    std::vector<Eigen::VectorXd> trajIG;
    trajIG.reserve(nPts);
    for (int i = 0; i < nPts; ++i) {
        const double s = static_cast<double>(i) / (nPts - 1);
        Eigen::VectorXd state(11);
        state.setZero();
        state.head<6>() = X0_mee + (XF_mee - X0_mee) * s;
        state[6] = 1.0;
        state[7] = tf * s;
        state[9] = 0.5;
        trajIG.push_back(state);
    }

    constexpr int nSeg = 160;
    auto phase = ode.phase(TranscriptionModes::LGL5, trajIG, nSeg);
    phase.set_control_mode(ControlModes::BlockConstant);

    Eigen::VectorXd front_val(8);
    front_val << X0_mee[0], X0_mee[1], X0_mee[2], X0_mee[3], X0_mee[4], X0_mee[5], 1.0, 0.0;
    phase.add_boundary_value(PhaseRegionFlags::Front, {"p", "f", "g", "h", "k", "L", "m", "t"},
                             front_val);

    phase.add_lu_norm_bound(PhaseRegionFlags::Path, {"u"}, 0.000001, 1.0, 1.0);

    phase.add_boundary_value(PhaseRegionFlags::Back, "t", tf);
    Eigen::VectorXd back_mee(6);
    back_mee << XF_mee[0], XF_mee[1], XF_mee[2], XF_mee[3], XF_mee[4], XF_mee[5];
    phase.add_boundary_value(PhaseRegionFlags::Back, {"p", "f", "g", "h", "k", "L"}, back_mee);

    phase.add_value_objective(PhaseRegionFlags::Back, "m", -1.0);

    phase.set_num_partitions(8, 8);
    phase.optimizer().set_opt_ls_mode("AUGLANG");
    phase.optimizer().set_max_ls_iters(2);
    phase.optimizer().set_max_acc_iters(200);
    phase.optimizer().set_bound_fraction(0.997);
    phase.optimizer().set_print_level(1);
    phase.optimizer().set_delta_h(1.0e-6);
    phase.optimizer().set_econ_tol(1.0e-9);

    std::cout << "=== Dionysus Low Thrust: Earth-to-Asteroid Mass-Optimal Transfer ===\n";
    auto flag = phase.optimize();

    auto traj = phase.return_traj();
    double final_mass_nd = traj.back()[6];
    double final_mass_kg = final_mass_nd * mass_dim;
    double mass_expended = mass_dim - final_mass_kg;

    std::cout << "\n=== Results ===\n";
    std::cout << "  Final Mass:    " << std::fixed << std::setprecision(2) << final_mass_kg
              << " kg\n";
    std::cout << "  Mass Expended: " << std::fixed << std::setprecision(2) << mass_expended
              << " kg\n";

    // Reference final mass from the Python DionysusLowThrust.py example at
    // matching IG/mesh/tolerance. Update both sides if the Python example
    // changes.
    constexpr double kPythonFinalMassKg = 2716.62;
    const bool converged = flag <= PSIOPT::ConvergenceFlags::ACCEPTABLE &&
                           std::abs(final_mass_kg - kPythonFinalMassKg) < 1.0;

    if (converged) {
        std::cout << "\nDionysusLowThrust: PASSED\n";
        return EXIT_SUCCESS;
    }
    std::cerr << "\nDionysusLowThrust: FAILED (final_mass=" << final_mass_kg
              << " kg, expected " << kPythonFinalMassKg << " kg)\n";
    return EXIT_FAILURE;
}
