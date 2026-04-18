///////////////////////////////////////////////////////////////////////////////
// Delta III Launch Vehicle — C++ example (TU-split)
//
// Maximum payload-to-orbit for a multi-stage rocket.  Four phases model the
// staged burn sequence (6 SRBs + core, 3 SRBs + core, core only, upper stage).
// The control vector u(3) is the thrust direction; it is normalised inside the
// ODE so no unit-norm path constraint is needed.
//
// State  : [Rx, Ry, Rz, Vx, Vy, Vz, m]   (7)
// Control: [ux, uy, uz]                     (3)
//
// Concrete RocketODE VectorFunction + the target-orbit expression tree live
// in delta3_launch_ode.cpp behind GenericFunction<-1,-1> factories.  This TU
// builds four erased phases and links them via the Problem OCP wrapper.
//
// Corresponds to the Python example in examples/Delta3Launch.py.
///////////////////////////////////////////////////////////////////////////////

#include "delta3_launch_ode.h"

#include <tycho/tycho.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tycho;
using namespace tycho::oc;
using namespace tycho::solvers;
using namespace tycho::astro;

int main() {
    const auto &c = tycho_examples::delta3_constants();

    // Target orbital elements
    const double at = 24361140.0 / c.Lstar;
    const double et = 0.7308;
    const double Ot = 269.8 * M_PI / 180.0;
    const double Wt = 130.5 * M_PI / 180.0;
    const double istart = 28.5 * M_PI / 180.0;

    // Initial state: launch site on rotating Earth
    Eigen::VectorXd y0(6);
    y0.setZero();
    y0[0] = std::cos(istart) * c.Re;
    y0[1] = 0.0;
    y0[2] = std::sin(istart) * c.Re;

    // Velocity from Earth rotation: V = -R × omega
    Eigen::Vector3d R0 = y0.head<3>();
    Eigen::Vector3d omega_earth(0.0, 0.0, c.We);
    Eigen::Vector3d V0 = -R0.cross(omega_earth);
    V0[0] += 0.00001 / c.Vstar; // small perturbation to avoid zero Vr norm
    y0[3] = V0[0];
    y0[4] = V0[1];
    y0[5] = V0[2];

    // Target state from classical orbital elements
    Vector6<double> oelems;
    oelems << at, et, istart, Ot, Wt, -0.05; // last element = mean anomaly
    auto yf_vec = classic_to_cartesian(oelems, c.mu);
    Eigen::VectorXd yf = yf_vec;

    const int nPts = 1000;
    std::vector<Eigen::VectorXd> IG1, IG2, IG3, IG4;

    for (int i = 0; i < nPts; ++i) {
        const double t = c.tf_phase4 * static_cast<double>(i) / (nPts - 1);
        Eigen::VectorXd X(11);
        X.setZero();

        // Linearly interpolate position/velocity
        Eigen::VectorXd rv = y0 + (yf - y0) * (t / c.tf_phase4);
        X.head<6>() = rv;
        X[7] = t;
        X[8] = 0.0;
        X[9] = 1.0;
        X[10] = 0.0;

        if (t < c.tf_phase1) {
            X[6] = c.m0_phase1 + (c.mf_phase1 - c.m0_phase1) * (t / c.tf_phase1);
            IG1.push_back(X);
        } else if (t < c.tf_phase2) {
            X[6] = c.m0_phase2 +
                   (c.mf_phase2 - c.m0_phase2) * ((t - c.tf_phase1) / (c.tf_phase2 - c.tf_phase1));
            IG2.push_back(X);
        } else if (t < c.tf_phase3) {
            X[6] = c.m0_phase3 +
                   (c.mf_phase3 - c.m0_phase3) * ((t - c.tf_phase2) / (c.tf_phase3 - c.tf_phase2));
            IG3.push_back(X);
        } else {
            X[6] = c.m0_phase4 +
                   (c.mf_phase4 - c.m0_phase4) * ((t - c.tf_phase3) / (c.tf_phase4 - c.tf_phase3));
            IG4.push_back(X);
        }
    }

    // Erased ODEs for each phase
    ODE ode1(tycho_examples::make_rocket_ode(c.T_phase1, c.mdot_phase1), 7, 3, 0);
    ODE ode2(tycho_examples::make_rocket_ode(c.T_phase2, c.mdot_phase2), 7, 3, 0);
    ODE ode3(tycho_examples::make_rocket_ode(c.T_phase3, c.mdot_phase3), 7, 3, 0);
    ODE ode4(tycho_examples::make_rocket_ode(c.T_phase4, c.mdot_phase4), 7, 3, 0);

    constexpr int nSegs = 40;
    const auto tmode = TranscriptionModes::LGL3;

    // Helper for VectorXi construction
    auto make_idx = [](std::initializer_list<int> vals) {
        Eigen::VectorXi v(vals.size());
        int i = 0;
        for (auto val : vals) v[i++] = val;
        return v;
    };

    // Phase index arrays used repeatedly
    auto pos_idx = make_idx({0, 1, 2});
    auto ctrl_idx = make_idx({8, 9, 10});
    auto rv_idx = make_idx({0, 1, 2, 3, 4, 5});

    ///////////////////////////////////////////////////////////////////////////
    // Phase 1: 6 SRBs + core
    ///////////////////////////////////////////////////////////////////////////
    auto phase1 = ode1.phase(tmode, IG1, nSegs);
    phase1.set_control_mode(ControlModes::HighestOrderSpline);

    phase1.add_lu_norm_bound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0, ScaleModes::AUTO);

    // Front BC: full state [R, V, m, t]
    auto front_idx_8 = make_idx({0, 1, 2, 3, 4, 5, 6, 7});
    Eigen::VectorXd front_val_8(8);
    front_val_8 << y0[0], y0[1], y0[2], y0[3], y0[4], y0[5], c.m0_phase1, 0.0;
    phase1.add_boundary_value(PhaseRegionFlags::Front, front_idx_8, front_val_8, ScaleModes::AUTO);

    // Altitude above (slightly relaxed) Earth radius
    phase1.add_lower_norm_bound(PhaseRegionFlags::Path, pos_idx, c.Re * 0.999999, 1.0,
                                ScaleModes::AUTO);

    // Back BC: final time
    Eigen::VectorXi t_idx(1);
    t_idx << 7;
    Eigen::VectorXd tf1_val(1);
    tf1_val << c.tf_phase1;
    phase1.add_boundary_value(PhaseRegionFlags::Back, t_idx, tf1_val, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 2: 3 SRBs + core
    ///////////////////////////////////////////////////////////////////////////
    auto phase2 = ode2.phase(tmode, IG2, nSegs);
    phase2.set_control_mode(ControlModes::HighestOrderSpline);

    phase2.add_lower_norm_bound(PhaseRegionFlags::Path, pos_idx, c.Re, 1.0, ScaleModes::AUTO);
    phase2.add_lu_norm_bound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0, ScaleModes::AUTO);

    Eigen::VectorXi m_idx(1);
    m_idx << 6;
    Eigen::VectorXd m0_p2_val(1);
    m0_p2_val << c.m0_phase2;
    phase2.add_boundary_value(PhaseRegionFlags::Front, m_idx, m0_p2_val, ScaleModes::AUTO);

    Eigen::VectorXd tf2_val(1);
    tf2_val << c.tf_phase2;
    phase2.add_boundary_value(PhaseRegionFlags::Back, t_idx, tf2_val, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 3: core only
    ///////////////////////////////////////////////////////////////////////////
    auto phase3 = ode3.phase(tmode, IG3, nSegs);
    phase3.set_control_mode(ControlModes::HighestOrderSpline);

    phase3.add_lower_norm_bound(PhaseRegionFlags::Path, pos_idx, c.Re, 1.0, ScaleModes::AUTO);
    phase3.add_lu_norm_bound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0, ScaleModes::AUTO);

    Eigen::VectorXd m0_p3_val(1);
    m0_p3_val << c.m0_phase3;
    phase3.add_boundary_value(PhaseRegionFlags::Front, m_idx, m0_p3_val, ScaleModes::AUTO);

    Eigen::VectorXd tf3_val(1);
    tf3_val << c.tf_phase3;
    phase3.add_boundary_value(PhaseRegionFlags::Back, t_idx, tf3_val, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Phase 4: upper stage
    ///////////////////////////////////////////////////////////////////////////
    auto phase4 = ode4.phase(tmode, IG4, nSegs);
    phase4.set_control_mode(ControlModes::HighestOrderSpline);

    phase4.add_lower_norm_bound(PhaseRegionFlags::Path, pos_idx, c.Re, 1.0, ScaleModes::AUTO);
    phase4.add_lu_norm_bound(PhaseRegionFlags::Path, ctrl_idx, 0.5, 1.5, 1.0, ScaleModes::AUTO);

    Eigen::VectorXd m0_p4_val(1);
    m0_p4_val << c.m0_phase4;
    phase4.add_boundary_value(PhaseRegionFlags::Front, m_idx, m0_p4_val, ScaleModes::AUTO);

    // Upper bound on final time
    phase4.add_upper_var_bound(PhaseRegionFlags::Back, 7, c.tf_phase4, 1.0, ScaleModes::AUTO);

    // Terminal orbital element constraint — returned pre-erased from the ODE TU.
    auto target_orbit_fn = tycho_examples::make_target_orbit_residuals(at, et, istart, Ot, Wt);
    phase4.add_equal_con(PhaseRegionFlags::Back, target_orbit_fn, rv_idx, ScaleModes::AUTO);

    // Maximize final mass (minimise -m)
    phase4.add_value_objective(PhaseRegionFlags::Back, 6, -1.0, ScaleModes::AUTO);

    ///////////////////////////////////////////////////////////////////////////
    // Optimal Control Problem — link phases
    ///////////////////////////////////////////////////////////////////////////
    OptimalControlProblem ocp;
    ocp.add_phase(phase1);
    ocp.add_phase(phase2);
    ocp.add_phase(phase3);
    ocp.add_phase(phase4);

    // Continuity in everything except mass (var 6)
    auto link_vars = make_idx({0, 1, 2, 3, 4, 5, 7, 8, 9, 10});
    ocp.add_forward_link_equal_con(phase1, phase4, link_vars);

    ocp.optimizer().set_opt_ls_mode("L1");
    ocp.optimizer().set_soe_ls_mode("L1");
    ocp.optimizer().set_max_ls_iters(2);
    ocp.optimizer().set_print_level(1);

    ///////////////////////////////////////////////////////////////////////////
    // Solve
    ///////////////////////////////////////////////////////////////////////////
    std::cout << "Solving Delta III launch vehicle problem ...\n" << std::flush;

    const auto status = ocp.solve_optimize();

    if (status > PSIOPT::ConvergenceFlags::ACCEPTABLE) {
        std::cerr << "Delta3Launch: FAILED (status " << static_cast<int>(status) << ")\n";
        return 1;
    }

    auto traj4 = phase4.return_traj();
    if (traj4.empty()) {
        std::cerr << "Delta3Launch: solver converged but trajectory is empty\n";
        return 1;
    }

    const double final_mass_kg = traj4.back()[6] * c.Mstar;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Delta3Launch: Final mass = " << final_mass_kg << " kg\n";

    return 0;
}
